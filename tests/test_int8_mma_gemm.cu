/* test_int8_mma_gemm.cu — INT8 tensor-core slice GEMM vs BF16 GEMM on REAL weight matrices.
 *
 * Weights: layer-0 q/k/v/o/gate/up/down of a real BF16 safetensors model, quantized here with the
 * pin rule (symmetric, 64-block over the flattened tensor, scale = amax/127, rint, clamp +-127).
 * Inputs: RMSNorm(embedding rows of real tokens) for q/k/v/gate/up, and silu(gate)*up from the
 * BF16 reference for down. These are real-weight-derived, not captured activations (that needs
 * piece B).
 *
 * Reported per matrix, all against the BF16 GEMM (bf16 X, bf16 W, double accumulate):
 *   emu  : kernel vs a host emulation of the same W8A8 arithmetic  -> kernel CORRECTNESS gate
 *   w8   : INT8 weights dequantized, bf16 X                         -> weight-quantization floor
 *   w8a8 : the kernel                                                -> what this path costs in fidelity
 *
 * Build: nvcc -O2 -arch=sm_86 -Iinclude tests/test_int8_mma_gemm.cu cuda/int8_mma_gemm.cu
 * Run:   ./a.out [bf16 model.safetensors]
 */
#include "int8_mma_gemm.h"
#include "safetensors_c.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { /* round-to-nearest-even, matches __float2bfloat16_rn for finite values */
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static const TensorEntry *need(const SafeTensors *st, const char *name) {
    const TensorEntry *e = safetensors_find(st, name);
    if (!e) { printf("missing tensor %s\n", name); exit(2); }
    if (strcmp(e->dtype, "BF16") != 0 || e->n_dims != 2) { printf("%s is not a 2-D BF16 tensor\n", name); exit(2); }
    return e;
}

static void quant_w(const uint16_t *w, int64_t n, std::vector<int8_t> &q, std::vector<float> &sc) {
    int64_t nb = (n + 63) / 64;
    q.assign(n, 0); sc.assign(nb, 1.f);
    for (int64_t b = 0; b < nb; b++) {
        float am = 0.f;
        for (int64_t i = b * 64; i < n && i < (b + 1) * 64; i++) am = fmaxf(am, fabsf(bf2f(w[i])));
        float s = am > 0.f ? am / 127.f : 1.f;
        sc[b] = s;
        for (int64_t i = b * 64; i < n && i < (b + 1) * 64; i++) {
            int v = (int)rintf(bf2f(w[i]) / s);
            q[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
    }
}

struct Err { double rel, maxabs; };
static Err cmp(const std::vector<double> &ref, const std::vector<float> &got) {
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        double d = (double)got[i] - ref[i];
        num += d * d; den += ref[i] * ref[i];
        mx = fmax(mx, fabs(d));
    }
    return { sqrt(num / (den > 0 ? den : 1)), mx };
}

/* Run one matrix. w: BF16 [n,k]; x: bf16 bits [m,k]. Returns the BF16 reference (double) in `yref`. */
static int run_matrix(const char *label, const uint16_t *w, int n, int k, const std::vector<uint16_t> &x, int m,
                      std::vector<double> &yref, double *emu_rel, double *w8_rel, double *w8a8_rel) {
    std::vector<int8_t> q; std::vector<float> sc;
    quant_w(w, (int64_t)n * k, q, sc);
    int nkb = k / 64;

    std::vector<double> y_bf((size_t)m * n), y_w8((size_t)m * n);
    std::vector<float> wd((size_t)n * k);
    for (int64_t i = 0; i < (int64_t)n * k; i++) wd[i] = (float)q[i] * sc[i / 64];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double a = 0, b = 0;
            for (int kk = 0; kk < k; kk++) {
                double xv = bf2f(x[(size_t)i * k + kk]);
                a += xv * bf2f(w[(size_t)j * k + kk]);
                b += xv * wd[(size_t)j * k + kk];
            }
            y_bf[(size_t)i * n + j] = a; y_w8[(size_t)i * n + j] = b;
        }

    /* host emulation of the exact W8A8 arithmetic */
    std::vector<int8_t> xq((size_t)m * k); std::vector<float> sx((size_t)m * nkb);
    for (int i = 0; i < m; i++)
        for (int kb = 0; kb < nkb; kb++) {
            float am = 0;
            for (int t = 0; t < 64; t++) am = fmaxf(am, fabsf(bf2f(x[(size_t)i * k + kb * 64 + t])));
            float s = am > 0 ? am / 127.f : 1.f;
            sx[(size_t)i * nkb + kb] = s;
            for (int t = 0; t < 64; t++) {
                int v = (int)rintf(bf2f(x[(size_t)i * k + kb * 64 + t]) / s);
                xq[(size_t)i * k + kb * 64 + t] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
            }
        }
    std::vector<double> y_emu((size_t)m * n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0.f;
            for (int kb = 0; kb < nkb; kb++) {
                int ic = 0;
                for (int t = 0; t < 64; t++) ic += (int)xq[(size_t)i * k + kb * 64 + t] * (int)q[(size_t)j * k + kb * 64 + t];
                acc += ((float)ic * sx[(size_t)i * nkb + kb]) * sc[(size_t)j * nkb + kb];
            }
            y_emu[(size_t)i * n + j] = acc;
        }

    /* device */
    uint16_t *d_xh; int8_t *d_q, *d_xq; float *d_s, *d_sx, *d_y;
    CK(cudaMalloc(&d_xh, x.size() * 2)); CK(cudaMemcpy(d_xh, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_q, q.size())); CK(cudaMemcpy(d_q, q.data(), q.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_s, sc.size() * 4)); CK(cudaMemcpy(d_s, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_xq, int8_mma_xq_bytes(m, k))); CK(cudaMalloc(&d_sx, int8_mma_sx_bytes(m, k)));
    CK(cudaMalloc(&d_y, (size_t)m * n * 4)); CK(cudaMemset(d_y, 0xFF, (size_t)m * n * 4));
    Int8MmaLinear lin = { d_q, d_s, n, k };
    CK(int8_mma_gemm((const __nv_bfloat16 *)d_xh, lin, d_y, m, d_xq, d_sx, 0));
    CK(cudaDeviceSynchronize());
    std::vector<float> y((size_t)m * n);
    CK(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost));

    /* activation quantization must match the host rule byte for byte */
    std::vector<int8_t> gxq((size_t)m * k);
    CK(cudaMemcpy(gxq.data(), d_xq, gxq.size(), cudaMemcpyDeviceToHost));
    size_t xq_bad = 0;
    for (size_t i = 0; i < gxq.size(); i++) xq_bad += gxq[i] != xq[i];

    Err e_emu = cmp(y_emu, y), e_w8 = cmp(y_bf, std::vector<float>(y_w8.begin(), y_w8.end())), e_k = cmp(y_bf, y);
    printf("%-10s n=%5d k=%5d m=%d | xq bytes differing=%zu | kernel vs emu rel=%.2e max|d|=%.2e | vs BF16: w8 floor rel=%.3e  w8a8 rel=%.3e\n",
           label, n, k, m, xq_bad, e_emu.rel, e_emu.maxabs, e_w8.rel, e_k.rel);
    *emu_rel = e_emu.rel; *w8_rel = e_w8.rel; *w8a8_rel = e_k.rel;
    yref = y_bf;
    cudaFree(d_xh); cudaFree(d_q); cudaFree(d_s); cudaFree(d_xq); cudaFree(d_sx); cudaFree(d_y);
    return xq_bad == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/workspace/pins/DeepSeek-R1-Distill-Qwen-1.5B-BF16/model.safetensors";
    printf("=== TEST INT8 MMA GEMM vs BF16 (real weights) ===\nfile: %s\n", path);
    SafeTensors st;
    if (safetensors_open(&st, path) != 0) { printf("open failed\n"); return 2; }

    const TensorEntry *emb = need(&st, "model.embed_tokens.weight");
    const TensorEntry *ln = safetensors_find(&st, "model.layers.0.input_layernorm.weight");
    if (!ln) { printf("missing input_layernorm\n"); return 2; }
    int hid = (int)emb->shape[1];
    const int m = 37; /* deliberately not a multiple of 64 */
    std::vector<uint16_t> x((size_t)m * hid);
    for (int i = 0; i < m; i++) {
        int tok = (int)((i * 2654435761u + 12345u) % (uint32_t)emb->shape[0]);
        const uint16_t *row = (const uint16_t *)emb->data + (size_t)tok * hid;
        double ss = 0; for (int j = 0; j < hid; j++) { double v = bf2f(row[j]); ss += v * v; }
        float inv = 1.f / sqrtf((float)(ss / hid) + 1e-6f);
        const uint16_t *lw = (const uint16_t *)ln->data;
        for (int j = 0; j < hid; j++) x[(size_t)i * hid + j] = f2bf(bf2f(row[j]) * inv * bf2f(lw[j]));
    }

    int fails = 0;
    double worst_emu = 0;
    struct { const char *name; const char *label; } mats[] = {
        {"model.layers.0.self_attn.q_proj.weight", "q_proj"}, {"model.layers.0.self_attn.k_proj.weight", "k_proj"},
        {"model.layers.0.self_attn.v_proj.weight", "v_proj"}, {"model.layers.0.self_attn.o_proj.weight", "o_proj"},
        {"model.layers.0.mlp.gate_proj.weight", "gate_proj"}, {"model.layers.0.mlp.up_proj.weight", "up_proj"},
    };
    std::vector<double> ref_gate, ref_up, tmp;
    for (auto &mt : mats) {
        const TensorEntry *e = need(&st, mt.name);
        double a, b, c;
        fails += run_matrix(mt.label, (const uint16_t *)e->data, (int)e->shape[0], (int)e->shape[1], x, m, tmp, &a, &b, &c);
        worst_emu = fmax(worst_emu, a);
        if (!strcmp(mt.label, "gate_proj")) ref_gate = tmp;
        if (!strcmp(mt.label, "up_proj")) ref_up = tmp;
    }
    /* ragged n: first 100 rows of q_proj */
    {
        const TensorEntry *e = need(&st, mats[0].name);
        double a, b, c;
        fails += run_matrix("q_proj[:100]", (const uint16_t *)e->data, 100, (int)e->shape[1], x, m, tmp, &a, &b, &c);
        worst_emu = fmax(worst_emu, a);
    }
    /* down_proj on silu(gate)*up from the BF16 reference */
    {
        const TensorEntry *e = need(&st, "model.layers.0.mlp.down_proj.weight");
        int inter = (int)e->shape[1];
        std::vector<uint16_t> xd((size_t)m * inter);
        for (size_t i = 0; i < xd.size(); i++) {
            double gt = ref_gate[i];
            xd[i] = f2bf((float)(gt / (1.0 + exp(-gt)) * ref_up[i]));
        }
        double a, b, c;
        fails += run_matrix("down_proj", (const uint16_t *)e->data, (int)e->shape[0], inter, xd, m, tmp, &a, &b, &c);
        worst_emu = fmax(worst_emu, a);
    }

    /* misuse is refused */
    {
        Int8MmaLinear bad = { (const int8_t *)1, (const float *)1, 64, 96 };
        cudaError_t r = int8_mma_gemm((const __nv_bfloat16 *)1, bad, (float *)1, 1, (int8_t *)1, (float *)1, 0);
        if (r != cudaErrorInvalidValue) { printf("FAIL: k_in %% 64 != 0 not refused\n"); fails++; }
    }

    const double EMU_TOL = 1e-5; /* pre-registered: same arithmetic, only fp32 contraction order may differ */
    if (worst_emu > EMU_TOL) { printf("FAIL: worst kernel-vs-emulation rel err %.2e > %.0e\n", worst_emu, EMU_TOL); fails++; }
    safetensors_close(&st);
    if (fails) { printf("INT8 MMA GEMM: FAILED (%d)\n", fails); return 1; }
    printf("INT8 MMA GEMM PASSED (kernel correctness). Fidelity vs BF16 is reported above, not gated.\n");
    return 0;
}
