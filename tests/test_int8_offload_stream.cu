/* test_int8_offload_stream.cu — Mode 2 (INT8 + CPU offload, issue #18): INT8 layers live in pinned host RAM,
 * stream through the 2-slot VRAM ring, and run on int8_mma_gemm. The streamed outputs must equal the
 * all-resident outputs bit for bit; timing and stall are measured, not asserted.
 *
 * Weights: every layer's q/k/v/o/gate/up/down of a real BF16 safetensors model, quantized with the pin rule
 * (64-block over the flattened tensor, scale = amax/127, rint, clamp +-127). Inputs are RMSNorm(embedding rows
 * of real tokens) for q/k/v/gate/up/o-shaped k=hid matrices; down takes the first `inter` values formed from
 * the gate GEMM output of the same layer. This tests transport + kernel equivalence, NOT model fidelity (that
 * needs the full forward pass).
 *
 * Build: nvcc -O2 -arch=sm_86 -Iinclude tests/test_int8_offload_stream.cu cuda/int8_mma_gemm.cu cuda/glint_plug_streamer.cu
 * Run:   ./a.out [bf16 model.safetensors]
 */
#include "glint_plug_streamer.h"
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
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFFu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }

enum { NM = 7 };
static const char *kMat[NM] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"};

struct Mat { int n, k; uint64_t off_q, off_s; };
struct LayerLayout { Mat m[NM]; uint64_t bytes; };

static uint64_t align256(uint64_t v) { return (v + 255) & ~255ull; }

static void quant_into(const uint16_t *w, int64_t n, int8_t *q, float *sc) {
    int64_t nb = (n + 63) / 64;
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

static const TensorEntry *need(const SafeTensors *st, const char *name) {
    const TensorEntry *e = safetensors_find(st, name);
    if (!e || strcmp(e->dtype, "BF16") != 0 || e->n_dims != 2) { printf("missing/non-BF16-2D tensor %s\n", name); exit(2); }
    return e;
}

__global__ void k_mk_down(const float *g, const float *u, __nv_bfloat16 *o, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float gv = g[i]; o[i] = __float2bfloat16_rn(gv / (1.f + expf(-gv)) * u[i]); }
}

/* Runs the 7 GEMMs of one layer from a device blob; y[i] are device outputs of size m*n_i. */
static void run_layer(const uint8_t *blob, const LayerLayout &L, const __nv_bfloat16 *d_x, __nv_bfloat16 *d_xd,
                      float *const *d_y, int m, int8_t *xq, float *sx, cudaStream_t s) {
    for (int i = 0; i < NM; i++) {
        Int8MmaLinear lin = { (const int8_t *)(blob + L.m[i].off_q), (const float *)(blob + L.m[i].off_s), L.m[i].n, L.m[i].k };
        const __nv_bfloat16 *x = (i == 6) ? d_xd : d_x;
        CK(int8_mma_gemm(x, lin, d_y[i], m, xq, sx, s));
        if (i == 5) { /* gate (i=4) and up (i=5) done: build down's input = bf16(silu(gate)*up), on device */
            size_t nel = (size_t)m * L.m[6].k;
            k_mk_down<<<(unsigned)((nel + 255) / 256), 256, 0, s>>>(d_y[4], d_y[5], d_xd, nel);
            CK(cudaGetLastError());
        }
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/workspace/pins/DeepSeek-R1-Distill-Qwen-1.5B-BF16/model.safetensors";
    printf("=== TEST INT8 OFFLOAD STREAM (Mode 2) ===\nfile: %s\n", path);
    SafeTensors st;
    if (safetensors_open(&st, path) != 0) { printf("open failed\n"); return 2; }

    int nl = 0;
    char nm[256];
    for (;; nl++) { snprintf(nm, sizeof nm, "model.layers.%d.self_attn.q_proj.weight", nl); if (!safetensors_find(&st, nm)) break; }
    if (nl < 2 || nl > 64) { printf("unexpected layer count %d\n", nl); return 2; }

    const TensorEntry *emb = need(&st, "model.embed_tokens.weight");
    int hid = (int)emb->shape[1];
    const int m = 37;

    /* layout from layer 0 (all layers share shapes) */
    LayerLayout L; uint64_t off = 0;
    for (int i = 0; i < NM; i++) {
        snprintf(nm, sizeof nm, "model.layers.0.%s.weight", kMat[i]);
        const TensorEntry *e = need(&st, nm);
        L.m[i].n = (int)e->shape[0]; L.m[i].k = (int)e->shape[1];
        if (L.m[i].k % 64) { printf("k not multiple of 64\n"); return 2; }
        L.m[i].off_q = off; off = align256(off + (uint64_t)L.m[i].n * L.m[i].k);
        L.m[i].off_s = off; off = align256(off + (uint64_t)L.m[i].n * L.m[i].k / 64 * 4);
    }
    L.bytes = off;
    printf("layers=%d hid=%d inter=%d layer blob=%.1f MiB total INT8=%.0f MiB\n", nl, hid, L.m[6].k, L.bytes / 1048576.0, L.bytes * (double)nl / 1048576.0);

    /* pinned host copies of every layer's blob */
    std::vector<uint8_t *> host(nl);
    std::vector<uint64_t> lb(nl, L.bytes);
    for (int l = 0; l < nl; l++) {
        CK(cudaMallocHost(&host[l], L.bytes));
        memset(host[l], 0, L.bytes);
        for (int i = 0; i < NM; i++) {
            snprintf(nm, sizeof nm, "model.layers.%d.%s.weight", l, kMat[i]);
            const TensorEntry *e = need(&st, nm);
            if ((int)e->shape[0] != L.m[i].n || (int)e->shape[1] != L.m[i].k) { printf("shape mismatch %s\n", nm); return 2; }
            quant_into((const uint16_t *)e->data, (int64_t)L.m[i].n * L.m[i].k, (int8_t *)(host[l] + L.m[i].off_q), (float *)(host[l] + L.m[i].off_s));
        }
    }

    /* real-token activations: RMSNorm(embedding rows) */
    std::vector<uint16_t> x((size_t)m * hid);
    snprintf(nm, sizeof nm, "model.layers.0.input_layernorm.weight");
    const TensorEntry *ln = safetensors_find(&st, nm);
    if (!ln) { printf("missing input_layernorm\n"); return 2; }
    for (int i = 0; i < m; i++) {
        int tok = (int)((i * 2654435761u + 12345u) % (uint32_t)emb->shape[0]);
        const uint16_t *row = (const uint16_t *)emb->data + (size_t)tok * hid;
        double ss = 0; for (int j = 0; j < hid; j++) { double v = bf2f(row[j]); ss += v * v; }
        float inv = 1.f / sqrtf((float)(ss / hid) + 1e-6f);
        const uint16_t *lw = (const uint16_t *)ln->data;
        for (int j = 0; j < hid; j++) x[(size_t)i * hid + j] = f2bf(bf2f(row[j]) * inv * bf2f(lw[j]));
    }

    cudaStream_t cs, ks;
    CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking));
    __nv_bfloat16 *d_x, *d_xd; int8_t *d_xq; float *d_sx; float *d_y[NM];
    CK(cudaMalloc(&d_x, x.size() * 2)); CK(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_xd, (size_t)m * L.m[6].k * 2));
    int kmax = 0; for (int i = 0; i < NM; i++) kmax = L.m[i].k > kmax ? L.m[i].k : kmax;
    CK(cudaMalloc(&d_xq, int8_mma_xq_bytes(m, kmax))); CK(cudaMalloc(&d_sx, int8_mma_sx_bytes(m, kmax)));
    for (int i = 0; i < NM; i++) CK(cudaMalloc(&d_y[i], (size_t)m * L.m[i].n * 4));

    size_t ytot = 0; for (int i = 0; i < NM; i++) ytot += (size_t)m * L.m[i].n;
    std::vector<float> y_res((size_t)nl * ytot), y_str((size_t)nl * ytot);
    auto fetch = [&](std::vector<float> &dst, int l) {
        size_t o = (size_t)l * ytot;
        for (int i = 0; i < NM; i++) { size_t c = (size_t)m * L.m[i].n; CK(cudaMemcpy(dst.data() + o, d_y[i], c * 4, cudaMemcpyDeviceToHost)); o += c; }
    };

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    /* ---- resident reference: every layer on the device ---- */
    std::vector<uint8_t *> dev(nl);
    for (int l = 0; l < nl; l++) { CK(cudaMalloc(&dev[l], L.bytes)); CK(cudaMemcpy(dev[l], host[l], L.bytes, cudaMemcpyHostToDevice)); }
    CK(cudaEventRecord(e0, ks));
    for (int l = 0; l < nl; l++) { run_layer(dev[l], L, d_x, d_xd, d_y, m, d_xq, d_sx, ks); CK(cudaStreamSynchronize(ks)); fetch(y_res, l); }
    CK(cudaEventRecord(e1, ks)); CK(cudaEventSynchronize(e1));
    float t_res_ms; CK(cudaEventElapsedTime(&t_res_ms, e0, e1));
    for (int l = 0; l < nl; l++) cudaFree(dev[l]);

    /* ---- streamed: only the 2-slot ring is on the device ---- */
    GlintPlugStreamer *s = NULL;
    if (glint_plug_streamer_create(&s, nl, lb.data(), cs, ks) != 0) { printf("streamer create failed\n"); return 2; }
    for (int l = 0; l < nl; l++) if (glint_plug_streamer_set_host_data(s, l, host[l], L.bytes) != 0) { printf("set_host_data %d failed\n", l); return 2; }
    if (glint_plug_streamer_start(s) != 0) { printf("start failed\n"); return 2; }
    CK(cudaEventRecord(e0, ks));
    for (int l = 0; l < nl; l++) {
        const uint8_t *slot = NULL;
        int rc = glint_plug_streamer_acquire(s, l, &slot);
        if (rc != 0) { printf("acquire %d rc=%d\n", l, rc); return 2; }
        run_layer(slot, L, d_x, d_xd, d_y, m, d_xq, d_sx, ks);
        rc = glint_plug_streamer_release(s, l);
        if (rc != 0) { printf("release %d rc=%d\n", l, rc); return 2; }
        CK(cudaStreamSynchronize(ks)); fetch(y_str, l);
    }
    CK(cudaEventRecord(e1, ks)); CK(cudaEventSynchronize(e1));
    float t_str_ms; CK(cudaEventElapsedTime(&t_str_ms, e0, e1));
    GlintStreamerStats stats; glint_plug_streamer_get_stats(s, &stats);

    size_t bad = 0, nonfinite = 0, zero_layers = 0;
    for (size_t i = 0; i < y_res.size(); i++) { bad += memcmp(&y_res[i], &y_str[i], 4) != 0; nonfinite += !isfinite(y_res[i]); }
    for (int l = 0; l < nl; l++) { double a = 0; for (size_t i = 0; i < ytot; i++) a += fabs(y_res[(size_t)l * ytot + i]); zero_layers += a == 0; }
    int distinct = memcmp(y_res.data(), y_res.data() + ytot, ytot * 4) != 0;

    printf("outputs compared: %zu floats | bitwise mismatches=%zu | non-finite=%zu | all-zero layers=%zu | layer0!=layer1: %d\n",
           y_res.size(), bad, nonfinite, zero_layers, distinct);
    printf("resident %.2f ms | streamed %.2f ms | streamed bytes=%llu | stall total=%.3f ms (layer0 %.3f)\n",
           t_res_ms, t_str_ms, (unsigned long long)stats.total_h2d_bytes, stats.total_stall_ms, stats.per_layer_stall_ms[0]);
    printf("VRAM weights resident=%.0f MiB vs streamed ring=%.0f MiB\n", L.bytes * (double)nl / 1048576.0, 2.0 * L.bytes / 1048576.0);

    glint_plug_streamer_destroy(s);
    int ok = bad == 0 && nonfinite == 0 && zero_layers == 0 && distinct;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
