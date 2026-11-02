/* tests/test_int8_tile_gemm.cu: int8_tile_gemm_bf16 against references on the dequantized weights.
 * Synthetic shapes cover M, N and K tails, odd K and scale blocks that straddle rows; they are
 * checked against a host double GEMM. With a pin path, real Phi-4-mini linears are checked against
 * the host double GEMM for small M and against cuBLAS on the bf16-dequantized weights for M = 2048.
 * Error is |y - ref| / sum_k |x_k * w_k|, so the bound does not depend on cancellation in the dot.
 * Usage: test_int8_tile_gemm [--quick] [PIN.safetensors]
 * Exit: 0 PASS, 1 FAIL, 2 usage, 3 setup or CUDA error.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "../include/int8_tile_gemm.h"
#include "../include/safetensors_c.h"

#define TOL_VS_DOUBLE 1e-4
#define TOL_VS_CUBLAS 2e-4

#define CK(call)                                                                              \
    do {                                                                                      \
        cudaError_t e_ = (call);                                                              \
        if (e_ != cudaSuccess) {                                                              \
            printf("SETUP ERROR: %s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); \
            exit(3);                                                                          \
        }                                                                                     \
    } while (0)

#define CKB(call)                                                                             \
    do {                                                                                      \
        cublasStatus_t e_ = (call);                                                           \
        if (e_ != CUBLAS_STATUS_SUCCESS) {                                                    \
            printf("SETUP ERROR: %s:%d cuBLAS status %d\n", __FILE__, __LINE__, (int)e_);      \
            exit(3);                                                                          \
        }                                                                                     \
    } while (0)

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(g_rng >> 33);
}
static float rnd_unit(void) { return (float)rnd() / 2147483648.0f; }

static int g_failures = 0;

static float dequant(const int8_t *q, const float *scale, size_t i) {
    return __bfloat162float(__float2bfloat16((float)q[i] * scale[i / INT8_PIN_BLOCK]));
}

static __nv_bfloat16 *random_x(int m, int k) {
    __nv_bfloat16 *x = (__nv_bfloat16 *)malloc((size_t)m * k * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < (size_t)m * k; i++) x[i] = __float2bfloat16(4.0f * rnd_unit() - 2.0f);
    return x;
}

static float *run_tile(const __nv_bfloat16 *h_x, const int8_t *h_q, const float *h_s, int m, int n, int k) {
    size_t n_scale = ((size_t)n * k + INT8_PIN_BLOCK - 1) / INT8_PIN_BLOCK;
    __nv_bfloat16 *d_x;
    int8_t *d_q;
    float *d_s, *d_y;
    CK(cudaMalloc(&d_x, (size_t)m * k * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_q, (size_t)n * k));
    CK(cudaMalloc(&d_s, n_scale * sizeof(float)));
    CK(cudaMalloc(&d_y, (size_t)m * n * sizeof(float)));
    CK(cudaMemcpy(d_x, h_x, (size_t)m * k * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_q, h_q, (size_t)n * k, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_s, h_s, n_scale * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_y, 0xff, (size_t)m * n * sizeof(float)));
    Int8PinLinear w = {d_q, d_s, n, k};
    CK(int8_tile_gemm_bf16(d_x, w, d_y, m, 0));
    CK(cudaDeviceSynchronize());
    float *y = (float *)malloc((size_t)m * n * sizeof(float));
    CK(cudaMemcpy(y, d_y, (size_t)m * n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaFree(d_x));
    CK(cudaFree(d_q));
    CK(cudaFree(d_s));
    CK(cudaFree(d_y));
    return y;
}

static void check_vs_double(const char *label, const __nv_bfloat16 *x, const int8_t *q, const float *s,
                            int m, int n, int k) {
    float *y = run_tile(x, q, s, m, n, k);
    float *w = (float *)malloc((size_t)n * k * sizeof(float));
    for (size_t i = 0; i < (size_t)n * k; i++) w[i] = dequant(q, s, i);
    double worst = 0.0;
    int bad = 0;
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            double dot = 0.0, absdot = 0.0;
            const float *wr = w + (size_t)c * k;
            const __nv_bfloat16 *xr = x + (size_t)r * k;
            for (int i = 0; i < k; i++) {
                double p = (double)__bfloat162float(xr[i]) * (double)wr[i];
                dot += p;
                absdot += fabs(p);
            }
            double got = y[(size_t)r * n + c];
            double ratio = fabs(got - dot) / (absdot > 0.0 ? absdot : 1e-30);
            if (!(ratio <= TOL_VS_DOUBLE)) {
                if (bad < 3) printf("  %s y[%d,%d] = %.8g, want %.8g\n", label, r, c, got, dot);
                bad++;
            }
            if (!(ratio <= worst)) worst = ratio;
        }
    }
    printf("%-44s M=%-5d N=%-6d K=%-5d max err/absdot %.3e  %s\n", label, m, n, k, worst,
           bad ? "FAIL" : "ok");
    if (bad) g_failures++;
    free(w);
    free(y);
}

static void synthetic_case(int m, int n, int k) {
    size_t n_scale = ((size_t)n * k + INT8_PIN_BLOCK - 1) / INT8_PIN_BLOCK;
    int8_t *q = (int8_t *)malloc((size_t)n * k);
    float *s = (float *)malloc(n_scale * sizeof(float));
    for (size_t i = 0; i < (size_t)n * k; i++) q[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (size_t i = 0; i < n_scale; i++) s[i] = 1e-3f + 2e-2f * rnd_unit();
    __nv_bfloat16 *x = random_x(m, k);
    check_vs_double("synthetic", x, q, s, m, n, k);
    free(q);
    free(s);
    free(x);
}

__global__ void dequant_bf16_kernel(const int8_t *q, const float *s, __nv_bfloat16 *w, __nv_bfloat16 *w_abs,
                                    size_t numel) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    __nv_bfloat16 v = __float2bfloat16((float)q[i] * s[i / INT8_PIN_BLOCK]);
    w[i] = v;
    w_abs[i] = __habs(v);
}

__global__ void abs_bf16_kernel(const __nv_bfloat16 *x, __nv_bfloat16 *x_abs, size_t numel) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) x_abs[i] = __habs(x[i]);
}

static void cublas_bf16(cublasHandle_t h, const __nv_bfloat16 *w, const __nv_bfloat16 *x, float *y, int m, int n,
                        int k) {
    float alpha = 1.0f, beta = 0.0f;
    CKB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w, CUDA_R_16BF, k, x, CUDA_R_16BF, k, &beta, y,
                     CUDA_R_32F, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}

static void check_vs_cublas(const char *label, const __nv_bfloat16 *h_x, const int8_t *h_q, const float *h_s, int m,
                            int n, int k) {
    size_t numel = (size_t)n * k;
    size_t n_scale = numel / INT8_PIN_BLOCK;
    __nv_bfloat16 *d_x, *d_x_abs, *d_w, *d_w_abs;
    int8_t *d_q;
    float *d_s, *d_y_tile, *d_y_ref, *d_y_abs;
    CK(cudaMalloc(&d_x, (size_t)m * k * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_x_abs, (size_t)m * k * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_w, numel * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_w_abs, numel * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_q, numel));
    CK(cudaMalloc(&d_s, n_scale * sizeof(float)));
    CK(cudaMalloc(&d_y_tile, (size_t)m * n * sizeof(float)));
    CK(cudaMalloc(&d_y_ref, (size_t)m * n * sizeof(float)));
    CK(cudaMalloc(&d_y_abs, (size_t)m * n * sizeof(float)));
    CK(cudaMemcpy(d_x, h_x, (size_t)m * k * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_q, h_q, numel, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_s, h_s, n_scale * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_y_tile, 0xff, (size_t)m * n * sizeof(float)));
    dequant_bf16_kernel<<<(unsigned)((numel + 255) / 256), 256>>>(d_q, d_s, d_w, d_w_abs, numel);
    abs_bf16_kernel<<<(unsigned)(((size_t)m * k + 255) / 256), 256>>>(d_x, d_x_abs, (size_t)m * k);
    CK(cudaGetLastError());

    cublasHandle_t h;
    CKB(cublasCreate(&h));
    cublas_bf16(h, d_w, d_x, d_y_ref, m, n, k);
    cublas_bf16(h, d_w_abs, d_x_abs, d_y_abs, m, n, k);
    Int8PinLinear w = {d_q, d_s, n, k};
    CK(int8_tile_gemm_bf16(d_x, w, d_y_tile, m, 0));
    CK(cudaDeviceSynchronize());
    CKB(cublasDestroy(h));

    size_t outs = (size_t)m * n;
    float *y_tile = (float *)malloc(outs * sizeof(float));
    float *y_ref = (float *)malloc(outs * sizeof(float));
    float *y_abs = (float *)malloc(outs * sizeof(float));
    CK(cudaMemcpy(y_tile, d_y_tile, outs * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y_ref, d_y_ref, outs * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y_abs, d_y_abs, outs * sizeof(float), cudaMemcpyDeviceToHost));
    double worst = 0.0;
    size_t bad = 0, equal = 0;
    for (size_t i = 0; i < outs; i++) {
        double ratio = fabs((double)y_tile[i] - (double)y_ref[i]) / (y_abs[i] > 0.0f ? (double)y_abs[i] : 1e-30);
        if (!(ratio <= TOL_VS_CUBLAS)) bad++;
        if (!(ratio <= worst)) worst = ratio;
        if (y_tile[i] == y_ref[i]) equal++;
    }
    printf("%-44s M=%-5d N=%-6d K=%-5d max |tile-cublas|/absdot %.3e  bit-equal %.2f %%  %s\n", label, m, n, k, worst,
           100.0 * (double)equal / (double)outs, bad ? "FAIL" : "ok");
    if (bad) g_failures++;
    free(y_tile);
    free(y_ref);
    free(y_abs);
    CK(cudaFree(d_x));
    CK(cudaFree(d_x_abs));
    CK(cudaFree(d_w));
    CK(cudaFree(d_w_abs));
    CK(cudaFree(d_q));
    CK(cudaFree(d_s));
    CK(cudaFree(d_y_tile));
    CK(cudaFree(d_y_ref));
    CK(cudaFree(d_y_abs));
}

static int pin_linear(const SafeTensors *st, const char *stem, const int8_t **q, const float **s, int *n, int *k) {
    char name[320];
    snprintf(name, sizeof(name), "%s.weight", stem);
    const TensorEntry *eq = safetensors_find(st, name);
    snprintf(name, sizeof(name), "%s.weight.int8_scale", stem);
    const TensorEntry *es = safetensors_find(st, name);
    if (!eq || !es || strcmp(eq->dtype, "I8") || strcmp(es->dtype, "F32") || eq->n_dims != 2) {
        printf("SETUP ERROR: %s is not a bf16_to_int8_pin_v1 linear\n", stem);
        exit(3);
    }
    *n = (int)eq->shape[0];
    *k = (int)eq->shape[1];
    uint64_t want_scale = ((uint64_t)*n * *k + INT8_PIN_BLOCK - 1) / INT8_PIN_BLOCK * sizeof(float);
    if (eq->end - eq->begin != (uint64_t)*n * *k || es->end - es->begin != want_scale) {
        printf("SETUP ERROR: %s has inconsistent byte counts\n", stem);
        exit(3);
    }
    *q = (const int8_t *)eq->data;
    *s = (const float *)es->data;
    return 0;
}

int main(int argc, char **argv) {
    int quick = 0;
    const char *pin_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) quick = 1;
        else if (!pin_path && argv[i][0] != '-') pin_path = argv[i];
        else {
            printf("usage: %s [--quick] [PIN.safetensors]\n", argv[0]);
            return 2;
        }
    }

    static const int ms[] = {1, 2, 3, 7, 16, 17, 63, 64, 65, 130};
    static const int ns[] = {1, 8, 63, 64, 65, 100, 129};
    static const int ks[] = {1, 16, 37, 63, 64, 80, 129, 200, 1000};
    for (size_t a = 0; a < sizeof(ms) / sizeof(ms[0]); a++)
        for (size_t b = 0; b < sizeof(ns) / sizeof(ns[0]); b++)
            for (size_t c = 0; c < sizeof(ks) / sizeof(ks[0]); c++) {
                if (quick && (a + b + c) % 7 != 0) continue;
                synthetic_case(ms[a], ns[b], ks[c]);
            }

    if (pin_path) {
        SafeTensors st;
        if (safetensors_open(&st, pin_path) != 0) {
            printf("SETUP ERROR: cannot open %s\n", pin_path);
            return 3;
        }
        static const char *stems[] = {
            "model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.down_proj",
            "model.layers.31.mlp.down_proj"};
        static const int small_ms[] = {1, 17, 64};
        for (size_t i = 0; i < sizeof(stems) / sizeof(stems[0]); i++) {
            const int8_t *q;
            const float *s;
            int n, k;
            pin_linear(&st, stems[i], &q, &s, &n, &k);
            for (size_t j = 0; j < sizeof(small_ms) / sizeof(small_ms[0]); j++) {
                if (quick && j != 1) continue;
                __nv_bfloat16 *x = random_x(small_ms[j], k);
                check_vs_double(stems[i], x, q, s, small_ms[j], n, k);
                free(x);
            }
            if (!quick) {
                __nv_bfloat16 *x = random_x(2048, k);
                check_vs_cublas(stems[i], x, q, s, 2048, n, k);
                free(x);
            }
        }
        safetensors_close(&st);
    }

    printf(g_failures ? "VERDICT: FAIL (%d cases)\n" : "VERDICT: PASS\n", g_failures);
    return g_failures ? 1 : 0;
}
