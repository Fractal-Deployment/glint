/* p7_abi_selftest.cu — the attach gate's measurement. Runs the library's own entry points through
 * dlopen-visible symbols: a 4-layer INT8 stack built with the pin rule, run once with every layer
 * resident and once streamed through the 2-slot ring, on int8_mma_gemm. Passes only if the two
 * outputs agree bit for bit, are finite and not all zero, and match a CPU float reference of the
 * kernel's arithmetic. Returns 0 on pass; a negative code names the failed stage. */
#include "int8_mma_gemm.h"
#include "glint_plug_streamer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define ST_LAYERS 4
#define ST_M 70
#define ST_K 256
#define ST_N 192

static uint32_t rng_state = 0x9e3779b9u;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((rng_state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

static void quant_pin(const std::vector<float> &w, int n, int k, int8_t *q, float *s) {
    for (int r = 0; r < n; r++)
        for (int kb = 0; kb < k / 64; kb++) {
            float am = 0.f;
            for (int j = 0; j < 64; j++) am = fmaxf(am, fabsf(w[(size_t)r * k + kb * 64 + j]));
            float sc = am > 0.f ? am / 127.f : 1.f;
            s[(size_t)r * (k / 64) + kb] = sc;
            for (int j = 0; j < 64; j++) {
                int v = (int)rintf(w[(size_t)r * k + kb * 64 + j] / sc);
                q[(size_t)r * k + kb * 64 + j] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
            }
        }
}

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

#define CK(x) do { if ((x) != cudaSuccess) return -1; } while (0)

extern "C" const char *p7_abi_version(void);

extern "C" int p7_abi_selftest(char *msg, size_t msg_n) {
    const size_t qb = (size_t)ST_N * ST_K, sb = (size_t)ST_N * (ST_K / 64) * sizeof(float);
    const size_t blob = ((qb + 255) / 256) * 256 + sb;
    std::vector<uint8_t> host((size_t)ST_LAYERS * blob, 0);
    std::vector<std::vector<float>> wref(ST_LAYERS);
    for (int l = 0; l < ST_LAYERS; l++) {
        std::vector<float> w((size_t)ST_N * ST_K);
        for (auto &v : w) v = 0.05f * frand() * (1.f + (l & 1));
        uint8_t *b = host.data() + (size_t)l * blob;
        quant_pin(w, ST_N, ST_K, (int8_t *)b, (float *)(b + ((qb + 255) / 256) * 256));
    }
    std::vector<uint16_t> xb((size_t)ST_M * ST_K);
    for (auto &v : xb) v = f2bf(frand());

    __nv_bfloat16 *dx; float *dy_res, *dy_str, *dsx; int8_t *dxq; uint8_t *dres;
    CK(cudaMalloc(&dx, xb.size() * 2));
    CK(cudaMemcpy(dx, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy_res, (size_t)ST_M * ST_N * 4));
    CK(cudaMalloc(&dy_str, (size_t)ST_M * ST_N * 4));
    CK(cudaMalloc(&dxq, int8_mma_xq_bytes(ST_M, ST_K)));
    CK(cudaMalloc(&dsx, int8_mma_sx_bytes(ST_M, ST_K)));
    CK(cudaMalloc(&dres, host.size()));
    CK(cudaMemcpy(dres, host.data(), host.size(), cudaMemcpyHostToDevice));

    std::vector<float> res((size_t)ST_LAYERS * ST_M * ST_N), str(res.size());
    for (int l = 0; l < ST_LAYERS; l++) {
        const uint8_t *b = dres + (size_t)l * blob;
        Int8MmaLinear lin = {(const int8_t *)b, (const float *)(b + ((qb + 255) / 256) * 256), ST_N, ST_K};
        if (int8_mma_gemm(dx, lin, dy_res, ST_M, dxq, dsx, 0) != cudaSuccess) return -2;
        CK(cudaMemcpy(res.data() + (size_t)l * ST_M * ST_N, dy_res, (size_t)ST_M * ST_N * 4, cudaMemcpyDeviceToHost));
    }

    cudaStream_t cs, ks;
    CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
    CK(cudaStreamCreate(&ks));
    uint64_t lb[ST_LAYERS];
    for (int l = 0; l < ST_LAYERS; l++) lb[l] = blob;
    GlintPlugStreamer *s = NULL;
    if (glint_plug_streamer_create(&s, ST_LAYERS, lb, cs, ks) != 0) return -3;
    for (int l = 0; l < ST_LAYERS; l++)
        if (glint_plug_streamer_set_host_data(s, l, host.data() + (size_t)l * blob, blob) != 0) return -3;
    if (glint_plug_streamer_start(s) != 0) return -4;
    for (int l = 0; l < ST_LAYERS; l++) {
        const uint8_t *b = NULL;
        if (glint_plug_streamer_acquire(s, l, &b) != 0) return -4;
        Int8MmaLinear lin = {(const int8_t *)b, (const float *)(b + ((qb + 255) / 256) * 256), ST_N, ST_K};
        if (int8_mma_gemm(dx, lin, dy_str, ST_M, dxq, dsx, ks) != cudaSuccess) return -2;
        CK(cudaMemcpyAsync(str.data() + (size_t)l * ST_M * ST_N, dy_str, (size_t)ST_M * ST_N * 4, cudaMemcpyDeviceToHost, ks));
        CK(cudaStreamSynchronize(ks));
        if (glint_plug_streamer_release(s, l) != 0) return -4;
    }
    GlintStreamerStats st;
    glint_plug_streamer_get_stats(s, &st);
    glint_plug_streamer_destroy(s);

    long mism = 0, nonfin = 0, nz = 0;
    for (size_t i = 0; i < res.size(); i++) {
        if (memcmp(&res[i], &str[i], 4) != 0) mism++;
        if (!isfinite(res[i])) nonfin++;
        if (res[i] != 0.f) nz++;
    }
    /* CPU float reference of the kernel's arithmetic, layer 0. */
    double maxrel = 0;
    {
        const uint8_t *b = host.data();
        const int8_t *wq = (const int8_t *)b;
        const float *sw = (const float *)(b + ((qb + 255) / 256) * 256);
        std::vector<int8_t> xq((size_t)ST_M * ST_K);
        std::vector<float> sx((size_t)ST_M * (ST_K / 64));
        std::vector<float> x((size_t)ST_M * ST_K);
        for (size_t i = 0; i < x.size(); i++) x[i] = bf2f(xb[i]);
        quant_pin(x, ST_M, ST_K, xq.data(), sx.data());
        for (int m = 0; m < ST_M; m++)
            for (int n = 0; n < ST_N; n++) {
                float acc = 0.f;
                for (int kb = 0; kb < ST_K / 64; kb++) {
                    int ic = 0;
                    for (int j = 0; j < 64; j++) ic += (int)xq[(size_t)m * ST_K + kb * 64 + j] * (int)wq[(size_t)n * ST_K + kb * 64 + j];
                    acc += ((float)ic * sx[(size_t)m * (ST_K / 64) + kb]) * sw[(size_t)n * (ST_K / 64) + kb];
                }
                double r = res[(size_t)m * ST_N + n];
                double rel = fabs(r - acc) / (fabs((double)acc) + 1e-6);
                if (fabs(r - acc) > 1e-6 && rel > maxrel) maxrel = rel;
            }
    }
    cudaFree(dx); cudaFree(dy_res); cudaFree(dy_str); cudaFree(dxq); cudaFree(dsx); cudaFree(dres);
    cudaStreamDestroy(cs); cudaStreamDestroy(ks);

    snprintf(msg, msg_n,
             "abi=%s layers=%d streamed_bytes=%llu mismatches=%ld nonfinite=%ld nonzero=%ld/%zu cpu_max_rel=%.3e",
             p7_abi_version(), ST_LAYERS, (unsigned long long)st.total_h2d_bytes, mism, nonfin, nz, res.size(), maxrel);
    if (mism || nonfin) return -5;
    if (nz < (long)(res.size() / 2)) return -6;
    if (maxrel > 1e-4) return -7;
    if (st.total_h2d_bytes != (uint64_t)ST_LAYERS * blob) return -8;
    return 0;
}

/* Flat-argument entry for FFI callers (Mojo): no struct by value. Returns cudaError_t as int. */
extern "C" int p7_int8_gemm_flat(const void *x_bf16, const void *q, const void *scale, int n_out, int k_in,
                                 void *y, int m, void *xq, void *sx, void *stream) {
    Int8MmaLinear w = {(const int8_t *)q, (const float *)scale, n_out, k_in};
    return (int)int8_mma_gemm((const __nv_bfloat16 *)x_bf16, w, (float *)y, m, (int8_t *)xq, (float *)sx, (cudaStream_t)stream);
}

/* Fused late plug over FFI: base + res4 plug slice in one kernel (int8_mma_gemm_plug_fused). */
extern "C" int p7_int8_gemm_plug_flat(const void *x_bf16, const void *q, const void *scale, const void *plug_q4,
                                      const void *plug_scale, int n_out, int k_in, void *y, int m, void *xq, void *sx,
                                      void *stream) {
    Int8MmaLinear w = {(const int8_t *)q, (const float *)scale, n_out, k_in};
    return (int)int8_mma_gemm_plug_fused((const __nv_bfloat16 *)x_bf16, w, (const uint8_t *)plug_q4,
                                         (const float *)plug_scale, (float *)y, m, (int8_t *)xq, (float *)sx,
                                         (cudaStream_t)stream);
}
