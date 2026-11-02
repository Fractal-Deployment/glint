/* Late-plug kernel on real weights: layer L linear NAME from the INT8 pin, its res4 plug, and the original
 * BF16. Reference = dequantized quantized-activation times the BF16 weight in double (the activation
 * quantization is shared by both arms, so it cancels out of the comparison). PASS if base+plug error vs the
 * reference is at least 5x lower than base alone, and a poisoned plug_i8 scratch does not change the result.
 * Usage: test_late_plug PIN PLUG MODEL_DIR TENSOR_NAME */
#include "safetensors_c.h"
#include "int8_mma_gemm.h"
#include <dirent.h>
#include <math.h>
#include <vector>
#include <string>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s\n", cudaGetErrorString(e_)); return 3; } } while (0)
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    SafeTensors pin, plug, src; const TensorEntry *w = NULL;
    if (safetensors_open(&pin, argv[1]) || safetensors_open(&plug, argv[2])) return 3;
    DIR *d = opendir(argv[3]); struct dirent *de;
    while (d && (de = readdir(d))) {
        size_t L = strlen(de->d_name); if (L < 13 || strcmp(de->d_name + L - 12, ".safetensors")) continue;
        std::string p = std::string(argv[3]) + "/" + de->d_name;
        if (safetensors_open(&src, p.c_str())) return 3;
        if ((w = safetensors_find(&src, argv[4]))) break;
        safetensors_close(&src);
    }
    if (!w) { printf("tensor not found\n"); return 3; }
    int N = (int)w->shape[0], K = (int)w->shape[1], M = 64; size_t n = (size_t)N * K;
    std::string nm = argv[4];
    const TensorEntry *q = safetensors_find(&pin, nm.c_str()), *sb = safetensors_find(&pin, (nm + ".int8_scale").c_str());
    const TensorEntry *pq = safetensors_find(&plug, (nm + ".plug_q4").c_str()), *ps = safetensors_find(&plug, (nm + ".plug_scale").c_str());
    if (!q || !sb || !pq || !ps) { printf("missing pin/plug entries\n"); return 3; }
    std::vector<uint16_t> xb((size_t)M * K); uint32_t r = 12345;
    for (auto &v : xb) { r = r * 1664525u + 1013904223u; v = f2bf(((r >> 8) & 0xFFFF) / 32768.f - 1.f); }
    int8_t *dq, *dxq, *dpi; float *dsb, *dps, *dsx, *dy; uint8_t *dpq; __nv_bfloat16 *dx;
    CK(cudaMalloc(&dq, n)); CK(cudaMalloc(&dsb, sb->end - sb->begin)); CK(cudaMalloc(&dpq, pq->end - pq->begin));
    CK(cudaMalloc(&dps, ps->end - ps->begin)); CK(cudaMalloc(&dpi, n)); CK(cudaMalloc(&dx, xb.size() * 2));
    CK(cudaMalloc(&dxq, (size_t)M * K)); CK(cudaMalloc(&dsx, (size_t)M * (K / 64) * 4)); CK(cudaMalloc(&dy, (size_t)M * N * 4));
    CK(cudaMemcpy(dq, q->data, n, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsb, sb->data, sb->end - sb->begin, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpq, pq->data, pq->end - pq->begin, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dps, ps->data, ps->end - ps->begin, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
    Int8MmaLinear lin = {dq, dsb, N, K};
    std::vector<float> yb((size_t)M * N), yp(yb.size()), yp2(yb.size());
    CK(int8_mma_gemm(dx, lin, dy, M, dxq, dsx, 0)); CK(cudaMemcpy(yb.data(), dy, yb.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemset(dpi, 0x55, n));
    CK(int8_mma_gemm_plug(dx, lin, dpq, dps, dy, M, dxq, dsx, dpi, 0)); CK(cudaMemcpy(yp.data(), dy, yp.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemset(dpi, 0x7f, n));
    CK(int8_mma_gemm_plug(dx, lin, dpq, dps, dy, M, dxq, dsx, dpi, 0)); CK(cudaMemcpy(yp2.data(), dy, yp2.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<int8_t> xq((size_t)M * K); std::vector<float> sx((size_t)M * (K / 64));
    CK(cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(sx.data(), dsx, sx.size() * 4, cudaMemcpyDeviceToHost));
    const uint16_t *wb = (const uint16_t *)w->data; double eb = 0, ep = 0, ref2 = 0; long mism = 0;
    for (int m = 0; m < M; m += 4) for (int j = 0; j < N; j += 3) {
        double acc = 0;
        for (int k = 0; k < K; k++) acc += (double)xq[(size_t)m * K + k] * sx[(size_t)m * (K / 64) + k / 64] * bf2f(wb[(size_t)j * K + k]);
        size_t i = (size_t)m * N + j; eb += (yb[i] - acc) * (yb[i] - acc); ep += (yp[i] - acc) * (yp[i] - acc); ref2 += acc * acc;
    }
    for (size_t i = 0; i < yp.size(); i++) mism += memcmp(&yp[i], &yp2[i], 4) != 0;
    printf("tensor=%s base_rel=%.4e plug_rel=%.4e ratio=%.1fx poison_mismatches=%ld\n", argv[4], sqrt(eb / ref2), sqrt(ep / ref2), sqrt(eb / ep), mism);
    int ok = sqrt(eb / ep) >= 5.0 && mism == 0;
    printf("%s\n", ok ? "PASS" : "FAIL"); return ok ? 0 : 1;
}
