/* tests/bench_packed_tier.cu — measured per-token time of a PACKED mixed-width plug store (spec Amendment 9, glint #92).
 * Replaces the arithmetic in TIERED_PLUG_phi4mini.md ("assuming free unpacking") with a measurement. Synthetic data, speed only:
 * no accuracy is computed, the values are random. Store: K-major per linear; a channel of band A holds n_out 4-bit codes
 * (n_out/2 bytes), a channel of band B holds n_out 3-bit codes packed 8 codes per 3 bytes (3*n_out/8 bytes). All 32 layers
 * x 4 linears (QKV, O, GATE_UP, DOWN, Phi-4-mini shapes) live in pinned mapped host memory (separate regions per layer, so
 * nothing is reused across layers). One "token" = all 128 linears in order on one stream: a fused kernel reads the selected channels
 * straight from host memory over PCIe (zero-copy), unpacks, and accumulates x[c]*cell into y (fp32) with split-K atomics.
 * Selection is a fixed-seed random subset of channels per layer and linear (it fixes counts and bytes, not which channels).
 * Output (TSV): arm, k1_frac, k2_frac, bits_B, bytes_per_token, ms_per_token, GB_per_s, plug_only_tok_per_s.
 * usage: bench_packed_tier [reps=10]
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

struct Lin { int n_out, k_in; };
static const Lin LIN[4] = {{5120, 3072}, {3072, 3072}, {16384, 3072}, {3072, 8192}};
static const int NL = 32, SPLIT = 8;
__constant__ float c_cell[16];

/* grid (n_out/256, SPLIT, 1); block 32 threads, thread t = rows 8t..8t+7 of the 256-row tile. */
__global__ void k_corr(const uint8_t *s4, const uint8_t *s3, const int *idx4, int n4, const int *idx3, int n3, const float *x, float *y, int n_out, int pk) {
    const int tile = blockIdx.x, sp = blockIdx.y, t = threadIdx.x; float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int j = sp; j < n4; j += SPLIT) {
        const int c = idx4[j]; const float xc = x[c];
        uint32_t w = *(const uint32_t *)(s4 + (size_t)j * (n_out / 2) + tile * 128 + t * 4);
#pragma unroll
        for (int r = 0; r < 8; r++) acc[r] += xc * c_cell[(w >> (4 * r)) & 15];
    }
    for (int j = sp; j < n3; j += SPLIT) {
        const int c = idx3[j]; const float xc = x[c];
        const uint8_t *p = s3 + (size_t)j * (3 * n_out / 8) + tile * 96 + t * 3;
        uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
#pragma unroll
        for (int r = 0; r < 8; r++) acc[r] += xc * c_cell[(w >> (3 * r)) & 7];
    }
#pragma unroll
    for (int r = 0; r < 8; r++) atomicAdd(&y[tile * 256 + t * 8 + r], acc[r]);
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 10;
    struct Arm { const char *name; double f1, f2; int bB; } arms[] = {
        {"4bit_act50", 0.50, 0.0, 4}, {"4bit_act25", 0.25, 0.0, 4}, {"TA_10_30_3bit", 0.10, 0.30, 3}, {"TA_10_30_asif4bit", 0.10, 0.30, 4},
        {"F3_10_10_3bit", 0.10, 0.10, 3}, {"F4_05_10_3bit", 0.05, 0.10, 3}, {"T_C_25_25_3bit", 0.25, 0.25, 3}};
    /* store sizes: band A region sized for the largest k1, band B for the largest k2, per linear and layer */
    const double max1 = 0.50, max2 = 0.30;
    size_t off4[NL][4], off3[NL][4], tot4 = 0, tot3 = 0;
    for (int l = 0; l < NL; l++) for (int j = 0; j < 4; j++) {
        off4[l][j] = tot4; tot4 += (size_t)(max1 * LIN[j].k_in + 1) * (LIN[j].n_out / 2); tot4 = (tot4 + 255) & ~(size_t)255;
        off3[l][j] = tot3; tot3 += (size_t)(max2 * LIN[j].k_in + 1) * (3 * LIN[j].n_out / 8); tot3 = (tot3 + 255) & ~(size_t)255;
    }
    uint8_t *h4, *h3, *d4, *d3; CK(cudaHostAlloc((void **)&h4, tot4, cudaHostAllocMapped)); CK(cudaHostAlloc((void **)&h3, tot3, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void **)&d4, h4, 0)); CK(cudaHostGetDevicePointer((void **)&d3, h3, 0));
    for (size_t i = 0; i < tot4; i++) h4[i] = (uint8_t)(i * 2654435761u >> 13); for (size_t i = 0; i < tot3; i++) h3[i] = (uint8_t)(i * 40503u >> 5);
    float cell[16]; for (int i = 0; i < 16; i++) cell[i] = (i - 7.5f) / 8.f; CK(cudaMemcpyToSymbol(c_cell, cell, sizeof cell));
    int maxK = 8192, maxN = 16384; float *dx, *dy; CK(cudaMalloc((void **)&dx, maxK * 4)); CK(cudaMalloc((void **)&dy, maxN * 4));
    std::vector<float> hx(maxK); for (int i = 0; i < maxK; i++) hx[i] = (float)((i * 37) % 101) / 101.f - 0.5f; CK(cudaMemcpy(dx, hx.data(), maxK * 4, cudaMemcpyHostToDevice));
    int *didx; CK(cudaMalloc((void **)&didx, 2 * 8192 * 4 * 4 * NL));
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking)); cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    printf("arm\tk1_frac\tk2_frac\tbits_B\tbytes_per_token\tms_per_token\tGB_per_s\tplug_only_tok_per_s\n");
    for (auto &a : arms) {
        /* index lists: per (layer, linear) a random subset, band A first then band B (disjoint) */
        std::vector<int> n4(NL * 4), n3(NL * 4); std::vector<int> all(NL * 4 * 2 * 8192, 0); size_t ip = 0; std::vector<size_t> p4(NL * 4), p3(NL * 4);
        double bytes = 0;
        for (int l = 0; l < NL; l++) for (int j = 0; j < 4; j++) {
            int K = LIN[j].k_in, k1 = (int)(a.f1 * K), k2 = (int)(a.f2 * K); std::vector<int> perm(K); for (int i = 0; i < K; i++) perm[i] = i;
            uint32_t s = 1234567u + l * 31 + j; for (int i = K - 1; i > 0; i--) { s = s * 1664525u + 1013904223u; std::swap(perm[i], perm[(s >> 8) % (i + 1)]); }
            std::sort(perm.begin(), perm.begin() + k1); std::sort(perm.begin() + k1, perm.begin() + k1 + k2);
            n4[l * 4 + j] = k1; n3[l * 4 + j] = k2; p4[l * 4 + j] = ip; for (int i = 0; i < k1; i++) all[ip++] = perm[i]; p3[l * 4 + j] = ip; for (int i = 0; i < k2; i++) all[ip++] = perm[k1 + i];
            bytes += (double)k1 * LIN[j].n_out / 2 + (double)k2 * (a.bB == 3 ? 3.0 * LIN[j].n_out / 8 : LIN[j].n_out / 2);
        }
        CK(cudaMemcpy(didx, all.data(), ip * 4, cudaMemcpyHostToDevice));
        /* "asif4bit" arm: band B read with the 4-bit stride (same code, bytes of a 4-bit band) to isolate the unpack cost */
        auto token = [&]() {
            for (int l = 0; l < NL; l++) for (int j = 0; j < 4; j++) {
                CK(cudaMemsetAsync(dy, 0, LIN[j].n_out * 4, st));
                dim3 g(LIN[j].n_out / 256, SPLIT);
                if (a.bB == 3) k_corr<<<g, 32, 0, st>>>(d4 + off4[l][j], d3 + off3[l][j], didx + p4[l * 4 + j], n4[l * 4 + j], didx + p3[l * 4 + j], n3[l * 4 + j], dx, dy, LIN[j].n_out, 0);
                else k_corr<<<g, 32, 0, st>>>(d4 + off4[l][j], d4 + off4[l][j], didx + p4[l * 4 + j], n4[l * 4 + j] + n3[l * 4 + j], didx + p3[l * 4 + j], 0, dx, dy, LIN[j].n_out, 1);
            }
        };
        /* the 4-bit-only and "as 4-bit" arms put all channels (k1 [+k2 as 4-bit]) in band A for a clean 4-bit stride read */
        token(); CK(cudaStreamSynchronize(st));
        CK(cudaEventRecord(e0, st)); for (int r = 0; r < reps; r++) token(); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= reps;
        printf("%s\t%.2f\t%.2f\t%d\t%.0f\t%.2f\t%.2f\t%.1f\n", a.name, a.f1, a.f2, a.bB, bytes, ms, bytes / (ms * 1e6), 1000.0 / ms); fflush(stdout);
    }
    return 0;
}
