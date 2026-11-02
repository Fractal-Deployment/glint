#ifndef P7_ACT_MAP_H
#define P7_ACT_MAP_H
#include <vector>
/* Stage 0b activation map (glint #12): per (layer, linear) statistics of the linear's input X, on device.
 * ARM0_ACT_MAP=out.json. Works under every --gemm path, so the same map can be taken on the BF16
 * (cublas) path and the INT8 path and compared. */
struct ActStat { double n, sum2, sumabs, maxabs, outl6, blk_amax_sum, blk_n; };
static ActStat *g_act = NULL;
__global__ void k_act_stat(const __nv_bfloat16 *x, size_t total, int k, double rms_hint, ActStat *st) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.f;
    if (i < total) v = __bfloat162float(x[i]);
    float a = fabsf(v);
    double s2 = (double)v * v, sa = a, o6 = (a > 6.0 * rms_hint && rms_hint > 0) ? 1.0 : 0.0;
    float am = a;
    for (int o = 16; o > 0; o >>= 1) { am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o)); }
    __shared__ float sh[8];
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = am;
    __syncthreads();
    if (threadIdx.x == 0 && i < total) {
        /* blockDim 64: two warps = one pin block */
        float b = fmaxf(sh[0], sh[1]);
        atomicAdd(&st->blk_amax_sum, (double)b); atomicAdd(&st->blk_n, 1.0);
    }
    if (i < total) {
        atomicAdd(&st->n, 1.0); atomicAdd(&st->sum2, s2); atomicAdd(&st->sumabs, sa); atomicAdd(&st->outl6, o6);
        unsigned long long *p = (unsigned long long *)&st->maxabs, old = *p, assumed;
        do { assumed = old; if (__longlong_as_double(assumed) >= a) break;
             old = atomicCAS(p, assumed, (unsigned long long)__double_as_longlong((double)a)); } while (assumed != old);
    }
}
static void act_capture(int layer, int which, const __nv_bfloat16 *x, int m) {
    if (!g_act) return;
    int k = ARM0_LINEAR[which].k_in;
    size_t total = (size_t)m * k;
    k_act_stat<<<(unsigned)((total + 63) / 64), 64>>>(x, total, k, 1.0, &g_act[layer * ARM0_N_LINEARS + which]);
}


static void act_map_init(const char *path) {
    if (!path) return;
    CK(cudaMalloc(&g_act, sizeof(ActStat) * N_LAYERS * ARM0_N_LINEARS));
    CK(cudaMemset(g_act, 0, sizeof(ActStat) * N_LAYERS * ARM0_N_LINEARS));
}
static void act_map_write(const char *act_map, const char *path_name) {
    if (!act_map) return;
        std::vector<ActStat> h(N_LAYERS * ARM0_N_LINEARS);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h.data(), g_act, sizeof(ActStat) * h.size(), cudaMemcpyDeviceToHost));
        FILE *f = fopen(act_map, "w");
        if (!f) setup_fail("cannot write %s", act_map);
        fprintf(f, "{\"schema\":\"p7_activation_map_v1\",\"path\":\"%s\",\"tokens\":2047,\"entries\":[", path_name);
        for (int l = 0; l < N_LAYERS; l++)
            for (int j = 0; j < ARM0_N_LINEARS; j++) {
                const ActStat &a = h[l * ARM0_N_LINEARS + j];
                fprintf(f, "%s{\"layer\":%d,\"linear\":\"%s\",\"k_in\":%d,\"n\":%.0f,\"rms\":%.6e,\"mean_abs\":%.6e,\"max_abs\":%.6e,\"mean_block_amax\":%.6e,\"max_over_rms\":%.4f}",
                        (l || j) ? "," : "", l, ARM0_LINEAR[j].name, ARM0_LINEAR[j].k_in, a.n, sqrt(a.sum2 / a.n), a.sumabs / a.n, a.maxabs,
                        a.blk_amax_sum / a.blk_n, a.maxabs / sqrt(a.sum2 / a.n));
            }
        fprintf(f, "]}\n");
        fclose(f);
        printf("Activation map written: %s\n", act_map);
}
#endif
