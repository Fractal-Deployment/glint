/* GLINT H-TILE S1 sketch — CUDA, Ampere sm_86.
 * Replace NF4 codebook[nibble]*absmax with GLINT_CELLS[hole][plug]*absmax.
 * Plug[L] resident in VRAM. memcpyAsync plug[L+1] on stream 1 — never in TILE_K.
 * S5: swap FMA for mma.sync.bf16. train_ok=false.
 *
 * nvcc -arch=sm_86 -I../include -c glint_htile.cu
 */
#include "../include/glint_meta.h"
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef G_HTILE_M
#define G_HTILE_M 64
#define G_HTILE_N 64
#define G_HTILE_K 64
#endif

__global__ void glint_htile_s1_f32(
    const uint8_t *__restrict__ hole,
    const uint8_t *__restrict__ plug,
    const float *__restrict__ absmax,
    const __nv_bfloat16 *__restrict__ x,
    float *__restrict__ y,
    int M, int N, int K, int blocksize)
{
    const int n0 = blockIdx.x * G_HTILE_N;
    const int m0 = blockIdx.y * G_HTILE_M;
    const int tn = threadIdx.x;
    const int tm = threadIdx.y;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += G_HTILE_K) {
        /* S3: cp.async hole+plug+x tiles here */
        if ((m0 + tm) < M && (n0 + tn) < N) {
            const int n = n0 + tn;
            for (int kk = 0; kk < G_HTILE_K && (k0 + kk) < K; ++kk) {
                const int i = n * K + (k0 + kk);
                const uint8_t hb = hole[i >> 1];
                const uint8_t pb = plug[i >> 1];
                const uint8_t h = (i & 1) ? (uint8_t)((hb >> 4) & 15) : (uint8_t)(hb & 15);
                const uint8_t p = (i & 1) ? (uint8_t)((pb >> 4) & 15) : (uint8_t)(pb & 15);
                const float w = glint_inflate(h, p, absmax[i / blocksize]);
                acc += w * __bfloat162float(x[(m0 + tm) * K + (k0 + kk)]);
            }
        }
    }
    if ((m0 + tm) < M && (n0 + tn) < N)
        y[(m0 + tm) * N + (n0 + tn)] = acc;
}
