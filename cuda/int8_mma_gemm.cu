/* int8_mma_gemm.cu — slice-at-a-time INT8 tensor-core GEMM over a bf16_to_int8_pin_v1 linear.
 * Target sm_80+ (built and tested for sm_86). See include/int8_mma_gemm.h for the contract. */
#include "int8_mma_gemm.h"

#define BM 64
#define BN 64
#define BK 64        /* one pin scale block == one K slice */
#define SPAD 80      /* smem row stride in bytes: 64 + 16 keeps the 32-bit fragment loads conflict-free */

/* One block per (row, 64-block): xq = rint(x / s), s = amax / 127, same rule as quant_i8.cu. */
__global__ void k_quant_act(const __nv_bfloat16 *x, int m, int k_in, int8_t *xq, float *sx) {
    int nkb = k_in / BK;
    int row = blockIdx.x / nkb;
    int kb = blockIdx.x % nkb;
    int t = threadIdx.x; /* 64 threads */
    size_t i = (size_t)row * k_in + (size_t)kb * BK + t;
    float w = __bfloat162float(x[i]);
    float a = fabsf(w);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float sh[2];
    if ((t & 31) == 0) sh[t >> 5] = a;
    __syncthreads();
    float am = fmaxf(sh[0], sh[1]);
    float s = (am > 0.f) ? (am / 127.f) : 1.f;
    if (t == 0) sx[(size_t)row * nkb + kb] = s;
    int v = (int)rintf(w / s);
    v = v > 127 ? 127 : (v < -127 ? -127 : v);
    xq[i] = (int8_t)v;
}

__device__ __forceinline__ void mma_s8(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                       uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/* 128 threads = 4 warps; warp w owns output rows [16w, 16w+16) of the 64x64 tile and all 8 n-tiles. */
__global__ void __launch_bounds__(128)
k_int8_mma_gemm(const int8_t *xq, const float *sx, const int8_t *wq, const float *sw,
                float *y, int m, int n, int k_in, int accumulate) {
    __shared__ __align__(16) int8_t sA[BM][SPAD];
    __shared__ __align__(16) int8_t sB[BN][SPAD];
    __shared__ float sSx[BM];
    __shared__ float sSw[BN];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int nkb = k_in / BK;
    const int rb = warp * 16;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }

    for (int kb = 0; kb < nkb; kb++) {
        /* Stage this K slice: 64 rows x 64 bytes of X and of W, 16 bytes per thread-load. */
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * 128, r = idx >> 2, c = (idx & 3) * 16;
            uint4 va = make_uint4(0, 0, 0, 0), vb = make_uint4(0, 0, 0, 0);
            if (m0 + r < m) va = *(const uint4 *)(xq + (size_t)(m0 + r) * k_in + (size_t)kb * BK + c);
            if (n0 + r < n) vb = *(const uint4 *)(wq + (size_t)(n0 + r) * k_in + (size_t)kb * BK + c);
            *(uint4 *)&sA[r][c] = va;
            *(uint4 *)&sB[r][c] = vb;
        }
        if (tid < BM) sSx[tid] = (m0 + tid < m) ? sx[(size_t)(m0 + tid) * nkb + kb] : 0.f;
        else sSw[tid - BM] = (n0 + tid - BM < n) ? sw[(size_t)(n0 + tid - BM) * nkb + kb] : 0.f;
        __syncthreads();

        int ic[8][4];
#pragma unroll
        for (int i = 0; i < 8; i++) { ic[i][0] = ic[i][1] = ic[i][2] = ic[i][3] = 0; }
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            uint32_t a0 = *(const uint32_t *)&sA[rb + g][ks * 32 + t * 4];
            uint32_t a1 = *(const uint32_t *)&sA[rb + g + 8][ks * 32 + t * 4];
            uint32_t a2 = *(const uint32_t *)&sA[rb + g][ks * 32 + 16 + t * 4];
            uint32_t a3 = *(const uint32_t *)&sA[rb + g + 8][ks * 32 + 16 + t * 4];
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                uint32_t b0 = *(const uint32_t *)&sB[nt * 8 + g][ks * 32 + t * 4];
                uint32_t b1 = *(const uint32_t *)&sB[nt * 8 + g][ks * 32 + 16 + t * 4];
                mma_s8(ic[nt], a0, a1, a2, a3, b0, b1);
            }
        }
        const float x0 = sSx[rb + g], x1 = sSx[rb + g + 8];
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            float w0 = sSw[nt * 8 + t * 2], w1 = sSw[nt * 8 + t * 2 + 1];
            acc[nt][0] += ((float)ic[nt][0] * x0) * w0;
            acc[nt][1] += ((float)ic[nt][1] * x0) * w1;
            acc[nt][2] += ((float)ic[nt][2] * x1) * w0;
            acc[nt][3] += ((float)ic[nt][3] * x1) * w1;
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int col = n0 + nt * 8 + t * 2;
        int r0 = m0 + rb + g, r1 = r0 + 8;
        if (r0 < m) {
            if (col < n) y[(size_t)r0 * n + col] = (accumulate ? y[(size_t)r0 * n + col] : 0.f) + acc[nt][0];
            if (col + 1 < n) y[(size_t)r0 * n + col + 1] = (accumulate ? y[(size_t)r0 * n + col + 1] : 0.f) + acc[nt][1];
        }
        if (r1 < m) {
            if (col < n) y[(size_t)r1 * n + col] = (accumulate ? y[(size_t)r1 * n + col] : 0.f) + acc[nt][2];
            if (col + 1 < n) y[(size_t)r1 * n + col + 1] = (accumulate ? y[(size_t)r1 * n + col + 1] : 0.f) + acc[nt][3];
        }
    }
}

extern "C" cudaError_t int8_mma_gemm(const __nv_bfloat16 *x, Int8MmaLinear w, float *y, int m,
                                     int8_t *xq, float *sx, cudaStream_t stream) {
    if (!x || !y || !xq || !sx || !w.q || !w.scale || m < 1 || w.n_out < 1 || w.k_in < BK ||
        (w.k_in % BK) != 0)
        return cudaErrorInvalidValue;
    int nkb = w.k_in / BK;
    k_quant_act<<<m * nkb, 64, 0, stream>>>(x, m, w.k_in, xq, sx);
    dim3 grid((w.n_out + BN - 1) / BN, (m + BM - 1) / BM);
    k_int8_mma_gemm<<<grid, 128, 0, stream>>>(xq, sx, w.q, w.scale, y, m, w.n_out, w.k_in, 0);
    return cudaGetLastError();
}

/* Unpack a p7_plug_res4_v1 slice: element 2i in the low nibble, biased +8, to int8 in [-7, 7]. */
__global__ void k_unpack_res4(const uint8_t *q4, int8_t *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int nib = (q4[i >> 1] >> ((i & 1) * 4)) & 0xF;
    out[i] = (int8_t)(nib - 8);
}

extern "C" cudaError_t int8_mma_gemm_plug(const __nv_bfloat16 *x, Int8MmaLinear w, const uint8_t *plug_q4,
                                          const float *plug_scale, float *y, int m, int8_t *xq, float *sx,
                                          int8_t *plug_i8, cudaStream_t stream) {
    if (!x || !y || !xq || !sx || !w.q || !w.scale || !plug_q4 || !plug_scale || !plug_i8 || m < 1 ||
        w.n_out < 1 || w.k_in < BK || (w.k_in % BK) != 0)
        return cudaErrorInvalidValue;
    int nkb = w.k_in / BK;
    size_t n = (size_t)w.n_out * w.k_in;
    k_quant_act<<<m * nkb, 64, 0, stream>>>(x, m, w.k_in, xq, sx);
    dim3 grid((w.n_out + BN - 1) / BN, (m + BM - 1) / BM);
    k_int8_mma_gemm<<<grid, 128, 0, stream>>>(xq, sx, w.q, w.scale, y, m, w.n_out, w.k_in, 0);
    k_unpack_res4<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(plug_q4, plug_i8, n);
    /* the plug shares the base's activation quantization; it adds into y */
    k_int8_mma_gemm<<<grid, 128, 0, stream>>>(xq, sx, plug_i8, plug_scale, y, m, w.n_out, w.k_in, 1);
    return cudaGetLastError();
}
