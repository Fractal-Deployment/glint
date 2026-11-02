#include "../include/int8_tile_gemm.h"

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int STAGES = 3;
constexpr int THREADS = 128;
constexpr int X_STRIDE = BK + 8;
constexpr int W_STRIDE = BK + 16;

static_assert(BK == INT8_PIN_BLOCK, "a W tile row spans at most two scale blocks only when BK <= the block");

struct __align__(16) Stage {
    __nv_bfloat16 x[BM][X_STRIDE];
    int8_t w[BN][W_STRIDE];
    float scale[BN][2];
    int hi_from[BN];
};

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem, bool valid) {
    unsigned dst = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async4(void *smem, const void *gmem, bool valid) {
    unsigned dst = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(gmem), "r"(valid ? 4 : 0));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <bool ASYNC>
__device__ void load_stage(Stage &st, const __nv_bfloat16 *__restrict__ x, Int8PinLinear w, int m, int m0, int n0,
                           int k0) {
    const int tid = threadIdx.x;
    const int k = w.k_in;
    if (ASYNC) {
        for (int c = tid; c < BM * BK / 8; c += THREADS) {
            int r = c / (BK / 8), col = (c % (BK / 8)) * 8;
            int gm = m0 + r, gk = k0 + col;
            bool valid = gm < m && gk < k;
            cp_async16(&st.x[r][col], valid ? x + (size_t)gm * k + gk : x, valid);
        }
        for (int c = tid; c < BN * BK / 16; c += THREADS) {
            int r = c / (BK / 16), col = (c % (BK / 16)) * 16;
            int gn = n0 + r, gk = k0 + col;
            bool valid = gn < w.n_out && gk < k;
            cp_async16(&st.w[r][col], valid ? w.q + (size_t)gn * k + gk : w.q, valid);
        }
    } else {
        for (int c = tid; c < BM * BK; c += THREADS) {
            int r = c / BK, col = c % BK;
            int gm = m0 + r, gk = k0 + col;
            st.x[r][col] = gm < m && gk < k ? x[(size_t)gm * k + gk] : __float2bfloat16(0.0f);
        }
        for (int c = tid; c < BN * BK; c += THREADS) {
            int r = c / BK, col = c % BK;
            int gn = n0 + r, gk = k0 + col;
            st.w[r][col] = gn < w.n_out && gk < k ? w.q[(size_t)gn * k + gk] : (int8_t)0;
        }
    }
    const size_t n_scale = ((size_t)w.n_out * k + INT8_PIN_BLOCK - 1) / INT8_PIN_BLOCK;
    for (int c = tid; c < BN * 2; c += THREADS) {
        int r = c / 2, half = c % 2;
        int gn = n0 + r;
        size_t flat = (size_t)gn * k + k0;
        size_t blk = flat / INT8_PIN_BLOCK + half;
        bool valid = gn < w.n_out && blk < n_scale;
        if (ASYNC) cp_async4(&st.scale[r][half], valid ? w.scale + blk : w.scale, valid);
        else st.scale[r][half] = valid ? w.scale[blk] : 0.0f;
        if (half == 0) st.hi_from[r] = INT8_PIN_BLOCK - (int)(flat % INT8_PIN_BLOCK);
    }
}

__device__ __forceinline__ uint32_t unfold_pair(const int8_t *row, int kk, float s_lo, float s_hi, int hi_from) {
    float a = (float)row[kk] * (kk < hi_from ? s_lo : s_hi);
    float b = (float)row[kk + 1] * (kk + 1 < hi_from ? s_lo : s_hi);
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t *>(&v);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void *smem) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <bool ASYNC>
__global__ void __launch_bounds__(THREADS) int8_tile_gemm_kernel(const __nv_bfloat16 *__restrict__ x,
                                                                 Int8PinLinear w, float *__restrict__ y, int m) {
    __shared__ Stage ring[STAGES];

    const int n0 = blockIdx.x * BN;
    const int m0 = blockIdx.y * BM;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int wm = (warp / 2) * 32;
    const int wn = (warp % 2) * 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int ktiles = (w.k_in + BK - 1) / BK;

    float acc[2][4][4] = {};

    for (int s = 0; s < STAGES - 1; s++) {
        if (s < ktiles) load_stage<ASYNC>(ring[s], x, w, m, m0, n0, s * BK);
        cp_async_commit();
    }

    for (int kt = 0; kt < ktiles; kt++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();

        int next = kt + STAGES - 1;
        if (next < ktiles) load_stage<ASYNC>(ring[next % STAGES], x, w, m, m0, n0, next * BK);
        cp_async_commit();

        const Stage &st = ring[kt % STAGES];
        for (int k16 = 0; k16 < BK; k16 += 16) {
            uint32_t a[2][4];
            for (int i = 0; i < 2; i++) ldmatrix_x4(a[i], &st.x[wm + i * 16 + lane % 16][k16 + (lane / 16) * 8]);
            for (int j = 0; j < 4; j++) {
                int n = wn + j * 8 + g;
                float s_lo = st.scale[n][0], s_hi = st.scale[n][1];
                int hi_from = st.hi_from[n];
                uint32_t b[2] = {unfold_pair(st.w[n], k16 + 2 * t, s_lo, s_hi, hi_from),
                                 unfold_pair(st.w[n], k16 + 2 * t + 8, s_lo, s_hi, hi_from)};
                for (int i = 0; i < 2; i++) mma_bf16_16816(acc[i][j], a[i], b);
            }
        }
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            int c = n0 + wn + j * 8 + 2 * t;
            for (int h = 0; h < 2; h++) {
                int r = m0 + wm + i * 16 + g + h * 8;
                if (r >= m) continue;
                float *yr = y + (size_t)r * w.n_out;
                if (c < w.n_out) yr[c] = acc[i][j][2 * h];
                if (c + 1 < w.n_out) yr[c + 1] = acc[i][j][2 * h + 1];
            }
        }
    }
}

}

cudaError_t int8_tile_gemm_bf16(const __nv_bfloat16 *x, Int8PinLinear w, float *y, int m, cudaStream_t stream) {
    if (m < 0 || w.n_out <= 0 || w.k_in <= 0 || !x || !w.q || !w.scale || !y) return cudaErrorInvalidValue;
    if (m == 0) return cudaSuccess;
    dim3 grid((w.n_out + BN - 1) / BN, (m + BM - 1) / BM);
    bool aligned = w.k_in % 16 == 0 && (uintptr_t)x % 16 == 0 && (uintptr_t)w.q % 16 == 0 &&
                   (uintptr_t)w.scale % 4 == 0;
    if (aligned) int8_tile_gemm_kernel<true><<<grid, THREADS, 0, stream>>>(x, w, y, m);
    else int8_tile_gemm_kernel<false><<<grid, THREADS, 0, stream>>>(x, w, y, m);
    return cudaGetLastError();
}
