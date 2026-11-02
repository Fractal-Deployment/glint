#include "../include/residual_correction.h"

namespace {

constexpr int SELECT_THREADS = 256;
constexpr int SELECT_MAX_K = 8192;

__device__ __forceinline__ int code_at(const uint8_t *row, int n, int bits) {
    if (bits == 8) return (int8_t)row[n];
    int b = row[n >> 1];
    int nib = (n & 1) ? (b >> 4) : (b & 15);
    return nib >= 8 ? nib - 16 : nib;
}

__device__ __forceinline__ float resid_step(ResidLinear r) { return r.range / (float)((1 << (r.bits - 1)) - 1); }

__device__ __forceinline__ float scale_at(ResidLinear r, int n, int k) {
    return r.scale[((size_t)n * r.k_in + k) / RESID_SCALE_BLOCK];
}

__device__ __forceinline__ int quantize_residual(float w, float w_base, float s, ResidLinear r, ResidBuildStats *stats) {
    if (s == 0.0f) return 0;
    float res = w - w_base;
    atomicMax(&stats->max_rel_bits, __float_as_uint(fabsf(res / s)));
    float qmax = (float)((1 << (r.bits - 1)) - 1);
    float c = rintf(res / (s * resid_step(r)));
    if (fabsf(c) > qmax) {
        atomicAdd(&stats->clamped, 1ull);
        c = fminf(fmaxf(c, -qmax), qmax);
    }
    return (int)c;
}

__global__ void build_kernel(const __nv_bfloat16 *__restrict__ w, const __nv_bfloat16 *__restrict__ w_base,
                             ResidLinear r, uint8_t *__restrict__ codes, ResidBuildStats *stats) {
    const int per_byte = 8 / r.bits, row_bytes = r.n_out / per_byte;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)r.k_in * row_bytes) return;
    int k = (int)(t % r.k_in);
    int p = (int)(t / r.k_in);
    unsigned byte = 0;
    for (int h = 0; h < per_byte; h++) {
        int n = per_byte * p + h;
        size_t i = (size_t)n * r.k_in + k;
        int c = quantize_residual(__bfloat162float(w[i]), __bfloat162float(w_base[i]), scale_at(r, n, k), r, stats);
        byte |= ((unsigned)c & ((1u << r.bits) - 1)) << (h * r.bits);
    }
    codes[(size_t)k * row_bytes + p] = (uint8_t)byte;
}

__global__ void col_norm_kernel(const uint8_t *__restrict__ codes, ResidLinear r, float *__restrict__ col_norm) {
    int k = blockIdx.x;
    const uint8_t *row = codes + (size_t)k * r.n_out * r.bits / 8;
    float sum = 0.0f;
    for (int n = threadIdx.x; n < r.n_out; n += blockDim.x) {
        float v = scale_at(r, n, k) * (float)code_at(row, n, r.bits) * resid_step(r);
        sum += v * v;
    }
    __shared__ float s_sum[256];
    s_sum[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) col_norm[k] = sqrtf(s_sum[0]);
}

__global__ void select_topk_kernel(const __nv_bfloat16 *__restrict__ x, int k_in, const float *__restrict__ chan_weight,
                                   int ksel, __nv_bfloat16 *__restrict__ xg, uint8_t *__restrict__ chan_used) {
    __shared__ unsigned keys[SELECT_MAX_K];
    __shared__ unsigned hist[256];
    __shared__ unsigned s_prefix;
    __shared__ int s_need;
    __shared__ int s_ties[SELECT_THREADS];
    const __nv_bfloat16 *xr = x + (size_t)blockIdx.x * k_in;
    __nv_bfloat16 *out = xg + (size_t)blockIdx.x * k_in;
    const int tid = threadIdx.x;

    for (int k = tid; k < k_in; k += blockDim.x) {
        float score = fabsf(__bfloat162float(xr[k]));
        if (chan_weight) score *= chan_weight[k];
        keys[k] = __float_as_uint(score);
    }
    if (tid == 0) {
        s_prefix = 0;
        s_need = ksel;
    }
    __syncthreads();

    unsigned mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[tid] = 0;
        __syncthreads();
        unsigned prefix = s_prefix;
        for (int k = tid; k < k_in; k += blockDim.x)
            if ((keys[k] & mask) == prefix) atomicAdd(&hist[(keys[k] >> shift) & 255], 1u);
        __syncthreads();
        if (tid == 0) {
            int need = s_need, above = 0;
            for (int d = 255; d >= 0; d--) {
                if (above + (int)hist[d] >= need) {
                    s_prefix = prefix | ((unsigned)d << shift);
                    s_need = need - above;
                    break;
                }
                above += hist[d];
            }
        }
        __syncthreads();
        mask |= 255u << shift;
    }
    const unsigned threshold = s_prefix;
    const int ties_taken = s_need;

    const int per = (k_in + blockDim.x - 1) / blockDim.x;
    const int lo = tid * per, hi = min(lo + per, k_in);
    int ties = 0;
    for (int k = lo; k < hi; k++) ties += keys[k] == threshold;
    s_ties[tid] = ties;
    __syncthreads();
    if (tid == 0) {
        int run = 0;
        for (int i = 0; i < blockDim.x; i++) {
            int c = s_ties[i];
            s_ties[i] = run;
            run += c;
        }
    }
    __syncthreads();
    int rank = s_ties[tid];
    for (int k = lo; k < hi; k++) {
        bool keep = keys[k] > threshold;
        if (keys[k] == threshold) keep = rank++ < ties_taken;
        out[k] = keep ? xr[k] : __float2bfloat16(0.0f);
        if (keep && chan_used) chan_used[k] = 1;
    }
}

__global__ void gather_kernel(const __nv_bfloat16 *__restrict__ x, int m, int k_in, const uint16_t *__restrict__ chan,
                              int c, __nv_bfloat16 *__restrict__ xg) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)m * c) return;
    size_t row = t / c;
    xg[t] = x[row * k_in + chan[t % c]];
}

__global__ void dequant_kernel(ResidCols cols, ResidLinear r, __nv_bfloat16 *__restrict__ rg) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)r.n_out * cols.c) return;
    int j = (int)(t % cols.c);
    int n = (int)(t / cols.c);
    int k = cols.chan ? cols.chan[j] : j;
    const uint8_t *row = cols.codes + (size_t)j * r.n_out * r.bits / 8;
    rg[t] = __float2bfloat16(scale_at(r, n, k) * (float)code_at(row, n, r.bits) * resid_step(r));
}

__global__ void accumulate_abs_kernel(const __nv_bfloat16 *__restrict__ x, int m, int k_in, float *__restrict__ sum_abs) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= k_in) return;
    float sum = 0.0f;
    for (int row = 0; row < m; row++) sum += fabsf(__bfloat162float(x[(size_t)row * k_in + k]));
    sum_abs[k] += sum;
}

unsigned blocks_for(size_t n) { return (unsigned)((n + 255) / 256); }

}

cudaError_t resid_build(const __nv_bfloat16 *w, const __nv_bfloat16 *w_base, ResidLinear r, uint8_t *codes,
                        float *col_norm, ResidBuildStats *stats, cudaStream_t stream) {
    if ((r.bits != 4 && r.bits != 8) || (r.bits == 4 && r.n_out % 2 != 0)) return cudaErrorInvalidValue;
    size_t bytes = resid_cols_bytes(r.k_in, r.n_out, r.bits);
    build_kernel<<<blocks_for(bytes), 256, 0, stream>>>(w, w_base, r, codes, stats);
    col_norm_kernel<<<r.k_in, 256, 0, stream>>>(codes, r, col_norm);
    return cudaGetLastError();
}

cudaError_t resid_select_topk(const __nv_bfloat16 *x, int m, int k_in, const float *chan_weight, int ksel,
                              __nv_bfloat16 *xg, uint8_t *chan_used, cudaStream_t stream) {
    if (k_in > SELECT_MAX_K || ksel < 1 || ksel > k_in) return cudaErrorInvalidValue;
    select_topk_kernel<<<m, SELECT_THREADS, 0, stream>>>(x, k_in, chan_weight, ksel, xg, chan_used);
    return cudaGetLastError();
}

cudaError_t resid_gather(const __nv_bfloat16 *x, int m, int k_in, const uint16_t *chan, int c, __nv_bfloat16 *xg,
                         cudaStream_t stream) {
    gather_kernel<<<blocks_for((size_t)m * c), 256, 0, stream>>>(x, m, k_in, chan, c, xg);
    return cudaGetLastError();
}

cudaError_t resid_dequant(ResidCols cols, ResidLinear r, __nv_bfloat16 *rg, cudaStream_t stream) {
    if (!cols.chan && cols.c != r.k_in) return cudaErrorInvalidValue;
    dequant_kernel<<<blocks_for((size_t)r.n_out * cols.c), 256, 0, stream>>>(cols, r, rg);
    return cudaGetLastError();
}

cudaError_t resid_accumulate_abs(const __nv_bfloat16 *x, int m, int k_in, float *sum_abs, cudaStream_t stream) {
    accumulate_abs_kernel<<<blocks_for(k_in), 256, 0, stream>>>(x, m, k_in, sum_abs);
    return cudaGetLastError();
}
