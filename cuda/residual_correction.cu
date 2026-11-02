#include "../include/residual_correction.h"

namespace {

constexpr int SELECT_THREADS = 256;
constexpr int SELECT_MAX_K = 8192;

__device__ __forceinline__ int code_at(const uint8_t *row, int n) {
    int b = row[n >> 1];
    int nib = (n & 1) ? (b >> 4) : (b & 15);
    return nib >= 8 ? nib - 16 : nib;
}

__device__ __forceinline__ int quantize_residual(float w, int8_t q, float s, ResidBuildStats *stats) {
    float r = w - __bfloat162float(__float2bfloat16((float)q * s));
    if (s == 0.0f) return 0;
    float rel = fabsf(r / s);
    atomicMax(&stats->max_rel_bits, __float_as_uint(rel));
    float c = rintf(r / (s * RESID_STEP));
    if (fabsf(c) > 7.0f) {
        atomicAdd(&stats->clamped, 1ull);
        c = fminf(fmaxf(c, -7.0f), 7.0f);
    }
    return (int)c;
}

__global__ void build_kernel(const __nv_bfloat16 *__restrict__ w, Int8PinLinear base, uint8_t *__restrict__ codes,
                             ResidBuildStats *stats) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half = base.n_out / 2;
    if (t >= (size_t)base.k_in * half) return;
    int k = (int)(t % base.k_in);
    int p = (int)(t / base.k_in);
    int nib[2];
    for (int h = 0; h < 2; h++) {
        size_t i = (size_t)(2 * p + h) * base.k_in + k;
        nib[h] = quantize_residual(__bfloat162float(w[i]), base.q[i], base.scale[i / INT8_PIN_BLOCK], stats) & 15;
    }
    codes[(size_t)k * half + p] = (uint8_t)(nib[0] | (nib[1] << 4));
}

__global__ void col_norm_kernel(const uint8_t *__restrict__ codes, Int8PinLinear base, float *__restrict__ col_norm) {
    int k = blockIdx.x;
    const uint8_t *row = codes + (size_t)k * (base.n_out / 2);
    float sum = 0.0f;
    for (int n = threadIdx.x; n < base.n_out; n += blockDim.x) {
        float v = base.scale[((size_t)n * base.k_in + k) / INT8_PIN_BLOCK] * (float)code_at(row, n) * RESID_STEP;
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

__global__ void dequant_kernel(ResidCols r, Int8PinLinear base, __nv_bfloat16 *__restrict__ rg) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)base.n_out * r.c) return;
    int j = (int)(t % r.c);
    int n = (int)(t / r.c);
    int k = r.chan ? r.chan[j] : j;
    float s = base.scale[((size_t)n * base.k_in + k) / INT8_PIN_BLOCK];
    rg[t] = __float2bfloat16(s * (float)code_at(r.codes + (size_t)j * (base.n_out / 2), n) * RESID_STEP);
}

__global__ void accumulate_abs_kernel(const __nv_bfloat16 *__restrict__ x, int m, int k_in, float *__restrict__ sum_abs) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= k_in) return;
    float sum = 0.0f;
    for (int row = 0; row < m; row++) sum += fabsf(__bfloat162float(x[(size_t)row * k_in + k]));
    sum_abs[k] += sum;
}

unsigned blocks_for(size_t n) { return (unsigned)((n + 255) / 256); }

}  // namespace

cudaError_t resid_build(const __nv_bfloat16 *w, Int8PinLinear base, uint8_t *codes, float *col_norm,
                        ResidBuildStats *stats, cudaStream_t stream) {
    if (base.n_out % 2 != 0) return cudaErrorInvalidValue;
    build_kernel<<<blocks_for((size_t)base.k_in * (base.n_out / 2)), 256, 0, stream>>>(w, base, codes, stats);
    col_norm_kernel<<<base.k_in, 256, 0, stream>>>(codes, base, col_norm);
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

cudaError_t resid_dequant(ResidCols r, Int8PinLinear base, __nv_bfloat16 *rg, cudaStream_t stream) {
    if (!r.chan && r.c != base.k_in) return cudaErrorInvalidValue;
    dequant_kernel<<<blocks_for((size_t)base.n_out * r.c), 256, 0, stream>>>(r, base, rg);
    return cudaGetLastError();
}

cudaError_t resid_accumulate_abs(const __nv_bfloat16 *x, int m, int k_in, float *sum_abs, cudaStream_t stream) {
    accumulate_abs_kernel<<<blocks_for(k_in), 256, 0, stream>>>(x, m, k_in, sum_abs);
    return cudaGetLastError();
}
