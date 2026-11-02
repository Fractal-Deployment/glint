#ifndef GLINT_RESIDUAL_CORRECTION_H
#define GLINT_RESIDUAL_CORRECTION_H

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define RESID_SCALE_BLOCK 64

/* The residual R = W_bf16 - W_base of a linear whose base was quantized per block of 64 along the flattened
 * row (the INT8 and INT4 pins), stored in units of the base's own block scale:
 * R[n, k] ~= scale[(n * k_in + k) / 64] * code * range / qmax with qmax = 2^(bits - 1) - 1 and code in
 * [-qmax, qmax]. Rounding to a base grid bounds |R| by scale / 2 and the bf16 dequant of the base adds at most
 * qmax_base * scale * 2^-8, so range 1 covers an INT8 base (127/256) and range 0.55 an INT4 base (7/256).
 *
 * Codes are channel-major so one input channel k is one contiguous row of n_out * bits / 8 bytes. At 4 bits,
 * byte n / 2 of a row holds output n, low nibble for even n, and n_out must be even; at 8 bits, byte n is a
 * signed code. */
typedef struct {
    const float *scale;
    int n_out;
    int k_in;
    int bits;
    float range;
} ResidLinear;

typedef struct {
    const uint8_t *codes;
    const uint16_t *chan; /* channel of each code row; NULL means row j is channel j and c == k_in */
    int c;
} ResidCols;

typedef struct {
    unsigned long long clamped;
    unsigned int max_rel_bits; /* float bits of max |R / scale| over the linear */
} ResidBuildStats;

static inline int resid_qmax(int bits) { return (1 << (bits - 1)) - 1; }

static inline uint64_t resid_cols_bytes(int c, int n_out, int bits) {
    return (uint64_t)c * (uint64_t)n_out * (uint64_t)bits / 8;
}

/* codes [k_in][row bytes] and col_norm[k] = ||R_hat[:, k]||_2 from the BF16 weight w and the base dequantized
 * to bf16, w_base, both [n_out, k_in]. stats is a device pointer the kernel accumulates into. Refuses bits other
 * than 4 and 8, and an odd n_out at 4 bits. */
cudaError_t resid_build(const __nv_bfloat16 *w, const __nv_bfloat16 *w_base, ResidLinear r, uint8_t *codes,
                        float *col_norm, ResidBuildStats *stats, cudaStream_t stream);

/* Per row of x [m][k_in], keeps the ksel channels with the largest |x| * chan_weight (chan_weight NULL means
 * 1), lowest index first among equal scores, and writes xg [m][k_in] with every other channel zero.
 * chan_used, when not NULL, gets 1 at every channel some row kept. */
cudaError_t resid_select_topk(const __nv_bfloat16 *x, int m, int k_in, const float *chan_weight, int ksel,
                              __nv_bfloat16 *xg, uint8_t *chan_used, cudaStream_t stream);

/* xg [m][c] = x [m][chan[j]]. */
cudaError_t resid_gather(const __nv_bfloat16 *x, int m, int k_in, const uint16_t *chan, int c, __nv_bfloat16 *xg,
                         cudaStream_t stream);

/* rg [n_out][c] = bf16(R_hat[n, chan[j]]). */
cudaError_t resid_dequant(ResidCols cols, ResidLinear r, __nv_bfloat16 *rg, cudaStream_t stream);

/* y [m][n_out] += xg [m][c] * rg [n_out][c]^T, fp32 accumulate. */
static inline cublasStatus_t resid_accumulate(cublasHandle_t h, const __nv_bfloat16 *xg, const __nv_bfloat16 *rg,
                                              float *y, int m, int n_out, int c) {
    const float one = 1.0f;
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n_out, m, c, &one, rg, CUDA_R_16BF, c, xg, CUDA_R_16BF, c, &one,
                        y, CUDA_R_32F, n_out, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
}

/* sum_abs[k] += sum over rows of |x[m][k]|. */
cudaError_t resid_accumulate_abs(const __nv_bfloat16 *x, int m, int k_in, float *sum_abs, cudaStream_t stream);

#endif
