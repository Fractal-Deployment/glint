#ifndef GLINT_INT8_MMA_GEMM_H
#define GLINT_INT8_MMA_GEMM_H

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INT8_MMA_BLOCK 64

/* One bf16_to_int8_pin_v1 linear on the device, as the pin stores it:
 * q is I8 [n_out, k_in] row-major, scale is F32 [n_out * k_in / 64] (one per 64
 * consecutive weights of the flattened tensor; with k_in % 64 == 0 that is
 * scale[n * (k_in / 64) + kb]), and w[n, k] = q[n, k] * scale[n * (k_in / 64) + k / 64]. */
typedef struct {
    const int8_t *q;
    const float *scale;
    int n_out;
    int k_in;
} Int8MmaLinear;

/* Scratch for the activation quantization: xq [m, k_in] bytes and sx [m, k_in / 64] floats. */
static inline size_t int8_mma_xq_bytes(int m, int k_in) { return (size_t)m * (size_t)k_in; }
static inline size_t int8_mma_sx_bytes(int m, int k_in) { return (size_t)m * (size_t)(k_in / INT8_MMA_BLOCK) * sizeof(float); }

/* Y[m, n] = sum_k X[m, k] * W[n, k] on INT8 tensor cores (mma.sync m16n8k32 s8, s32 accumulate).
 *
 * X bf16 [m, k_in] is quantized per (row, 64-block) to int8 with the pin's rule
 * (scale = amax / 127, round-to-nearest-even, clamp +-127) into xq/sx, then each 64-wide
 * K slice of W and X is staged in shared memory, multiplied as int8 x int8 -> int32, scaled by
 * sx * sw and added into fp32, and discarded. The weights are never unfolded to a wider type.
 *
 * Requires k_in % 64 == 0 and m, n_out >= 1. Returns cudaErrorInvalidValue otherwise.
 * y is fp32 [m, n_out] row-major. xq/sx are caller-provided device scratch (sizes above). */
cudaError_t int8_mma_gemm(const __nv_bfloat16 *x, Int8MmaLinear w, float *y, int m,
                          int8_t *xq, float *sx, cudaStream_t stream);

/* Late plug: y = X * (base + plug)^T, where the plug is a p7_plug_res4_v1 slice for this linear
 * (plug_q4: ceil(n_out*k_in/2) bytes, plug_scale: one F32 per 64 weights, the base's block order).
 * The activation is quantized once and shared; the plug is unpacked into plug_i8 (n_out*k_in bytes of
 * caller scratch) and added into y by a second accumulating pass. */
cudaError_t int8_mma_gemm_plug(const __nv_bfloat16 *x, Int8MmaLinear w, const uint8_t *plug_q4,
                               const float *plug_scale, float *y, int m, int8_t *xq, float *sx,
                               int8_t *plug_i8, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
