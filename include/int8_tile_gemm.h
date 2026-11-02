#ifndef GLINT_INT8_TILE_GEMM_H
#define GLINT_INT8_TILE_GEMM_H

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define INT8_PIN_BLOCK 64

/* One bf16_to_int8_pin_v1 linear resident on the device, as the pin stores it:
 * q is I8 [n_out, k_in] row-major, scale is F32 [ceil(n_out * k_in / 64)],
 * and w[i] = q[i] * scale[i / 64] over the flattened index i. */
typedef struct {
    const int8_t *q;
    const float *scale;
    int n_out;
    int k_in;
} Int8PinLinear;

/* Y[m, n] = sum_k X[m, k] * bf16(q[n, k] * scale) for X bf16 [m, k_in] row-major
 * and Y fp32 [m, n_out] row-major. Each weight tile is unfolded to bf16 in
 * registers, rounded exactly as a bf16 dequant of the pin would be, and fed to
 * bf16 mma.sync with fp32 accumulate. */
cudaError_t int8_tile_gemm_bf16(const __nv_bfloat16 *x, Int8PinLinear w, float *y, int m,
                                cudaStream_t stream);

#endif
