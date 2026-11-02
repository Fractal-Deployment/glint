#ifndef GLINT_INT4_TILE_GEMM_H
#define GLINT_INT4_TILE_GEMM_H

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define INT4_PIN_BLOCK 64
#define INT4_QMAX 7

/* One bf16_to_int4_pin_v1 linear resident on the device. Over the flattened row-major index i of [n_out, k_in],
 * w[i] = code(i) * scale[i / 64], where code(i) is the two's-complement nibble of q[i / 2], low nibble for even i,
 * in [-7, 7]. q holds ceil(n_out * k_in / 2) bytes and scale F32 [ceil(n_out * k_in / 64)]. */
typedef struct {
    const uint8_t *q;
    const float *scale;
    int n_out;
    int k_in;
} Int4PinLinear;

/* Y[m, n] = sum_k X[m, k] * bf16(code(n, k) * scale) for X bf16 [m, k_in] row-major and Y fp32 [m, n_out]
 * row-major. Each weight tile is unfolded to bf16 in registers and fed to bf16 mma.sync with fp32 accumulate.
 * Needs sm_80 or newer. Before checking its arguments it returns, and clears, any error already pending on the
 * calling thread, without launching. */
cudaError_t int4_tile_gemm_bf16(const __nv_bfloat16 *x, Int4PinLinear w, float *y, int m, cudaStream_t stream);

/* w_bf16[i] = bf16(code(i) * scale[i / 64]), the same rounding the tile GEMM applies. */
cudaError_t int4_dequant_bf16(Int4PinLinear w, __nv_bfloat16 *w_bf16, cudaStream_t stream);

#endif
