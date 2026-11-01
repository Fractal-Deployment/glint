#ifndef GLINT_HTILE_H_
#define GLINT_HTILE_H_
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* y[M,N] = x[M,K] * W^T with W[n,k] = GLINT_CELLS[hole][plug] * absmax[(n*K+k)/blocksize].
 * Enqueued on `stream`; a plug from glint_plug_streamer must be read on its compute stream. */
int launch_glint_htile_s3_f32_stream(
    const uint8_t *d_hole, const uint8_t *d_plug, const float *d_absmax,
    const __nv_bfloat16 *d_x, float *d_y,
    int M, int N, int K, int blocksize, cudaStream_t stream);
/* Same kernel on the legacy default stream. */
int launch_glint_htile_s3_f32(
    const uint8_t *d_hole, const uint8_t *d_plug, const float *d_absmax,
    const __nv_bfloat16 *d_x, float *d_y,
    int M, int N, int K, int blocksize);
#ifdef __cplusplus
}
#endif
#endif
