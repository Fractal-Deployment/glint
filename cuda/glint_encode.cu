/* GLINT encode on GPU. One thread per weight after a block absmax reduce.
 * nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC -o libglint_encode.so \
 *      cuda/glint_encode.cu -Iinclude
 */
#include "../include/glint_encode.h"
#include "../include/glint_cells.h"
#include "../include/glint_meta.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define GLINT_PARENT_N 16
#define GLINT_CHILD_N 16

__constant__ float d_parent[16];
__constant__ float d_cells[16][16];

static const float kParent[16] = {
    -1.0f,
    -0.696192801f,
    -0.525073051f,
    -0.394917488f,
    -0.284441382f,
    -0.184773430f,
    -0.091050036f,
    0.0f,
    0.079580300f,
    0.160930201f,
    0.246112302f,
    0.337915242f,
    0.440709829f,
    0.562617004f,
    0.722956836f,
    1.0f,
};

__global__ void k_bf16_to_f32(const uint16_t *src, float *dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    dst[i] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&src[i]));
}

__global__ void k_absmax(const float *w, float *absmax, int64_t n, int bs) {
    int b = (int)blockIdx.x;
    int64_t base = (int64_t)b * bs;
    float m = 0.f;
    for (int t = (int)threadIdx.x; t < bs; t += (int)blockDim.x) {
        int64_t i = base + t;
        if (i < n) {
            float a = fabsf(w[i]);
            if (a > m)
                m = a;
        }
    }
    __shared__ float sh[256];
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            float o = sh[threadIdx.x + s];
            if (o > sh[threadIdx.x])
                sh[threadIdx.x] = o;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        absmax[b] = sh[0];
}

__device__ __forceinline__ int nearest16(const float *tab, float v) {
    int best = 0;
    float bd = fabsf(v - tab[0]);
#pragma unroll
    for (int k = 1; k < 16; ++k) {
        float d = fabsf(v - tab[k]);
        if (d < bd) {
            bd = d;
            best = k;
        }
    }
    return best;
}

__global__ void k_encode_unpacked(
    const float *w, const float *absmax, uint8_t *hole, uint8_t *plug, int64_t n, int bs)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float s = absmax[i / bs];
    float v = (s == 0.f) ? 0.f : (w[i] / s);
    int h = nearest16(d_parent, v);
    int p = nearest16(d_cells[h], v);
    hole[i] = (uint8_t)h;
    plug[i] = (uint8_t)p;
}

__global__ void k_pack(const uint8_t *src, uint8_t *dst, int64_t n) {
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t npack = (n + 1) / 2;
    if (e >= npack)
        return;
    uint8_t a = src[e * 2];
    uint8_t b = (e * 2 + 1 < n) ? src[e * 2 + 1] : 0;
    dst[e] = (uint8_t)((a & 15) | ((b & 15) << 4));
}

static int load_cells() {
    static int once = 0;
    if (once)
        return 0;
    float host_cells[16][16];
    memcpy(host_cells, GLINT_CELLS, sizeof(host_cells));
    if (cudaMemcpyToSymbol(d_parent, kParent, sizeof(kParent)) != cudaSuccess)
        return -1;
    if (cudaMemcpyToSymbol(d_cells, host_cells, sizeof(host_cells)) != cudaSuccess)
        return -1;
    once = 1;
    return 0;
}

int glint_encode_n_absmax(int64_t n, int blocksize) {
    if (blocksize <= 0)
        return 0;
    return (int)((n + blocksize - 1) / blocksize);
}

int glint_encode_n_packed(int64_t n) { return (int)((n + 1) / 2); }

static int encode_dev_f32(const float *d_w, int64_t n, int blocksize, uint8_t *h_hole, uint8_t *h_plug, float *h_am) {
    if (load_cells() != 0)
        return -2;
    int nblocks = glint_encode_n_absmax(n, blocksize);
    int npack = glint_encode_n_packed(n);
    float *d_am = NULL;
    uint8_t *d_hu = NULL, *d_pu = NULL, *d_hp = NULL, *d_pp = NULL;
    int rc = -3;
    if (cudaMalloc(&d_am, (size_t)nblocks * sizeof(float)) != cudaSuccess)
        goto done;
    if (cudaMalloc(&d_hu, (size_t)n) != cudaSuccess)
        goto done;
    if (cudaMalloc(&d_pu, (size_t)n) != cudaSuccess)
        goto done;
    if (cudaMalloc(&d_hp, (size_t)npack) != cudaSuccess)
        goto done;
    if (cudaMalloc(&d_pp, (size_t)npack) != cudaSuccess)
        goto done;

    k_absmax<<<nblocks, 256>>>(d_w, d_am, n, blocksize);
    {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        k_encode_unpacked<<<blocks, threads>>>(d_w, d_am, d_hu, d_pu, n, blocksize);
        int pblocks = (int)((npack + threads - 1) / threads);
        k_pack<<<pblocks, threads>>>(d_hu, d_hp, n);
        k_pack<<<pblocks, threads>>>(d_pu, d_pp, n);
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
        goto done;
    if (cudaMemcpy(h_am, d_am, (size_t)nblocks * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        goto done;
    if (cudaMemcpy(h_hole, d_hp, (size_t)npack, cudaMemcpyDeviceToHost) != cudaSuccess)
        goto done;
    if (cudaMemcpy(h_plug, d_pp, (size_t)npack, cudaMemcpyDeviceToHost) != cudaSuccess)
        goto done;
    rc = 0;
done:
    cudaFree(d_am);
    cudaFree(d_hu);
    cudaFree(d_pu);
    cudaFree(d_hp);
    cudaFree(d_pp);
    return rc;
}

int glint_encode_f32_host(const float *w, int64_t n, int blocksize, uint8_t *hole_packed, uint8_t *plug_packed, float *absmax) {
    float *d_w = NULL;
    int rc = -1;
    if (cudaMalloc(&d_w, (size_t)n * sizeof(float)) != cudaSuccess)
        return -1;
    if (cudaMemcpy(d_w, w, (size_t)n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_w);
        return -1;
    }
    rc = encode_dev_f32(d_w, n, blocksize, hole_packed, plug_packed, absmax);
    cudaFree(d_w);
    return rc;
}

int glint_encode_bf16_host(const uint16_t *w_bf16, int64_t n, int blocksize, uint8_t *hole_packed, uint8_t *plug_packed, float *absmax) {
    uint16_t *d_b = NULL;
    float *d_w = NULL;
    int rc = -1;
    if (cudaMalloc(&d_b, (size_t)n * sizeof(uint16_t)) != cudaSuccess)
        return -1;
    if (cudaMalloc(&d_w, (size_t)n * sizeof(float)) != cudaSuccess) {
        cudaFree(d_b);
        return -1;
    }
    if (cudaMemcpy(d_b, w_bf16, (size_t)n * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)
        goto done;
    {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        k_bf16_to_f32<<<blocks, threads>>>(d_b, d_w, n);
    }
    rc = encode_dev_f32(d_w, n, blocksize, hole_packed, plug_packed, absmax);
done:
    cudaFree(d_b);
    cudaFree(d_w);
    return rc;
}
