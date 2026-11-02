/* Each read spins first, so a refill that skips its slot_free wait lands before
 * the read. With 1 or 2 layers every refill rewrites a slot's own bytes, so
 * only 3+ layers can observe a skipped wait.
 */
#include "../include/glint_plug_streamer.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const uint64_t kLayerBytes = 4 * 1024 * 1024;
static const int kPasses = 3;
static const double kDelayMs = 5.0;
static int g_failed = 0;
static long long g_delay_cycles = 0;

static void check(int ok, const char *what, int n_layers) {
    printf("[%s] n_layers=%d: %s\n", ok ? "PASS" : "FAIL", n_layers, what);
    g_failed += !ok;
}

__global__ void k_spin(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {
    }
}

__global__ void k_verify_plug(const uint8_t *plug, int layer, uint64_t nbytes, int *d_err) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nbytes && plug[idx] != (uint8_t)((layer * 37 + (idx % 251)) & 0xFF))
        atomicAdd(d_err, 1);
}

static GlintPlugStreamer *make_streamer(int n_layers, cudaStream_t copy_stream, cudaStream_t compute_stream) {
    uint64_t layer_bytes[64];
    for (int l = 0; l < n_layers; ++l) layer_bytes[l] = kLayerBytes;
    GlintPlugStreamer *s = NULL;
    if (glint_plug_streamer_create(&s, n_layers, layer_bytes, copy_stream, compute_stream) != 0) {
        fprintf(stderr, "glint_plug_streamer_create(n_layers=%d) failed\n", n_layers);
        exit(2);
    }
    uint8_t *host_buf = (uint8_t *)malloc(kLayerBytes);
    for (int l = 0; l < n_layers; ++l) {
        for (uint64_t i = 0; i < kLayerBytes; ++i) host_buf[i] = (uint8_t)((l * 37 + (i % 251)) & 0xFF);
        if (glint_plug_streamer_set_host_data(s, l, host_buf, kLayerBytes) != 0) {
            fprintf(stderr, "set_host_data(%d) failed\n", l);
            exit(2);
        }
    }
    free(host_buf);
    return s;
}

static void run_passes(int n_layers, cudaStream_t copy_stream, cudaStream_t compute_stream) {
    GlintPlugStreamer *s = make_streamer(n_layers, copy_stream, compute_stream);
    int *d_err = NULL;
    cudaMalloc(&d_err, sizeof(int));
    cudaMemset(d_err, 0, sizeof(int));
    int api_ok = 1;
    const int threads = 256;
    const int blocks = (int)((kLayerBytes + threads - 1) / threads);
    for (int p = 0; p < kPasses; ++p) {
        api_ok &= glint_plug_streamer_start(s) == 0;
        for (int l = 0; l < n_layers; ++l) {
            const uint8_t *dev_plug = NULL;
            api_ok &= glint_plug_streamer_acquire(s, l, &dev_plug) == 0 && dev_plug;
            k_spin<<<1, 1, 0, compute_stream>>>(g_delay_cycles);
            k_verify_plug<<<blocks, threads, 0, compute_stream>>>(dev_plug, l, kLayerBytes, d_err);
            api_ok &= glint_plug_streamer_release(s, l) == 0;
        }
    }
    int h_err = 0;
    api_ok &= cudaDeviceSynchronize() == cudaSuccess;
    cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_err);
    GlintStreamerStats stats;
    api_ok &= glint_plug_streamer_get_stats(s, &stats) == 0;
    glint_plug_streamer_destroy(s);

    char what[128];
    check(api_ok, "start/acquire/release return 0 on every pass", n_layers);
    snprintf(what, sizeof(what), "%d delayed passes read only their own layer's bytes (%d mismatches)", kPasses, h_err);
    check(h_err == 0, what, n_layers);
    check(stats.total_h2d_bytes == (uint64_t)kPasses * n_layers * kLayerBytes, "every layer copied once per pass", n_layers);
}

static void release_order(cudaStream_t copy_stream, cudaStream_t compute_stream) {
    const int n = 3;
    GlintPlugStreamer *s = make_streamer(n, copy_stream, compute_stream);
    const uint8_t *dev_plug = NULL;
    check(glint_plug_streamer_release(s, 0) == -11, "release(0) before start is refused", n);
    check(glint_plug_streamer_start(s) == 0, "start", n);
    check(glint_plug_streamer_release(s, 0) == -11, "release(0) before acquire(0) is refused", n);
    check(glint_plug_streamer_acquire(s, 0, &dev_plug) == 0, "acquire(0)", n);
    check(glint_plug_streamer_release(s, 1) == -11, "release(1) before release(0) is refused", n);
    check(glint_plug_streamer_release(s, 0) == 0, "release(0)", n);
    check(glint_plug_streamer_release(s, 0) == -11, "second release(0) is refused", n);
    check(glint_plug_streamer_release(s, 1) == -11, "release(1) before acquire(1) is refused", n);
    check(glint_plug_streamer_acquire(s, 1, &dev_plug) == 0 && glint_plug_streamer_release(s, 1) == 0 &&
              glint_plug_streamer_acquire(s, 2, &dev_plug) == 0 && glint_plug_streamer_release(s, 2) == 0,
          "in-order acquire/release of layers 1 and 2", n);
    check(glint_plug_streamer_start(s) == 0, "next pass starts after the last release", n);
    cudaDeviceSynchronize();
    glint_plug_streamer_destroy(s);
}

int main(void) {
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    g_delay_cycles = (long long)(kDelayMs * clock_khz);

    cudaStream_t copy_stream, compute_stream;
    cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);

    const int layer_counts[] = {1, 2, 3, 32};
    for (int i = 0; i < (int)(sizeof(layer_counts) / sizeof(layer_counts[0])); ++i)
        run_passes(layer_counts[i], copy_stream, compute_stream);
    release_order(copy_stream, compute_stream);

    cudaStreamDestroy(copy_stream);
    cudaStreamDestroy(compute_stream);
    printf("%s: %d failed checks\n", g_failed ? "FAIL" : "PASS", g_failed);
    return g_failed ? 1 : 0;
}
