/* test_plug_evict_real.cu — plug streamer on REAL weights with poison-after-release.
 *
 * Each "layer" plug is one real tensor from a safetensors file (mmap'd via safetensors_c.h).
 * For every layer: acquire -> compare slot byte-for-byte to a resident device copy
 * -> overwrite the slot with poison -> release. If a later layer ever read a slot before its
 * copy landed, or an earlier layer's data were still needed after release, the compare fails.
 * Two full passes run so a stale slot from pass 1 cannot satisfy pass 2.
 * Control: with the poison moved BEFORE the compare, mismatches must appear; otherwise the
 * harness could not see poison at all.
 *
 * Build: nvcc -O2 -arch=sm_86 -Iinclude tests/test_plug_evict_real.cu cuda/glint_plug_streamer.cu
 * Run:   ./a.out [model.safetensors] [n_layers]
 */
#include "glint_plug_streamer.h"
#include "safetensors_c.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 2; } } while (0)

__global__ void k_compare(const uint8_t *slot, const uint8_t *ref, uint64_t n, unsigned long long *bad) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && slot[i] != ref[i]) atomicAdd(bad, 1ull);
}
__global__ void k_poison(uint8_t *slot, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) slot[i] = 0xA5;
}

static unsigned long long run_pass(GlintPlugStreamer *s, int n_layers, const std::vector<uint64_t> &bytes,
                                   const std::vector<uint8_t *> &ref, cudaStream_t compute,
                                   unsigned long long *d_bad, int poison_before_read) {
    cudaMemset(d_bad, 0, sizeof(unsigned long long));
    if (glint_plug_streamer_start(s) != 0) return ~0ull;
    for (int l = 0; l < n_layers; ++l) {
        const uint8_t *plug = NULL;
        if (glint_plug_streamer_acquire(s, l, &plug) != 0) return ~0ull;
        int blocks = (int)((bytes[l] + 255) / 256);
        if (poison_before_read) k_poison<<<blocks, 256, 0, compute>>>((uint8_t *)plug, bytes[l]);
        k_compare<<<blocks, 256, 0, compute>>>(plug, ref[l], bytes[l], d_bad);
        if (!poison_before_read) k_poison<<<blocks, 256, 0, compute>>>((uint8_t *)plug, bytes[l]);
        if (glint_plug_streamer_release(s, l) != 0) return ~0ull;
    }
    cudaStreamSynchronize(compute);
    unsigned long long bad = 0;
    cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost);
    return bad;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/workspace/pins/debug_hf_smollm2/SmolLM2-135M/model.safetensors";
    int want = argc > 2 ? atoi(argv[2]) : 16;
    printf("=== TEST PLUG STREAMER: REAL WEIGHTS + POISON-AFTER-RELEASE ===\nfile: %s\n", path);

    SafeTensors st;
    int rc = safetensors_open(&st, path);
    if (rc != 0) { printf("safetensors_open rc=%d\n", rc); return 2; }

    /* Layers = the first `want` tensors of at least 64 KiB, in file order. */
    std::vector<const TensorEntry *> pick;
    for (int i = 0; i < st.n_entries && (int)pick.size() < want; ++i)
        if (st.entries[i].end - st.entries[i].begin >= 65536) pick.push_back(&st.entries[i]);
    int n = (int)pick.size();
    if (n < 3) { printf("need >=3 tensors >=64KiB, got %d\n", n); return 2; }

    std::vector<uint64_t> bytes(n);
    std::vector<uint8_t *> ref(n);
    uint64_t total = 0;
    for (int l = 0; l < n; ++l) {
        bytes[l] = pick[l]->end - pick[l]->begin;
        total += bytes[l];
        CK(cudaMalloc(&ref[l], bytes[l]));
        CK(cudaMemcpy(ref[l], pick[l]->data, bytes[l], cudaMemcpyHostToDevice));
    }
    printf("layers=%d total=%.1f MiB (real tensors, mixed sizes)\n", n, total / 1048576.0);

    cudaStream_t copy, compute;
    CK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    GlintPlugStreamer *s = NULL;
    if (glint_plug_streamer_create(&s, n, bytes.data(), copy, compute) != 0) { printf("create failed\n"); return 2; }
    for (int l = 0; l < n; ++l)
        if (glint_plug_streamer_set_host_data(s, l, pick[l]->data, bytes[l]) != 0) { printf("set_host_data %d failed\n", l); return 2; }

    unsigned long long *d_bad;
    CK(cudaMalloc(&d_bad, sizeof(*d_bad)));

    int fails = 0;
    for (int pass = 0; pass < 2; ++pass) {
        glint_plug_streamer_reset_stats(s);
        unsigned long long bad = run_pass(s, n, bytes, ref, compute, d_bad, 0);
        printf("pass %d (poison after read): mismatched bytes = %llu\n", pass, bad);
        if (bad != 0) fails++;
    }
    glint_plug_streamer_reset_stats(s);
    unsigned long long ctl = run_pass(s, n, bytes, ref, compute, d_bad, 1);
    printf("control (poison BEFORE read): mismatched bytes = %llu (must be > 0)\n", ctl);
    if (ctl == 0) { printf("control did not detect poison: harness is blind\n"); fails++; }

    glint_plug_streamer_destroy(s);
    safetensors_close(&st);
    if (fails) { printf("PLUG EVICT REAL: FAILED (%d)\n", fails); return 1; }
    printf("PLUG EVICT REAL PASSED.\n");
    return 0;
}
