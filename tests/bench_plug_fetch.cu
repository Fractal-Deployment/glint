/* tests/bench_plug_fetch.cu — time to bring a selected fraction of one Phi-4-mini layer's plug from pinned host RAM into VRAM.
 * Layout under test: K-major (channel c of a linear holds its n_out plug nibbles contiguously, n_out/2 bytes), the layout a
 * channel- or block-selected fetch needs. A unit is one channel ("channel") or 64 adjacent channels ("block"). The four linears
 * of the layer (QKV, O, GATE_UP, DOWN) are fetched back to back; the selection is a fixed-seed random subset of units
 * (it fixes the unit count and size, not which units an activation would pick). Adjacent selected units are merged into runs.
 * Methods: "memcpy" = one cudaMemcpyAsync per run; "zerocopy" = a gather kernel reading mapped pinned memory over PCIe (DecDEC style).
 * Output (TSV): gran, f, method, bytes, runs, ms_per_layer, GB_per_s.   usage: bench_plug_fetch [reps=20]
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <vector>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

struct Lin { int n_out, k_in; };
static const Lin LIN[4] = {{5120, 3072}, {3072, 3072}, {16384, 3072}, {3072, 8192}};

__global__ void k_gather(const uint4 *src, uint4 *dst, const uint64_t *off16, const uint64_t *len16, int nruns) {
    int r = blockIdx.x; if (r >= nruns) return;
    uint64_t o = off16[r], n = len16[r];
    for (uint64_t i = threadIdx.x; i < n; i += blockDim.x) dst[o + i] = src[o + i];
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    size_t base[4], total = 0;
    for (int j = 0; j < 4; j++) { base[j] = total; total += (size_t)LIN[j].k_in * (LIN[j].n_out / 2); total = (total + 15) & ~(size_t)15; }
    uint8_t *h = NULL, *d = NULL, *hd = NULL;
    CK(cudaHostAlloc((void **)&h, total, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostGetDevicePointer((void **)&hd, h, 0));
    CK(cudaMalloc((void **)&d, total));
    memset(h, 0xA5, total);
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    printf("layer_plug_bytes\t%zu\n", total);
    printf("gran\tf\tmethod\tbytes\truns\tms_per_layer\tGB_per_s\n");
    const double FR[] = {1.0, 0.5, 0.25, 0.1, 0.05, 0.01};
    for (int g = 0; g < 2; g++) {                                     /* 0 channel, 1 block of 64 */
        for (double f : FR) {
            std::vector<uint64_t> off, len;                           /* byte runs over the layer buffer */
            uint64_t bytes = 0;
            for (int j = 0; j < 4; j++) {
                const uint64_t chb = (uint64_t)LIN[j].n_out / 2, usz = g ? 64 * chb : chb;
                const int nu = g ? LIN[j].k_in / 64 : LIN[j].k_in;
                int ns = (int)std::ceil(f * nu - 1e-9); if (ns < 1) ns = 1; if (ns > nu) ns = nu;
                std::vector<int> idx(nu); for (int i = 0; i < nu; i++) idx[i] = i;
                unsigned s = 12345u + j; for (int i = nu - 1; i > 0; i--) { s = s * 1664525u + 1013904223u; std::swap(idx[i], idx[s % (unsigned)(i + 1)]); }
                std::sort(idx.begin(), idx.begin() + ns);
                for (int i = 0; i < ns;) {
                    int k = i; while (k + 1 < ns && idx[k + 1] == idx[k] + 1) k++;
                    off.push_back(base[j] + (uint64_t)idx[i] * usz); len.push_back((uint64_t)(k - i + 1) * usz); bytes += len.back(); i = k + 1;
                }
            }
            for (int method = 0; method < 2; method++) {
                std::vector<uint64_t> o16(off.size()), l16(len.size());
                for (size_t r = 0; r < off.size(); r++) { o16[r] = off[r] / 16; l16[r] = (len[r] + 15) / 16; }
                uint64_t *d_o = NULL, *d_l = NULL;
                CK(cudaMalloc((void **)&d_o, o16.size() * 8)); CK(cudaMalloc((void **)&d_l, l16.size() * 8));
                CK(cudaMemcpy(d_o, o16.data(), o16.size() * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_l, l16.data(), l16.size() * 8, cudaMemcpyHostToDevice));
                auto run = [&]() {
                    if (method == 0) for (size_t r = 0; r < off.size(); r++) CK(cudaMemcpyAsync(d + off[r], h + off[r], len[r], cudaMemcpyHostToDevice, st));
                    else k_gather<<<(unsigned)off.size(), 256, 0, st>>>((const uint4 *)hd, (uint4 *)d, d_o, d_l, (int)off.size());
                };
                for (int w = 0; w < 3; w++) run();
                CK(cudaStreamSynchronize(st));
                CK(cudaEventRecord(e0, st)); for (int r = 0; r < reps; r++) run(); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
                float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= reps;
                printf("%s\t%.2f\t%s\t%llu\t%zu\t%.4f\t%.2f\n", g ? "block" : "channel", f, method ? "zerocopy" : "memcpy", (unsigned long long)bytes, off.size(), ms, bytes / (ms * 1e-3) / 1e9);
                fflush(stdout);
                CK(cudaFree(d_o)); CK(cudaFree(d_l));
            }
        }
    }
    return 0;
}
