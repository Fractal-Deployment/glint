/* Real Phi-4-mini BF16 weights -> CUDA GLINT encode -> plugs streamed through
 * the two-slot streamer into H-TILE. Gates: streamed output is bit-identical to
 * the same H-TILE run with plugs resident over back-to-back passes, and
 * resident output matches a host decode GEMM within an f32 summation bound.
 * Poisoned passes overwrite each released slot with random bytes. That copy
 * waits on the reader, which also orders every later refill behind it, so only
 * the delayed passes, which spin before each launch, expose a skipped slot wait.
 * Error against the BF16 GEMM of the original weights is printed, not gated.
 *
 * ./test_streamer_real_weights <model-00001-of-00002.safetensors> [n_layers=2] [M=4]
 */
#include "glint_cells.h"
#include "glint_encode.h"
#include "glint_htile.h"
#include "glint_plug_streamer.h"
#include "safetensors_c.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define CK(call)                                                                   \
    do {                                                                           \
        cudaError_t e_ = (call);                                                   \
        if (e_ != cudaSuccess) {                                                   \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call,          \
                    cudaGetErrorString(e_));                                       \
            exit(2);                                                               \
        }                                                                          \
    } while (0)

#define REQUIRE(cond, ...)                                                         \
    do {                                                                           \
        if (!(cond)) {                                                             \
            fprintf(stderr, __VA_ARGS__);                                          \
            fprintf(stderr, "\n");                                                 \
            exit(2);                                                               \
        }                                                                          \
    } while (0)

static const int kBlocksize = 64;
static const double kDelayMs = 20.0;
enum PassKind { PLAIN, POISON, DELAY };
static const PassKind kPassKinds[] = {PLAIN, PLAIN, POISON, POISON, DELAY, DELAY};
static const int kPasses = sizeof(kPassKinds) / sizeof(kPassKinds[0]);
static const char *kPassKindNames[] = {"plain", "poisoned", "delayed"};
static const char *kMatrices[] = {"self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"};

typedef struct {
    char name[128];
    int N, K;
    uint64_t n_packed, n_absmax;
    const uint16_t *w_bf16;
    uint8_t *hole, *plug;
    float *absmax;
    __nv_bfloat16 *x;
    uint8_t *d_hole, *d_plug;
    float *d_absmax;
    __nv_bfloat16 *d_x;
    float *d_y_resident;
    float *d_y_streamed[kPasses];
} Unit;

__global__ void k_spin(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {
    }
}

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint64_t next_u64(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

static float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static void load_unit(Unit *u, const SafeTensors *st, int layer, const char *matrix, int M) {
    snprintf(u->name, sizeof(u->name), "model.layers.%d.%s.weight", layer, matrix);
    const TensorEntry *e = safetensors_find(st, u->name);
    REQUIRE(e, "missing tensor %s", u->name);
    REQUIRE(strcmp(e->dtype, "BF16") == 0 && e->n_dims == 2, "%s: want 2-D BF16, got %s %d-D", u->name, e->dtype, e->n_dims);
    u->N = (int)e->shape[0];
    u->K = (int)e->shape[1];
    const int64_t n = (int64_t)u->N * u->K;
    REQUIRE(e->end - e->begin == (uint64_t)n * 2, "%s: byte range does not match shape", u->name);
    u->w_bf16 = (const uint16_t *)e->data;
    u->n_packed = (uint64_t)glint_encode_n_packed(n);
    u->n_absmax = (uint64_t)glint_encode_n_absmax(n, kBlocksize);
    u->hole = (uint8_t *)malloc(u->n_packed);
    u->plug = (uint8_t *)malloc(u->n_packed);
    u->absmax = (float *)malloc(u->n_absmax * sizeof(float));
    int rc = glint_encode_bf16_host(u->w_bf16, n, kBlocksize, u->hole, u->plug, u->absmax);
    REQUIRE(rc == 0, "%s: glint_encode_bf16_host rc=%d", u->name, rc);

    u->x = (__nv_bfloat16 *)malloc((size_t)M * u->K * sizeof(__nv_bfloat16));
    for (int64_t i = 0; i < (int64_t)M * u->K; ++i)
        u->x[i] = __float2bfloat16((float)((double)(next_u64() >> 11) / 9007199254740992.0 * 2.0 - 1.0));

    CK(cudaMalloc(&u->d_hole, u->n_packed));
    CK(cudaMalloc(&u->d_plug, u->n_packed));
    CK(cudaMalloc(&u->d_absmax, u->n_absmax * sizeof(float)));
    CK(cudaMalloc(&u->d_x, (size_t)M * u->K * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&u->d_y_resident, (size_t)M * u->N * sizeof(float)));
    for (int p = 0; p < kPasses; ++p)
        CK(cudaMalloc(&u->d_y_streamed[p], (size_t)M * u->N * sizeof(float)));
    CK(cudaMemcpy(u->d_hole, u->hole, u->n_packed, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(u->d_plug, u->plug, u->n_packed, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(u->d_absmax, u->absmax, u->n_absmax * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(u->d_x, u->x, (size_t)M * u->K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
}

static float *download(const float *d, size_t n) {
    float *h = (float *)malloc(n * sizeof(float));
    CK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

/* Worst-case error of f32 recursive summation of K rounded products. */
static double f32_dot_error_bound(int K, double sum_abs_terms) {
    return (double)(K + 2) * ldexp(1.0, -23) * sum_abs_terms;
}

static int compare_host(const Unit *u, int M, const float *y_gpu) {
    double worst_ratio = 0.0, max_abs = 0.0, err2 = 0.0, ref2 = 0.0;
    int64_t over = 0;
    for (int n = 0; n < u->N; ++n) {
        for (int m = 0; m < M; ++m) {
            double glint = 0.0, mag = 0.0, bf16 = 0.0;
            const __nv_bfloat16 *xr = u->x + (int64_t)m * u->K;
            for (int k = 0; k < u->K; ++k) {
                const int64_t i = (int64_t)n * u->K + k;
                const int h = (u->hole[i >> 1] >> (4 * (i & 1))) & 15;
                const int p = (u->plug[i >> 1] >> (4 * (i & 1))) & 15;
                const float w = GLINT_CELLS[h][p] * u->absmax[i / kBlocksize];
                const double xv = (double)__bfloat162float(xr[k]);
                glint += (double)w * xv;
                mag += fabs((double)w * xv);
                bf16 += (double)bf16_to_f32(u->w_bf16[i]) * xv;
            }
            const double got = (double)y_gpu[(int64_t)m * u->N + n];
            const double diff = fabs(got - glint);
            if (diff > max_abs) max_abs = diff;
            const double bound = f32_dot_error_bound(u->K, mag);
            over += diff > bound;
            if (bound > 0.0 && diff / bound > worst_ratio) worst_ratio = diff / bound;
            err2 += (got - bf16) * (got - bf16);
            ref2 += bf16 * bf16;
        }
    }
    const int ok = over == 0;
    printf("  %-40s host-decode max|d|=%.3e worst d/bound=%.3e [%s]  vs-BF16 rel_fro=%.4e\n",
           u->name, max_abs, worst_ratio, ok ? "PASS" : "FAIL", sqrt(err2 / ref2));
    return ok;
}

int main(int argc, char **argv) {
    REQUIRE(argc >= 2, "usage: %s <phi4-mini model-00001-of-00002.safetensors> [n_layers=2] [M=4]", argv[0]);
    const int n_layers = argc > 2 ? atoi(argv[2]) : 2;
    const int M = argc > 3 ? atoi(argv[3]) : 4;
    const int n_units = n_layers * 4;
    REQUIRE(n_layers >= 1 && n_units <= 64 && M >= 1, "need 1 <= n_layers <= 16 and M >= 1");

    SafeTensors st;
    int rc = safetensors_open(&st, argv[1]);
    REQUIRE(rc == 0, "safetensors_open(%s) rc=%d", argv[1], rc);
    printf("shard %s: %d tensors, %zu bytes\n", argv[1], st.n_entries, st.file_size);

    Unit *units = (Unit *)calloc((size_t)n_units, sizeof(Unit));
    uint64_t plug_bytes[64];
    uint64_t max_plug = 0;
    for (int u = 0; u < n_units; ++u) {
        load_unit(&units[u], &st, u / 4, kMatrices[u % 4], M);
        plug_bytes[u] = units[u].n_packed;
        if (plug_bytes[u] > max_plug) max_plug = plug_bytes[u];
        printf("unit %d %s N=%d K=%d plug=%llu B\n", u, units[u].name, units[u].N, units[u].K,
               (unsigned long long)plug_bytes[u]);
    }

    cudaStream_t copy_stream, compute_stream;
    CK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));

    for (int u = 0; u < n_units; ++u) {
        Unit *x = &units[u];
        REQUIRE(launch_glint_htile_s3_f32_stream(x->d_hole, x->d_plug, x->d_absmax, x->d_x, x->d_y_resident,
                                          M, x->N, x->K, kBlocksize, compute_stream) == 0, "resident launch %d", u);
    }
    CK(cudaDeviceSynchronize());

    GlintPlugStreamer *streamer = NULL;
    REQUIRE(glint_plug_streamer_create(&streamer, n_units, plug_bytes, copy_stream, compute_stream) == 0, "streamer create");
    for (int u = 0; u < n_units; ++u)
        REQUIRE(glint_plug_streamer_set_host_data(streamer, u, units[u].plug, plug_bytes[u]) == 0, "set_host_data %d", u);

    uint8_t *poison = NULL;
    CK(cudaHostAlloc((void **)&poison, max_plug, cudaHostAllocDefault));
    for (uint64_t i = 0; i < max_plug; ++i) poison[i] = (uint8_t)next_u64();
    cudaEvent_t released[64];
    for (int u = 0; u < n_units; ++u) CK(cudaEventCreateWithFlags(&released[u], cudaEventDisableTiming));
    int clock_khz = 0;
    CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    const long long delay_cycles = (long long)(kDelayMs * clock_khz);

    for (int p = 0; p < kPasses; ++p) {
        REQUIRE(glint_plug_streamer_start(streamer) == 0, "start pass %d", p);
        if (n_units > 1) {
            const uint8_t *early = NULL;
            REQUIRE(glint_plug_streamer_acquire(streamer, 1, &early) == -10, "acquire(1) before release(0) must be refused");
        }
        for (int u = 0; u < n_units; ++u) {
            Unit *x = &units[u];
            const uint8_t *d_slot = NULL;
            REQUIRE(glint_plug_streamer_acquire(streamer, u, &d_slot) == 0 && d_slot, "acquire %d", u);
            if (kPassKinds[p] == DELAY) k_spin<<<1, 1, 0, compute_stream>>>(delay_cycles);
            REQUIRE(launch_glint_htile_s3_f32_stream(x->d_hole, d_slot, x->d_absmax, x->d_x, x->d_y_streamed[p],
                                              M, x->N, x->K, kBlocksize, compute_stream) == 0, "streamed launch %d", u);
            REQUIRE(glint_plug_streamer_release(streamer, u) == 0, "release %d", u);
            if (kPassKinds[p] != POISON) continue;
            CK(cudaEventRecord(released[u], compute_stream));
            CK(cudaStreamWaitEvent(copy_stream, released[u], 0));
            CK(cudaMemcpyAsync((void *)d_slot, poison, plug_bytes[u], cudaMemcpyHostToDevice, copy_stream));
        }
    }
    CK(cudaDeviceSynchronize());

    int failed = 0;
    printf("streamed vs resident (bitwise, %d back-to-back passes):\n", kPasses);
    for (int u = 0; u < n_units; ++u) {
        const size_t ny = (size_t)M * units[u].N;
        float *res = download(units[u].d_y_resident, ny);
        for (int p = 0; p < kPasses; ++p) {
            float *str = download(units[u].d_y_streamed[p], ny);
            size_t mismatched = 0;
            for (size_t i = 0; i < ny; ++i) mismatched += memcmp(&res[i], &str[i], 4) != 0;
            printf("  %-40s pass %d (%s): %zu / %zu outputs differ [%s]\n", units[u].name, p,
                   kPassKindNames[kPassKinds[p]], mismatched, ny, mismatched ? "FAIL" : "PASS");
            failed += mismatched != 0;
            free(str);
        }
        free(res);
    }

    printf("resident H-TILE vs host decode GEMM, and vs BF16 GEMM of the source weights (reported):\n");
    for (int u = 0; u < n_units; ++u) {
        float *res = download(units[u].d_y_resident, (size_t)M * units[u].N);
        failed += !compare_host(&units[u], M, res);
        free(res);
    }

    glint_plug_streamer_destroy(streamer);
    for (int u = 0; u < n_units; ++u) {
        Unit *x = &units[u];
        cudaEventDestroy(released[u]);
        cudaFree(x->d_hole);
        cudaFree(x->d_plug);
        cudaFree(x->d_absmax);
        cudaFree(x->d_x);
        cudaFree(x->d_y_resident);
        for (int p = 0; p < kPasses; ++p) cudaFree(x->d_y_streamed[p]);
        free(x->hole);
        free(x->plug);
        free(x->absmax);
        free(x->x);
    }
    cudaFreeHost(poison);
    free(units);
    cudaStreamDestroy(copy_stream);
    cudaStreamDestroy(compute_stream);
    safetensors_close(&st);
    printf("%s: %d failed checks\n", failed ? "FAIL" : "PASS", failed);
    return failed ? 1 : 0;
}
