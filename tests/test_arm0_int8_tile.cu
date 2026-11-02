#include "arm0_forward.h"
#include <vector>
#include <algorithm>
#include "../include/int8_tile_gemm.h"
#include "../include/int8_mma_gemm.h"
#include "../include/glint_plug_streamer.h"

enum GemmPath { GEMM_TILE, GEMM_CUBLAS_DEQUANT, GEMM_MMA_W8A8, GEMM_MMA_STREAM };

__global__ void dequant_bf16_kernel(const int8_t *q, const float *s, __nv_bfloat16 *w, size_t numel) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) w[i] = __float2bfloat16((float)q[i] * s[i / INT8_PIN_BLOCK]);
}

struct PinLinears {
    GemmPath path;
    cublasHandle_t cublas;
    Int8PinLinear packed[N_LAYERS][ARM0_N_LINEARS];
    __nv_bfloat16 *bf16[N_LAYERS][ARM0_N_LINEARS];
    size_t weight_bytes;
    int8_t *xq;
    float *sx;
    GlintPlugStreamer *streamer;
    uint64_t off_q[ARM0_N_LINEARS], off_s[ARM0_N_LINEARS];
    const uint8_t *slot;
    uint8_t *pq[N_LAYERS][ARM0_N_LINEARS];
    float *ps[N_LAYERS][ARM0_N_LINEARS];
    int8_t *plug_i8;
    size_t plug_bytes;
    int resolve;
    int8_t *rq;
    float *sr;
};

static uint64_t align256(uint64_t v) { return (v + 255) & ~255ull; }

/* One layer blob = the 4 linears' q bytes then scales, each 256-aligned; identical layout for every layer. */
static uint64_t stream_layout(PinLinears *lin) {
    uint64_t off = 0;
    for (int j = 0; j < ARM0_N_LINEARS; j++) {
        uint64_t numel = (uint64_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in;
        lin->off_q[j] = off; off = align256(off + numel);
        lin->off_s[j] = off; off = align256(off + numel / INT8_PIN_BLOCK * sizeof(float));
    }
    return off;
}

static void load_linear(PinLinears *lin, const SafeTensors *pin, int layer, Arm0Linear which) {
    int n_out = ARM0_LINEAR[which].n_out, k_in = ARM0_LINEAR[which].k_in;
    size_t numel = (size_t)n_out * k_in;
    size_t scale_bytes = (numel + INT8_PIN_BLOCK - 1) / INT8_PIN_BLOCK * sizeof(float);
    char name[256], scale_name[300];
    snprintf(name, sizeof(name), "model.layers.%d.%s.weight", layer, ARM0_LINEAR[which].name);
    snprintf(scale_name, sizeof(scale_name), "%s.int8_scale", name);
    const TensorEntry *eq = need_tensor(pin, NULL, name, "I8", numel);
    const TensorEntry *es = need_tensor(pin, NULL, scale_name, "F32", scale_bytes);
    int8_t *d_q;
    float *d_s;
    CK(cudaMalloc(&d_q, numel));
    CK(cudaMalloc(&d_s, scale_bytes));
    CK(cudaMemcpy(d_q, eq->data, numel, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_s, es->data, scale_bytes, cudaMemcpyHostToDevice));
    if (lin->path == GEMM_TILE || lin->path == GEMM_MMA_W8A8) {
        lin->packed[layer][which] = Int8PinLinear{d_q, d_s, n_out, k_in};
        lin->weight_bytes += numel + scale_bytes;
        return;
    }
    CK(cudaMalloc(&lin->bf16[layer][which], numel * sizeof(__nv_bfloat16)));
    dequant_bf16_kernel<<<(unsigned)((numel + 255) / 256), 256>>>(d_q, d_s, lin->bf16[layer][which], numel);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaFree(d_q));
    CK(cudaFree(d_s));
    lin->weight_bytes += numel * sizeof(__nv_bfloat16);
}

static void pin_linear(void *ctx, int layer, Arm0Linear which, const __nv_bfloat16 *x, float *y, int m) {
    const PinLinears *lin = (const PinLinears *)ctx;
    if (lin->path == GEMM_TILE) {
        CK(int8_tile_gemm_bf16(x, lin->packed[layer][which], y, m, 0));
        return;
    }
    if (lin->path == GEMM_MMA_STREAM) {
        PinLinears *w = (PinLinears *)ctx;
        if (which == ARM0_QKV && glint_plug_streamer_acquire(w->streamer, layer, &w->slot) != 0) setup_fail("acquire layer %d", layer);
        Int8MmaLinear l = {(const int8_t *)(w->slot + w->off_q[which]), (const float *)(w->slot + w->off_s[which]),
                           ARM0_LINEAR[which].n_out, ARM0_LINEAR[which].k_in};
        CK(int8_mma_gemm(x, l, y, m, w->xq, w->sx, 0));
        if (which == ARM0_DOWN && glint_plug_streamer_release(w->streamer, layer) != 0) setup_fail("release layer %d", layer);
        return;
    }
    if (lin->path == GEMM_MMA_W8A8) {
        const Int8PinLinear &p = lin->packed[layer][which];
        Int8MmaLinear w = {p.q, p.scale, p.n_out, p.k_in};
        if (lin->resolve)
            CK(int8_mma_gemm_resolve(x, w, lin->pq[layer][which], lin->ps[layer][which], y, m, lin->xq, lin->sx, lin->rq, lin->sr, lin->pq[layer][which] ? lin->plug_i8 : NULL, 0));
        else if (lin->pq[layer][which])
            CK(int8_mma_gemm_plug(x, w, lin->pq[layer][which], lin->ps[layer][which], y, m, lin->xq, lin->sx, lin->plug_i8, 0));
        else
            CK(int8_mma_gemm(x, w, y, m, lin->xq, lin->sx, 0));
        return;
    }
    int n_out = ARM0_LINEAR[which].n_out, k_in = ARM0_LINEAR[which].k_in;
    float alpha = 1.0f, beta = 0.0f;
    CKB(cublasGemmEx(lin->cublas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, m, k_in, &alpha, lin->bf16[layer][which],
                     CUDA_R_16BF, k_in, x, CUDA_R_16BF, k_in, &beta, y, CUDA_R_32F, n_out, CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT));
}

static size_t vram_used(void) {
    size_t free_b, total_b;
    CK(cudaMemGetInfo(&free_b, &total_b));
    return total_b - free_b;
}

static Arm0Eval g_eval;
static PinLinears g_lin;
static Arm0Result g_result;

int main(int argc, char **argv) {
    GemmPath path = GEMM_TILE;
    int argi = 1;
    if (argi + 1 < argc && strcmp(argv[argi], "--gemm") == 0) {
        if (strcmp(argv[argi + 1], "tile") == 0) path = GEMM_TILE;
        else if (strcmp(argv[argi + 1], "cublas") == 0) path = GEMM_CUBLAS_DEQUANT;
        else if (strcmp(argv[argi + 1], "mma") == 0) path = GEMM_MMA_W8A8;
        else if (strcmp(argv[argi + 1], "mma-stream") == 0) path = GEMM_MMA_STREAM;
        else argi = argc;
        argi += 2;
    }
    if (argc - argi != 3) {
        printf("usage: %s [--gemm tile|cublas|mma|mma-stream] PIN.safetensors <model_snapshot_dir> <artifacts_dir>\n", argv[0]);
        return EXIT_USAGE;
    }
    const char *pin_path = argv[argi];
    const char *model_dir = argv[argi + 1];
    const char *path_name = path == GEMM_TILE ? "INT8 tile GEMM (packed resident)" : path == GEMM_MMA_STREAM ? "INT8 tensor-core W8A8 (streamed from pinned RAM, 2-slot ring)" : path == GEMM_MMA_W8A8 ? "INT8 tensor-core W8A8 (packed resident)" : "INT8 dequant-to-BF16 + cuBLAS";

    arm0_read_eval(argv[argi + 2], &g_eval);

    SafeTensors st1, st2, pin;
    char p1[1024], p2[1024];
    snprintf(p1, sizeof(p1), "%s/model-00001-of-00002.safetensors", model_dir);
    snprintf(p2, sizeof(p2), "%s/model-00002-of-00002.safetensors", model_dir);
    if (safetensors_open(&st1, p1) != 0 || safetensors_open(&st2, p2) != 0)
        setup_fail("cannot open safetensors shards in %s", model_dir);
    if (safetensors_open(&pin, pin_path) != 0) setup_fail("cannot open INT8 pin %s", pin_path);
    Arm0Model model;
    arm0_upload_model(&st1, &st2, &model);

    CK(cudaDeviceSynchronize());
    size_t used_before_linears = vram_used();
    g_lin.path = path;
    if (path == GEMM_MMA_W8A8 || path == GEMM_MMA_STREAM) {
        int kmax = 0;
        for (int j = 0; j < ARM0_N_LINEARS; j++) kmax = ARM0_LINEAR[j].k_in > kmax ? ARM0_LINEAR[j].k_in : kmax;
        CK(cudaMalloc(&g_lin.xq, int8_mma_xq_bytes(2048, kmax)));
        CK(cudaMalloc(&g_lin.sx, int8_mma_sx_bytes(2048, kmax)));
        if (getenv("ARM0_RESOLVE") && path == GEMM_MMA_W8A8) {
            g_lin.resolve = 1;
            CK(cudaMalloc(&g_lin.rq, int8_mma_xq_bytes(2048, kmax)));
            CK(cudaMalloc(&g_lin.sr, int8_mma_sx_bytes(2048, kmax)));
            printf("Resolution mode: activation residual second pass ON\n");
        }
    }
    cudaStream_t copy_stream = 0;
    if (path == GEMM_MMA_STREAM) {
        uint64_t blob = stream_layout(&g_lin);
        std::vector<uint64_t> lb(N_LAYERS, blob);
        CK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        if (glint_plug_streamer_create(&g_lin.streamer, N_LAYERS, lb.data(), copy_stream, 0) != 0) setup_fail("streamer create");
        uint8_t *tmp = (uint8_t *)malloc(blob);
        for (int l = 0; l < N_LAYERS; l++) {
            memset(tmp, 0, blob);
            for (int j = 0; j < ARM0_N_LINEARS; j++) {
                size_t numel = (size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in;
                size_t scale_bytes = numel / INT8_PIN_BLOCK * sizeof(float);
                char name[256], scale_name[300];
                snprintf(name, sizeof(name), "model.layers.%d.%s.weight", l, ARM0_LINEAR[j].name);
                snprintf(scale_name, sizeof(scale_name), "%s.int8_scale", name);
                const TensorEntry *eq = need_tensor(&pin, NULL, name, "I8", numel);
                const TensorEntry *es = need_tensor(&pin, NULL, scale_name, "F32", scale_bytes);
                memcpy(tmp + g_lin.off_q[j], eq->data, numel);
                memcpy(tmp + g_lin.off_s[j], es->data, scale_bytes);
            }
            if (glint_plug_streamer_set_host_data(g_lin.streamer, l, tmp, blob) != 0) setup_fail("set_host_data %d", l);
        }
        free(tmp);
        g_lin.weight_bytes = 2 * blob;
        if (glint_plug_streamer_start(g_lin.streamer) != 0) setup_fail("streamer start");
    } else
    for (int l = 0; l < N_LAYERS; l++)
        for (int j = 0; j < ARM0_N_LINEARS; j++) load_linear(&g_lin, &pin, l, (Arm0Linear)j);
    /* ARM0_PLUG_FILE (p7_plug_res4_v1) + ARM0_PLUG_LAYERS (comma list): resident mma path only. */
    const char *plug_file = getenv("ARM0_PLUG_FILE"), *plug_layers = getenv("ARM0_PLUG_LAYERS");
    if (plug_file && plug_layers && *plug_layers && path == GEMM_MMA_W8A8) {
        SafeTensors plug;
        if (safetensors_open(&plug, plug_file) != 0) setup_fail("cannot open plug %s", plug_file);
        size_t maxn = 0;
        for (int j = 0; j < ARM0_N_LINEARS; j++) { size_t n = (size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in; if (n > maxn) maxn = n; }
        CK(cudaMalloc(&g_lin.plug_i8, maxn));
        for (const char *c = plug_layers; *c;) {
            int l = atoi(c);
            if (l < 0 || l >= N_LAYERS) setup_fail("bad plug layer %d", l);
            for (int j = 0; j < ARM0_N_LINEARS; j++) {
                size_t numel = (size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in;
                char name[256], qn[300], sn[300];
                snprintf(name, sizeof(name), "model.layers.%d.%s.weight", l, ARM0_LINEAR[j].name);
                snprintf(qn, sizeof(qn), "%s.plug_q4", name); snprintf(sn, sizeof(sn), "%s.plug_scale", name);
                size_t qb = (numel + 1) / 2, sb = numel / INT8_PIN_BLOCK * sizeof(float);
                const TensorEntry *eq = need_tensor(&plug, NULL, qn, "U8", qb), *es = need_tensor(&plug, NULL, sn, "F32", sb);
                CK(cudaMalloc(&g_lin.pq[l][j], qb)); CK(cudaMalloc(&g_lin.ps[l][j], sb));
                CK(cudaMemcpy(g_lin.pq[l][j], eq->data, qb, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(g_lin.ps[l][j], es->data, sb, cudaMemcpyHostToDevice));
                g_lin.plug_bytes += qb + sb;
            }
            while (*c && *c != ',') c++;
            if (*c == ',') c++;
        }
        printf("Plugged layers: %s | plug bytes %zu\n", plug_layers, g_lin.plug_bytes);
        safetensors_close(&plug);
    }
    CK(cudaDeviceSynchronize());
    size_t used_after_linears = vram_used();
    printf("Linear path: %s\n", path_name);
    printf("Linear weight tensor bytes:  %zu (%.3f GiB)\n", g_lin.weight_bytes, g_lin.weight_bytes / 1073741824.0);
    printf("VRAM used by linears (cudaMemGetInfo delta): %zu (%.3f GiB)\n", used_after_linears - used_before_linears,
           (used_after_linears - used_before_linears) / 1073741824.0);

    CKB(cublasCreate(&g_lin.cublas));
    arm0_forward(&model, g_eval.tokens, pin_linear, &g_lin, g_lin.cublas, NULL, &g_result);
    size_t used_peak = vram_used();
    if (path == GEMM_MMA_STREAM) {
        GlintStreamerStats ss;
        glint_plug_streamer_get_stats(g_lin.streamer, &ss);
        printf("Streamed bytes: %llu | stall total: %.3f ms (layer0 %.3f ms)\n", (unsigned long long)ss.total_h2d_bytes, ss.total_stall_ms, ss.per_layer_stall_ms[0]);
    }

    double total_ce = 0.0;
    int top1_matches = 0;
    uint64_t top1_hash = 1469598103934665603ull;
    for (int i = 0; i < 2047; i++) {
        total_ce += g_result.losses[i];
        if (g_result.top1[i] == (int32_t)g_eval.oracle_top1[i]) top1_matches++;
        top1_hash = (top1_hash ^ (uint32_t)g_result.top1[i]) * 1099511628211ull;
    }
    double ppl = exp(total_ce / 2047.0);

    printf("\n=== %s ===\n", path_name);
    printf("Forward Pass Time:      %.3f s\n", g_result.seconds);
    printf("Mean Cross Entropy:     %.9f\n", total_ce / 2047.0);
    printf("Measured Perplexity:    %.6f\n", ppl);
    printf("HuggingFace Oracle PPL: %.6f\n", g_eval.oracle_ppl);
    printf("PPL Delta vs Oracle:    %.4f %%\n", fabs(ppl - g_eval.oracle_ppl) / g_eval.oracle_ppl * 100.0);
    printf("Top-1 Matches:          %d / 2047 (%.2f %%)\n", top1_matches, top1_matches / 20.47);
    if (g_result.has_kl) {
        std::vector<float> k(g_result.kl, g_result.kl + 2047);
        double sum = 0; int agree = 0;
        for (int i = 0; i < 2047; i++) { sum += k[i]; agree += g_result.top1[i] == g_result.ref_top1[i]; }
        std::sort(k.begin(), k.end());
        printf("KL(BF16 || this) per token: mean %.6e  median %.6e  p99 %.6e  max %.6e\n", sum / 2047, k[1023], k[(int)(2047 * 0.99)], k[2046]);
        printf("Top-1 agreement with BF16 CUDA anchor: %d / 2047 (%.2f %%)\n", agree, agree / 20.47);
    }
    printf("Top-1 FNV-1a 64:        %016llx\n", (unsigned long long)top1_hash);
    printf("VRAM used at end of forward: %zu (%.3f GiB)\n", used_peak, used_peak / 1073741824.0);
    printf("VERDICT: REPORT ONLY (INT8 pin %s, no gate)\n", pin_path);

    safetensors_close(&st1);
    safetensors_close(&st2);
    safetensors_close(&pin);
    cublasDestroy(g_lin.cublas);
    return EXIT_GATE_PASS;
}
