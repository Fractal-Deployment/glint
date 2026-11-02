#include "arm0_forward.h"
#include "../include/int4_tile_gemm.h"

enum GemmPath { GEMM_TILE, GEMM_CUBLAS_DEQUANT };

struct PinLinears {
    GemmPath path;
    cublasHandle_t cublas;
    Int4PinLinear packed[N_LAYERS][ARM0_N_LINEARS];
    __nv_bfloat16 *bf16[N_LAYERS][ARM0_N_LINEARS];
    size_t weight_bytes;
};

static void load_linear(PinLinears *lin, const SafeTensors *pin, int layer, Arm0Linear which) {
    int n_out = ARM0_LINEAR[which].n_out, k_in = ARM0_LINEAR[which].k_in;
    size_t numel = (size_t)n_out * k_in;
    size_t q_bytes = numel / 2, scale_bytes = numel / INT4_PIN_BLOCK * sizeof(float);
    char name[256], scale_name[300];
    snprintf(name, sizeof(name), "model.layers.%d.%s.weight", layer, ARM0_LINEAR[which].name);
    snprintf(scale_name, sizeof(scale_name), "%s.int4_scale", name);
    const TensorEntry *eq = need_tensor(pin, NULL, name, "U8", q_bytes);
    const TensorEntry *es = need_tensor(pin, NULL, scale_name, "F32", scale_bytes);
    uint8_t *d_q;
    float *d_s;
    CK(cudaMalloc(&d_q, q_bytes));
    CK(cudaMalloc(&d_s, scale_bytes));
    CK(cudaMemcpy(d_q, eq->data, q_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_s, es->data, scale_bytes, cudaMemcpyHostToDevice));
    Int4PinLinear w{d_q, d_s, n_out, k_in};
    if (lin->path == GEMM_TILE) {
        lin->packed[layer][which] = w;
        lin->weight_bytes += q_bytes + scale_bytes;
        return;
    }
    CK(cudaMalloc(&lin->bf16[layer][which], numel * sizeof(__nv_bfloat16)));
    CK(int4_dequant_bf16(w, lin->bf16[layer][which], 0));
    CK(cudaDeviceSynchronize());
    CK(cudaFree(d_q));
    CK(cudaFree(d_s));
    lin->weight_bytes += numel * sizeof(__nv_bfloat16);
}

static void pin_linear(void *ctx, int layer, Arm0Linear which, const __nv_bfloat16 *x, float *y, int m) {
    const PinLinears *lin = (const PinLinears *)ctx;
    if (lin->path == GEMM_TILE) {
        CK(int4_tile_gemm_bf16(x, lin->packed[layer][which], y, m, 0));
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
        else argi = argc;
        argi += 2;
    }
    if (argc - argi != 3) {
        printf("usage: %s [--gemm tile|cublas] PIN.safetensors <model_snapshot_dir> <artifacts_dir>\n", argv[0]);
        return EXIT_USAGE;
    }
    const char *pin_path = argv[argi];
    const char *model_dir = argv[argi + 1];
    const char *path_name = path == GEMM_TILE ? "INT4 tile GEMM (packed resident)" : "INT4 dequant-to-BF16 + cuBLAS";

    arm0_read_eval(argv[argi + 2], &g_eval);

    SafeTensors st1, st2, pin;
    char p1[1024], p2[1024];
    snprintf(p1, sizeof(p1), "%s/model-00001-of-00002.safetensors", model_dir);
    snprintf(p2, sizeof(p2), "%s/model-00002-of-00002.safetensors", model_dir);
    if (safetensors_open(&st1, p1) != 0 || safetensors_open(&st2, p2) != 0)
        setup_fail("cannot open safetensors shards in %s", model_dir);
    if (safetensors_open(&pin, pin_path) != 0) setup_fail("cannot open INT4 pin %s", pin_path);
    Arm0Model model;
    arm0_upload_model(&st1, &st2, &model);

    CK(cudaDeviceSynchronize());
    size_t used_before_linears = vram_used();
    g_lin.path = path;
    for (int l = 0; l < N_LAYERS; l++)
        for (int j = 0; j < ARM0_N_LINEARS; j++) load_linear(&g_lin, &pin, l, (Arm0Linear)j);
    CK(cudaDeviceSynchronize());
    size_t used_after_linears = vram_used();
    printf("Linear path: %s\n", path_name);
    printf("Linear weight tensor bytes:  %zu (%.3f GiB)\n", g_lin.weight_bytes, g_lin.weight_bytes / 1073741824.0);
    printf("VRAM used by linears (cudaMemGetInfo delta): %zu (%.3f GiB)\n", used_after_linears - used_before_linears,
           (used_after_linears - used_before_linears) / 1073741824.0);

    CKB(cublasCreate(&g_lin.cublas));
    arm0_forward(&model, g_eval.tokens, pin_linear, &g_lin, g_lin.cublas, NULL, NULL, &g_result);
    size_t used_peak = vram_used();

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
    printf("Top-1 FNV-1a 64:        %016llx\n", (unsigned long long)top1_hash);
    printf("VRAM used at end of forward: %zu (%.3f GiB)\n", used_peak, used_peak / 1073741824.0);
    printf("VERDICT: REPORT ONLY (INT4 pin %s, no gate)\n", pin_path);

    safetensors_close(&st1);
    safetensors_close(&st2);
    safetensors_close(&pin);
    cublasDestroy(g_lin.cublas);
    return EXIT_GATE_PASS;
}
