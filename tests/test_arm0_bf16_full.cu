/* tests/test_arm0_bf16_full.cu — Full 32-Layer Arm (0) BF16 Evaluation on 2048 Tokens
 * Loads unquantized BF16 weights directly from model-00001-of-00002.safetensors and model-00002-of-00002.safetensors.
 * Executes on RTX 3060 via Tensor Core cuBLAS and online softmax causal attention.
 * Verifies Gate 1:
 *   - Top-1 argmax token agreement >= 95.0% vs HuggingFace (amended 2026-11-02, GATE_RULES.md C.1) Oracle
 *   - Perplexity within 1.0% of HuggingFace Oracle
 * Exit: 0 gate PASS (or INT8 report), 1 gate FAIL, 2 usage, 3 setup, I/O or CUDA error.
 */

#include "arm0_forward.h"

#define INT8_BLOCK 64

__global__ void int8_block_dequant_kernel(const int8_t* __restrict__ q, const float* __restrict__ scale,
                                          __nv_bfloat16* __restrict__ out, size_t numel) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) out[i] = __float2bfloat16((float)q[i] * scale[i / INT8_BLOCK]);
}

static __nv_bfloat16* upload_linear(const SafeTensors *st1, const SafeTensors *st2, const SafeTensors *int8_pin,
                                    const char *name, size_t numel) {
    __nv_bfloat16 *d_w;
    CK(cudaMalloc(&d_w, numel * sizeof(__nv_bfloat16)));
    if (!int8_pin) {
        CK(cudaMemcpy(d_w, need_tensor(st1, st2, name, "BF16", numel * sizeof(__nv_bfloat16))->data,
                      numel * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        return d_w;
    }
    char scale_name[300];
    snprintf(scale_name, sizeof(scale_name), "%s.int8_scale", name);
    size_t scale_bytes = (numel + INT8_BLOCK - 1) / INT8_BLOCK * sizeof(float);
    const TensorEntry *eq = need_tensor(int8_pin, NULL, name, "I8", numel);
    const TensorEntry *es = need_tensor(int8_pin, NULL, scale_name, "F32", scale_bytes);
    int8_t *d_q;
    float *d_s;
    CK(cudaMalloc(&d_q, numel));
    CK(cudaMalloc(&d_s, scale_bytes));
    CK(cudaMemcpy(d_q, eq->data, numel, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_s, es->data, scale_bytes, cudaMemcpyHostToDevice));
    int8_block_dequant_kernel<<<(unsigned)((numel + 255) / 256), 256>>>(d_q, d_s, d_w, numel);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaFree(d_q));
    CK(cudaFree(d_s));
    return d_w;
}

struct CublasLinears {
    cublasHandle_t cublas;
    __nv_bfloat16 *w[N_LAYERS][ARM0_N_LINEARS];
};

static void cublas_linear(void *ctx, int layer, Arm0Linear which, const __nv_bfloat16 *x, float *y, int m) {
    const CublasLinears *lin = (const CublasLinears *)ctx;
    int n_out = ARM0_LINEAR[which].n_out, k_in = ARM0_LINEAR[which].k_in;
    float alpha = 1.0f, beta = 0.0f;
    CKB(cublasGemmEx(lin->cublas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, m, k_in, &alpha, lin->w[layer][which], CUDA_R_16BF,
                     k_in, x, CUDA_R_16BF, k_in, &beta, y, CUDA_R_32F, n_out, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}

static Arm0Result g_result;
static float g_dump[N_LAYERS + 1][N_DUMP][HIDDEN_DIM];

int main(int argc, char** argv) {
    const char* dump_dir = NULL;
    const char* int8_pin_path = NULL;
    int argi = 1;
    for (; argi + 1 < argc && argv[argi][0] == '-'; argi += 2) {
        if (strcmp(argv[argi], "--dump") == 0) dump_dir = argv[argi + 1];
        else if (strcmp(argv[argi], "--int8") == 0) int8_pin_path = argv[argi + 1];
        else break;
    }
    if (argc - argi != 2) {
        printf("usage: %s [--dump DIR] [--int8 PIN.safetensors] <model_snapshot_dir> <artifacts_dir>\n", argv[0]);
        return EXIT_USAGE;
    }
    const char* model_dir = argv[argi];
    const char* artifacts_dir = argv[argi + 1];
    char path[1024];

    printf("=========================================================\n");
    printf("Arm (0) Full 32-Layer BF16 Evaluator (2048 Tokens)\n");
    printf("=========================================================\n");

    if (dump_dir && mkdir(dump_dir, 0755) != 0 && errno != EEXIST)
        setup_fail("cannot create dump dir %s: %s", dump_dir, strerror(errno));

    static Arm0Eval ev;
    arm0_read_eval(artifacts_dir, &ev);

    SafeTensors st1, st2;
    char p1[1024], p2[1024];
    snprintf(p1, sizeof(p1), "%s/model-00001-of-00002.safetensors", model_dir);
    snprintf(p2, sizeof(p2), "%s/model-00002-of-00002.safetensors", model_dir);
    if (safetensors_open(&st1, p1) != 0 || safetensors_open(&st2, p2) != 0)
        setup_fail("cannot open safetensors shards in %s", model_dir);
    Arm0Model model;
    arm0_upload_model(&st1, &st2, &model);

    SafeTensors st_int8;
    const SafeTensors* int8_pin = NULL;
    if (int8_pin_path) {
        if (safetensors_open(&st_int8, int8_pin_path) != 0) setup_fail("cannot open INT8 pin %s", int8_pin_path);
        int8_pin = &st_int8;
    }

    static CublasLinears lin;
    for (int l = 0; l < N_LAYERS; l++) {
        for (int j = 0; j < ARM0_N_LINEARS; j++) {
            char name[256];
            snprintf(name, sizeof(name), "model.layers.%d.%s.weight", l, ARM0_LINEAR[j].name);
            lin.w[l][j] = upload_linear(&st1, &st2, int8_pin, name, (size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in);
        }
    }
    printf("Uploaded all 32 layers of %s weights to VRAM.\n", int8_pin ? "INT8-dequantized" : "BF16");

    CKB(cublasCreate(&lin.cublas));
    arm0_forward(&model, ev.tokens, cublas_linear, &lin, lin.cublas, dump_dir ? g_dump : NULL, &g_result);
    const float* h_losses = g_result.losses;
    const int32_t* h_top1 = g_result.top1;
    const uint32_t* oracle_top1 = ev.oracle_top1;
    const double oracle_ppl = ev.oracle_ppl;
    double total_sec = g_result.seconds;

    double total_ce = 0.0;
    int top1_matches = 0;
    int printed = 0;
    for (int i = 0; i < 2047; i++) {
        total_ce += h_losses[i];
        if (h_top1[i] == (int32_t)oracle_top1[i]) {
            top1_matches++;
        } else if (printed < 10) {
            printf("Mismatch at pos %d: our_top1=%d oracle_top1=%d loss=%.4f\n",
                   i, h_top1[i], oracle_top1[i], h_losses[i]);
            printed++;
        }
    }
    double mean_ce = total_ce / 2047.0;
    double ppl = exp(mean_ce);
    double top1_acc = (double)top1_matches / 2047.0 * 100.0;

    double ppl_delta_pct = fabs(ppl - oracle_ppl) / oracle_ppl * 100.0;

    printf("\n=== %s Evaluation Results ===\n", int8_pin ? "INT8 pin" : "Arm (0) BF16");
    printf("Forward Pass Time:      %.3f s (%.1f tok/s)\n", total_sec, 2048.0 / total_sec);
    printf("Mean Cross Entropy:     %.6f\n", mean_ce);
    printf("Measured Perplexity:    %.6f\n", ppl);
    printf("HuggingFace Oracle PPL: %.6f\n", oracle_ppl);
    printf("PPL Delta vs Oracle:    %.4f %%\n", ppl_delta_pct);
    printf("Top-1 Matches:          %d / 2047\n", top1_matches);
    printf("Top-1 Agreement:        %.2f %%\n", top1_acc);

    int pass = top1_acc >= 95.0 && ppl_delta_pct <= 1.0;
    if (int8_pin) printf("VERDICT: REPORT ONLY (INT8 pin %s, no gate)\n", int8_pin_path);
    else printf(pass ? "VERDICT: PASS (Gate 1 Anchor Verified)\n" : "VERDICT: FAIL (Gate 1 Anchor Mismatch)\n");
    fflush(stdout);

    if (dump_dir) {
        snprintf(path, sizeof(path), "%s/cuda_hidden.bin", dump_dir);
        FILE* fh = fopen(path, "wb");
        if (!fh) setup_fail("cannot write %s", path);
        fwrite(g_dump, sizeof(g_dump), 1, fh);
        fclose(fh);
        snprintf(path, sizeof(path), "%s/cuda_pos.bin", dump_dir);
        FILE* fp = fopen(path, "wb");
        if (!fp) setup_fail("cannot write %s", path);
        fwrite(h_losses, sizeof(float), 2047, fp);
        fwrite(h_top1, sizeof(int32_t), 2047, fp);
        fclose(fp);
    }

    safetensors_close(&st1);
    safetensors_close(&st2);
    cublasDestroy(lin.cublas);
    return (pass || int8_pin) ? EXIT_GATE_PASS : EXIT_GATE_FAIL;
}
