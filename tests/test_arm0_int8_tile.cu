/* Compiled with the Arm 0 anchor (PR #30 owns that file) so every non-linear kernel and ruler check is its own. */
#define main arm0_bf16_main
#include "test_arm0_bf16_full.cu"
#undef main

#include "../include/int8_tile_gemm.h"

enum GemmPath { GEMM_TILE, GEMM_CUBLAS_DEQUANT };

struct Linear {
    Int8PinLinear packed;
    __nv_bfloat16 *bf16;
};

struct LayerWeights {
    __nv_bfloat16 *d_input_norm;
    __nv_bfloat16 *d_post_norm;
    Linear qkv, o, gate_up, down;
};

static size_t g_weight_bytes = 0;

static Linear load_linear(GemmPath path, const SafeTensors *st1, const SafeTensors *st2, const SafeTensors *pin,
                          const char *name, int n_out, int k_in) {
    size_t numel = (size_t)n_out * k_in;
    Linear lin = {};
    if (path == GEMM_CUBLAS_DEQUANT) {
        lin.bf16 = upload_linear(st1, st2, pin, name, numel);
        g_weight_bytes += numel * sizeof(__nv_bfloat16);
        return lin;
    }
    char scale_name[300];
    snprintf(scale_name, sizeof(scale_name), "%s.int8_scale", name);
    size_t scale_bytes = numel / INT8_PIN_BLOCK * sizeof(float);
    const TensorEntry *eq = need_tensor(pin, NULL, name, "I8", numel);
    const TensorEntry *es = need_tensor(pin, NULL, scale_name, "F32", scale_bytes);
    int8_t *d_q;
    float *d_s;
    CK(cudaMalloc(&d_q, numel));
    CK(cudaMalloc(&d_s, scale_bytes));
    CK(cudaMemcpy(d_q, eq->data, numel, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_s, es->data, scale_bytes, cudaMemcpyHostToDevice));
    lin.packed = Int8PinLinear{d_q, d_s, n_out, k_in};
    g_weight_bytes += numel + scale_bytes;
    return lin;
}

static void linear_forward(GemmPath path, cublasHandle_t cublas, const Linear &lin, int n_out, int k_in,
                           const __nv_bfloat16 *x, float *y, int m) {
    if (path == GEMM_TILE) {
        CK(int8_tile_gemm_bf16(x, lin.packed, y, m, 0));
        return;
    }
    float alpha = 1.0f, beta = 0.0f;
    CKB(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, m, k_in, &alpha, lin.bf16, CUDA_R_16BF, k_in, x,
                     CUDA_R_16BF, k_in, &beta, y, CUDA_R_32F, n_out, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}

static size_t vram_used(void) {
    size_t free_b, total_b;
    CK(cudaMemGetInfo(&free_b, &total_b));
    return total_b - free_b;
}

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
    const char *artifacts_dir = argv[argi + 2];
    const char *path_name = path == GEMM_TILE ? "INT8 tile GEMM (packed resident)" : "INT8 dequant-to-BF16 + cuBLAS";
    char fpath[1024];

    size_t meta_len, tokens_len, oracle_len;
    snprintf(fpath, sizeof(fpath), "%s/hf_meta.json", artifacts_dir);
    char *meta = read_file(fpath, &meta_len);
    const double oracle_ppl = strtod(json_value(meta, "ppl"), NULL);
    if (!(oracle_ppl > 1.0)) setup_fail("hf_meta.json ppl %f is not a perplexity", oracle_ppl);
    char want_tokens_sha[65], want_top1_sha[65];
    json_hex64(meta, "tokens_sha256", want_tokens_sha);
    json_hex64(meta, "top1_sha256", want_top1_sha);

    snprintf(fpath, sizeof(fpath), "%s/eval_tokens_2048.txt", artifacts_dir);
    char *tokens_txt = read_file(fpath, &tokens_len);
    require_sha256(fpath, tokens_txt, tokens_len, want_tokens_sha, "tokens_sha256");
    int32_t tokens[2048];
    const char *tp = tokens_txt;
    for (int i = 0; i < 2048; i++) {
        char *end;
        long v = strtol(tp, &end, 10);
        if (end == tp) setup_fail("cannot parse token %d of %s", i, fpath);
        tokens[i] = (int32_t)v;
        tp = end;
    }

    snprintf(fpath, sizeof(fpath), "%s/oracle_top1_2048.bin", artifacts_dir);
    char *oracle_bytes = read_file(fpath, &oracle_len);
    if (oracle_len != 2047 * sizeof(uint32_t)) setup_fail("%s has %zu bytes, want %zu", fpath, oracle_len, 2047 * sizeof(uint32_t));
    require_sha256(fpath, oracle_bytes, oracle_len, want_top1_sha, "top1_sha256");
    uint32_t oracle_top1[2047];
    memcpy(oracle_top1, oracle_bytes, sizeof(oracle_top1));

    SafeTensors st1, st2, pin;
    char p1[1024], p2[1024];
    snprintf(p1, sizeof(p1), "%s/model-00001-of-00002.safetensors", model_dir);
    snprintf(p2, sizeof(p2), "%s/model-00002-of-00002.safetensors", model_dir);
    if (safetensors_open(&st1, p1) != 0 || safetensors_open(&st2, p2) != 0)
        setup_fail("cannot open safetensors shards in %s", model_dir);
    if (safetensors_open(&pin, pin_path) != 0) setup_fail("cannot open INT8 pin %s", pin_path);

    const TensorEntry *e_embed = need_tensor(&st1, &st2, "model.embed_tokens.weight", "BF16",
                                             (uint64_t)VOCAB_SIZE * HIDDEN_DIM * sizeof(__nv_bfloat16));
    const TensorEntry *e_final_norm = need_tensor(&st1, &st2, "model.norm.weight", "BF16", HIDDEN_DIM * sizeof(__nv_bfloat16));
    const __nv_bfloat16 *embed_table = (const __nv_bfloat16 *)e_embed->data;
    __nv_bfloat16 *d_embed_tokens, *d_final_norm;
    CK(cudaMalloc(&d_embed_tokens, (size_t)VOCAB_SIZE * HIDDEN_DIM * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_final_norm, HIDDEN_DIM * sizeof(__nv_bfloat16)));
    CK(cudaMemcpy(d_embed_tokens, embed_table, (size_t)VOCAB_SIZE * HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_final_norm, e_final_norm->data, HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    float *h_x = (float *)malloc(2048 * HIDDEN_DIM * sizeof(float));
    for (int t = 0; t < 2048; t++)
        for (int d = 0; d < HIDDEN_DIM; d++)
            h_x[t * HIDDEN_DIM + d] = __bfloat162float(embed_table[tokens[t] * HIDDEN_DIM + d]);

    CK(cudaDeviceSynchronize());
    size_t used_before_linears = vram_used();
    LayerWeights layers[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        char name[256];
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
        CK(cudaMalloc(&layers[l].d_input_norm, HIDDEN_DIM * sizeof(__nv_bfloat16)));
        CK(cudaMemcpy(layers[l].d_input_norm, need_tensor(&st1, &st2, name, "BF16", HIDDEN_DIM * sizeof(__nv_bfloat16))->data, HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
        CK(cudaMalloc(&layers[l].d_post_norm, HIDDEN_DIM * sizeof(__nv_bfloat16)));
        CK(cudaMemcpy(layers[l].d_post_norm, need_tensor(&st1, &st2, name, "BF16", HIDDEN_DIM * sizeof(__nv_bfloat16))->data, HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.qkv_proj.weight", l);
        layers[l].qkv = load_linear(path, &st1, &st2, &pin, name, 5120, HIDDEN_DIM);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", l);
        layers[l].o = load_linear(path, &st1, &st2, &pin, name, HIDDEN_DIM, HIDDEN_DIM);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_up_proj.weight", l);
        layers[l].gate_up = load_linear(path, &st1, &st2, &pin, name, 2 * INTERMEDIATE_DIM, HIDDEN_DIM);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", l);
        layers[l].down = load_linear(path, &st1, &st2, &pin, name, HIDDEN_DIM, INTERMEDIATE_DIM);
    }
    CK(cudaDeviceSynchronize());
    size_t used_after_linears = vram_used();
    printf("Linear path: %s\n", path_name);
    printf("Linear weight tensor bytes:  %zu (%.3f GiB)\n", g_weight_bytes, g_weight_bytes / 1073741824.0);
    printf("VRAM used by linears (cudaMemGetInfo delta): %zu (%.3f GiB)\n", used_after_linears - used_before_linears,
           (used_after_linears - used_before_linears) / 1073741824.0);

    float *d_x, *d_h, *d_qkv, *d_q, *d_k, *d_v, *d_attn_out, *d_attn_proj, *d_gate_up, *d_act, *d_down;
    __nv_bfloat16 *d_h_bf16, *d_attn_out_bf16, *d_act_bf16, *d_x_final_bf16;
    CK(cudaMalloc(&d_x, 2048 * HIDDEN_DIM * sizeof(float)));
    CK(cudaMalloc(&d_h, 2048 * HIDDEN_DIM * sizeof(float)));
    CK(cudaMalloc(&d_h_bf16, 2048 * HIDDEN_DIM * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_qkv, 2048 * 5120 * sizeof(float)));
    CK(cudaMalloc(&d_q, 2048 * NUM_Q_HEADS * HEAD_DIM * sizeof(float)));
    CK(cudaMalloc(&d_k, 2048 * NUM_KV_HEADS * HEAD_DIM * sizeof(float)));
    CK(cudaMalloc(&d_v, 2048 * NUM_KV_HEADS * HEAD_DIM * sizeof(float)));
    CK(cudaMalloc(&d_attn_out, 2048 * NUM_Q_HEADS * HEAD_DIM * sizeof(float)));
    CK(cudaMalloc(&d_attn_out_bf16, 2048 * NUM_Q_HEADS * HEAD_DIM * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_attn_proj, 2048 * HIDDEN_DIM * sizeof(float)));
    CK(cudaMalloc(&d_gate_up, 2048 * (2 * INTERMEDIATE_DIM) * sizeof(float)));
    CK(cudaMalloc(&d_act, 2048 * INTERMEDIATE_DIM * sizeof(float)));
    CK(cudaMalloc(&d_act_bf16, 2048 * INTERMEDIATE_DIM * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_down, 2048 * HIDDEN_DIM * sizeof(float)));
    CK(cudaMalloc(&d_x_final_bf16, 2048 * HIDDEN_DIM * sizeof(__nv_bfloat16)));

    int32_t *d_targets, *d_top1;
    float *d_losses, *d_logits_chunk;
    CK(cudaMalloc(&d_targets, 2047 * sizeof(int32_t)));
    CK(cudaMalloc(&d_top1, 2047 * sizeof(int32_t)));
    CK(cudaMalloc(&d_losses, 2047 * sizeof(float)));
    CK(cudaMalloc(&d_logits_chunk, 256 * VOCAB_SIZE * sizeof(float)));
    int32_t h_targets[2047];
    for (int i = 0; i < 2047; i++) h_targets[i] = tokens[i + 1];
    CK(cudaMemcpy(d_targets, h_targets, 2047 * sizeof(int32_t), cudaMemcpyHostToDevice));

    cublasHandle_t cublas;
    CKB(cublasCreate(&cublas));
    float alpha = 1.0f, beta = 0.0f;

    clock_t t0 = clock();
    CK(cudaMemcpy(d_x, h_x, 2048 * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));
    for (int l = 0; l < N_LAYERS; l++) {
        rms_norm_kernel<<<2048, 128>>>(d_x, layers[l].d_input_norm, d_h, d_h_bf16, 2048, HIDDEN_DIM, RMS_EPS);
        linear_forward(path, cublas, layers[l].qkv, 5120, HIDDEN_DIM, d_h_bf16, d_qkv, 2048);
        int total_el = 2048 * (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
        phi3_rope_kernel<<<(total_el + 255) / 256, 256>>>(d_qkv, d_q, d_k, d_v, 2048);
        causal_gqa_online_attention_kernel<<<dim3(2048, NUM_Q_HEADS), 128>>>(d_q, d_k, d_v, d_attn_out, d_attn_out_bf16, 2048);
        linear_forward(path, cublas, layers[l].o, HIDDEN_DIM, HIDDEN_DIM, d_attn_out_bf16, d_attn_proj, 2048);
        residual_add_kernel<<<(2048 * HIDDEN_DIM + 255) / 256, 256>>>(d_x, d_attn_proj, 2048 * HIDDEN_DIM);
        rms_norm_kernel<<<2048, 128>>>(d_x, layers[l].d_post_norm, d_h, d_h_bf16, 2048, HIDDEN_DIM, RMS_EPS);
        linear_forward(path, cublas, layers[l].gate_up, 2 * INTERMEDIATE_DIM, HIDDEN_DIM, d_h_bf16, d_gate_up, 2048);
        swiglu_kernel<<<(2048 * INTERMEDIATE_DIM + 255) / 256, 256>>>(d_gate_up, d_act, d_act_bf16, 2048);
        linear_forward(path, cublas, layers[l].down, HIDDEN_DIM, INTERMEDIATE_DIM, d_act_bf16, d_down, 2048);
        residual_add_kernel<<<(2048 * HIDDEN_DIM + 255) / 256, 256>>>(d_x, d_down, 2048 * HIDDEN_DIM);
        CK(cudaGetLastError());
    }
    rms_norm_kernel<<<2048, 128>>>(d_x, d_final_norm, d_h, d_x_final_bf16, 2048, HIDDEN_DIM, RMS_EPS);
    CK(cudaGetLastError());

    for (int offset = 0; offset < 2047; offset += 256) {
        int cur = offset + 256 <= 2047 ? 256 : 2047 - offset;
        CKB(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, VOCAB_SIZE, cur, HIDDEN_DIM, &alpha, d_embed_tokens,
                         CUDA_R_16BF, HIDDEN_DIM, d_x_final_bf16 + offset * HIDDEN_DIM, CUDA_R_16BF, HIDDEN_DIM,
                         &beta, d_logits_chunk, CUDA_R_32F, VOCAB_SIZE, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
        compute_ce_and_argmax_kernel<<<cur, 256>>>(d_logits_chunk, d_targets + offset, d_losses + offset,
                                                   d_top1 + offset, cur, VOCAB_SIZE);
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());
    double total_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    size_t used_peak = vram_used();

    float h_losses[2047];
    int32_t h_top1[2047];
    CK(cudaMemcpy(h_losses, d_losses, 2047 * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_top1, d_top1, 2047 * sizeof(int32_t), cudaMemcpyDeviceToHost));
    double total_ce = 0.0;
    int top1_matches = 0;
    uint64_t top1_hash = 1469598103934665603ull;
    for (int i = 0; i < 2047; i++) {
        total_ce += h_losses[i];
        if (h_top1[i] == (int32_t)oracle_top1[i]) top1_matches++;
        top1_hash = (top1_hash ^ (uint32_t)h_top1[i]) * 1099511628211ull;
    }
    double ppl = exp(total_ce / 2047.0);

    printf("\n=== %s ===\n", path_name);
    printf("Forward Pass Time:      %.3f s\n", total_sec);
    printf("Mean Cross Entropy:     %.9f\n", total_ce / 2047.0);
    printf("Measured Perplexity:    %.6f\n", ppl);
    printf("HuggingFace Oracle PPL: %.6f\n", oracle_ppl);
    printf("PPL Delta vs Oracle:    %.4f %%\n", fabs(ppl - oracle_ppl) / oracle_ppl * 100.0);
    printf("Top-1 Matches:          %d / 2047 (%.2f %%)\n", top1_matches, top1_matches / 20.47);
    printf("Top-1 FNV-1a 64:        %016llx\n", (unsigned long long)top1_hash);
    printf("VRAM used at end of forward: %zu (%.3f GiB)\n", used_peak, used_peak / 1073741824.0);
    printf("VERDICT: REPORT ONLY (INT8 pin %s, no gate)\n", pin_path);

    safetensors_close(&st1);
    safetensors_close(&st2);
    safetensors_close(&pin);
    cublasDestroy(cublas);
    free(h_x);
    return EXIT_GATE_PASS;
}
