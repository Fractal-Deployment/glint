#ifndef GLINT_ARM0_FORWARD_H
#define GLINT_ARM0_FORWARD_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <errno.h>
#include <openssl/sha.h>
#include <stdarg.h>
#include <sys/stat.h>

#include "../include/safetensors_c.h"

#define N_LAYERS 32
#define HIDDEN_DIM 3072
#define NUM_Q_HEADS 24
#define NUM_KV_HEADS 8
#define HEAD_DIM 128
#define ROTARY_DIM 96
#define INTERMEDIATE_DIM 8192
#define VOCAB_SIZE 200064
#define RMS_EPS 1e-5f
#define ROPE_THETA 10000.0f
#define ATTENTION_SCALING 1.1902380714238083f
#define ORIGINAL_MAX_POSITION 4096
static_assert(2048 <= ORIGINAL_MAX_POSITION, "longrope short_factor applies only below original_max_position_embeddings");

enum { EXIT_GATE_PASS = 0, EXIT_GATE_FAIL = 1, EXIT_USAGE = 2, EXIT_SETUP = 3 };

static void setup_fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("SETUP ERROR: ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    exit(EXIT_SETUP);
}

#define CK(call)                                                                       \
    do {                                                                               \
        cudaError_t e_ = (call);                                                       \
        if (e_ != cudaSuccess) setup_fail("%s:%d %s: %s", __FILE__, __LINE__, #call,   \
                                          cudaGetErrorString(e_));                     \
    } while (0)

#define CKB(call)                                                                      \
    do {                                                                               \
        cublasStatus_t e_ = (call);                                                    \
        if (e_ != CUBLAS_STATUS_SUCCESS) setup_fail("%s:%d cuBLAS status %d", __FILE__, \
                                                    __LINE__, (int)e_);                \
    } while (0)

__device__ __forceinline__ float bf16r(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ out,
    __nv_bfloat16* __restrict__ out_bf16,
    int M, int dim, float eps)
{
    int row = blockIdx.x;
    if (row >= M) return;
    const float* in_row = x + (size_t)row * dim;
    float* out_row = out + (size_t)row * dim;
    __nv_bfloat16* out_bf16_row = out_bf16 ? (out_bf16 + (size_t)row * dim) : NULL;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = in_row[i];
        sum_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float s_mean_sq;
    __shared__ float s_warp[4];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) s_warp[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0 && lane < 4) {
        float b = s_warp[lane];
        for (int offset = 2; offset > 0; offset /= 2)
            b += __shfl_down_sync(0xf, b, offset);
        if (lane == 0) s_mean_sq = b / (float)dim;
    }
    __syncthreads();

    float rscale = rsqrtf(s_mean_sq + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float w = __bfloat162float(weight[i]);
        float val = bf16r(bf16r(in_row[i] * rscale) * w);
        out_row[i] = val;
        if (out_bf16_row) out_bf16_row[i] = __float2bfloat16(val);
    }
}

__global__ void phi3_rope_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
    if (idx >= total_elements) return;

    int d = idx % HEAD_DIM;
    int head = (idx / HEAD_DIM) % (NUM_Q_HEADS + 2 * NUM_KV_HEADS);
    int m = idx / ((NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM);

    int pos = m;
    const float* qkv_row = qkv + (size_t)m * 5120;

    int i = d % (ROTARY_DIM / 2);
    float inv_freq = 1.0f / powf(ROPE_THETA, (2.0f * (float)i) / (float)ROTARY_DIM);
    float angle = (float)pos * inv_freq;
    float cos_v = bf16r(cosf(angle) * ATTENTION_SCALING);
    float sin_v = bf16r(sinf(angle) * ATTENTION_SCALING);

    if (head < NUM_Q_HEADS) {
        int h = head;
        int in_base = h * HEAD_DIM;
        float val;
        if (d < ROTARY_DIM / 2) {
            float x0 = bf16r(qkv_row[in_base + d]);
            float x1 = bf16r(qkv_row[in_base + d + ROTARY_DIM / 2]);
            val = bf16r(bf16r(x0 * cos_v) - bf16r(x1 * sin_v));
        } else if (d < ROTARY_DIM) {
            float x0 = bf16r(qkv_row[in_base + d - ROTARY_DIM / 2]);
            float x1 = bf16r(qkv_row[in_base + d]);
            val = bf16r(bf16r(x1 * cos_v) + bf16r(x0 * sin_v));
        } else {
            val = bf16r(qkv_row[in_base + d]);
        }
        q[(size_t)m * NUM_Q_HEADS * HEAD_DIM + h * HEAD_DIM + d] = val;
    } else if (head < NUM_Q_HEADS + NUM_KV_HEADS) {
        int h = head - NUM_Q_HEADS;
        int in_base = NUM_Q_HEADS * HEAD_DIM + h * HEAD_DIM;
        float val;
        if (d < ROTARY_DIM / 2) {
            float x0 = bf16r(qkv_row[in_base + d]);
            float x1 = bf16r(qkv_row[in_base + d + ROTARY_DIM / 2]);
            val = bf16r(bf16r(x0 * cos_v) - bf16r(x1 * sin_v));
        } else if (d < ROTARY_DIM) {
            float x0 = bf16r(qkv_row[in_base + d - ROTARY_DIM / 2]);
            float x1 = bf16r(qkv_row[in_base + d]);
            val = bf16r(bf16r(x1 * cos_v) + bf16r(x0 * sin_v));
        } else {
            val = bf16r(qkv_row[in_base + d]);
        }
        k[(size_t)m * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + d] = val;
    } else {
        int h = head - (NUM_Q_HEADS + NUM_KV_HEADS);
        int in_base = (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM + h * HEAD_DIM;
        v[(size_t)m * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + d] = bf16r(qkv_row[in_base + d]);
    }
}

__global__ void causal_gqa_online_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    __nv_bfloat16* __restrict__ out_bf16,
    int M)
{
    int tq = blockIdx.x;
    int hq = blockIdx.y;
    if (tq >= M || hq >= NUM_Q_HEADS) return;

    int hkv = hq / 3;
    int d = threadIdx.x;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    float q_val = q[((size_t)tq * NUM_Q_HEADS + hq) * HEAD_DIM + d] * scale;

    float m_val = -1e30f;
    float d_val = 0.0f;
    float acc = 0.0f;

    __shared__ float s_red[128];

    for (int tk = 0; tk <= tq; tk++) {
        const float* k_row = k + ((size_t)tk * NUM_KV_HEADS + hkv) * HEAD_DIM;
        float prod = q_val * k_row[d];
        s_red[d] = prod;
        __syncthreads();

        if (d < 64) s_red[d] += s_red[d + 64];
        __syncthreads();
        if (d < 32) {
            float b = s_red[d] + s_red[d + 32];
            for (int offset = 16; offset > 0; offset /= 2)
                b += __shfl_down_sync(0xffffffff, b, offset);
            s_red[d] = b;
        }
        __syncthreads();

        float score = s_red[0];
        float m_new = fmaxf(m_val, score);
        float alpha = expf(m_val - m_new);
        float beta = expf(score - m_new);

        d_val = d_val * alpha + beta;
        m_val = m_new;

        const float* v_row = v + ((size_t)tk * NUM_KV_HEADS + hkv) * HEAD_DIM;
        float v_val = v_row[d];
        acc = acc * alpha + bf16r(beta) * v_val;
        __syncthreads();
    }

    float final_out = (d_val > 0.0f) ? (acc / d_val) : 0.0f;
    out[((size_t)tq * NUM_Q_HEADS + hq) * HEAD_DIM + d] = final_out;
    if (out_bf16) out_bf16[((size_t)tq * NUM_Q_HEADS + hq) * HEAD_DIM + d] = __float2bfloat16(final_out);
}

__global__ void swiglu_kernel(const float* __restrict__ gate_up, float* __restrict__ out, __nv_bfloat16* __restrict__ out_bf16, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * INTERMEDIATE_DIM;
    if (idx >= total) return;

    int m = idx / INTERMEDIATE_DIM;
    int i = idx % INTERMEDIATE_DIM;

    const float* row = gate_up + (size_t)m * (2 * INTERMEDIATE_DIM);
    float g = bf16r(row[i]);
    float u = bf16r(row[INTERMEDIATE_DIM + i]);
    float silu_g = bf16r(g / (1.0f + expf(-g)));
    float val = bf16r(silu_g * u);
    out[(size_t)m * INTERMEDIATE_DIM + i] = val;
    if (out_bf16) out_bf16[(size_t)m * INTERMEDIATE_DIM + i] = __float2bfloat16(val);
}

__global__ void residual_add_kernel(float* __restrict__ x, const float* __restrict__ res, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        x[i] = bf16r(x[i] + bf16r(res[i]));
    }
}

__global__ void compute_ce_and_argmax_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ targets,
    float* __restrict__ losses,
    int32_t* __restrict__ top1,
    int M, int vocab_size)
{
    int row = blockIdx.x;
    if (row >= M) return;

    const float* l_row = logits + (size_t)row * vocab_size;
    int target = targets[row];

    float max_val = -1e30f;
    int max_idx = 200064;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        float val = bf16r(l_row[v]);
        if (val > max_val || (val == max_val && v < max_idx)) {
            max_val = val;
            max_idx = v;
        }
    }

    __shared__ float s_max[256];
    __shared__ int s_idx[256];
    s_max[threadIdx.x] = max_val;
    s_idx[threadIdx.x] = max_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            float other_val = s_max[threadIdx.x + stride];
            int other_idx = s_idx[threadIdx.x + stride];
            if (other_val > s_max[threadIdx.x] || (other_val == s_max[threadIdx.x] && other_idx < s_idx[threadIdx.x])) {
                s_max[threadIdx.x] = other_val;
                s_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }
    float block_max = s_max[0];
    if (threadIdx.x == 0 && top1) {
        top1[row] = s_idx[0];
    }
    __syncthreads();

    float sum_exp = 0.0f;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        sum_exp += expf(bf16r(l_row[v]) - block_max);
    }

    __shared__ float s_sum[256];
    s_sum[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float target_l = (target >= 0 && target < vocab_size) ? bf16r(l_row[target]) : 0.0f;
        float log_sum_exp = block_max + logf(fmaxf(s_sum[0], 1e-20f));
        losses[row] = log_sum_exp - target_l;
    }
}

__device__ float block_reduce(float v, bool is_max) {
    __shared__ float s_red[256];
    s_red[threadIdx.x] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            float o = s_red[threadIdx.x + stride];
            s_red[threadIdx.x] = is_max ? fmaxf(s_red[threadIdx.x], o) : s_red[threadIdx.x] + o;
        }
        __syncthreads();
    }
    float r = s_red[0];
    __syncthreads();
    return r;
}

/* KL(P_ref || Q) per row, both softmaxes in fp32 over the full vocab; Q reads the logits rounded to bf16
 * as compute_ce_and_argmax_kernel does. */
__global__ void kl_vs_ref_kernel(const float* __restrict__ logits, const __nv_bfloat16* __restrict__ ref,
                                 float* __restrict__ kl, int vocab_size) {
    const float* q = logits + (size_t)blockIdx.x * vocab_size;
    const __nv_bfloat16* r = ref + (size_t)blockIdx.x * vocab_size;
    float max_q = -1e30f, max_r = -1e30f;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        max_q = fmaxf(max_q, bf16r(q[v]));
        max_r = fmaxf(max_r, __bfloat162float(r[v]));
    }
    max_q = block_reduce(max_q, true);
    max_r = block_reduce(max_r, true);
    float sum_q = 0.0f, sum_r = 0.0f;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        sum_q += expf(bf16r(q[v]) - max_q);
        sum_r += expf(__bfloat162float(r[v]) - max_r);
    }
    float lse_q = max_q + logf(block_reduce(sum_q, false));
    float lse_r = max_r + logf(block_reduce(sum_r, false));
    float acc = 0.0f;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        float lr = __bfloat162float(r[v]) - lse_r;
        acc += expf(lr) * (lr - (bf16r(q[v]) - lse_q));
    }
    acc = block_reduce(acc, false);
    if (threadIdx.x == 0) kl[blockIdx.x] = acc;
}

__global__ void to_bf16_kernel(const float* __restrict__ in, __nv_bfloat16* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

static const int DUMP_POS[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976,
    1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000,
    2047};
#define N_DUMP ((int)(sizeof(DUMP_POS) / sizeof(DUMP_POS[0])))

static void dump_rows(float (*dump)[N_DUMP][HIDDEN_DIM], int slot, const float* d_src) {
    CK(cudaDeviceSynchronize());
    for (int i = 0; i < N_DUMP; i++)
        CK(cudaMemcpy(dump[slot][i], d_src + (size_t)DUMP_POS[i] * HIDDEN_DIM, HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost));
}

static const TensorEntry* need_tensor(const SafeTensors *st1, const SafeTensors *st2, const char *name,
                                      const char *dtype, uint64_t bytes) {
    const TensorEntry *e = safetensors_find(st1, name);
    if (!e && st2) e = safetensors_find(st2, name);
    if (!e) setup_fail("tensor %s missing", name);
    if (strcmp(e->dtype, dtype) != 0 || e->end - e->begin != bytes)
        setup_fail("tensor %s is %s with %llu bytes, want %s with %llu", name, e->dtype,
                   (unsigned long long)(e->end - e->begin), dtype, (unsigned long long)bytes);
    return e;
}

static char* read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) setup_fail("cannot open %s", path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char*)malloc((size_t)n + 1);
    if (n < 0 || fread(buf, 1, (size_t)n, f) != (size_t)n) setup_fail("cannot read %s", path);
    fclose(f);
    buf[n] = '\0';
    *len = (size_t)n;
    return buf;
}

static void sha256_hex(const void *data, size_t len, char out[65]) {
    unsigned char md[SHA256_DIGEST_LENGTH];
    SHA256((const unsigned char*)data, len, md);
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) snprintf(out + 2 * i, 3, "%02x", md[i]);
}

static const char* json_value(const char *json, const char *key) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char *p = strstr(json, pat);
    if (!p) setup_fail("hf_meta.json has no \"%s\"", key);
    p += strlen(pat);
    while (*p == ' ') p++;
    return p;
}

static void json_hex64(const char *json, const char *key, char out[65]) {
    const char *p = json_value(json, key);
    if (*p != '"' || strspn(p + 1, "0123456789abcdef") != 64 || p[65] != '"')
        setup_fail("hf_meta.json \"%s\" is not a 64-digit hex string", key);
    memcpy(out, p + 1, 64);
    out[64] = '\0';
}

static void require_sha256(const char *path, const void *data, size_t len, const char *want, const char *meta_key) {
    char got[65];
    sha256_hex(data, len, got);
    if (strcmp(got, want) != 0)
        setup_fail("%s sha256 %s does not match hf_meta.json %s %s", path, got, meta_key, want);
}

enum Arm0Linear { ARM0_QKV, ARM0_O, ARM0_GATE_UP, ARM0_DOWN, ARM0_N_LINEARS };

static const struct {
    const char *name;
    int n_out;
    int k_in;
} ARM0_LINEAR[ARM0_N_LINEARS] = {
    {"self_attn.qkv_proj", 5120, HIDDEN_DIM},
    {"self_attn.o_proj", HIDDEN_DIM, HIDDEN_DIM},
    {"mlp.gate_up_proj", 2 * INTERMEDIATE_DIM, HIDDEN_DIM},
    {"mlp.down_proj", HIDDEN_DIM, INTERMEDIATE_DIM},
};

typedef void (*Arm0LinearFn)(void *ctx, int layer, Arm0Linear which, const __nv_bfloat16 *x, float *y, int m);

struct Arm0Eval {
    int32_t tokens[2048];
    uint32_t oracle_top1[2047];
    double oracle_ppl;
};

struct Arm0Model {
    const __nv_bfloat16 *h_embed;
    __nv_bfloat16 *d_embed;
    __nv_bfloat16 *d_final_norm;
    __nv_bfloat16 *d_input_norm[N_LAYERS];
    __nv_bfloat16 *d_post_norm[N_LAYERS];
};

#define ARM0_MAX_REFS 2

/* Host bf16 logits [2047][VOCAB_SIZE] for positions 0..2046. Each non-NULL kl_ref gets
 * KL(reference || this forward) per position; save_logits gets this forward's logits rounded to bf16. */
struct Arm0Refs {
    const __nv_bfloat16 *kl_ref[ARM0_MAX_REFS];
    __nv_bfloat16 *save_logits;
};

struct Arm0Result {
    float losses[2047];
    int32_t top1[2047];
    float kl[ARM0_MAX_REFS][2047];
    double seconds;
};

static void arm0_read_eval(const char *artifacts_dir, Arm0Eval *ev) {
    char path[1024];
    size_t meta_len, tokens_len, oracle_len;
    snprintf(path, sizeof(path), "%s/hf_meta.json", artifacts_dir);
    char *meta = read_file(path, &meta_len);
    ev->oracle_ppl = strtod(json_value(meta, "ppl"), NULL);
    if (!(ev->oracle_ppl > 1.0)) setup_fail("hf_meta.json ppl %f is not a perplexity", ev->oracle_ppl);
    char want_tokens_sha[65], want_top1_sha[65];
    json_hex64(meta, "tokens_sha256", want_tokens_sha);
    json_hex64(meta, "top1_sha256", want_top1_sha);

    snprintf(path, sizeof(path), "%s/eval_tokens_2048.txt", artifacts_dir);
    char *tokens_txt = read_file(path, &tokens_len);
    require_sha256(path, tokens_txt, tokens_len, want_tokens_sha, "tokens_sha256");
    const char *tp = tokens_txt;
    for (int i = 0; i < 2048; i++) {
        char *end;
        long v = strtol(tp, &end, 10);
        if (end == tp) setup_fail("cannot parse token %d of %s", i, path);
        ev->tokens[i] = (int32_t)v;
        tp = end;
    }
    printf("Read 2048 tokens successfully.\n");

    snprintf(path, sizeof(path), "%s/oracle_top1_2048.bin", artifacts_dir);
    char *oracle_bytes = read_file(path, &oracle_len);
    if (oracle_len != 2047 * sizeof(uint32_t)) setup_fail("%s has %zu bytes, want %zu", path, oracle_len, 2047 * sizeof(uint32_t));
    require_sha256(path, oracle_bytes, oracle_len, want_top1_sha, "top1_sha256");
    memcpy(ev->oracle_top1, oracle_bytes, sizeof(ev->oracle_top1));
    printf("Read 2047 oracle top-1 tokens; sha256 matches hf_meta.json (oracle ppl %.6f).\n", ev->oracle_ppl);
    free(meta);
    free(tokens_txt);
    free(oracle_bytes);
}

static __nv_bfloat16 *upload_bf16(const SafeTensors *st1, const SafeTensors *st2, const char *name, size_t numel) {
    __nv_bfloat16 *d;
    CK(cudaMalloc(&d, numel * sizeof(__nv_bfloat16)));
    CK(cudaMemcpy(d, need_tensor(st1, st2, name, "BF16", numel * sizeof(__nv_bfloat16))->data,
                  numel * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    return d;
}

static void arm0_upload_model(const SafeTensors *st1, const SafeTensors *st2, Arm0Model *model) {
    model->h_embed = (const __nv_bfloat16 *)need_tensor(st1, st2, "model.embed_tokens.weight", "BF16",
                                                        (uint64_t)VOCAB_SIZE * HIDDEN_DIM * sizeof(__nv_bfloat16))->data;
    model->d_embed = upload_bf16(st1, st2, "model.embed_tokens.weight", (size_t)VOCAB_SIZE * HIDDEN_DIM);
    model->d_final_norm = upload_bf16(st1, st2, "model.norm.weight", HIDDEN_DIM);
    for (int l = 0; l < N_LAYERS; l++) {
        char name[256];
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
        model->d_input_norm[l] = upload_bf16(st1, st2, name, HIDDEN_DIM);
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
        model->d_post_norm[l] = upload_bf16(st1, st2, name, HIDDEN_DIM);
    }
}

static void arm0_forward(const Arm0Model *model, const int32_t tokens[2048], Arm0LinearFn linear, void *ctx,
                         cublasHandle_t cublas, float (*dump)[N_DUMP][HIDDEN_DIM], const Arm0Refs *refs,
                         Arm0Result *out) {
    float *h_x = (float *)malloc(2048 * HIDDEN_DIM * sizeof(float));
    for (int t = 0; t < 2048; t++)
        for (int d = 0; d < HIDDEN_DIM; d++)
            h_x[t * HIDDEN_DIM + d] = __bfloat162float(model->h_embed[tokens[t] * HIDDEN_DIM + d]);

    float *d_x, *d_h, *d_qkv, *d_q, *d_k, *d_v, *d_attn_out, *d_attn_proj;
    float *d_gate_up, *d_act, *d_down;
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
    __nv_bfloat16 *d_bf16_chunk = NULL;
    float *d_kl = NULL;
    if (refs) {
        CK(cudaMalloc(&d_bf16_chunk, 256 * (size_t)VOCAB_SIZE * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&d_kl, ARM0_MAX_REFS * 2047 * sizeof(float)));
        CK(cudaMemset(d_kl, 0, ARM0_MAX_REFS * 2047 * sizeof(float)));
    }
    CK(cudaMemcpy(d_targets, tokens + 1, 2047 * sizeof(int32_t), cudaMemcpyHostToDevice));
    float alpha = 1.0f, beta = 0.0f;

    printf("Starting forward pass over 32 layers...\n");
    clock_t t0 = clock();
    CK(cudaMemcpy(d_x, h_x, 2048 * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));
    if (dump) dump_rows(dump, 0, d_x);
    for (int l = 0; l < N_LAYERS; l++) {
        rms_norm_kernel<<<2048, 128>>>(d_x, model->d_input_norm[l], d_h, d_h_bf16, 2048, HIDDEN_DIM, RMS_EPS);
        linear(ctx, l, ARM0_QKV, d_h_bf16, d_qkv, 2048);
        int total_el = 2048 * (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
        phi3_rope_kernel<<<(total_el + 255) / 256, 256>>>(d_qkv, d_q, d_k, d_v, 2048);
        causal_gqa_online_attention_kernel<<<dim3(2048, NUM_Q_HEADS), 128>>>(d_q, d_k, d_v, d_attn_out,
                                                                             d_attn_out_bf16, 2048);
        linear(ctx, l, ARM0_O, d_attn_out_bf16, d_attn_proj, 2048);
        residual_add_kernel<<<(2048 * HIDDEN_DIM + 255) / 256, 256>>>(d_x, d_attn_proj, 2048 * HIDDEN_DIM);
        rms_norm_kernel<<<2048, 128>>>(d_x, model->d_post_norm[l], d_h, d_h_bf16, 2048, HIDDEN_DIM, RMS_EPS);
        linear(ctx, l, ARM0_GATE_UP, d_h_bf16, d_gate_up, 2048);
        swiglu_kernel<<<(2048 * INTERMEDIATE_DIM + 255) / 256, 256>>>(d_gate_up, d_act, d_act_bf16, 2048);
        linear(ctx, l, ARM0_DOWN, d_act_bf16, d_down, 2048);
        residual_add_kernel<<<(2048 * HIDDEN_DIM + 255) / 256, 256>>>(d_x, d_down, 2048 * HIDDEN_DIM);
        CK(cudaGetLastError());
        if (dump && l < N_LAYERS - 1) dump_rows(dump, l + 1, d_x);
    }
    rms_norm_kernel<<<2048, 128>>>(d_x, model->d_final_norm, d_h, d_x_final_bf16, 2048, HIDDEN_DIM, RMS_EPS);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    if (dump) dump_rows(dump, N_LAYERS, d_h);

    for (int offset = 0; offset < 2047; offset += 256) {
        int cur = offset + 256 <= 2047 ? 256 : 2047 - offset;
        CKB(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, VOCAB_SIZE, cur, HIDDEN_DIM, &alpha, model->d_embed,
                         CUDA_R_16BF, HIDDEN_DIM, d_x_final_bf16 + offset * HIDDEN_DIM, CUDA_R_16BF, HIDDEN_DIM,
                         &beta, d_logits_chunk, CUDA_R_32F, VOCAB_SIZE, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
        compute_ce_and_argmax_kernel<<<cur, 256>>>(d_logits_chunk, d_targets + offset, d_losses + offset,
                                                   d_top1 + offset, cur, VOCAB_SIZE);
        size_t chunk_bytes = (size_t)cur * VOCAB_SIZE * sizeof(__nv_bfloat16);
        for (int r = 0; refs && r < ARM0_MAX_REFS; r++) {
            if (!refs->kl_ref[r]) continue;
            CK(cudaMemcpy(d_bf16_chunk, refs->kl_ref[r] + (size_t)offset * VOCAB_SIZE, chunk_bytes,
                          cudaMemcpyHostToDevice));
            kl_vs_ref_kernel<<<cur, 256>>>(d_logits_chunk, d_bf16_chunk, d_kl + r * 2047 + offset, VOCAB_SIZE);
        }
        if (refs && refs->save_logits) {
            size_t n = (size_t)cur * VOCAB_SIZE;
            to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256>>>(d_logits_chunk, d_bf16_chunk, n);
            CK(cudaMemcpy(refs->save_logits + (size_t)offset * VOCAB_SIZE, d_bf16_chunk, chunk_bytes,
                          cudaMemcpyDeviceToHost));
        }
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());
    out->seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;
    CK(cudaMemcpy(out->losses, d_losses, 2047 * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(out->top1, d_top1, 2047 * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (refs) CK(cudaMemcpy(out->kl, d_kl, sizeof(out->kl), cudaMemcpyDeviceToHost));
    void *bufs[] = {d_x, d_h, d_qkv, d_q, d_k, d_v, d_attn_out, d_attn_proj, d_gate_up, d_act, d_down, d_h_bf16,
                    d_attn_out_bf16, d_act_bf16, d_x_final_bf16, d_targets, d_top1, d_losses, d_logits_chunk,
                    d_bf16_chunk, d_kl};
    for (void *b : bufs) CK(cudaFree(b));
    free(h_x);
}

#endif
