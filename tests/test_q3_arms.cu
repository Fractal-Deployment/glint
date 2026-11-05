/* tests/test_q3_arms.cu — Qwen3-4B-Base arms on the CUDA harness (glint #85): the BF16 anchor, GLINT hole + plug, and GLINT fold only,
 * 2047 tokens, scored against the HF oracle in ARTIFACTS (eval_tokens_2048.txt, oracle_top1_2048.bin, oracle_ref_2048.json).
 * Same method as test_arm0_bf16_full.cu (bf16 rounding where HF rounds; cuBLAS GEMMs on bf16 weights), with the Qwen3 differences:
 * separate q/k/v and gate/up projections, per-head QK RMSNorm before RoPE, full-width RoPE (theta 1e6), GQA 32:8, eps 1e-6, tied
 * embedding (vocab 151936), three shards. Linears come from the BF16 model (Q3_MODE=bf16) or from a glint_pin_v1 (hole + plug + absmax)
 * decoded to bf16 one matrix at a time: Q3_MODE=full uses GLINT_CELLS[hole][plug] * absmax, Q3_MODE=fold uses PARENT[hole] * absmax.
 * Embedding and norms always come from the BF16 model. ARM0_LOGITS_OUT / ARM0_LOGITS_REF write / compare logits (KL(ref || arm)).
 * usage: test_q3_arms BF16_MODEL_DIR GLINT_PIN_DIR ARTIFACTS_DIR   (env Q3_MODE=bf16|full|fold)
 */
#include "arm0_forward.h"
#include "nf4.h"
#include "../include/glint_cells.h"
#include <algorithm>
#include <string>
#include <vector>

#define Q3_H 2560
#define Q3_I 9728
#define Q3_L 36
#define Q3_NQ 32
#define Q3_NKV 8
#define Q3_HD 128
#define Q3_QD 4096
#define Q3_KD 1024
#define Q3_V 151936
#define Q3_T 2048
#define Q3_NT 2047
#define Q3_EPS 1e-6f

static __constant__ float c_cells[16][16];
static __constant__ float c_parent[16];

enum { L_Q, L_K, L_V, L_O, L_G, L_U, L_D, N_LIN };
static const char *LNAME[N_LIN] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"};
static const int LN[N_LIN] = {Q3_QD, Q3_KD, Q3_KD, Q3_H, Q3_I, Q3_I, Q3_H};   /* n_out */
static const int LK[N_LIN] = {Q3_H, Q3_H, Q3_H, Q3_QD, Q3_H, Q3_H, Q3_I};     /* k_in */

__global__ void k_q3_qkprep(const float *q, const float *k, const float *v, const __nv_bfloat16 *qn, const __nv_bfloat16 *kn,
                            const float *cos_t, const float *sin_t, float *qo, float *ko, float *vo) {
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    if (hh >= Q3_NQ + Q3_NKV) { vo[(size_t)t * Q3_KD + (hh - Q3_NQ - Q3_NKV) * Q3_HD + d] = bf16r(v[(size_t)t * Q3_KD + (hh - Q3_NQ - Q3_NKV) * Q3_HD + d]); return; }
    const bool isq = hh < Q3_NQ;
    const float *src = isq ? q + (size_t)t * Q3_QD + hh * Q3_HD : k + (size_t)t * Q3_KD + (hh - Q3_NQ) * Q3_HD;
    const __nv_bfloat16 *w = isq ? qn : kn;
    __shared__ float sr[Q3_HD], sn[Q3_HD];
    const float x = bf16r(src[d]); sr[d] = x * x; __syncthreads();
    for (int st = Q3_HD / 2; st > 0; st >>= 1) { if (d < st) sr[d] += sr[d + st]; __syncthreads(); }
    const float rs = rsqrtf(sr[0] / (float)Q3_HD + Q3_EPS);
    sn[d] = bf16r(bf16r(x * rs) * __bfloat162float(w[d])); __syncthreads();
    const int i = d % (Q3_HD / 2); const float cs = cos_t[(size_t)t * (Q3_HD / 2) + i], sn_ = sin_t[(size_t)t * (Q3_HD / 2) + i];
    float val;
    if (d < Q3_HD / 2) val = bf16r(bf16r(sn[d] * cs) - bf16r(sn[d + Q3_HD / 2] * sn_));
    else               val = bf16r(bf16r(sn[d] * cs) + bf16r(sn[d - Q3_HD / 2] * sn_));
    if (isq) qo[(size_t)t * Q3_QD + hh * Q3_HD + d] = val; else ko[(size_t)t * Q3_KD + (hh - Q3_NQ) * Q3_HD + d] = val;
}

__global__ void k_q3_attn(const float *q, const float *k, const float *v, float *out, __nv_bfloat16 *out_bf16) { /* grid (T, NQ), block 128 */
    const int tq = blockIdx.x, hq = blockIdx.y, hkv = hq / (Q3_NQ / Q3_NKV), d = threadIdx.x;
    const float scale = 1.0f / sqrtf((float)Q3_HD);
    const float q_val = q[((size_t)tq * Q3_NQ + hq) * Q3_HD + d] * scale;
    float m_val = -1e30f, d_val = 0.0f, acc = 0.0f;
    __shared__ float s_red[128];
    for (int tk = 0; tk <= tq; tk++) {
        const float *k_row = k + ((size_t)tk * Q3_NKV + hkv) * Q3_HD;
        s_red[d] = q_val * k_row[d]; __syncthreads();
        if (d < 64) s_red[d] += s_red[d + 64];
        __syncthreads();
        if (d < 32) {
            float b = s_red[d] + s_red[d + 32];
            for (int o = 16; o > 0; o /= 2) b += __shfl_down_sync(0xffffffff, b, o);
            s_red[d] = b;
        }
        __syncthreads();
        const float score = s_red[0];
        const float m_new = fmaxf(m_val, score), alpha = expf(m_val - m_new), beta = expf(score - m_new);
        d_val = d_val * alpha + beta; m_val = m_new;
        acc = acc * alpha + bf16r(beta) * v[((size_t)tk * Q3_NKV + hkv) * Q3_HD + d];
        __syncthreads();
    }
    const float fo = (d_val > 0.0f) ? (acc / d_val) : 0.0f;
    out[((size_t)tq * Q3_NQ + hq) * Q3_HD + d] = fo;
    out_bf16[((size_t)tq * Q3_NQ + hq) * Q3_HD + d] = __float2bfloat16(fo);
}

__global__ void k_q3_swiglu(const float *g, const float *u, float *out, __nv_bfloat16 *out_bf16, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    const float gg = bf16r(g[i]), uu = bf16r(u[i]);
    const float sg = bf16r(gg / (1.0f + expf(-gg)));
    const float val = bf16r(sg * uu);
    out[i] = val; out_bf16[i] = __float2bfloat16(val);
}

__global__ void k_q3_decode(const uint8_t *hole, const uint8_t *plug, const float *am, __nv_bfloat16 *w, size_t n, int fold) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const uint8_t hb = hole[i >> 1];
    const int h = (i & 1) ? (hb >> 4) : (hb & 15);
    float v;
    if (fold) v = c_parent[h] * am[i >> 6];
    else { const uint8_t pb = plug[i >> 1]; const int p = (i & 1) ? (pb >> 4) : (pb & 15); v = c_cells[h][p] * am[i >> 6]; }
    w[i] = __float2bfloat16(v);
}

static const TensorEntry *find3(const SafeTensors st[3], const char *name, const char *dtype, size_t bytes) {
    for (int s = 0; s < 3; s++) {
        const TensorEntry *e = safetensors_find(&st[s], name);
        if (e) { if (strcmp(e->dtype, dtype) || e->end - e->begin != bytes) setup_fail("%s: want %s %zu bytes, have %s %llu", name, dtype, bytes, e->dtype, (unsigned long long)(e->end - e->begin)); return e; }
    }
    setup_fail("missing tensor %s", name); return NULL;
}
static void open3(SafeTensors st[3], const char *dir) {
    for (int s = 0; s < 3; s++) { char p[1100]; snprintf(p, sizeof(p), "%s/model-0000%d-of-00003.safetensors", dir, s + 1); if (safetensors_open(&st[s], p) != 0) setup_fail("cannot open %s", p); }
}
static __nv_bfloat16 *up_bf16(const SafeTensors st[3], const std::string &name, size_t numel) {
    __nv_bfloat16 *d; CK(cudaMalloc(&d, numel * 2)); CK(cudaMemcpy(d, find3(st, name.c_str(), "BF16", numel * 2)->data, numel * 2, cudaMemcpyHostToDevice)); return d;
}

int main(int argc, char **argv) {
    if (argc != 4) { printf("usage: %s BF16_MODEL_DIR GLINT_PIN_DIR ARTIFACTS_DIR   (env Q3_MODE=bf16|full|fold)\n", argv[0]); return 2; }
    const char *mode_s = getenv("Q3_MODE") ? getenv("Q3_MODE") : "bf16";
    const int mode = !strcmp(mode_s, "bf16") ? 0 : !strcmp(mode_s, "full") ? 1 : !strcmp(mode_s, "fold") ? 2 : -1;
    if (mode < 0) setup_fail("Q3_MODE must be bf16|full|fold");
    SafeTensors mst[3], gst[3]; open3(mst, argv[1]); if (mode) open3(gst, argv[2]);
    /* eval tokens and oracle */
    std::vector<int32_t> tok; { char p[1100]; snprintf(p, sizeof(p), "%s/eval_tokens_2048.txt", argv[3]); FILE *f = fopen(p, "r"); if (!f) setup_fail("cannot read %s", p); int v; while (fscanf(f, "%d", &v) == 1) tok.push_back(v); fclose(f); }
    if (tok.size() != Q3_T) setup_fail("eval tokens: %zu", tok.size());
    std::vector<uint32_t> oracle(Q3_NT); { char p[1100]; snprintf(p, sizeof(p), "%s/oracle_top1_2048.bin", argv[3]); FILE *f = fopen(p, "rb"); if (!f || fread(oracle.data(), 4, Q3_NT, f) != Q3_NT) setup_fail("cannot read %s", p); fclose(f); }
    double oracle_ppl = 0; { char p[1100]; snprintf(p, sizeof(p), "%s/oracle_ref_2048.json", argv[3]); FILE *f = fopen(p, "r"); std::string js; char b[4096]; size_t n; while (f && (n = fread(b, 1, sizeof b, f)) > 0) js.append(b, n); if (f) fclose(f);
        size_t pp = js.find("\"ppl\""); if (pp == std::string::npos) setup_fail("no ppl in oracle json"); oracle_ppl = atof(js.c_str() + js.find(':', pp) + 1); }
    printf("Qwen3-4B-Base arms: mode=%s oracle PPL %.6f\n", mode_s, oracle_ppl);
    CK(cudaMemcpyToSymbol(c_cells, GLINT_CELLS, sizeof(float) * 256)); CK(cudaMemcpyToSymbol(c_parent, NF4_CODE, sizeof(float) * 16));

    /* embedding, norms (BF16 model always) */
    const TensorEntry *emb_e = find3(mst, "model.embed_tokens.weight", "BF16", (size_t)Q3_V * Q3_H * 2);
    __nv_bfloat16 *d_embed = up_bf16(mst, "model.embed_tokens.weight", (size_t)Q3_V * Q3_H);
    __nv_bfloat16 *d_fnorm = up_bf16(mst, "model.norm.weight", Q3_H);
    std::vector<__nv_bfloat16 *> in_n(Q3_L), post_n(Q3_L), qn(Q3_L), kn(Q3_L);
    /* linears */
    std::vector<__nv_bfloat16 *> Wbf(Q3_L * N_LIN, nullptr); std::vector<uint8_t *> Hh(Q3_L * N_LIN, nullptr), Pp(Q3_L * N_LIN, nullptr); std::vector<float *> Am(Q3_L * N_LIN, nullptr);
    for (int l = 0; l < Q3_L; l++) {
        std::string pre = "model.layers." + std::to_string(l) + ".";
        in_n[l] = up_bf16(mst, pre + "input_layernorm.weight", Q3_H); post_n[l] = up_bf16(mst, pre + "post_attention_layernorm.weight", Q3_H);
        qn[l] = up_bf16(mst, pre + "self_attn.q_norm.weight", Q3_HD); kn[l] = up_bf16(mst, pre + "self_attn.k_norm.weight", Q3_HD);
        for (int j = 0; j < N_LIN; j++) {
            const size_t n = (size_t)LN[j] * LK[j]; std::string nm = pre + LNAME[j] + ".weight"; const int idx = l * N_LIN + j;
            if (mode == 0) { Wbf[idx] = up_bf16(mst, nm, n); continue; }
            const size_t nb = (n + 1) / 2, ab = (n + 63) / 64 * 4;
            CK(cudaMalloc(&Hh[idx], nb)); CK(cudaMemcpy(Hh[idx], find3(gst, nm.c_str(), "U8", nb)->data, nb, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&Am[idx], ab)); CK(cudaMemcpy(Am[idx], find3(gst, (nm + ".absmax").c_str(), "F32", ab)->data, ab, cudaMemcpyHostToDevice));
            if (mode == 1) { CK(cudaMalloc(&Pp[idx], nb)); CK(cudaMemcpy(Pp[idx], find3(gst, (nm + ".glint_plug").c_str(), "U8", nb)->data, nb, cudaMemcpyHostToDevice)); }
        }
    }
    size_t maxn = 0; for (int j = 0; j < N_LIN; j++) maxn = std::max(maxn, (size_t)LN[j] * LK[j]);
    __nv_bfloat16 *d_wscratch = nullptr; if (mode) CK(cudaMalloc(&d_wscratch, maxn * 2));
    printf("Weights uploaded (%s).\n", mode == 0 ? "BF16 anchor" : mode == 1 ? "GLINT hole + plug" : "GLINT fold only (parent decode)");

    /* RoPE tables: double precision on the host, rounded to bf16 */
    std::vector<float> cth((size_t)Q3_T * 64), sth((size_t)Q3_T * 64);
    for (int pos = 0; pos < Q3_T; pos++) for (int i = 0; i < 64; i++) {
        double inv = 1.0 / pow(1000000.0, (2.0 * i) / (double)Q3_HD), a = (double)pos * inv; float c = (float)cos(a), s = (float)sin(a);
        uint32_t u; memcpy(&u, &c, 4); u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u; memcpy(&c, &u, 4);
        memcpy(&u, &s, 4); u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u; memcpy(&s, &u, 4); cth[(size_t)pos * 64 + i] = c; sth[(size_t)pos * 64 + i] = s; }
    float *d_cos, *d_sin; CK(cudaMalloc(&d_cos, cth.size() * 4)); CK(cudaMalloc(&d_sin, sth.size() * 4));
    CK(cudaMemcpy(d_cos, cth.data(), cth.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_sin, sth.data(), sth.size() * 4, cudaMemcpyHostToDevice));

    /* buffers */
    float *x, *hf, *qb, *kb, *vb, *qo, *ko, *vo, *attn, *proj, *gb, *ub, *act, *down; __nv_bfloat16 *hbf, *attnb, *actb, *xfb;
    CK(cudaMalloc(&x, (size_t)Q3_T * Q3_H * 4)); CK(cudaMalloc(&hf, (size_t)Q3_T * Q3_H * 4)); CK(cudaMalloc(&hbf, (size_t)Q3_T * Q3_H * 2));
    CK(cudaMalloc(&qb, (size_t)Q3_T * Q3_QD * 4)); CK(cudaMalloc(&kb, (size_t)Q3_T * Q3_KD * 4)); CK(cudaMalloc(&vb, (size_t)Q3_T * Q3_KD * 4));
    CK(cudaMalloc(&qo, (size_t)Q3_T * Q3_QD * 4)); CK(cudaMalloc(&ko, (size_t)Q3_T * Q3_KD * 4)); CK(cudaMalloc(&vo, (size_t)Q3_T * Q3_KD * 4));
    CK(cudaMalloc(&attn, (size_t)Q3_T * Q3_QD * 4)); CK(cudaMalloc(&attnb, (size_t)Q3_T * Q3_QD * 2)); CK(cudaMalloc(&proj, (size_t)Q3_T * Q3_H * 4));
    CK(cudaMalloc(&gb, (size_t)Q3_T * Q3_I * 4)); CK(cudaMalloc(&ub, (size_t)Q3_T * Q3_I * 4)); CK(cudaMalloc(&act, (size_t)Q3_T * Q3_I * 4));
    CK(cudaMalloc(&actb, (size_t)Q3_T * Q3_I * 2)); CK(cudaMalloc(&down, (size_t)Q3_T * Q3_H * 4)); CK(cudaMalloc(&xfb, (size_t)Q3_T * Q3_H * 2));
    int32_t *d_tgt, *d_top1; float *d_loss, *d_lg; CK(cudaMalloc(&d_tgt, Q3_NT * 4)); CK(cudaMalloc(&d_top1, Q3_NT * 4)); CK(cudaMalloc(&d_loss, Q3_NT * 4)); CK(cudaMalloc(&d_lg, (size_t)256 * Q3_V * 4));
    CK(cudaMemcpy(d_tgt, tok.data() + 1, Q3_NT * 4, cudaMemcpyHostToDevice));
    const char *lo = getenv("ARM0_LOGITS_OUT"), *lr = getenv("ARM0_LOGITS_REF"); FILE *fo = lo ? fopen(lo, "wb") : NULL, *fr = lr ? fopen(lr, "rb") : NULL;
    if ((lo && !fo) || (lr && !fr)) setup_fail("cannot open logits files");
    float *d_ref = NULL, *d_kl = NULL, *hbuf = NULL; int32_t *d_rt1 = NULL, *d_qt1 = NULL;
    if (fo || fr) hbuf = (float *)malloc((size_t)256 * Q3_V * 4);
    if (fr) { CK(cudaMalloc(&d_ref, (size_t)256 * Q3_V * 4)); CK(cudaMalloc(&d_kl, Q3_NT * 4)); CK(cudaMalloc(&d_rt1, Q3_NT * 4)); CK(cudaMalloc(&d_qt1, Q3_NT * 4)); }
    cublasHandle_t cb; CKB(cublasCreate(&cb)); const float one = 1.f, zero = 0.f;

    auto linear = [&](int l, int j, const __nv_bfloat16 *xin, float *yout) {
        const int idx = l * N_LIN + j; const size_t n = (size_t)LN[j] * LK[j]; const __nv_bfloat16 *W = Wbf[idx];
        if (mode) { k_q3_decode<<<(unsigned)((n + 255) / 256), 256>>>(Hh[idx], Pp[idx], Am[idx], d_wscratch, n, mode == 2); W = d_wscratch; }
        CKB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, LN[j], Q3_T, LK[j], &one, W, CUDA_R_16BF, LK[j], xin, CUDA_R_16BF, LK[j], &zero, yout, CUDA_R_32F, LN[j], CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    };

    /* embedding gather on the host (BF16 -> float) */
    std::vector<float> hx((size_t)Q3_T * Q3_H); const __nv_bfloat16 *emb = (const __nv_bfloat16 *)emb_e->data;
    for (int t = 0; t < Q3_T; t++) for (int d = 0; d < Q3_H; d++) hx[(size_t)t * Q3_H + d] = __bfloat162float(emb[(size_t)tok[t] * Q3_H + d]);
    CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    printf("Starting forward pass over %d layers...\n", Q3_L); cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventRecord(e0));
    for (int l = 0; l < Q3_L; l++) {
        rms_norm_kernel<<<Q3_T, 128>>>(x, in_n[l], hf, hbf, Q3_T, Q3_H, Q3_EPS);
        linear(l, L_Q, hbf, qb); linear(l, L_K, hbf, kb); linear(l, L_V, hbf, vb);
        k_q3_qkprep<<<dim3(Q3_T, Q3_NQ + 2 * Q3_NKV), Q3_HD>>>(qb, kb, vb, qn[l], kn[l], d_cos, d_sin, qo, ko, vo);
        k_q3_attn<<<dim3(Q3_T, Q3_NQ), Q3_HD>>>(qo, ko, vo, attn, attnb);
        linear(l, L_O, attnb, proj);
        residual_add_kernel<<<(Q3_T * Q3_H + 255) / 256, 256>>>(x, proj, Q3_T * Q3_H);
        rms_norm_kernel<<<Q3_T, 128>>>(x, post_n[l], hf, hbf, Q3_T, Q3_H, Q3_EPS);
        linear(l, L_G, hbf, gb); linear(l, L_U, hbf, ub);
        k_q3_swiglu<<<(unsigned)(((size_t)Q3_T * Q3_I + 255) / 256), 256>>>(gb, ub, act, actb, (size_t)Q3_T * Q3_I);
        linear(l, L_D, actb, down);
        residual_add_kernel<<<(Q3_T * Q3_H + 255) / 256, 256>>>(x, down, Q3_T * Q3_H);
        CK(cudaGetLastError());
    }
    rms_norm_kernel<<<Q3_T, 128>>>(x, d_fnorm, hf, xfb, Q3_T, Q3_H, Q3_EPS);
    for (int off = 0; off < Q3_NT; off += 256) {
        const int cur = off + 256 <= Q3_NT ? 256 : Q3_NT - off;
        CKB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, Q3_V, cur, Q3_H, &one, d_embed, CUDA_R_16BF, Q3_H, xfb + (size_t)off * Q3_H, CUDA_R_16BF, Q3_H, &zero, d_lg, CUDA_R_32F, Q3_V, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
        compute_ce_and_argmax_kernel<<<cur, 256>>>(d_lg, d_tgt + off, d_loss + off, d_top1 + off, cur, Q3_V);
        if (fo) { CK(cudaMemcpy(hbuf, d_lg, (size_t)cur * Q3_V * 4, cudaMemcpyDeviceToHost)); if (fwrite(hbuf, 4, (size_t)cur * Q3_V, fo) != (size_t)cur * Q3_V) setup_fail("short write"); }
        if (fr) { if (fread(hbuf, 4, (size_t)cur * Q3_V, fr) != (size_t)cur * Q3_V) setup_fail("short read"); CK(cudaMemcpy(d_ref, hbuf, (size_t)cur * Q3_V * 4, cudaMemcpyHostToDevice));
                  kl_and_argmax_kernel<<<cur, 256>>>(d_ref, d_lg, d_kl + off, d_rt1 + off, d_qt1 + off, cur, Q3_V); }
        CK(cudaGetLastError());
    }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    std::vector<float> loss(Q3_NT); std::vector<int32_t> t1(Q3_NT), klt1(Q3_NT); std::vector<float> kl(Q3_NT);
    CK(cudaMemcpy(loss.data(), d_loss, Q3_NT * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(t1.data(), d_top1, Q3_NT * 4, cudaMemcpyDeviceToHost));
    double tot = 0; int match = 0; for (int i = 0; i < Q3_NT; i++) { tot += loss[i]; match += ((uint32_t)t1[i] == oracle[i]); }
    const double ce = tot / Q3_NT, ppl = exp(ce);
    printf("\n=== Qwen3-4B-Base %s ===\nForward Pass Time:      %.3f s\nMean Cross Entropy:     %.6f\nMeasured Perplexity:    %.6f\nHuggingFace Oracle PPL: %.6f\nPPL Delta vs Oracle:    %.4f %%\nTop-1 Matches:          %d / %d\nTop-1 Agreement:        %.2f %%\n",
           mode_s, ms / 1000.0, ce, ppl, oracle_ppl, fabs(ppl - oracle_ppl) / oracle_ppl * 100.0, match, Q3_NT, 100.0 * match / Q3_NT);
    if (fr) { CK(cudaMemcpy(kl.data(), d_kl, Q3_NT * 4, cudaMemcpyDeviceToHost)); std::vector<float> s = kl; std::sort(s.begin(), s.end()); double ks = 0; for (float v : kl) ks += v;
              printf("KL vs reference: mean %.6e p99 %.6e max %.6e (%d tokens)\n", ks / Q3_NT, s[(size_t)(0.99 * (Q3_NT - 1))], s[Q3_NT - 1], Q3_NT); }
    printf("VERDICT: REPORT ONLY (Qwen3 %s)\n", mode_s);
    if (fo) fclose(fo); if (fr) fclose(fr);
    return 0;
}
