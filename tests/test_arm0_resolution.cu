/* Product 2 "resolution" arms over the INT8 or INT4 pin: the base tile GEMM plus a 4- or 8-bit residual of the
 * BF16 weights held in pinned host RAM, added back for a chosen set of input channels. Every arm is scored against
 * HF BF16 on the ruler's 2048 tokens: perplexity, top-1 agreement and KL over the full vocabulary.
 * KL is also taken against this C/CUDA forward run with the BF16 weights, which isolates the quantization error
 * from the forward's own distance to HF.
 * Usage: test_arm0_resolution [--base int8|int4] [--resid-bits 4|8] PIN MODEL_DIR ARTIFACTS_DIR RES_ARTIFACTS_DIR
 *        HF_LOGITS OUT_TSV [FRAC...]   (defaults: int8 base, 4-bit residual)
 * Exit: 0 report written and every streamed arm bitwise equal to its resident twin, 1 not equal, 2 usage, 3 setup. */
#include <vector>
#include <algorithm>
#include <numeric>

#include "arm0_forward.h"
#include "../include/int8_tile_gemm.h"
#include "../include/int4_tile_gemm.h"
#include "../include/residual_correction.h"
#include "../include/glint_plug_streamer.h"

enum Mode { MODE_BASE, MODE_FULL, MODE_DYNAMIC, MODE_STATIC };
enum Source { SRC_RESIDENT, SRC_STREAMED };

struct LinearResid {
    Int8PinLinear b8;
    Int4PinLinear b4;
    ResidLinear res;
    uint8_t *h_codes;
    uint64_t codes_off;
    int col_off;
    std::vector<float> col_norm;
    std::vector<uint16_t> chan;
    uint8_t *d_codes;
    float *d_col_norm;
    float *d_abs_sum;
    uint16_t *d_chan;
    uint8_t *d_block;
};

struct Plan {
    const char *arm;
    double frac;
    Mode mode;
    bool weight_by_norm;
    int ksel[N_LAYERS][ARM0_N_LINEARS];
};

struct Ctx {
    cublasHandle_t cublas;
    LinearResid r[N_LAYERS][ARM0_N_LINEARS];
    const Plan *plan;
    Source source;
    bool calib_abs;
    GlintPlugStreamer *streamer;
    uint64_t plug_off[N_LAYERS][ARM0_N_LINEARS];
    const uint8_t *plug;
    __nv_bfloat16 *d_xg, *d_rg;
    uint8_t *d_used;
    float *d_yarm, *d_yfull;
    double *d_energy;
    double used_frac_sum;
    int used_n;
    size_t vram_in_run;
};

__global__ void add_kernel(float *__restrict__ y, const float *__restrict__ d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += d[i];
}

__global__ void energy_kernel(const float *__restrict__ full, const float *__restrict__ arm, size_t n, double *energy) {
    __shared__ double s_d[256], s_f[256];
    double d = 0, f = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        double a = full[i], b = arm[i];
        d += (a - b) * (a - b);
        f += a * a;
    }
    s_d[threadIdx.x] = d;
    s_f[threadIdx.x] = f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_d[threadIdx.x] += s_d[threadIdx.x + stride];
            s_f[threadIdx.x] += s_f[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&energy[0], s_d[0]);
        atomicAdd(&energy[1], s_f[0]);
    }
}

static std::vector<uint16_t> static_channels(const LinearResid &r, int c) {
    std::vector<uint16_t> sel(r.chan.begin(), r.chan.begin() + c);
    std::sort(sel.begin(), sel.end());
    return sel;
}

static bool g_int4_base;
static int g_resid_bits = 4;
static size_t g_vram_at_start;

/* Device memory in use now, less what was in use once the CUDA context existed and before this process
 * allocated anything. */
static size_t vram_used(void) {
    size_t free_b, total_b;
    CK(cudaMemGetInfo(&free_b, &total_b));
    return total_b - free_b - g_vram_at_start;
}

static int k_of(int j) { return ARM0_LINEAR[j].k_in; }
static int n_of(int j) { return ARM0_LINEAR[j].n_out; }

static void resolution_linear(void *vctx, int layer, Arm0Linear which, const __nv_bfloat16 *x, float *y, int m) {
    Ctx *c = (Ctx *)vctx;
    LinearResid &r = c->r[layer][which];
    CK(g_int4_base ? int4_tile_gemm_bf16(x, r.b4, y, m, 0) : int8_tile_gemm_bf16(x, r.b8, y, m, 0));
    if (c->calib_abs) CK(resid_accumulate_abs(x, m, r.res.k_in, r.d_abs_sum, 0));
    const Plan *p = c->plan;
    if (c->source == SRC_STREAMED && which == ARM0_QKV &&
        glint_plug_streamer_acquire(c->streamer, layer, &c->plug) != 0)
        setup_fail("streamer acquire layer %d", layer);
    int ksel = p->ksel[layer][which], k_in = r.res.k_in, n_out = r.res.n_out;
    size_t ny = (size_t)m * n_out;
    if (p->mode != MODE_BASE && ksel > 0) {
        CK(cudaMemset(c->d_yarm, 0, ny * sizeof(float)));
        const uint8_t *codes = c->source == SRC_STREAMED ? c->plug + c->plug_off[layer][which]
                               : p->mode == MODE_STATIC  ? r.d_block
                                                         : r.d_codes;
        if (p->mode == MODE_STATIC) {
            CK(resid_gather(x, m, k_in, r.d_chan, ksel, c->d_xg, 0));
            CK(resid_dequant(ResidCols{codes, r.d_chan, ksel}, r.res, c->d_rg, 0));
            CKB(resid_accumulate(c->cublas, c->d_xg, c->d_rg, c->d_yarm, m, n_out, ksel));
        } else {
            const __nv_bfloat16 *xs = x;
            if (p->mode == MODE_DYNAMIC) {
                CK(cudaMemset(c->d_used, 0, k_in));
                CK(resid_select_topk(x, m, k_in, p->weight_by_norm ? r.d_col_norm : NULL, ksel, c->d_xg, c->d_used, 0));
                std::vector<uint8_t> used(k_in);
                CK(cudaMemcpy(used.data(), c->d_used, k_in, cudaMemcpyDeviceToHost));
                c->used_frac_sum += std::accumulate(used.begin(), used.end(), 0) / (double)k_in;
                c->used_n++;
                xs = c->d_xg;
            }
            CK(resid_dequant(ResidCols{codes, NULL, k_in}, r.res, c->d_rg, 0));
            CKB(resid_accumulate(c->cublas, xs, c->d_rg, c->d_yarm, m, n_out, k_in));
        }
        add_kernel<<<(unsigned)((ny + 255) / 256), 256>>>(y, c->d_yarm, ny);
        CK(cudaGetLastError());
    }
    if (p->mode != MODE_BASE && p->mode != MODE_FULL && c->source == SRC_RESIDENT) {
        if (ksel == 0) CK(cudaMemset(c->d_yarm, 0, ny * sizeof(float)));
        CK(cudaMemset(c->d_yfull, 0, ny * sizeof(float)));
        CK(resid_dequant(ResidCols{r.d_codes, NULL, k_in}, r.res, c->d_rg, 0));
        CKB(resid_accumulate(c->cublas, x, c->d_rg, c->d_yfull, m, n_out, k_in));
        energy_kernel<<<256, 256>>>(c->d_yfull, c->d_yarm, ny, c->d_energy);
        CK(cudaGetLastError());
    }
    if (c->source == SRC_STREAMED && which == ARM0_DOWN && glint_plug_streamer_release(c->streamer, layer) != 0)
        setup_fail("streamer release layer %d", layer);
    if (layer == N_LAYERS - 1 && which == ARM0_DOWN) c->vram_in_run = vram_used();
}

static bool g_reorder_k_sum;
static size_t g_bf16_vram_in_run;

static void bf16_linear(void *vctx, int layer, Arm0Linear which, const __nv_bfloat16 *x, float *y, int m) {
    __nv_bfloat16 *(*w)[ARM0_N_LINEARS] = (__nv_bfloat16 *(*)[ARM0_N_LINEARS])((void **)vctx + 1);
    cublasHandle_t h = *(cublasHandle_t *)vctx;
    float one = 1.0f, zero = 0.0f;
    int k = k_of(which), parts = g_reorder_k_sum ? 2 : 1, kp = k / parts;
    for (int i = 0; i < parts; i++)
        CKB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n_of(which), m, kp, &one, w[layer][which] + i * kp, CUDA_R_16BF,
                         k, x + i * kp, CUDA_R_16BF, k, i ? &one : &zero, y, CUDA_R_32F, n_of(which), CUDA_R_32F,
                         CUBLAS_GEMM_DEFAULT));
    if (layer == N_LAYERS - 1 && which == ARM0_DOWN) g_bf16_vram_in_run = vram_used();
}

static const char *weight_name(int layer, int j, char *buf, size_t n) {
    snprintf(buf, n, "model.layers.%d.%s.weight", layer, ARM0_LINEAR[j].name);
    return buf;
}

static void read_tokens(const char *path, const char *want_sha, int32_t out[2048]) {
    size_t len;
    char *txt = read_file(path, &len);
    require_sha256(path, txt, len, want_sha, "calib_tokens_sha256");
    const char *p = txt;
    for (int i = 0; i < 2048; i++) {
        char *end;
        out[i] = (int32_t)strtol(p, &end, 10);
        if (end == p) setup_fail("cannot parse token %d of %s", i, path);
        p = end;
    }
    free(txt);
}

static __nv_bfloat16 *read_ref_logits(const char *path, const char *want_sha) {
    size_t bytes = (size_t)2047 * VOCAB_SIZE * sizeof(__nv_bfloat16);
    FILE *f = fopen(path, "rb");
    if (!f) setup_fail("cannot open %s", path);
    __nv_bfloat16 *h;
    CK(cudaHostAlloc((void **)&h, bytes, cudaHostAllocDefault));
    if (fread(h, 1, bytes, f) != bytes || fgetc(f) != EOF) setup_fail("%s is not %zu bytes", path, bytes);
    fclose(f);
    require_sha256(path, h, bytes, want_sha, "sha256");
    return h;
}

struct Score {
    double ppl, kl_mean, kl_max, kl_c_mean;
    int top1;
};

static Score score(const Arm0Result &res, const Arm0Eval &ev) {
    Score s{0, 0, 0, 0, 0};
    double ce = 0;
    for (int i = 0; i < 2047; i++) {
        ce += res.losses[i];
        s.kl_mean += res.kl[0][i];
        s.kl_c_mean += res.kl[1][i];
        s.kl_max = fmax(s.kl_max, res.kl[0][i]);
        s.top1 += res.top1[i] == (int32_t)ev.oracle_top1[i];
    }
    s.ppl = exp(ce / 2047.0);
    s.kl_mean /= 2047.0;
    s.kl_c_mean /= 2047.0;
    return s;
}

static uint64_t plan_bytes_per_token(const Plan &p) {
    if (p.mode == MODE_BASE) return 0;
    uint64_t b = 0;
    for (int l = 0; l < N_LAYERS; l++)
        for (int j = 0; j < ARM0_N_LINEARS; j++) {
            int ksel = p.mode == MODE_FULL ? k_of(j) : p.ksel[l][j];
            b += resid_cols_bytes(ksel, n_of(j), g_resid_bits) + (p.mode == MODE_DYNAMIC ? 2ull * ksel : 0);
        }
    return b;
}

static Plan uniform_plan(const char *arm, double frac, Mode mode, bool by_norm) {
    Plan p{arm, frac, mode, by_norm, {}};
    for (int l = 0; l < N_LAYERS; l++)
        for (int j = 0; j < ARM0_N_LINEARS; j++)
            p.ksel[l][j] = mode == MODE_FULL ? k_of(j) : std::max(1, (int)lround(frac * k_of(j)));
    return p;
}

static void allocate_budget_by_growth(double frac, const double growth[N_LAYERS], double layer_frac[N_LAYERS]) {
    double lo = 0, hi = 1e9;
    for (int it = 0; it < 200; it++) {
        double lam = 0.5 * (lo + hi), mean = 0;
        for (int l = 0; l < N_LAYERS; l++) mean += fmin(1.0, lam * growth[l]) / N_LAYERS;
        (mean < frac ? lo : hi) = lam;
    }
    for (int l = 0; l < N_LAYERS; l++) layer_frac[l] = fmin(1.0, hi * growth[l]);
}

static Plan trace_plan(double frac, const double growth[N_LAYERS]) {
    double layer_frac[N_LAYERS];
    allocate_budget_by_growth(frac, growth, layer_frac);
    Plan p{"trace", frac, MODE_DYNAMIC, false, {}};
    for (int l = 0; l < N_LAYERS; l++)
        for (int j = 0; j < ARM0_N_LINEARS; j++) p.ksel[l][j] = (int)lround(layer_frac[l] * k_of(j));
    return p;
}

static Ctx g_ctx;
static Arm0Eval g_eval;
static int32_t g_calib[2048];
typedef float Dump[N_LAYERS + 1][N_DUMP][HIDDEN_DIM];
static Arm0Result g_res, g_bf16_res[2], g_resident[2], g_streamed[2];
static Dump g_dump_a, g_dump_b, g_dump_resident[2], g_dump_streamed[2];

static uint64_t g_codes_bytes;
static int g_cols;
static uint8_t *g_codes, *g_block;
static float *g_col_norm, *g_abs_sum;
static uint16_t *g_chan;

template <typename T>
static T *dev_alloc(size_t bytes) {
    T *p;
    CK(cudaMalloc(&p, bytes));
    return p;
}

template <typename T>
static void dev_free(T *&p) {
    if (p) CK(cudaFree(p));
    p = NULL;
}

template <typename F>
static void each_linear(F f) {
    for (int l = 0; l < N_LAYERS; l++)
        for (int j = 0; j < ARM0_N_LINEARS; j++) f(l, j, g_ctx.r[l][j]);
}

static void codes_resident(bool on) {
    if (on == (g_codes != NULL)) return;
    if (!on) {
        dev_free(g_codes);
        each_linear([](int, int, LinearResid &r) { r.d_codes = NULL; });
        return;
    }
    g_codes = dev_alloc<uint8_t>(g_codes_bytes);
    each_linear([](int, int, LinearResid &r) {
        r.d_codes = g_codes + r.codes_off;
        CK(cudaMemcpy(r.d_codes, r.h_codes, resid_cols_bytes(r.res.k_in, r.res.n_out, r.res.bits), cudaMemcpyHostToDevice));
    });
}

static void col_norm_resident(bool on) {
    if (on == (g_col_norm != NULL)) return;
    if (!on) {
        dev_free(g_col_norm);
        each_linear([](int, int, LinearResid &r) { r.d_col_norm = NULL; });
        return;
    }
    g_col_norm = dev_alloc<float>((size_t)g_cols * sizeof(float));
    each_linear([](int, int, LinearResid &r) {
        r.d_col_norm = g_col_norm + r.col_off;
        CK(cudaMemcpy(r.d_col_norm, r.col_norm.data(), r.res.k_in * sizeof(float), cudaMemcpyHostToDevice));
    });
}

/* The channel list of a static plan and, with_block, its residual columns, each as one allocation. */
static void static_resident(const Plan *p, bool with_block) {
    dev_free(g_chan);
    dev_free(g_block);
    each_linear([](int, int, LinearResid &r) { r.d_chan = NULL, r.d_block = NULL; });
    if (!p) return;
    size_t n_chan = 0, n_block = 0;
    each_linear([&](int l, int j, LinearResid &r) {
        n_chan += p->ksel[l][j];
        n_block += p->ksel[l][j] * resid_cols_bytes(1, r.res.n_out, r.res.bits);
    });
    g_chan = dev_alloc<uint16_t>(n_chan * sizeof(uint16_t));
    if (with_block) g_block = dev_alloc<uint8_t>(n_block);
    n_chan = n_block = 0;
    each_linear([&](int l, int j, LinearResid &r) {
        int c = p->ksel[l][j];
        std::vector<uint16_t> sel = static_channels(r, c);
        r.d_chan = g_chan + n_chan;
        CK(cudaMemcpy(r.d_chan, sel.data(), c * sizeof(uint16_t), cudaMemcpyHostToDevice));
        n_chan += c;
        if (!with_block) return;
        uint64_t row = resid_cols_bytes(1, r.res.n_out, r.res.bits);
        std::vector<uint8_t> block(c * row);
        for (int i = 0; i < c; i++) memcpy(&block[i * row], r.h_codes + (size_t)sel[i] * row, row);
        r.d_block = g_block + n_block;
        CK(cudaMemcpy(r.d_block, block.data(), block.size(), cudaMemcpyHostToDevice));
        n_block += block.size();
    });
}

/* Allocates exactly the device state arm p reads from source src and frees everything else, so the VRAM read
 * during its forward is its own. Resident dynamic and static arms also hold the full residual and two output
 * buffers for the resid_left ruler. */
static void arm_device_state(const Plan &p, Source src) {
    bool corrects = p.mode != MODE_BASE, selects = p.mode == MODE_DYNAMIC || p.mode == MODE_STATIC;
    bool ruler = selects && src == SRC_RESIDENT;
    dev_free(g_ctx.d_xg);
    dev_free(g_ctx.d_rg);
    dev_free(g_ctx.d_used);
    dev_free(g_ctx.d_yarm);
    dev_free(g_ctx.d_yfull);
    dev_free(g_ctx.d_energy);
    codes_resident(src == SRC_RESIDENT && corrects);
    col_norm_resident(p.weight_by_norm);
    static_resident(p.mode == MODE_STATIC ? &p : NULL, src == SRC_RESIDENT);
    const size_t max_y = (size_t)2048 * 2 * INTERMEDIATE_DIM * sizeof(float);
    if (selects) g_ctx.d_xg = dev_alloc<__nv_bfloat16>((size_t)2048 * INTERMEDIATE_DIM * sizeof(__nv_bfloat16));
    if (corrects) g_ctx.d_rg = dev_alloc<__nv_bfloat16>((size_t)2 * INTERMEDIATE_DIM * HIDDEN_DIM * sizeof(__nv_bfloat16));
    if (corrects) g_ctx.d_yarm = dev_alloc<float>(max_y);
    if (p.mode == MODE_DYNAMIC) g_ctx.d_used = dev_alloc<uint8_t>(INTERMEDIATE_DIM);
    if (ruler) g_ctx.d_yfull = dev_alloc<float>(max_y);
    if (ruler) g_ctx.d_energy = dev_alloc<double>(2 * sizeof(double));
}

static void run(const Arm0Model *model, const int32_t *tokens, const Plan &p, Source src,
                float (*dump)[N_DUMP][HIDDEN_DIM], const Arm0Refs *ref, Arm0Result *out) {
    g_ctx.plan = &p;
    g_ctx.source = src;
    g_ctx.used_frac_sum = 0;
    g_ctx.used_n = 0;
    g_ctx.vram_in_run = 0;
    if (g_ctx.d_energy) CK(cudaMemset(g_ctx.d_energy, 0, 2 * sizeof(double)));
    if (src == SRC_STREAMED) {
        glint_plug_streamer_reset_stats(g_ctx.streamer);
        if (glint_plug_streamer_start(g_ctx.streamer) != 0) setup_fail("streamer start");
    }
    arm0_forward(model, tokens, resolution_linear, &g_ctx, g_ctx.cublas, dump, ref, out);
}

static void report(FILE *tsv, const Plan &p, const char *source, const Arm0Result &res, size_t vram,
                   uint64_t h2d_bytes, bool has_cbf16 = true) {
    Score s = score(res, g_eval);
    uint64_t bpt = plan_bytes_per_token(p);
    double used = g_ctx.used_n ? g_ctx.used_frac_sum / g_ctx.used_n : 0;
    double energy[2] = {0, 0};
    if (g_ctx.d_energy) CK(cudaMemcpy(energy, g_ctx.d_energy, sizeof(energy), cudaMemcpyDeviceToHost));
    double left = strncmp(p.arm, "bf16", 4) == 0 ? -1.0 : p.mode == MODE_BASE ? 1.0 : p.mode == MODE_FULL ? 0.0 : energy[1] > 0 ? energy[0] / energy[1] : -1.0;
    char cbf16[32];
    snprintf(cbf16, sizeof(cbf16), has_cbf16 ? "%.6f" : "n/a", s.kl_c_mean);
    printf("ROW %-12s frac %.3f %-9s ppl %.6f top1 %4d/2047 kl_hf %.6f kl_hf_max %.4f kl_cbf16 %s bytes/token "
           "%llu vram %.3f GiB h2d %llu used_union %.4f resid_left %.4f fwd %.1f s\n",
           p.arm, p.frac, source, s.ppl, s.top1, s.kl_mean, s.kl_max, cbf16, (unsigned long long)bpt,
           vram / 1073741824.0, (unsigned long long)h2d_bytes, used, left, res.seconds);
    fprintf(tsv, "%s\t%.3f\t%s\t%.6f\t%d\t%.6f\t%.4f\t%s\t%llu\t%zu\t%llu\t%.4f\t%.6f\n", p.arm, p.frac, source, s.ppl,
            s.top1, s.kl_mean, s.kl_max, cbf16, (unsigned long long)bpt, vram, (unsigned long long)h2d_bytes, used,
            left);
    fflush(tsv);
    fflush(stdout);
}

/* Per-token KL against HF, arm a minus arm b, on the same eval tokens: the mean and its standard error over the
 * 2047 positions. Neighbouring positions are correlated, so the standard error is a lower bound. */
static void paired(const char *what, const Arm0Result &a, const Arm0Result &b) {
    double sum = 0, sq = 0;
    for (int i = 0; i < 2047; i++) {
        double d = (double)a.kl[0][i] - b.kl[0][i];
        sum += d;
        sq += d * d;
    }
    double mean = sum / 2047, se = sqrt((sq / 2047 - mean * mean) / 2046);
    printf("PAIRED %-40s mean dKL_hf %+.6f  se %.6f  mean/se %+.1f\n", what, mean, se, mean / se);
}

static bool same(const char *what, const Arm0Result &a, const Arm0Result &b, const void *da, const void *db,
                 size_t dump_bytes) {
    bool eq = memcmp(a.losses, b.losses, sizeof(a.losses)) == 0 && memcmp(a.top1, b.top1, sizeof(a.top1)) == 0 &&
              memcmp(a.kl, b.kl, sizeof(a.kl)) == 0 && (!da || memcmp(da, db, dump_bytes) == 0);
    printf("EQUAL %s: %s (losses, top-1, KL%s)\n", what, eq ? "bitwise identical" : "DIFFERENT",
           da ? ", hidden states at 49 positions x 33 slots" : "");
    return eq;
}

static void run_bf16_and_keep_its_logits(const Arm0Model *model, const SafeTensors *st1, const SafeTensors *st2, Arm0Refs *refs,
                           FILE *tsv) {
    static void *ctx[1 + N_LAYERS * ARM0_N_LINEARS];
    CKB(cublasCreate((cublasHandle_t *)&ctx[0]));
    for (int l = 0; l < N_LAYERS; l++)
        for (int j = 0; j < ARM0_N_LINEARS; j++) {
            char name[256];
            ctx[1 + l * ARM0_N_LINEARS + j] =
                upload_bf16(st1, st2, weight_name(l, j, name, sizeof(name)), (size_t)n_of(j) * k_of(j));
        }
    CK(cudaDeviceSynchronize());
    arm0_forward(model, g_eval.tokens, bf16_linear, ctx, *(cublasHandle_t *)&ctx[0], NULL, refs, &g_res);
    report(tsv, uniform_plan("bf16", 1.0, MODE_BASE, false), "resident", g_res, g_bf16_vram_in_run, 0, false);
    g_bf16_res[0] = g_res;
    __nv_bfloat16 *saved = refs->save_logits;
    refs->kl_ref[1] = saved;
    refs->save_logits = NULL;
    g_reorder_k_sum = true;
    arm0_forward(model, g_eval.tokens, bf16_linear, ctx, *(cublasHandle_t *)&ctx[0], NULL, refs, &g_res);
    g_reorder_k_sum = false;
    report(tsv, uniform_plan("bf16_splitk", 1.0, MODE_BASE, false), "resident", g_res, g_bf16_vram_in_run, 0);
    g_bf16_res[1] = g_res;
    for (int i = 1; i <= N_LAYERS * ARM0_N_LINEARS; i++) CK(cudaFree(ctx[i]));
    cublasDestroy(*(cublasHandle_t *)&ctx[0]);
}

int main(int argc, char **argv) {
    int argi = 1;
    bool usage = false;
    for (; argi + 1 < argc && strncmp(argv[argi], "--", 2) == 0; argi += 2) {
        if (strcmp(argv[argi], "--base") == 0) {
            g_int4_base = strcmp(argv[argi + 1], "int4") == 0;
            usage |= !g_int4_base && strcmp(argv[argi + 1], "int8") != 0;
        } else if (strcmp(argv[argi], "--resid-bits") == 0) {
            g_resid_bits = atoi(argv[argi + 1]);
            usage |= g_resid_bits != 4 && g_resid_bits != 8;
        } else {
            usage = true;
        }
    }
    if (usage || argc - argi < 6) {
        printf("usage: %s [--base int8|int4] [--resid-bits 4|8] PIN MODEL_DIR ARTIFACTS_DIR RES_ARTIFACTS_DIR HF_LOGITS "
               "OUT_TSV [FRAC...]\n", argv[0]);
        return EXIT_USAGE;
    }
    const char *base_name = g_int4_base ? "INT4" : "INT8";
    const float resid_range = g_int4_base ? 0.55f : 1.0f;
    const char *pin_path = argv[argi], *model_dir = argv[argi + 1], *art = argv[argi + 2], *res_art = argv[argi + 3];
    const char *logits_path = argv[argi + 4], *tsv_path = argv[argi + 5];
    std::vector<double> fracs;
    for (int i = argi + 6; i < argc; i++) fracs.push_back(strtod(argv[i], NULL));

    CK(cudaFree(0));
    g_vram_at_start = vram_used();
    printf("VRAM in use with the CUDA context up, before any allocation: %.3f GiB (subtracted from every reading)\n",
           g_vram_at_start / 1073741824.0);

    arm0_read_eval(art, &g_eval);
    char path[1024], meta_sha[65], want[65], hf_tokens_sha[65];
    size_t len;
    snprintf(path, sizeof(path), "%s/hf_logits_meta.json", res_art);
    char *lmeta = read_file(path, &len);
    snprintf(path, sizeof(path), "%s/hf_meta.json", art);
    char *hmeta = read_file(path, &len);
    json_hex64(lmeta, "eval_tokens_sha256", want);
    json_hex64(hmeta, "tokens_sha256", hf_tokens_sha);
    if (strcmp(want, hf_tokens_sha) != 0) setup_fail("hf_logits_meta.json was made from other eval tokens");
    json_hex64(lmeta, "calib_tokens_sha256", want);
    snprintf(path, sizeof(path), "%s/calib_tokens_2048.txt", res_art);
    read_tokens(path, want, g_calib);
    json_hex64(lmeta, "sha256", meta_sha);
    static Arm0Refs refs;
    refs.kl_ref[0] = read_ref_logits(logits_path, meta_sha);
    CK(cudaHostAlloc((void **)&refs.save_logits, (size_t)2047 * VOCAB_SIZE * sizeof(__nv_bfloat16), cudaHostAllocDefault));
    const Arm0Refs *ref = &refs;
    printf("HF bf16 logits [2047, %d] read; sha256 matches hf_logits_meta.json.\n", VOCAB_SIZE);

    FILE *tsv = fopen(tsv_path, "w");
    if (!tsv) setup_fail("cannot write %s", tsv_path);
    fprintf(tsv, "arm\tfrac\tsource\tppl\ttop1\tkl_hf_mean\tkl_hf_max\tkl_cbf16_mean\tbytes_per_token\tvram_bytes\t"
                 "h2d_bytes\tused_union\tresid_left\n");

    SafeTensors st1, st2, pin;
    char p1[1024], p2[1024];
    snprintf(p1, sizeof(p1), "%s/model-00001-of-00002.safetensors", model_dir);
    snprintf(p2, sizeof(p2), "%s/model-00002-of-00002.safetensors", model_dir);
    if (safetensors_open(&st1, p1) != 0 || safetensors_open(&st2, p2) != 0)
        setup_fail("cannot open safetensors shards in %s", model_dir);
    Arm0Model model;
    arm0_upload_model(&st1, &st2, &model);
    CK(cudaDeviceSynchronize());
    printf("VRAM after embedding and norms: %.3f GiB\n", vram_used() / 1073741824.0);
    run_bf16_and_keep_its_logits(&model, &st1, &st2, &refs, tsv);
    if (safetensors_open(&pin, pin_path) != 0) setup_fail("cannot open %s pin %s", base_name, pin_path);

    __nv_bfloat16 *d_w = dev_alloc<__nv_bfloat16>((size_t)2 * INTERMEDIATE_DIM * HIDDEN_DIM * sizeof(__nv_bfloat16));
    __nv_bfloat16 *d_wb = dev_alloc<__nv_bfloat16>((size_t)2 * INTERMEDIATE_DIM * HIDDEN_DIM * sizeof(__nv_bfloat16));
    ResidBuildStats *d_stats = dev_alloc<ResidBuildStats>(sizeof(ResidBuildStats));
    CK(cudaMemset(d_stats, 0, sizeof(ResidBuildStats)));
    uint64_t base_bytes = 0;
    each_linear([](int, int j, LinearResid &r) {
        r.codes_off = g_codes_bytes;
        r.col_off = g_cols;
        g_codes_bytes += resid_cols_bytes(k_of(j), n_of(j), g_resid_bits);
        g_cols += k_of(j);
    });
    g_codes = dev_alloc<uint8_t>(g_codes_bytes);
    g_col_norm = dev_alloc<float>((size_t)g_cols * sizeof(float));
    size_t vram_before_base = vram_used();
    for (int l = 0; l < N_LAYERS; l++)
        for (int j = 0; j < ARM0_N_LINEARS; j++) {
            LinearResid &r = g_ctx.r[l][j];
            int n = n_of(j), k = k_of(j);
            size_t numel = (size_t)n * k, scale_bytes = numel / RESID_SCALE_BLOCK * sizeof(float);
            size_t q_bytes = g_int4_base ? numel / 2 : numel;
            char name[256], sname[300];
            weight_name(l, j, name, sizeof(name));
            snprintf(sname, sizeof(sname), "%s.%s", name, g_int4_base ? "int4_scale" : "int8_scale");
            uint8_t *d_q = dev_alloc<uint8_t>(q_bytes);
            float *d_s = dev_alloc<float>(scale_bytes);
            CK(cudaMemcpy(d_q, need_tensor(&pin, NULL, name, g_int4_base ? "U8" : "I8", q_bytes)->data, q_bytes,
                          cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_s, need_tensor(&pin, NULL, sname, "F32", scale_bytes)->data, scale_bytes,
                          cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_w, need_tensor(&st1, &st2, name, "BF16", numel * 2)->data, numel * 2, cudaMemcpyHostToDevice));
            r.b8 = Int8PinLinear{(const int8_t *)d_q, d_s, n, k};
            r.b4 = Int4PinLinear{d_q, d_s, n, k};
            r.res = ResidLinear{d_s, n, k, g_resid_bits, resid_range};
            CK(g_int4_base ? int4_dequant_bf16(r.b4, d_wb, 0) : int8_dequant_bf16(r.b8, d_wb, 0));
            base_bytes += q_bytes + scale_bytes;
            uint64_t cb = resid_cols_bytes(k, n, g_resid_bits);
            r.d_codes = g_codes + r.codes_off;
            r.d_col_norm = g_col_norm + r.col_off;
            CK(resid_build(d_w, d_wb, r.res, r.d_codes, r.d_col_norm, d_stats, 0));
            CK(cudaHostAlloc((void **)&r.h_codes, cb, cudaHostAllocDefault));
            CK(cudaMemcpy(r.h_codes, r.d_codes, cb, cudaMemcpyDeviceToHost));
            r.col_norm.resize(k);
            CK(cudaMemcpy(r.col_norm.data(), r.d_col_norm, k * sizeof(float), cudaMemcpyDeviceToHost));
        }
    size_t vram_base = vram_used() - vram_before_base;
    dev_free(d_w);
    dev_free(d_wb);
    ResidBuildStats stats;
    CK(cudaMemcpy(&stats, d_stats, sizeof(stats), cudaMemcpyDeviceToHost));
    dev_free(d_stats);
    float max_rel;
    memcpy(&max_rel, &stats.max_rel_bits, 4);
    printf("Residual built: %llu bytes pinned host (%d bits/weight over %llu weights, codes over |R/scale| <= %.2f), "
           "max |R/scale| %.4f, clamped %llu. %s base linears %llu bytes resident.\n",
           (unsigned long long)g_codes_bytes, g_resid_bits, (unsigned long long)(g_codes_bytes * 8 / g_resid_bits),
           resid_range, max_rel, stats.clamped, base_name, (unsigned long long)base_bytes);
    if (stats.clamped) setup_fail("%llu residuals fell outside the code range", stats.clamped);
    printf("VRAM taken by the %s base linears (cudaMemGetInfo delta over their upload): %.3f GiB\n", base_name,
           vram_base / 1073741824.0);
    codes_resident(false);
    col_norm_resident(false);
    CK(cudaDeviceSynchronize());
    printf("VRAM with the model and the %s base resident, no residual: %.3f GiB\n", base_name,
           vram_used() / 1073741824.0);

    CKB(cublasCreate(&g_ctx.cublas));

    Plan base = uniform_plan("base", 0, MODE_BASE, false), full = uniform_plan("full", 1, MODE_FULL, false);
    g_abs_sum = dev_alloc<float>((size_t)g_cols * sizeof(float));
    CK(cudaMemset(g_abs_sum, 0, (size_t)g_cols * sizeof(float)));
    each_linear([](int, int, LinearResid &r) { r.d_abs_sum = g_abs_sum + r.col_off; });
    g_ctx.calib_abs = true;
    arm_device_state(base, SRC_RESIDENT);
    run(&model, g_calib, base, SRC_RESIDENT, g_dump_a, NULL, &g_res);
    g_ctx.calib_abs = false;
    std::vector<float> abs_sum_all(g_cols);
    CK(cudaMemcpy(abs_sum_all.data(), g_abs_sum, abs_sum_all.size() * sizeof(float), cudaMemcpyDeviceToHost));
    dev_free(g_abs_sum);
    each_linear([](int, int, LinearResid &r) { r.d_abs_sum = NULL; });
    arm_device_state(full, SRC_RESIDENT);
    run(&model, g_calib, full, SRC_RESIDENT, g_dump_b, NULL, &g_res);
    double growth[N_LAYERS], prev = 0;
    for (int l = 0; l < N_LAYERS; l++) {
        double e = 0;
        for (int i = 0; i < N_DUMP; i++) {
            double d2 = 0, f2 = 0;
            for (int d = 0; d < HIDDEN_DIM; d++) {
                double a = g_dump_a[l + 1][i][d], b = g_dump_b[l + 1][i][d];
                d2 += (a - b) * (a - b);
                f2 += b * b;
            }
            e += sqrt(d2 / f2) / N_DUMP;
        }
        growth[l] = fmax(e - prev, 1e-6);
        printf("calib layer %2d: rel error base vs full %.5f, growth %.5f%s\n", l, e, e - prev,
               e - prev < 1e-6 ? " (clamped to 1e-6 for trace)" : "");
        prev = e;
    }
    each_linear([&](int, int, LinearResid &r) {
        const float *abs_sum = abs_sum_all.data() + r.col_off;
        r.chan.resize(r.res.k_in);
        std::iota(r.chan.begin(), r.chan.end(), 0);
        std::stable_sort(r.chan.begin(), r.chan.end(), [&](uint16_t a, uint16_t b) {
            return abs_sum[a] * r.col_norm[a] > abs_sum[b] * r.col_norm[b];
        });
    });

    arm_device_state(base, SRC_RESIDENT);
    run(&model, g_eval.tokens, base, SRC_RESIDENT, NULL, ref, &g_res);
    report(tsv, base, "resident", g_res, g_ctx.vram_in_run, 0);
    arm_device_state(full, SRC_RESIDENT);
    run(&model, g_eval.tokens, full, SRC_RESIDENT, NULL, ref, &g_res);
    report(tsv, full, "resident", g_res, g_ctx.vram_in_run, 0);

    std::vector<Plan> plans;
    for (double f : fracs) {
        plans.push_back(uniform_plan("decdec", f, MODE_DYNAMIC, false));
        plans.push_back(uniform_plan("decdec_norm", f, MODE_DYNAMIC, true));
        plans.push_back(uniform_plan("static", f, MODE_STATIC, false));
        plans.push_back(trace_plan(f, growth));
        const Plan &t = plans.back();
        printf("trace frac %.3f: layers with ksel 0:", f);
        for (int l = 0; l < N_LAYERS; l++)
            if (t.ksel[l][ARM0_QKV] == 0) printf(" %d", l);
        printf("\n");
    }
    std::vector<Arm0Result> results(plans.size());
    for (size_t i = 0; i < plans.size(); i++) {
        arm_device_state(plans[i], SRC_RESIDENT);
        run(&model, g_eval.tokens, plans[i], SRC_RESIDENT, NULL, ref, &results[i]);
        report(tsv, plans[i], "resident", results[i], g_ctx.vram_in_run, 0);
    }
    paired("bf16_splitk - bf16 (summation order only)", g_bf16_res[1], g_bf16_res[0]);
    const size_t arms_per_frac = plans.size() / std::max<size_t>(fracs.size(), 1);
    for (size_t i = 0; i < plans.size(); i++) {
        size_t decdec = i - i % arms_per_frac;
        if (i == decdec) continue;
        char what[64];
        snprintf(what, sizeof(what), "%s - decdec at frac %.2f", plans[i].arm, plans[i].frac);
        paired(what, results[i], results[decdec]);
    }
    if (fracs.empty()) return EXIT_GATE_PASS;

    double f_eq = fracs[fracs.size() / 2];
    const Plan twins[2] = {uniform_plan("static", f_eq, MODE_STATIC, false), uniform_plan("decdec", f_eq, MODE_DYNAMIC, false)};
    for (int t = 0; t < 2; t++) {
        arm_device_state(twins[t], SRC_RESIDENT);
        run(&model, g_eval.tokens, twins[t], SRC_RESIDENT, g_dump_resident[t], ref, &g_resident[t]);
    }

    cudaStream_t copy_stream;
    CK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    bool all_equal = true;
    for (int t = 0; t < 2; t++) {
        const Plan &p = twins[t];
        arm_device_state(p, SRC_STREAMED);
        uint64_t layer_bytes[N_LAYERS];
        std::vector<std::vector<uint8_t>> plugs(N_LAYERS);
        for (int l = 0; l < N_LAYERS; l++) {
            for (int j = 0; j < ARM0_N_LINEARS; j++) {
                LinearResid &r = g_ctx.r[l][j];
                uint64_t row = resid_cols_bytes(1, r.res.n_out, r.res.bits);
                g_ctx.plug_off[l][j] = plugs[l].size();
                if (p.mode == MODE_STATIC) {
                    for (uint16_t k : static_channels(r, p.ksel[l][j])) plugs[l].insert(plugs[l].end(), r.h_codes + k * row, r.h_codes + (k + 1) * row);
                } else {
                    plugs[l].insert(plugs[l].end(), r.h_codes, r.h_codes + r.res.k_in * row);
                }
            }
            layer_bytes[l] = plugs[l].size();
        }
        if (glint_plug_streamer_create(&g_ctx.streamer, N_LAYERS, layer_bytes, copy_stream, 0) != 0)
            setup_fail("streamer create");
        for (int l = 0; l < N_LAYERS; l++)
            if (glint_plug_streamer_set_host_data(g_ctx.streamer, l, plugs[l].data(), layer_bytes[l]) != 0)
                setup_fail("streamer set_host_data layer %d", l);
        plugs.clear();
        CK(cudaDeviceSynchronize());
        run(&model, g_eval.tokens, p, SRC_STREAMED, g_dump_streamed[t], ref, &g_streamed[t]);
        GlintStreamerStats ss;
        glint_plug_streamer_get_stats(g_ctx.streamer, &ss);
        report(tsv, p, "streamed", g_streamed[t], g_ctx.vram_in_run, ss.total_h2d_bytes);
        printf("streamer: %llu H2D bytes over the pass, stall %.1f ms\n", (unsigned long long)ss.total_h2d_bytes,
               ss.total_stall_ms);
        glint_plug_streamer_destroy(g_ctx.streamer);
        g_ctx.streamer = NULL;
        char what[96];
        snprintf(what, sizeof(what), "%s frac %.3f streamed vs resident", p.arm, p.frac);
        all_equal &= same(what, g_resident[t], g_streamed[t], g_dump_resident[t], g_dump_streamed[t], sizeof(Dump));
    }
    printf("VERDICT: %s\n", all_equal ? "REPORT (streamed arms bitwise equal to resident)" : "FAIL (streamed differs)");
    return all_equal ? EXIT_GATE_PASS : EXIT_GATE_FAIL;
}
