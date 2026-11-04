/* tests/test_arm0_bf16_full.cu — Full 32-Layer Arm (0) BF16 Evaluation on 2048 Tokens
 * Loads unquantized BF16 weights directly from model-00001-of-00002.safetensors and model-00002-of-00002.safetensors.
 * Executes on RTX 3060 via Tensor Core cuBLAS and online softmax causal attention.
 * Verifies Gate 1:
 *   - Top-1 argmax token agreement >= 95.0% vs HuggingFace (amended 2026-11-02, GATE_RULES.md C.1) Oracle
 *   - Perplexity within 1.0% of HuggingFace Oracle
 * Exit: 0 gate PASS (or INT8 report), 1 gate FAIL, 2 usage, 3 setup, I/O or CUDA error.
 */

#include "arm0_forward.h"
#include "act_map.h"
#include "nf4.h"
#include "../include/glint_htile.h"
#include "../include/glint_plug_streamer.h"
#include "../include/glint_cells.h"
#include <algorithm>
#include <vector>

#define INT8_BLOCK 64

__global__ void int8_block_dequant_kernel(const int8_t* __restrict__ q, const float* __restrict__ scale,
                                          __nv_bfloat16* __restrict__ out, size_t numel) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) out[i] = __float2bfloat16((float)q[i] * scale[i / INT8_BLOCK]);
}

static double g_nf4_se, g_nf4_ref2, g_nf4_max; static uint64_t g_nf4_n; static FILE *g_nf4_map; static int g_nf4_first;
static __nv_bfloat16* upload_linear(const SafeTensors *st1, const SafeTensors *st2, const SafeTensors *int8_pin,
                                    const char *name, size_t numel) {
    __nv_bfloat16 *d_w;
    CK(cudaMalloc(&d_w, numel * sizeof(__nv_bfloat16)));
    if (!int8_pin) {
        const TensorEntry *e = need_tensor(st1, st2, name, "BF16", numel * sizeof(__nv_bfloat16));
        if (getenv("ARM0_NF4")) { /* NF4 comparison point and S0-WMAP-NF4 */
            std::vector<uint16_t> rt(numel);
            double se = 0, ref2 = 0, mx = 0;
            nf4_roundtrip((const uint16_t *)e->data, rt.data(), numel, &se, &ref2, &mx);
            g_nf4_se += se; g_nf4_ref2 += ref2; g_nf4_n += numel; if (mx > g_nf4_max) g_nf4_max = mx;
            if (g_nf4_map) fprintf(g_nf4_map, "%s{\"name\":\"%s\",\"rmse\":%.6e,\"max_abs\":%.6e,\"rel_l2\":%.6e}", g_nf4_first++ ? "," : "", name, sqrt(se / numel), mx, sqrt(se / ref2));
            CK(cudaMemcpy(d_w, rt.data(), numel * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
            return d_w;
        }
        CK(cudaMemcpy(d_w, e->data, numel * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
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

/* --glint PINDIR [--hole-only]: linears through the H-TILE kernel on a glint_pin_v1 (hole + plug + absmax). */
struct GlintLin { uint8_t *hole, *plug; float *am; };
static int g_glint = 0, g_hole_only = 0;
static GlintLin g_gl[N_LAYERS][ARM0_N_LINEARS];
static uint8_t *g_zero_plug = NULL;
/* --glint-stream PINDIR: hole resident, plugs staged in pinned host RAM and brought in one layer ahead through the
 * 2-slot ring (glint_plug_streamer). One layer's plug is its four linears back to back (QKV, O, GATE_UP, DOWN). */
static int g_stream = 0, g_res_layers = 0; /* ARM0_STREAM_RESIDENT_LAYERS=N: the first N layers keep their plug in VRAM, the rest stream */
static size_t g_vram_base_free = 0;
static GlintPlugStreamer *g_streamer = NULL;
static cudaStream_t g_copy_stream = NULL;
static const uint8_t *g_plug_src[N_LAYERS][ARM0_N_LINEARS];
static uint64_t g_plug_off[N_LAYERS][ARM0_N_LINEARS], g_layer_bytes[N_LAYERS];
static const uint8_t *g_cur_dev_plug = NULL;
static void glint_upload(const SafeTensors *a, const SafeTensors *b, int l, int j) {
    size_t n = (size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in;
    char nm[256], pn[300], an[300];
    snprintf(nm, sizeof(nm), "model.layers.%d.%s.weight", l, ARM0_LINEAR[j].name);
    snprintf(pn, sizeof(pn), "%s.glint_plug", nm); snprintf(an, sizeof(an), "%s.absmax", nm);
    size_t nb = (n + 1) / 2, ab = (n + 63) / 64 * sizeof(float);
    const TensorEntry *eh = need_tensor(a, b, nm, "U8", nb), *ep = need_tensor(a, b, pn, "U8", nb), *ea = need_tensor(a, b, an, "F32", ab);
    GlintLin &g = g_gl[l][j];
    CK(cudaMalloc(&g.hole, nb)); CK(cudaMemcpy(g.hole, eh->data, nb, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&g.am, ab)); CK(cudaMemcpy(g.am, ea->data, ab, cudaMemcpyHostToDevice));
    if (g_stream && l >= g_res_layers) { /* plug stays on the host; the ring brings it in */
        g_plug_src[l][j] = (const uint8_t *)ep->data; g_plug_off[l][j] = g_layer_bytes[l]; g_layer_bytes[l] += nb; g.plug = NULL;
    } else if (g_hole_only) { if (!g_zero_plug) { CK(cudaMalloc(&g_zero_plug, 8192ull * 3072 / 2 * 2)); CK(cudaMemset(g_zero_plug, 0, 8192ull * 3072)); } g.plug = g_zero_plug; }
    else { CK(cudaMalloc(&g.plug, nb)); CK(cudaMemcpy(g.plug, ep->data, nb, cudaMemcpyHostToDevice)); }
}

/* --glint-demand PIN: demand-map harness, docs/p7_mode3_fold/DEMAND_MAP_SPEC_2026-11-04.md.
 * Each GLINT linear is y = x*Wparent^T + mask(x)*Wcorr^T with Wparent = PARENT[hole]*absmax and
 * Wcorr = (GLINT_CELLS[hole][plug] - PARENT[hole])*absmax, both decoded to bf16 for one linear at a time and run
 * through cuBLAS. mask(x) keeps, per token, the top fraction f of 64-channel input blocks by the chosen rule.
 * Env: ARM0_DEMAND_RULE=act|random|static|oracle, ARM0_DEMAND_F (default 1), ARM0_DEMAND_SEED (42),
 * ARM0_DEMAND_ONLY_LAYER (restrict one layer, others full), ARM0_DEMAND_STATIC_OUT/IN (calibration file),
 * ARM0_DEMAND_STATS (json: bytes fetched, per-layer churn and predictability). */
enum { DEM_MAXBLK = 128, DEM_CALIB_TOKENS = 256 };
static int g_demand = 0, g_rule = 0, g_only_layer = -1, g_dem_m = 0, g_mask_parent = 0, g_gran_ch = 0;
static float g_frac = 1.0f;
static unsigned g_seed = 42;
static const char *g_static_out = NULL;
static __constant__ float c_cells[16][16];
static __constant__ float c_parent[16];
static __nv_bfloat16 *g_wp = NULL, *g_wd = NULL, *g_xm = NULL;
static uint8_t *g_sel = NULL;
static float *g_colnorm = NULL, *g_static_dev = NULL;
static double *g_dstats = NULL;                                  /* [N_LAYERS][N_LINEARS][4]: sel, jaccard, recall, pairs */
static double g_host_sel[N_LAYERS][ARM0_N_LINEARS];              /* selected blocks summed over tokens for f in {0,1} */

__global__ void k_dem_decode(const uint8_t *hole, const uint8_t *plug, const float *am, __nv_bfloat16 *wp, __nv_bfloat16 *wd, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t hb = hole[i >> 1], pb = plug[i >> 1];
    int h = (i & 1) ? (hb >> 4) : (hb & 15), p = (i & 1) ? (pb >> 4) : (pb & 15);
    float a = am[i >> 6], par = c_parent[h];
    wp[i] = __float2bfloat16(par * a);
    wd[i] = __float2bfloat16((c_cells[h][p] - par) * a);
}
__global__ void k_dem_colnorm(const __nv_bfloat16 *wd, int n_out, int k_in, float *out) { /* grid = k_in/64, block 256 */
    int b = blockIdx.x; double s = 0;
    for (int r = threadIdx.x; r < n_out; r += blockDim.x) {
        const __nv_bfloat16 *row = wd + (size_t)r * k_in + (size_t)b * 64;
        for (int c = 0; c < 64; c++) { float v = __bfloat162float(row[c]); s += (double)v * v; }
    }
    __shared__ double sh[256]; sh[threadIdx.x] = s; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st]; __syncthreads(); }
    if (threadIdx.x == 0) out[b] = (float)sqrt(sh[0]);
}
__device__ static float dem_hash01(unsigned seed, unsigned a, unsigned t, unsigned b) {
    unsigned h = seed * 0x9E3779B1u ^ a * 0x85EBCA6Bu ^ t * 0xC2B2AE35u ^ b * 0x27D4EB2Fu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return (h >> 8) * (1.0f / 16777216.0f);
}
/* grid = tokens, block 128. rule: 0 act, 1 random, 2 static, 3 oracle. Writes sel[t][blk] and the masked input xm. */
__global__ void k_dem_select(const __nv_bfloat16 *x, __nv_bfloat16 *xm, uint8_t *sel, int k_in, int nblk, int nsel, int rule,
                             unsigned seed, unsigned salt, const float *stat, const float *colnorm) {
    int t = blockIdx.x; __shared__ float sc[DEM_MAXBLK]; __shared__ uint8_t sl[DEM_MAXBLK];
    const __nv_bfloat16 *xr = x + (size_t)t * k_in;
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
        float s = 0.f;
        for (int c = 0; c < 64; c++) { float v = __bfloat162float(xr[b * 64 + c]); s += v * v; }
        float act = sqrtf(s);
        sc[b] = rule == 0 ? act : rule == 1 ? dem_hash01(seed, salt, (unsigned)t, (unsigned)b) : rule == 2 ? stat[b] : act * colnorm[b];
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
        int rank = 0; float me = sc[b];
        for (int j = 0; j < nblk; j++) rank += (sc[j] > me) || (sc[j] == me && j < b);
        sl[b] = rank < nsel; sel[(size_t)t * nblk + b] = sl[b];
    }
    __syncthreads();
    for (int c = threadIdx.x; c < k_in; c += blockDim.x) xm[(size_t)t * k_in + c] = sl[c >> 6] ? xr[c] : __float2bfloat16(0.f);
}
/* Channel granularity (spec: act and random only). Keeps the nsel input channels of each token with the largest key
 * (|x| for act, a hash for random), found by a binary search on the float bit pattern; ties may keep a few more.
 * grid = tokens, block 256. acc[0] += kept channels. */
__global__ void k_dem_select_ch(const __nv_bfloat16 *x, __nv_bfloat16 *xm, int k_in, int nsel, int rule, unsigned seed, unsigned salt, double *acc) {
    int t = blockIdx.x; const __nv_bfloat16 *xr = x + (size_t)t * k_in;
    __shared__ int cnt; __shared__ unsigned lo, hi;
    if (threadIdx.x == 0) { lo = 0; hi = 0x7f800000u; }
    __syncthreads();
    for (int it = 0; it < 32; it++) {
        if (lo >= hi) break;
        unsigned mid = lo + (hi - lo + 1) / 2;
        if (threadIdx.x == 0) cnt = 0;
        __syncthreads();
        int c = 0;
        for (int i = threadIdx.x; i < k_in; i += blockDim.x) {
            unsigned key = rule == 0 ? __float_as_uint(fabsf(__bfloat162float(xr[i]))) : __float_as_uint(dem_hash01(seed, salt, (unsigned)t, (unsigned)i));
            c += key >= mid;
        }
        atomicAdd(&cnt, c);
        __syncthreads();
        if (threadIdx.x == 0) { if (cnt >= nsel) lo = mid; else hi = mid - 1; }
        __syncthreads();
    }
    int kept = 0;
    for (int i = threadIdx.x; i < k_in; i += blockDim.x) {
        unsigned key = rule == 0 ? __float_as_uint(fabsf(__bfloat162float(xr[i]))) : __float_as_uint(dem_hash01(seed, salt, (unsigned)t, (unsigned)i));
        bool keep = key >= lo; kept += keep;
        xm[(size_t)t * k_in + i] = keep ? xr[i] : __float2bfloat16(0.f);
    }
    __shared__ int ktot; if (threadIdx.x == 0) ktot = 0; __syncthreads();
    atomicAdd(&ktot, kept); __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&acc[0], (double)ktot);
}
__global__ void k_dem_stats(const uint8_t *sel, int nblk, double *acc) { /* grid = tokens, block 128 */
    int t = blockIdx.x; __shared__ int cnt[3];
    if (threadIdx.x == 0) cnt[0] = cnt[1] = cnt[2] = 0;
    __syncthreads();
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
        int a = sel[(size_t)t * nblk + b], p = t > 0 ? sel[(size_t)(t - 1) * nblk + b] : 0;
        atomicAdd(&cnt[0], a); if (t > 0) { atomicAdd(&cnt[1], a & p); atomicAdd(&cnt[2], a | p); }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&acc[0], (double)cnt[0]);
        if (t > 0) { atomicAdd(&acc[1], cnt[2] ? (double)cnt[1] / cnt[2] : 1.0); atomicAdd(&acc[2], cnt[0] ? (double)cnt[1] / cnt[0] : 1.0); atomicAdd(&acc[3], 1.0); }
    }
}
__global__ void k_dem_static_acc(const __nv_bfloat16 *x, int m, int k_in, float *out) { /* grid = nblk, block 128: mean act norm over calibration tokens */
    int b = blockIdx.x; double s = 0; int mm = m < DEM_CALIB_TOKENS ? m : DEM_CALIB_TOKENS;
    for (int t = threadIdx.x; t < mm; t += blockDim.x) {
        double q = 0; for (int c = 0; c < 64; c++) { float v = __bfloat162float(x[(size_t)t * k_in + b * 64 + c]); q += (double)v * v; }
        s += sqrt(q);
    }
    __shared__ double sh[128]; sh[threadIdx.x] = s; __syncthreads();
    for (int st = 64; st > 0; st >>= 1) { if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st]; __syncthreads(); }
    if (threadIdx.x == 0) out[b] = (float)(sh[0] / mm);
}
static void demand_init() {
    CK(cudaMemcpyToSymbol(c_cells, GLINT_CELLS, sizeof(float) * 256));
    CK(cudaMemcpyToSymbol(c_parent, NF4_CODE, sizeof(float) * 16));
    size_t maxn = 0; for (int j = 0; j < ARM0_N_LINEARS; j++) { size_t n = (size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in; if (n > maxn) maxn = n; }
    CK(cudaMalloc(&g_wp, maxn * 2)); CK(cudaMalloc(&g_wd, maxn * 2));
    CK(cudaMalloc(&g_xm, 2048ull * 8192 * 2)); CK(cudaMalloc(&g_sel, 2048ull * DEM_MAXBLK));
    CK(cudaMalloc(&g_colnorm, DEM_MAXBLK * 4));
    size_t sn = (size_t)N_LAYERS * ARM0_N_LINEARS * DEM_MAXBLK;
    CK(cudaMalloc(&g_static_dev, sn * 4)); CK(cudaMemset(g_static_dev, 0, sn * 4));
    CK(cudaMalloc(&g_dstats, (size_t)N_LAYERS * ARM0_N_LINEARS * 4 * 8)); CK(cudaMemset(g_dstats, 0, (size_t)N_LAYERS * ARM0_N_LINEARS * 4 * 8));
    const char *r = getenv("ARM0_DEMAND_RULE");
    g_rule = !r || !strcmp(r, "act") ? 0 : !strcmp(r, "random") ? 1 : !strcmp(r, "static") ? 2 : !strcmp(r, "oracle") ? 3 : -1;
    if (g_rule < 0) setup_fail("ARM0_DEMAND_RULE must be act|random|static|oracle");
    if (getenv("ARM0_DEMAND_F")) g_frac = (float)atof(getenv("ARM0_DEMAND_F"));
    if (getenv("ARM0_DEMAND_SEED")) g_seed = (unsigned)atoi(getenv("ARM0_DEMAND_SEED"));
    if (getenv("ARM0_DEMAND_ONLY_LAYER")) g_only_layer = atoi(getenv("ARM0_DEMAND_ONLY_LAYER"));
    g_static_out = getenv("ARM0_DEMAND_STATIC_OUT");
    g_mask_parent = getenv("ARM0_DEMAND_MASK_PARENT") && atoi(getenv("ARM0_DEMAND_MASK_PARENT")) == 1;
    g_gran_ch = getenv("ARM0_DEMAND_GRAN") && !strcmp(getenv("ARM0_DEMAND_GRAN"), "channel");
    if (g_gran_ch && g_rule != 0 && g_rule != 1) setup_fail("channel granularity supports rules act and random only");
    if (g_rule == 2) {
        const char *in = getenv("ARM0_DEMAND_STATIC_IN"); if (!in) setup_fail("rule static needs ARM0_DEMAND_STATIC_IN");
        FILE *f = fopen(in, "rb"); if (!f) setup_fail("cannot read %s", in);
        std::vector<float> h(sn); if (fread(h.data(), 4, sn, f) != sn) setup_fail("short read %s", in); fclose(f);
        CK(cudaMemcpy(g_static_dev, h.data(), sn * 4, cudaMemcpyHostToDevice));
    }
    printf("Demand map: rule=%d f=%.4f seed=%u only_layer=%d\n", g_rule, g_frac, g_seed, g_only_layer);
}
static void demand_linear(cublasHandle_t cb, int layer, int which, const __nv_bfloat16 *x, float *y, int m) {
    const int n_out = ARM0_LINEAR[which].n_out, k_in = ARM0_LINEAR[which].k_in, nblk = k_in / 64; const size_t n = (size_t)n_out * k_in;
    g_dem_m = m; GlintLin &g = g_gl[layer][which];
    k_dem_decode<<<(unsigned)((n + 255) / 256), 256>>>(g.hole, g.plug, g.am, g_wp, g_wd, n);
    float one = 1.0f, zero = 0.0f;
    if (g_static_out) k_dem_static_acc<<<nblk, 128>>>(x, m, k_in, g_static_dev + ((size_t)layer * ARM0_N_LINEARS + which) * DEM_MAXBLK);
    float f = (g_only_layer >= 0 && layer != g_only_layer) ? 1.0f : g_frac;
    if (g_mask_parent && f <= 0.0f) setup_fail("ARM0_DEMAND_MASK_PARENT needs f > 0");
    const __nv_bfloat16 *xin = x;
    if (f >= 1.0f) g_host_sel[layer][which] += (double)m * (g_gran_ch ? k_in : nblk);
    else if (f > 0.0f && g_gran_ch) {
        int nsel = (int)ceil((double)f * k_in - 1e-6); if (nsel < 1) nsel = 1; if (nsel > k_in) nsel = k_in;
        k_dem_select_ch<<<m, 256>>>(x, g_xm, k_in, nsel, g_rule, g_seed, (unsigned)(layer * 8 + which), g_dstats + ((size_t)layer * ARM0_N_LINEARS + which) * 4);
        xin = g_xm;
    } else if (f > 0.0f) {
        int nsel = (int)ceil((double)f * nblk - 1e-6); if (nsel < 1) nsel = 1; if (nsel > nblk) nsel = nblk;
        if (g_rule == 3) k_dem_colnorm<<<nblk, 256>>>(g_wd, n_out, k_in, g_colnorm);
        const float *stat = g_static_dev + ((size_t)layer * ARM0_N_LINEARS + which) * DEM_MAXBLK;
        k_dem_select<<<m, 128>>>(x, g_xm, g_sel, k_in, nblk, nsel, g_rule, g_seed, (unsigned)(layer * 8 + which), stat, g_colnorm);
        k_dem_stats<<<m, 128>>>(g_sel, nblk, g_dstats + ((size_t)layer * ARM0_N_LINEARS + which) * 4);
        xin = g_xm;
    }
    /* ARM0_DEMAND_MASK_PARENT=1 (exploratory, not in the spec): the parent GEMM sees the masked input too, so low-activation
     * channels are skipped for the whole linear (a TEAL-style activation-sparsity test), not only for the plug correction. */
    const __nv_bfloat16 *xp = (g_mask_parent && f < 1.0f) ? xin : x;
    CKB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, n_out, m, k_in, &one, g_wp, CUDA_R_16BF, k_in, xp, CUDA_R_16BF, k_in, &zero, y, CUDA_R_32F, n_out, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    if (f > 0.0f)
        CKB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, n_out, m, k_in, &one, g_wd, CUDA_R_16BF, k_in, xin, CUDA_R_16BF, k_in, &one, y, CUDA_R_32F, n_out, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}
static void demand_report() {
    CK(cudaDeviceSynchronize());
    std::vector<double> h((size_t)N_LAYERS * ARM0_N_LINEARS * 4);
    CK(cudaMemcpy(h.data(), g_dstats, h.size() * 8, cudaMemcpyDeviceToHost));
    double sel_bytes = 0, all_bytes = 0; const int m = g_dem_m;
    const char *sp = getenv("ARM0_DEMAND_STATS"); FILE *js = sp ? fopen(sp, "w") : NULL;
    if (js) fprintf(js, "{\"rule\":%d,\"f\":%.6f,\"seed\":%u,\"only_layer\":%d,\"m\":%d,\"layers\":[", g_rule, g_frac, g_seed, g_only_layer, m);
    for (int l = 0; l < N_LAYERS; l++) {
        double jac = 0, rec = 0, pairs = 0, lsel = 0, lall = 0;
        for (int j = 0; j < ARM0_N_LINEARS; j++) {
            const double *a = &h[((size_t)l * ARM0_N_LINEARS + j) * 4]; int nblk = g_gran_ch ? ARM0_LINEAR[j].k_in : ARM0_LINEAR[j].k_in / 64;
            double pb = (double)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in / 2.0, frac = (a[0] + g_host_sel[l][j]) / ((double)m * nblk);
            sel_bytes += frac * pb; all_bytes += pb; lsel += frac * pb; lall += pb; jac += a[1]; rec += a[2]; pairs += a[3];
        }
        if (js) fprintf(js, "%s{\"layer\":%d,\"bytes_frac\":%.6f,\"jaccard_prev_token\":%.6f,\"recall_prev_token\":%.6f}", l ? "," : "", l, lsel / lall, pairs > 0 ? jac / pairs : -1.0, pairs > 0 ? rec / pairs : -1.0);
    }
    if (js) { fprintf(js, "],\"bytes_frac\":%.6f}\n", sel_bytes / all_bytes); fclose(js); }
    printf("DEMAND_STATS bytes_frac=%.6f\n", sel_bytes / all_bytes);
    if (g_static_out) {
        size_t sn = (size_t)N_LAYERS * ARM0_N_LINEARS * DEM_MAXBLK; std::vector<float> hs(sn);
        CK(cudaMemcpy(hs.data(), g_static_dev, sn * 4, cudaMemcpyDeviceToHost));
        FILE *f = fopen(g_static_out, "wb"); if (f) { fwrite(hs.data(), 4, sn, f); fclose(f); printf("static calibration written: %s\n", g_static_out); }
    }
}

struct CublasLinears {
    cublasHandle_t cublas;
    __nv_bfloat16 *w[N_LAYERS][ARM0_N_LINEARS];
};

static void cublas_linear(void *ctx, int layer, Arm0Linear which, const __nv_bfloat16 *x, float *y, int m) {
    const CublasLinears *lin = (const CublasLinears *)ctx;
    act_capture(layer, which, x, m);
    if (g_glint) {
        const GlintLin &g = g_gl[layer][which];
        if (g_demand) { demand_linear(lin->cublas, layer, which, x, y, m); return; }
        if (g_stream && layer >= g_res_layers) {
            if (which == ARM0_QKV && glint_plug_streamer_acquire(g_streamer, layer - g_res_layers, &g_cur_dev_plug) != 0)
                setup_fail("plug acquire failed layer %d", layer);
            if (!g_cur_dev_plug) setup_fail("plug for layer %d not acquired (linear order changed)", layer);
            if (launch_glint_htile_s3_f32_stream(g.hole, g_cur_dev_plug + g_plug_off[layer][which], g.am, x, y, m,
                                                 ARM0_LINEAR[which].n_out, ARM0_LINEAR[which].k_in, 64, 0) != 0)
                setup_fail("htile launch failed layer %d", layer);
            if (which == ARM0_DOWN) {
                if (glint_plug_streamer_release(g_streamer, layer - g_res_layers) != 0) setup_fail("plug release failed layer %d", layer);
                g_cur_dev_plug = NULL;
            }
            return;
        }
        if (launch_glint_htile_s3_f32(g.hole, g.plug, g.am, x, y, m, ARM0_LINEAR[which].n_out, ARM0_LINEAR[which].k_in, 64) != 0)
            setup_fail("htile launch failed layer %d", layer);
        return;
    }
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
    const char* glint_dir = NULL;
    int argi = 1;
    for (; argi + 1 < argc && argv[argi][0] == '-'; argi += 2) {
        if (strcmp(argv[argi], "--dump") == 0) dump_dir = argv[argi + 1];
        else if (strcmp(argv[argi], "--int8") == 0) int8_pin_path = argv[argi + 1];
        else if (strcmp(argv[argi], "--glint") == 0) { glint_dir = argv[argi + 1]; g_glint = 1; }
        else if (strcmp(argv[argi], "--glint-hole-only") == 0) { glint_dir = argv[argi + 1]; g_glint = 1; g_hole_only = 1; }
        else if (strcmp(argv[argi], "--glint-stream") == 0) { glint_dir = argv[argi + 1]; g_glint = 1; g_stream = 1; }
        else if (strcmp(argv[argi], "--glint-demand") == 0) { glint_dir = argv[argi + 1]; g_glint = 1; g_demand = 1; }
        else break;
    }
    if (argc - argi != 2) {
        printf("usage: %s [--dump DIR] [--int8 PIN.safetensors] <model_snapshot_dir> <artifacts_dir>\n", argv[0]);
        return EXIT_USAGE;
    }
    if (g_stream && getenv("ARM0_STREAM_RESIDENT_LAYERS")) { g_res_layers = atoi(getenv("ARM0_STREAM_RESIDENT_LAYERS")); if (g_res_layers < 0) g_res_layers = 0; if (g_res_layers > N_LAYERS) g_res_layers = N_LAYERS; }
    { size_t tot = 0; CK(cudaMemGetInfo(&g_vram_base_free, &tot)); }
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
    SafeTensors gs1, gs2;
    if (g_glint) {
        char a[1024], b[1024];
        snprintf(a, sizeof(a), "%s/model-00001-of-00002.safetensors", glint_dir);
        snprintf(b, sizeof(b), "%s/model-00002-of-00002.safetensors", glint_dir);
        if (safetensors_open(&gs1, a) != 0 || safetensors_open(&gs2, b) != 0) setup_fail("cannot open glint pin %s", glint_dir);
        for (int l = 0; l < N_LAYERS; l++) for (int j = 0; j < ARM0_N_LINEARS; j++) glint_upload(&gs1, &gs2, l, j);
        if (g_demand) demand_init();
        if (g_stream && g_res_layers < N_LAYERS) {
            CK(cudaStreamCreateWithFlags(&g_copy_stream, cudaStreamNonBlocking));
            if (glint_plug_streamer_create(&g_streamer, N_LAYERS - g_res_layers, g_layer_bytes + g_res_layers, g_copy_stream, 0) != 0) setup_fail("streamer create failed");
            for (int l = g_res_layers; l < N_LAYERS; l++) {
                std::vector<uint8_t> cat(g_layer_bytes[l]);
                for (int j = 0; j < ARM0_N_LINEARS; j++) {
                    size_t nb = ((size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in + 1) / 2;
                    memcpy(cat.data() + g_plug_off[l][j], g_plug_src[l][j], nb);
                }
                if (glint_plug_streamer_set_host_data(g_streamer, l - g_res_layers, cat.data(), g_layer_bytes[l]) != 0) setup_fail("streamer host data layer %d", l);
            }
            if (glint_plug_streamer_start(g_streamer) != 0) setup_fail("streamer start failed");
        }
        printf("Glint pin loaded (%s).\n", g_stream ? "hole resident; first N layers' plug resident, the rest streamed through the 2-slot ring" : (g_hole_only ? "hole only, plug zeroed" : "hole + plug"));
    }
    if (getenv("ARM0_NF4") && getenv("ARM0_NF4_WMAP")) {
        g_nf4_map = fopen(getenv("ARM0_NF4_WMAP"), "w");
        if (g_nf4_map) fprintf(g_nf4_map, "{\"schema\":\"nf4_weight_diff_map_v1\",\"scheme\":\"NF4 blocksize 64 absmax, bf16 dequant\",\"tensors\":[");
    }
    for (int l = 0; l < N_LAYERS && !g_glint; l++) {
        for (int j = 0; j < ARM0_N_LINEARS; j++) {
            char name[256];
            snprintf(name, sizeof(name), "model.layers.%d.%s.weight", l, ARM0_LINEAR[j].name);
            lin.w[l][j] = upload_linear(&st1, &st2, int8_pin, name, (size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in);
        }
    }
    if (getenv("ARM0_NF4")) {
        printf("NF4 weights: %llu linear elems, RMSE %.6e, max abs %.6e, rel L2 %.6e\n", (unsigned long long)g_nf4_n, sqrt(g_nf4_se / g_nf4_n), g_nf4_max, sqrt(g_nf4_se / g_nf4_ref2));
        if (g_nf4_map) { fprintf(g_nf4_map, "],\"n_elems\":%llu,\"rmse\":%.6e,\"max_abs\":%.6e,\"rel_l2\":%.6e}\n", (unsigned long long)g_nf4_n, sqrt(g_nf4_se / g_nf4_n), g_nf4_max, sqrt(g_nf4_se / g_nf4_ref2)); fclose(g_nf4_map); }
    }
    printf("Uploaded all 32 layers of %s weights to VRAM.\n", int8_pin ? "INT8-dequantized" : "BF16");

    const char *act_map = getenv("ARM0_ACT_MAP");
    act_map_init(act_map);
    CKB(cublasCreate(&lin.cublas));
    arm0_forward(&model, ev.tokens, cublas_linear, &lin, lin.cublas, dump_dir ? g_dump : NULL, &g_result);
    if (g_demand) demand_report();
    if (g_stream) {
        size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
        printf("VRAM_USED_MIB %.1f (resident plug layers %d of %d)\n", ((double)g_vram_base_free - (double)fr) / 1048576.0, g_res_layers, N_LAYERS);
    }
    if (g_stream && g_streamer) {
        GlintStreamerStats ss;
        if (glint_plug_streamer_get_stats(g_streamer, &ss) == 0) {
            double mx = 0; for (int l = 1; l < ss.n_layers; l++) if (ss.per_layer_stall_ms[l] > mx) mx = ss.per_layer_stall_ms[l];
            printf("STREAM_STATS h2d_bytes=%llu total_stall_ms=%.3f layer0_stall_ms=%.3f max_stall_ms_layers_1plus=%.3f\n",
                   (unsigned long long)ss.total_h2d_bytes, ss.total_stall_ms, ss.per_layer_stall_ms[0], mx);
        }
    }
    act_map_write(act_map, int8_pin ? "BF16 cuBLAS (INT8-dequantized weights)" : "BF16 cuBLAS (original BF16 weights)");
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
    if (g_result.has_kl) { /* KL(ref || arm) per token against ARM0_LOGITS_REF */
        std::vector<float> kv(g_result.kl, g_result.kl + 2047); double ks = 0; for (float v : kv) ks += v;
        std::sort(kv.begin(), kv.end());
        printf("KL vs reference: mean %.6e p99 %.6e max %.6e (2047 tokens)\n", ks / 2047.0, kv[(size_t)(0.99 * 2046)], kv[2046]);
    }

    int pass = top1_acc >= 95.0 && ppl_delta_pct <= 1.0;
    if (g_glint) printf("VERDICT: REPORT ONLY (glint %s)\n", g_hole_only ? "hole-only" : "hole+plug");
    else if (int8_pin) printf("VERDICT: REPORT ONLY (INT8 pin %s, no gate)\n", int8_pin_path);
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
    return (pass || int8_pin || g_glint) ? EXIT_GATE_PASS : EXIT_GATE_FAIL;
}
