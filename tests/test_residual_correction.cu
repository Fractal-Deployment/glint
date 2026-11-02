#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "../include/residual_correction.h"

#define CK(call)                                                                                     \
    do {                                                                                             \
        cudaError_t e_ = (call);                                                                     \
        if (e_ != cudaSuccess) {                                                                     \
            printf("SETUP ERROR: %s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); \
            exit(3);                                                                                 \
        }                                                                                            \
    } while (0)

static int g_failed = 0;

static void check(bool ok, const char *what) {
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) g_failed++;
}

template <typename T>
static T *to_dev(const std::vector<T> &h) {
    T *d;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
static std::vector<T> to_host(const T *d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

static float bf(float v) { return __bfloat162float(__float2bfloat16(v)); }

static uint32_t g_rng = 12345;
static float frand() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (g_rng >> 8) / 16777216.0f * 2.0f - 1.0f;
}

struct Linear {
    int n_out, k_in;
    std::vector<__nv_bfloat16> w;
    std::vector<int8_t> q;
    std::vector<float> s;
};

static Linear make_linear(int n_out, int k_in) {
    Linear L{n_out, k_in};
    size_t numel = (size_t)n_out * k_in;
    L.w.resize(numel);
    L.q.resize(numel);
    L.s.resize(numel / INT8_PIN_BLOCK);
    for (size_t i = 0; i < numel; i++) L.w[i] = __float2bfloat16(frand() * (1.0f + 3.0f * (i % 97 == 0)));
    for (size_t b = 0; b < L.s.size(); b++) {
        float amax = 0.0f;
        for (int j = 0; j < INT8_PIN_BLOCK; j++) amax = fmaxf(amax, fabsf(__bfloat162float(L.w[b * 64 + j])));
        L.s[b] = amax / 127.0f;
        for (int j = 0; j < INT8_PIN_BLOCK; j++)
            L.q[b * 64 + j] = (int8_t)lrintf(__bfloat162float(L.w[b * 64 + j]) / L.s[b]);
    }
    return L;
}

static float r_hat(const Linear &L, const std::vector<uint8_t> &codes, int n, int k) {
    int b = codes[(size_t)k * (L.n_out / 2) + n / 2];
    int nib = (n & 1) ? b >> 4 : b & 15;
    int c = nib >= 8 ? nib - 16 : nib;
    return L.s[((size_t)n * L.k_in + k) / 64] * (float)c * RESID_STEP;
}

static std::vector<int> ref_topk(const __nv_bfloat16 *x, int k_in, const float *wt, int ksel) {
    std::vector<int> idx(k_in);
    for (int k = 0; k < k_in; k++) idx[k] = k;
    auto score = [&](int k) { return fabsf(__bfloat162float(x[k])) * (wt ? wt[k] : 1.0f); };
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return score(a) > score(b); });
    idx.resize(ksel);
    std::sort(idx.begin(), idx.end());
    return idx;
}

int main() {
    const int n_out = 96, k_in = 320, m = 37;
    Linear L = make_linear(n_out, k_in);
    int8_t *d_q = to_dev(L.q);
    float *d_s = to_dev(L.s);
    __nv_bfloat16 *d_w = to_dev(L.w);
    Int8PinLinear base{d_q, d_s, n_out, k_in};

    uint8_t *d_codes;
    float *d_norm;
    ResidBuildStats *d_stats;
    CK(cudaMalloc(&d_codes, resid_cols_bytes(k_in, n_out)));
    CK(cudaMalloc(&d_norm, k_in * sizeof(float)));
    CK(cudaMalloc(&d_stats, sizeof(ResidBuildStats)));
    CK(cudaMemset(d_stats, 0, sizeof(ResidBuildStats)));
    CK(resid_build(d_w, base, d_codes, d_norm, d_stats, 0));
    std::vector<uint8_t> codes = to_host(d_codes, resid_cols_bytes(k_in, n_out));
    std::vector<float> norm = to_host(d_norm, k_in);
    ResidBuildStats stats = to_host(d_stats, 1)[0];
    float max_rel;
    memcpy(&max_rel, &stats.max_rel_bits, 4);

    double worst_err = 0.0, worst_base = 0.0, norm_err = 0.0;
    for (int k = 0; k < k_in; k++) {
        double sq = 0.0;
        for (int n = 0; n < n_out; n++) {
            size_t i = (size_t)n * k_in + k;
            float s = L.s[i / 64];
            float r = __bfloat162float(L.w[i]) - bf((float)L.q[i] * s);
            float rh = r_hat(L, codes, n, k);
            worst_err = fmax(worst_err, fabs(r - rh) / s);
            worst_base = fmax(worst_base, fabs(r) / s);
            sq += (double)rh * rh;
        }
        norm_err = fmax(norm_err, fabs(sqrt(sq) - norm[k]) / fmax(sqrt(sq), 1e-30));
    }
    printf("max |R/s| %.4f (device %.4f), max |R - R_hat| / s %.4f, clamped %llu\n", worst_base, max_rel, worst_err,
           stats.clamped);
    check(stats.clamped == 0 && max_rel <= RESID_RANGE, "every residual fits the 4-bit range without clamping");
    check(fabs(max_rel - worst_base) < 1e-6, "device max |R/s| matches the host");
    check(worst_err <= RESID_STEP / 2 * 1.0001, "R_hat is within half a step of R everywhere");
    check(norm_err < 1e-5, "column norms match the decoded residual");

    std::vector<__nv_bfloat16> x((size_t)m * k_in);
    for (size_t i = 0; i < x.size(); i++) x[i] = __float2bfloat16(roundf(frand() * 4.0f) / 4.0f);
    __nv_bfloat16 *d_x = to_dev(x), *d_xg;
    CK(cudaMalloc(&d_xg, x.size() * sizeof(__nv_bfloat16)));
    uint8_t *d_used;
    CK(cudaMalloc(&d_used, k_in));

    const int ksels[] = {1, 7, 64, k_in};
    for (int wi = 0; wi < 2; wi++) {
        const float *wt = wi ? norm.data() : NULL;
        for (int ksel : ksels) {
            CK(cudaMemset(d_used, 0, k_in));
            CK(resid_select_topk(d_x, m, k_in, wi ? d_norm : NULL, ksel, d_xg, d_used, 0));
            std::vector<__nv_bfloat16> xg = to_host(d_xg, x.size());
            std::vector<uint8_t> used = to_host(d_used, k_in), want_used(k_in, 0);
            bool ok = true;
            for (int r = 0; r < m; r++) {
                std::vector<int> sel = ref_topk(&x[(size_t)r * k_in], k_in, wt, ksel);
                std::vector<char> keep(k_in, 0);
                for (int k : sel) keep[k] = 1, want_used[k] = 1;
                for (int k = 0; k < k_in; k++) {
                    float want = keep[k] ? __bfloat162float(x[(size_t)r * k_in + k]) : 0.0f;
                    if (__bfloat162float(xg[(size_t)r * k_in + k]) != want) ok = false;
                }
            }
            char what[160];
            snprintf(what, sizeof(what), "top-%d by |x|%s keeps the reference channels, ties to the lowest index", ksel,
                     wi ? " * ||R col||" : "");
            check(ok && used == want_used, what);
        }
    }
    check(resid_select_topk(d_x, m, k_in, NULL, 0, d_xg, NULL, 0) == cudaErrorInvalidValue &&
              resid_select_topk(d_x, m, k_in, NULL, k_in + 1, d_xg, NULL, 0) == cudaErrorInvalidValue,
          "ksel outside [1, k_in] is refused");

    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return 3;
    __nv_bfloat16 *d_rg;
    float *d_y;
    CK(cudaMalloc(&d_rg, (size_t)n_out * k_in * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_y, (size_t)m * n_out * sizeof(float)));
    std::vector<float> y0((size_t)m * n_out);
    for (size_t i = 0; i < y0.size(); i++) y0[i] = frand();

    const int ksel = 40;
    CK(resid_select_topk(d_x, m, k_in, NULL, ksel, d_xg, NULL, 0));
    CK(resid_dequant(ResidCols{d_codes, NULL, k_in}, base, d_rg, 0));
    CK(cudaMemcpy(d_y, y0.data(), y0.size() * sizeof(float), cudaMemcpyHostToDevice));
    if (resid_accumulate(h, d_xg, d_rg, d_y, m, n_out, k_in) != CUBLAS_STATUS_SUCCESS) return 3;
    std::vector<float> y = to_host(d_y, y0.size());
    double worst = 0.0;
    for (int r = 0; r < m; r++) {
        std::vector<int> sel = ref_topk(&x[(size_t)r * k_in], k_in, NULL, ksel);
        for (int n = 0; n < n_out; n++) {
            double want = y0[(size_t)r * n_out + n];
            for (int k : sel) want += (double)__bfloat162float(x[(size_t)r * k_in + k]) * bf(r_hat(L, codes, n, k));
            worst = fmax(worst, fabs(want - y[(size_t)r * n_out + n]));
        }
    }
    printf("per-row correction max abs err vs double %.3g\n", worst);
    check(worst < 1e-5, "y += X[:, sel_row] R_hat[:, sel_row]^T matches double per row");

    std::vector<uint16_t> chan = {3, 64, 65, 127, 200, 319};
    const int c = (int)chan.size();
    std::vector<uint8_t> block((size_t)c * (n_out / 2));
    for (int j = 0; j < c; j++)
        memcpy(&block[(size_t)j * (n_out / 2)], &codes[(size_t)chan[j] * (n_out / 2)], n_out / 2);
    uint8_t *d_block = to_dev(block);
    uint16_t *d_chan = to_dev(chan);
    CK(resid_gather(d_x, m, k_in, d_chan, c, d_xg, 0));
    CK(resid_dequant(ResidCols{d_block, d_chan, c}, base, d_rg, 0));
    CK(cudaMemcpy(d_y, y0.data(), y0.size() * sizeof(float), cudaMemcpyHostToDevice));
    if (resid_accumulate(h, d_xg, d_rg, d_y, m, n_out, c) != CUBLAS_STATUS_SUCCESS) return 3;
    y = to_host(d_y, y0.size());
    worst = 0.0;
    for (int r = 0; r < m; r++)
        for (int n = 0; n < n_out; n++) {
            double want = y0[(size_t)r * n_out + n];
            for (int k : chan) want += (double)__bfloat162float(x[(size_t)r * k_in + k]) * bf(r_hat(L, codes, n, k));
            worst = fmax(worst, fabs(want - y[(size_t)r * n_out + n]));
        }
    printf("static-column correction max abs err vs double %.3g\n", worst);
    check(worst < 1e-5, "a gathered column block corrects exactly the channels it names");

    cublasDestroy(h);
    printf("VERDICT: %s (%d failed)\n", g_failed ? "FAIL" : "PASS", g_failed);
    return g_failed ? 1 : 0;
}
