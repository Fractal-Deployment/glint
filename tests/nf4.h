#ifndef P7_NF4_H
#define P7_NF4_H
/* NF4 (QLoRA normal-float 4-bit, blocksize 64, absmax scale) quantize-dequantize of BF16 weights, on host.
 * Used for S0-WMAP-NF4 and to run the NF4 comparison point through the same harness. */
#include <math.h>
#include <stdint.h>
#include <string.h>
static const float NF4_CODE[16] = {-1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f, 0.07958029955625534f,
    0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f, 0.44070982933044434f,
    0.5626170039176941f, 0.7229568362236023f, 1.0f};
static inline float nf4_bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t nf4_f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
/* out[i] = bf16(code[nearest(w[i]/absmax)] * absmax) per 64-block; accumulates squared error vs w. */
static inline void nf4_roundtrip(const uint16_t *w, uint16_t *out, uint64_t n, double *se, double *ref2, double *maxabs) {
    for (uint64_t b = 0; b < n; b += 64) {
        uint64_t e = b + 64 < n ? b + 64 : n;
        float am = 0.f;
        for (uint64_t j = b; j < e; j++) { float a = fabsf(nf4_bf2f(w[j])); if (a > am) am = a; }
        for (uint64_t j = b; j < e; j++) {
            float x = nf4_bf2f(w[j]), r = am > 0.f ? x / am : 0.f, best = 1e9f; int bi = 7;
            for (int c = 0; c < 16; c++) { float d = fabsf(r - NF4_CODE[c]); if (d < best) { best = d; bi = c; } }
            float dq = NF4_CODE[bi] * am;
            out[j] = nf4_f2bf(dq);
            double er = (double)x - (double)nf4_bf2f(out[j]);
            *se += er * er; *ref2 += (double)x * x; if (fabs(er) > *maxabs) *maxabs = fabs(er);
        }
    }
}
#endif
