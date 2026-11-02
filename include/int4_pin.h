#ifndef GLINT_INT4_PIN_H
#define GLINT_INT4_PIN_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "int4_tile_gemm.h"

/* bf16_to_int4_pin_v1, the INT4 sibling of bf16-to-int8's bf16_to_int8_pin_v1 and with its arithmetic:
 * per block of 64 elements along the flattened row, s = absmax / 7 in f64 (1.0 for an all-zero block),
 * code = clip(rint(w / s), -7, 7) in f64 with round-half-even, s stored as F32. Codes never take -8.
 * A linear <stem>.weight is stored as U8 [n_out, k_in / 2] of packed codes (low nibble = even flattened index)
 * and <stem>.weight.int4_scale as F32 [n_out * k_in / 64]. 4 bits of code plus 32 / 64 bits of scale give
 * 4.5 bits per weight. */
#define INT4_PIN_SCHEMA "bf16_to_int4_pin_v1"
#define INT4_PIN_BITS_PER_WEIGHT (4.0 + 32.0 / INT4_PIN_BLOCK)

static inline float int4_bf16_to_float(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* Quantizes w[0..n) as one block into codes[0..n) and returns its scale. clamped counts codes that rint put
 * outside [-7, 7], which only f64 rounding of absmax / 7 could cause. */
static inline float int4_quantize_block(const uint16_t *w, int n, int8_t *codes, unsigned long long *clamped) {
    double amax = 0.0;
    for (int i = 0; i < n; i++) amax = fmax(amax, fabs((double)int4_bf16_to_float(w[i])));
    double s = amax > 0.0 ? amax / INT4_QMAX : 1.0;
    for (int i = 0; i < n; i++) {
        double c = rint((double)int4_bf16_to_float(w[i]) / s);
        if (fabs(c) > INT4_QMAX) {
            (*clamped)++;
            c = c > 0 ? INT4_QMAX : -INT4_QMAX;
        }
        codes[i] = (int8_t)c;
    }
    return (float)s;
}

static inline void int4_pack(const int8_t *codes, size_t n, uint8_t *packed) {
    for (size_t i = 0; i < n; i += 2) {
        uint8_t lo = (uint8_t)codes[i] & 15, hi = i + 1 < n ? (uint8_t)codes[i + 1] & 15 : 0;
        packed[i / 2] = (uint8_t)(lo | (hi << 4));
    }
}

#endif
