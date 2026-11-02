/* bf16_to_int4_pin_v1 quantizer: rounding, scale, clamp range and nibble packing.
 * Exit: 0 all PASS, 1 any FAIL. */
#include <stdio.h>
#include <stdlib.h>

#include "../include/int4_pin.h"

static int g_failed = 0;

static void check(bool ok, const char *what) {
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) g_failed++;
}

static uint16_t bf16_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main() {
    unsigned long long clamped = 0;
    int8_t codes[INT4_PIN_BLOCK];

    const float ties[] = {7.0f, 2.5f, 3.5f, -2.5f, 0.5f, -0.5f, 1.5f, -1.5f, 6.5f, -6.5f};
    const int8_t want[] = {7, 2, 4, -2, 0, 0, 2, -2, 6, -6};
    const int n_ties = sizeof(ties) / sizeof(ties[0]);
    uint16_t w[INT4_PIN_BLOCK];
    for (int i = 0; i < n_ties; i++) w[i] = bf16_bits(ties[i]);
    float s = int4_quantize_block(w, n_ties, codes, &clamped);
    check(s == 1.0f && memcmp(codes, want, n_ties) == 0, "absmax 7 gives scale 1 and halves round to even");

    for (int i = 0; i < INT4_PIN_BLOCK; i++) w[i] = bf16_bits(i == 5 ? -3.0f : 0.25f);
    s = int4_quantize_block(w, INT4_PIN_BLOCK, codes, &clamped);
    check(s == (float)(3.0 / 7) && codes[5] == -7 && codes[0] == 1, "a negative absmax maps to -7");

    for (int i = 0; i < INT4_PIN_BLOCK; i++) w[i] = 0x8000 * (i % 2);
    s = int4_quantize_block(w, INT4_PIN_BLOCK, codes, &clamped);
    bool zero = s == 1.0f;
    for (int i = 0; i < INT4_PIN_BLOCK; i++) zero &= codes[i] == 0;
    check(zero, "an all-zero block (with -0.0) gets scale 1 and code 0");

    srand(7);
    double worst = 0.0;
    bool in_range = true;
    for (int b = 0; b < 20000; b++) {
        float mag = powf(2.0f, (float)(rand() % 40 - 30));
        for (int i = 0; i < INT4_PIN_BLOCK; i++) w[i] = bf16_bits(mag * ((float)rand() / RAND_MAX * 2.0f - 1.0f));
        s = int4_quantize_block(w, INT4_PIN_BLOCK, codes, &clamped);
        for (int i = 0; i < INT4_PIN_BLOCK; i++) {
            in_range &= codes[i] >= -7 && codes[i] <= 7;
            worst = fmax(worst, fabs((double)int4_bf16_to_float(w[i]) - codes[i] * (double)s) / s);
        }
    }
    printf("random blocks: max |w - code * scale| / scale %.9f, clamped %llu\n", worst, clamped);
    check(in_range && clamped == 0, "codes stay in [-7, 7] without clamping");
    check(worst <= 0.5 + 1e-6, "every weight is within half a step of its code");

    const int8_t pc[] = {1, -1, 7, -7, 3};
    uint8_t packed[3];
    int4_pack(pc, 5, packed);
    check(packed[0] == 0xF1 && packed[1] == 0x97 && packed[2] == 0x03,
          "even index in the low nibble, two's complement, odd tail padded with 0");

    printf("VERDICT: %s (%d failed)\n", g_failed ? "FAIL" : "PASS", g_failed);
    return g_failed ? 1 : 0;
}
