/* glint_entropy.c — M4-LOSSLESS size bound (glint #20). For every linear in a glint_pin_v1 shard set, the
 * order-0 Shannon entropy of the hole nibbles, the plug nibbles, and the plug conditioned on the hole
 * (joint 256-symbol entropy), in bits per weight. A lossless entropy coder cannot beat the joint entropy
 * at order 0; this is the byte bound, not a codec. Usage: glint_entropy SHARD... > report.json */
#include "safetensors_c.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
static double H(const double *c, int n, double tot) { double h = 0; for (int i = 0; i < n; i++) if (c[i] > 0) h -= c[i] / tot * log2(c[i] / tot); return h; }
int main(int argc, char **argv) {
    double T_n = 0, T_h = 0, T_p = 0, T_j = 0; int first = 1;
    printf("{\"schema\":\"glint_entropy_v1\",\"tensors\":[");
    for (int a = 1; a < argc; a++) {
        SafeTensors st; if (safetensors_open(&st, argv[a])) { fprintf(stderr, "open %s\n", argv[a]); return 3; }
        for (int i = 0; i < st.n_entries; i++) {
            const TensorEntry *e = &st.entries[i];
            size_t L = strlen(e->name); if (L < 12 || strcmp(e->name + L - 11, ".glint_plug")) continue;
            char hn[300]; memcpy(hn, e->name, L - 11); hn[L - 11] = 0;
            const TensorEntry *h = safetensors_find(&st, hn); if (!h) continue;
            uint64_t nb = e->end - e->begin, n = nb * 2;
            double ch[16] = {0}, cp[16] = {0}, cj[256] = {0};
            const uint8_t *ph = (const uint8_t *)h->data, *pp = (const uint8_t *)e->data;
            for (uint64_t j = 0; j < nb; j++) for (int s = 0; s < 2; s++) {
                int hv = (ph[j] >> (4 * s)) & 15, pv = (pp[j] >> (4 * s)) & 15;
                ch[hv]++; cp[pv]++; cj[hv * 16 + pv]++;
            }
            double hh = H(ch, 16, n), hp = H(cp, 16, n), hj = H(cj, 256, n);
            printf("%s{\"name\":\"%s\",\"n\":%llu,\"h_hole\":%.4f,\"h_plug\":%.4f,\"h_joint\":%.4f}", first ? "" : ",", hn, (unsigned long long)n, hh, hp, hj);
            first = 0; T_n += n; T_h += hh * n; T_p += hp * n; T_j += hj * n;
        }
        safetensors_close(&st);
    }
    printf("],\"bits_per_weight\":{\"raw\":8,\"hole\":%.4f,\"plug\":%.4f,\"joint\":%.4f},\"n_elems\":%.0f,\"note\":\"order-0 bound per tensor, absmax scales (0.5 bit/weight at F32 per 64) not included\"}\n", T_h / T_n, T_p / T_n, T_j / T_n, T_n);
    return 0;
}
