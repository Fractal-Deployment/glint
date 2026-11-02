/* weight_diff_map.c — Stage 0a weight map, S0-WMAP-INT8 (glint #11). C, no Python.
 * For every linear in a bf16_to_int8_pin_v1 pin, compares dequant(q, scale) against the original BF16
 * weights and prints one JSON map: per tensor RMSE, max abs error, relative L2, elements past the
 * half-LSB of their block scale, and per-block-scale max/mean amax so the error can be located.
 * Usage: weight_diff_map PIN.safetensors BF16_MODEL_DIR > map.json   Exit 0 ok, 3 setup error. */
#include "safetensors_c.h"
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s PIN.safetensors BF16_MODEL_DIR\n", argv[0]); return 3; }
    SafeTensors pin;
    if (safetensors_open(&pin, argv[1])) { fprintf(stderr, "cannot open pin\n"); return 3; }
    DIR *d = opendir(argv[2]);
    if (!d) { fprintf(stderr, "cannot open model dir\n"); return 3; }
    printf("{\"schema\":\"int8_weight_diff_map_v1\",\"pin\":\"%s\",\"tensors\":[", argv[1]);
    int first = 1, n_found = 0;
    double tot_se = 0, tot_n = 0, tot_ref2 = 0, tot_max = 0;
    struct dirent *de;
    while ((de = readdir(d))) {
        size_t L = strlen(de->d_name);
        if (L < 13 || strcmp(de->d_name + L - 12, ".safetensors")) continue;
        char path[1024]; snprintf(path, sizeof(path), "%s/%s", argv[2], de->d_name);
        SafeTensors src;
        if (safetensors_open(&src, path)) { fprintf(stderr, "cannot open %s\n", path); return 3; }
        for (int i = 0; i < src.n_entries; i++) {
            const TensorEntry *e = &src.entries[i];
            if (strcmp(e->dtype, "BF16") || e->n_dims != 2) continue;
            const TensorEntry *q = safetensors_find(&pin, e->name);
            char sn[320]; snprintf(sn, sizeof(sn), "%s.int8_scale", e->name);
            const TensorEntry *s = safetensors_find(&pin, sn);
            if (!q || !s || strcmp(q->dtype, "I8")) continue;
            uint64_t n = (uint64_t)e->shape[0] * e->shape[1];
            if (q->end - q->begin != n) { fprintf(stderr, "size mismatch %s\n", e->name); return 3; }
            const uint16_t *w = (const uint16_t *)e->data; const int8_t *qq = (const int8_t *)q->data;
            const float *sc = (const float *)s->data;
            double se = 0, ref2 = 0, mx = 0, half = 0, amax_sum = 0, amax_max = 0;
            uint64_t nb = (n + 63) / 64;
            for (uint64_t b = 0; b < nb; b++) {
                double am = (double)sc[b] * 127.0; amax_sum += am; if (am > amax_max) amax_max = am;
                for (uint64_t j = b * 64; j < n && j < (b + 1) * 64; j++) {
                    double r = bf2f(w[j]), dq = (double)qq[j] * sc[b], er = fabs(r - dq);
                    se += er * er; ref2 += r * r; if (er > mx) mx = er;
                    if (er > 0.5 * sc[b] * 1.001) half++;
                }
            }
            if (!first) printf(","); first = 0; n_found++;
            printf("{\"name\":\"%s\",\"shape\":[%lld,%lld],\"rmse\":%.6e,\"max_abs\":%.6e,\"rel_l2\":%.6e,"
                   "\"over_half_lsb\":%.0f,\"block_amax_mean\":%.6e,\"block_amax_max\":%.6e}",
                   e->name, (long long)e->shape[0], (long long)e->shape[1], sqrt(se / n), mx, sqrt(se / (ref2 > 0 ? ref2 : 1)),
                   half, amax_sum / nb, amax_max);
            tot_se += se; tot_n += n; tot_ref2 += ref2; if (mx > tot_max) tot_max = mx;
        }
        safetensors_close(&src);
    }
    closedir(d);
    printf("],\"n_linears\":%d,\"n_elems\":%.0f,\"rmse\":%.6e,\"max_abs\":%.6e,\"rel_l2\":%.6e}\n",
           n_found, tot_n, tot_n > 0 ? sqrt(tot_se / tot_n) : 0, tot_max, sqrt(tot_se / (tot_ref2 > 0 ? tot_ref2 : 1)));
    return n_found > 0 ? 0 : 3;
}
