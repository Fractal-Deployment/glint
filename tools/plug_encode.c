/* plug_encode.c — Piece E plug format "p7_plug_res4_v1" (glint #19/E). C, no Python.
 * A plug is the residual of the INT8 base against the original BF16, stored at 4 bits:
 *   r = bf16 - q*sb   per 64-block:  sp = max|r| / 7,  p = clamp(rint(r / sp), -7, 7)
 *   corrected weight = q*sb + p*sp
 * Layout per linear: <name>.plug_q4 (U8, ceil(n/2) bytes, element 2i in the low nibble, biased +8)
 *                    <name>.plug_scale (F32, one per 64 weights, same block order as the base).
 * Plug bytes are n/2 + n/16 per linear, i.e. 0.5625 B/elem against the base's 1.0625.
 * Usage: plug_encode PIN.safetensors BF16_MODEL_DIR OUT.safetensors [BITS=4] > error_report.json
 * BITS (2..4) is the M4-LOSSY knob: residual values are clamped to +-(2^(BITS-1)-1) but still stored in a nibble.
 * Reports per-linear RMSE vs BF16 for base alone and base+plug. Exit 0 ok, 3 setup error. */
#include "safetensors_c.h"
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

typedef struct { char name[256]; uint64_t n; uint8_t *q4; float *sp; double rb, rp, ref2; } Plug;

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) { fprintf(stderr, "usage: %s PIN MODEL_DIR OUT [BITS=4]\n", argv[0]); return 3; }
    const int bits = argc == 5 ? atoi(argv[4]) : 4;
    if (bits < 2 || bits > 4) { fprintf(stderr, "BITS must be 2..4\n"); return 3; }
    const int qmax = (1 << (bits - 1)) - 1;
    SafeTensors pin;
    if (safetensors_open(&pin, argv[1])) { fprintf(stderr, "cannot open pin\n"); return 3; }
    DIR *d = opendir(argv[2]);
    if (!d) return 3;
    Plug *pl = (Plug *)calloc(256, sizeof(Plug)); int np = 0;
    struct dirent *de;
    while ((de = readdir(d))) {
        size_t L = strlen(de->d_name);
        if (L < 13 || strcmp(de->d_name + L - 12, ".safetensors")) continue;
        char path[1024]; snprintf(path, sizeof(path), "%s/%s", argv[2], de->d_name);
        SafeTensors src; if (safetensors_open(&src, path)) return 3;
        for (int i = 0; i < src.n_entries; i++) {
            const TensorEntry *e = &src.entries[i];
            if (strcmp(e->dtype, "BF16") || e->n_dims != 2) continue;
            char sn[320]; snprintf(sn, sizeof(sn), "%s.int8_scale", e->name);
            const TensorEntry *q = safetensors_find(&pin, e->name), *s = safetensors_find(&pin, sn);
            if (!q || !s || strcmp(q->dtype, "I8")) continue;
            Plug *p = &pl[np++]; strncpy(p->name, e->name, 255);
            uint64_t n = p->n = (uint64_t)e->shape[0] * e->shape[1], nb = (n + 63) / 64;
            p->q4 = (uint8_t *)calloc((n + 1) / 2, 1); p->sp = (float *)calloc(nb, 4);
            const uint16_t *w = (const uint16_t *)e->data; const int8_t *qq = (const int8_t *)q->data; const float *sb = (const float *)s->data;
            for (uint64_t b = 0; b < nb; b++) {
                double am = 0;
                for (uint64_t j = b * 64; j < n && j < (b + 1) * 64; j++) { double r = fabs(bf2f(w[j]) - (double)qq[j] * sb[b]); if (r > am) am = r; }
                float sp = am > 0 ? (float)(am / (double)qmax) : 1.f; p->sp[b] = sp;
                for (uint64_t j = b * 64; j < n && j < (b + 1) * 64; j++) {
                    double r = bf2f(w[j]) - (double)qq[j] * sb[b];
                    int v = (int)rint(r / sp); v = v > qmax ? qmax : (v < -qmax ? -qmax : v);
                    p->q4[j >> 1] |= (uint8_t)((v + 8) << ((j & 1) * 4));
                    double e0 = r, e1 = r - (double)v * sp;
                    p->rb += e0 * e0; p->rp += e1 * e1; p->ref2 += (double)bf2f(w[j]) * bf2f(w[j]);
                }
            }
        }
        /* src stays mapped: q4/sp were copied; BF16 pointers are not retained */
        safetensors_close(&src);
    }
    closedir(d);
    /* safetensors writer */
    FILE *f = fopen(argv[3], "wb"); if (!f) return 3;
    size_t cap = (size_t)np * 700 + 256; char *h = (char *)malloc(cap); size_t o = 0; uint64_t off = 0;
    o += snprintf(h + o, cap - o, "{\"__metadata__\":{\"format\":\"p7_plug_res4_v1\"}");
    for (int i = 0; i < np; i++) {
        uint64_t qb = (pl[i].n + 1) / 2, sbyt = ((pl[i].n + 63) / 64) * 4;
        o += snprintf(h + o, cap - o, ",\"%s.plug_q4\":{\"dtype\":\"U8\",\"shape\":[%llu],\"data_offsets\":[%llu,%llu]}", pl[i].name,
                      (unsigned long long)qb, (unsigned long long)off, (unsigned long long)(off + qb)); off += qb;
        o += snprintf(h + o, cap - o, ",\"%s.plug_scale\":{\"dtype\":\"F32\",\"shape\":[%llu],\"data_offsets\":[%llu,%llu]}", pl[i].name,
                      (unsigned long long)(sbyt / 4), (unsigned long long)off, (unsigned long long)(off + sbyt)); off += sbyt;
    }
    o += snprintf(h + o, cap - o, "}");
    while (o % 8) h[o++] = ' ';
    uint64_t hl = o; fwrite(&hl, 8, 1, f); fwrite(h, 1, o, f);
    for (int i = 0; i < np; i++) { fwrite(pl[i].q4, 1, (pl[i].n + 1) / 2, f); fwrite(pl[i].sp, 4, (pl[i].n + 63) / 64, f); }
    fclose(f);
    double tb = 0, tp = 0, tn = 0, tr = 0, plug_bytes = 0;
    printf("{\"schema\":\"p7_plug_res4_error_v1\",\"linears\":[");
    for (int i = 0; i < np; i++) {
        printf("%s{\"name\":\"%s\",\"rmse_base\":%.6e,\"rmse_base_plus_plug\":%.6e,\"rel_l2_base\":%.6e,\"rel_l2_base_plus_plug\":%.6e}", i ? "," : "",
               pl[i].name, sqrt(pl[i].rb / pl[i].n), sqrt(pl[i].rp / pl[i].n), sqrt(pl[i].rb / pl[i].ref2), sqrt(pl[i].rp / pl[i].ref2));
        tb += pl[i].rb; tp += pl[i].rp; tn += pl[i].n; tr += pl[i].ref2; plug_bytes += (pl[i].n + 1) / 2 + ((pl[i].n + 63) / 64) * 4.0;
    }
    printf("],\"n_linears\":%d,\"rmse_base\":%.6e,\"rmse_base_plus_plug\":%.6e,\"rel_l2_base\":%.6e,\"rel_l2_base_plus_plug\":%.6e,\"plug_bytes\":%.0f,\"base_bytes\":%.0f}\n",
           np, sqrt(tb / tn), sqrt(tp / tn), sqrt(tb / tr), sqrt(tp / tr), plug_bytes, tn + tn / 16.0 * 1.0);
    return np > 0 ? 0 : 3;
}
