/* glint_huff.c — M4-LOSSLESS (glint #20): canonical Huffman over the hole and plug nibbles of a glint_pin_v1, per tensor
 * and per stream, with a decode that must reproduce the original bytes exactly. C, no Python.
 * Per stream the compressed size is 16 code lengths (16 bytes, 0 = unused symbol) + the bitstream rounded up to bytes; a
 * stream that would not get smaller is stored raw (flag byte counted). Absmax scales are left as they are.
 * Usage: glint_huff SHARD... > report.json   Exit 0 only if every stream round-trips bit-identically. */
#include "safetensors_c.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAXL 15
typedef struct { uint64_t f; int l, r, sym; } Node;

static void code_lengths(const uint64_t freq[16], int len[16]) {
    Node nd[32]; int n = 0, alive[32], na = 0;
    for (int s = 0; s < 16; s++) { len[s] = 0; if (freq[s]) { nd[n] = (Node){freq[s], -1, -1, s}; alive[na++] = n++; } }
    if (na == 1) { len[nd[alive[0]].sym] = 1; return; }
    while (na > 1) {
        int a = 0, b = 1; if (nd[alive[b]].f < nd[alive[a]].f) { int t = a; a = b; b = t; }
        for (int i = 2; i < na; i++) { if (nd[alive[i]].f < nd[alive[a]].f) { b = a; a = i; } else if (nd[alive[i]].f < nd[alive[b]].f) b = i; }
        nd[n] = (Node){nd[alive[a]].f + nd[alive[b]].f, alive[a], alive[b], -1};
        int ia = alive[a], ib = alive[b]; int w = 0;
        for (int i = 0; i < na; i++) if (alive[i] != ia && alive[i] != ib) alive[w++] = alive[i];
        alive[w++] = n++; na = w;
    }
    int stack[64][2], sp = 0; stack[sp][0] = alive[0]; stack[sp][1] = 0; sp++;
    while (sp) { sp--; int id = stack[sp][0], d = stack[sp][1]; if (nd[id].sym >= 0) len[nd[id].sym] = d; else { stack[sp][0] = nd[id].l; stack[sp][1] = d + 1; sp++; stack[sp][0] = nd[id].r; stack[sp][1] = d + 1; sp++; } }
}

/* returns compressed bytes (incl. 16-byte header + 1 flag), or -1 on a failed round trip */
static long long roundtrip(const uint8_t *packed, uint64_t nsym) {
    uint64_t freq[16] = {0};
    for (uint64_t i = 0; i < nsym; i++) freq[(packed[i >> 1] >> (4 * (i & 1))) & 15]++;
    int len[16]; code_lengths(freq, len);
    int maxl = 0; for (int s = 0; s < 16; s++) if (len[s] > maxl) maxl = len[s];
    if (maxl > MAXL) return (long long)((nsym + 1) / 2) + 1;      /* store raw */
    uint32_t code[16] = {0}; int count[MAXL + 2] = {0}; for (int s = 0; s < 16; s++) if (len[s]) count[len[s]]++;
    uint32_t next[MAXL + 2], c = 0; for (int l = 1; l <= maxl; l++) { c = (c + count[l - 1]) << 1; next[l] = c; if (l == 1) next[1] = 0; }
    { c = 0; for (int l = 1; l <= maxl; l++) { c = (c + (l > 1 ? count[l - 1] : 0)) << 1; next[l] = c; } }
    for (int s = 0; s < 16; s++) if (len[s]) code[s] = next[len[s]]++;
    uint64_t bits = 0; for (int s = 0; s < 16; s++) bits += freq[s] * (uint64_t)len[s];
    uint8_t *bs = (uint8_t *)calloc((bits + 7) / 8 + 8, 1); uint64_t pos = 0;
    for (uint64_t i = 0; i < nsym; i++) { int s = (packed[i >> 1] >> (4 * (i & 1))) & 15; for (int b = len[s] - 1; b >= 0; b--, pos++) if ((code[s] >> b) & 1) bs[pos >> 3] |= (uint8_t)(0x80 >> (pos & 7)); }
    /* decode with a 2^maxl table */
    uint32_t tsz = 1u << maxl; uint8_t *tsym = (uint8_t *)malloc(tsz), *tlen = (uint8_t *)malloc(tsz);
    for (int s = 0; s < 16; s++) if (len[s]) { uint32_t base = code[s] << (maxl - len[s]), span = 1u << (maxl - len[s]); for (uint32_t k = 0; k < span; k++) { tsym[base + k] = (uint8_t)s; tlen[base + k] = (uint8_t)len[s]; } }
    uint64_t rp = 0; long long ok = 1;
    for (uint64_t i = 0; i < nsym && ok; i++) {
        uint32_t w = 0; for (int b = 0; b < maxl; b++) { uint64_t q = rp + b; w = (w << 1) | (q < bits ? (bs[q >> 3] >> (7 - (q & 7))) & 1 : 0); }
        int s = tsym[w]; rp += tlen[w];
        if (s != ((packed[i >> 1] >> (4 * (i & 1))) & 15)) ok = 0;
    }
    free(bs); free(tsym); free(tlen);
    if (!ok || rp != bits) return -1;
    long long comp = 16 + 1 + (long long)((bits + 7) / 8);
    long long raw = 1 + (long long)((nsym + 1) / 2);
    return comp < raw ? comp : raw;
}

int main(int argc, char **argv) {
    unsigned long long raw = 0, comp = 0, nstreams = 0; int fails = 0;
    for (int a = 1; a < argc; a++) {
        SafeTensors st; if (safetensors_open(&st, argv[a])) { fprintf(stderr, "open %s\n", argv[a]); return 3; }
        for (int i = 0; i < st.n_entries; i++) {
            const TensorEntry *e = &st.entries[i]; size_t L = strlen(e->name);
            if (L < 12 || strcmp(e->name + L - 11, ".glint_plug")) continue;
            char hn[300]; memcpy(hn, e->name, L - 11); hn[L - 11] = 0;
            const TensorEntry *h = safetensors_find(&st, hn); if (!h) continue;
            uint64_t nb = e->end - e->begin, nsym = nb * 2;
            for (int w = 0; w < 2; w++) {
                const uint8_t *data = (const uint8_t *)(w ? e->data : h->data);
                long long c = roundtrip(data, nsym); nstreams++;
                if (c < 0) { fails++; fprintf(stderr, "ROUNDTRIP FAIL %s %s\n", hn, w ? "plug" : "hole"); continue; }
                raw += nb; comp += (unsigned long long)c;
            }
        }
        safetensors_close(&st);
    }
    printf("{\"schema\":\"glint_huff_v1\",\"streams\":%llu,\"roundtrip_failures\":%d,\"raw_index_bytes\":%llu,\"compressed_index_bytes\":%llu,\"saved_fraction\":%.5f,\"bit_identical\":%s}\n",
           nstreams, fails, raw, comp, raw ? 1.0 - (double)comp / (double)raw : 0.0, fails ? "false" : "true");
    return fails ? 1 : 0;
}
