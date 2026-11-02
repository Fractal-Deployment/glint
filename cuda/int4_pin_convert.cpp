/* Offline BF16 -> bf16_to_int4_pin_v1 converter for the Phi-4-mini linears (include/int4_pin.h has the format).
 * Every other tensor of the two BF16 shards is copied unchanged. Writes OUT_DIR/model.safetensors and
 * OUT_DIR/pin.json, then reads the written file back and checks every linear against the source.
 * Usage: int4_pin_convert MODEL_DIR OUT_DIR
 * Exit: 0 written and checked, 1 check failed, 2 usage, 3 setup or I/O error. */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <algorithm>

#include "../include/int4_pin.h"
#include "../include/safetensors_c.h"

static void fail(int code, const char *what, const char *arg) {
    printf("ERROR: %s %s\n", what, arg);
    exit(code);
}

static bool is_linear(const char *name) {
    static const char *suffix[] = {".self_attn.qkv_proj.weight", ".self_attn.o_proj.weight",
                                   ".mlp.gate_up_proj.weight", ".mlp.down_proj.weight"};
    if (strncmp(name, "model.layers.", 13) != 0) return false;
    size_t n = strlen(name);
    for (const char *s : suffix)
        if (n > strlen(s) && strcmp(name + n - strlen(s), s) == 0) return true;
    return false;
}

struct Out {
    std::string name, dtype;
    std::vector<int64_t> shape;
    const TensorEntry *src;
    bool scale;
    uint64_t begin, end;
};

struct Stats {
    unsigned long long clamped = 0, n_weights = 0, n_linears = 0;
    double max_err_over_scale = 0.0, sum_sq_err = 0.0;
};

static void quantize(const TensorEntry *e, std::vector<uint8_t> &packed, std::vector<float> &scale, Stats *st) {
    size_t numel = (size_t)e->shape[0] * e->shape[1];
    const uint16_t *w = (const uint16_t *)e->data;
    packed.assign(numel / 2, 0);
    scale.assign(numel / INT4_PIN_BLOCK, 0.0f);
    int8_t codes[INT4_PIN_BLOCK];
    for (size_t b = 0; b < scale.size(); b++) {
        const uint16_t *wb = w + b * INT4_PIN_BLOCK;
        scale[b] = int4_quantize_block(wb, INT4_PIN_BLOCK, codes, &st->clamped);
        int4_pack(codes, INT4_PIN_BLOCK, &packed[b * INT4_PIN_BLOCK / 2]);
    }
}

static size_t check_linear(const TensorEntry *src, const TensorEntry *q, const TensorEntry *s, Stats *st) {
    size_t numel = (size_t)src->shape[0] * src->shape[1], bad = 0;
    const uint16_t *w = (const uint16_t *)src->data;
    const uint8_t *p = (const uint8_t *)q->data;
    const float *sc = (const float *)s->data;
    for (size_t i = 0; i < numel; i++) {
        int nib = (p[i / 2] >> (i % 2 ? 4 : 0)) & 15, code = nib >= 8 ? nib - 16 : nib;
        double sv = sc[i / INT4_PIN_BLOCK], err = fabs((double)int4_bf16_to_float(w[i]) - code * sv);
        double rel = err / sv;
        if (code == -8 || !(rel <= 0.5 + 1e-6)) bad++;
        st->max_err_over_scale = fmax(st->max_err_over_scale, rel);
        st->sum_sq_err += err * err;
    }
    st->n_weights += numel;
    st->n_linears++;
    return bad;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("usage: %s MODEL_DIR OUT_DIR\n", argv[0]);
        return 2;
    }
    std::string shard[2] = {std::string(argv[1]) + "/model-00001-of-00002.safetensors",
                            std::string(argv[1]) + "/model-00002-of-00002.safetensors"};
    std::string out_path = std::string(argv[2]) + "/model.safetensors";
    SafeTensors st[2];
    for (int i = 0; i < 2; i++)
        if (safetensors_open(&st[i], shard[i].c_str()) != 0) fail(3, "cannot open", shard[i].c_str());

    std::vector<Out> outs;
    for (int i = 0; i < 2; i++)
        for (int t = 0; t < st[i].n_entries; t++) {
            const TensorEntry *e = &st[i].entries[t];
            if (!is_linear(e->name)) {
                outs.push_back({e->name, e->dtype, std::vector<int64_t>(e->shape, e->shape + e->n_dims), e, false, 0,
                                e->end - e->begin});
                continue;
            }
            if (strcmp(e->dtype, "BF16") != 0 || e->n_dims != 2 || e->shape[1] % INT4_PIN_BLOCK != 0)
                fail(3, "linear is not BF16 [n, k] with k % 64 == 0:", e->name);
            uint64_t numel = (uint64_t)e->shape[0] * e->shape[1];
            outs.push_back({e->name, "U8", {e->shape[0], e->shape[1] / 2}, e, false, 0, numel / 2});
            outs.push_back({std::string(e->name) + ".int4_scale", "F32", {(int64_t)(numel / INT4_PIN_BLOCK)}, e, true,
                            0, numel / INT4_PIN_BLOCK * 4});
        }
    std::sort(outs.begin(), outs.end(), [](const Out &a, const Out &b) { return a.name < b.name; });

    char meta[512];
    snprintf(meta, sizeof(meta),
             "{\"format\":\"pt\",\"schema\":\"%s\",\"quantization\":\"int4_symmetric_per_block_zp0\","
             "\"int4_blocksize\":\"%d\",\"int4_codes\":\"[-7,7]\",\"converted_from\":\"bf16\"}",
             INT4_PIN_SCHEMA, INT4_PIN_BLOCK);
    std::string header = std::string("{\"__metadata__\":") + meta;
    uint64_t off = 0;
    for (Out &o : outs) {
        uint64_t bytes = o.end;
        o.begin = off;
        o.end = off + bytes;
        off = o.end;
        header += ",\"" + o.name + "\":{\"dtype\":\"" + o.dtype + "\",\"shape\":[";
        for (size_t d = 0; d < o.shape.size(); d++) header += (d ? "," : "") + std::to_string(o.shape[d]);
        header += "],\"data_offsets\":[" + std::to_string(o.begin) + "," + std::to_string(o.end) + "]}";
    }
    header += "}";
    while (header.size() % 8) header += ' ';

    FILE *f = fopen(out_path.c_str(), "wb");
    if (!f) fail(3, "cannot write", out_path.c_str());
    uint64_t hlen = header.size();
    if (fwrite(&hlen, 8, 1, f) != 1 || fwrite(header.data(), 1, hlen, f) != hlen) fail(3, "write failed", out_path.c_str());
    Stats stats;
    std::vector<uint8_t> packed;
    std::vector<float> scale;
    const TensorEntry *quantized = NULL;
    for (const Out &o : outs) {
        const void *data = o.src->data;
        uint64_t bytes = o.end - o.begin;
        if (o.dtype == "U8" || o.scale) {
            if (quantized != o.src) {
                quantize(o.src, packed, scale, &stats);
                quantized = o.src;
            }
            data = o.scale ? (const void *)scale.data() : (const void *)packed.data();
        }
        if (fwrite(data, 1, bytes, f) != bytes) fail(3, "write failed", out_path.c_str());
    }
    if (fclose(f) != 0) fail(3, "close failed", out_path.c_str());

    SafeTensors pin;
    if (safetensors_open(&pin, out_path.c_str()) != 0) fail(3, "cannot reopen", out_path.c_str());
    size_t bad = 0, copied_bad = 0;
    for (const Out &o : outs) {
        const TensorEntry *e = safetensors_find(&pin, o.name.c_str());
        if (!e || e->end - e->begin != o.end - o.begin) fail(1, "missing or wrong size in output:", o.name.c_str());
        if (o.scale) continue;
        if (o.dtype == "U8") {
            const TensorEntry *s = safetensors_find(&pin, (o.name + ".int4_scale").c_str());
            bad += check_linear(o.src, e, s, &stats);
        } else if (memcmp(e->data, o.src->data, o.end - o.begin) != 0) {
            copied_bad++;
        }
    }
    double rmse = sqrt(stats.sum_sq_err / (double)stats.n_weights);
    uint64_t linear_bytes = stats.n_weights / 2 + stats.n_weights / INT4_PIN_BLOCK * 4;
    printf("%llu linears, %llu weights, %llu bytes of codes and scales (%.4f bits/weight), %zu tensors copied\n",
           stats.n_linears, stats.n_weights, (unsigned long long)linear_bytes,
           8.0 * linear_bytes / (double)stats.n_weights, outs.size() - (size_t)(2 * stats.n_linears));
    printf("read back: max |w - code * scale| / scale %.6f, rmse %.6e, clamped %llu, out-of-bound %zu, copy "
           "mismatches %zu\n",
           stats.max_err_over_scale, rmse, stats.clamped, bad, copied_bad);

    std::string json_path = std::string(argv[2]) + "/pin.json";
    FILE *j = fopen(json_path.c_str(), "w");
    if (!j) fail(3, "cannot write", json_path.c_str());
    fprintf(j,
            "{\n  \"schema\": \"%s\",\n  \"src\": \"%s\",\n  \"dst\": \"%s\",\n  \"int4_blocksize\": %d,\n"
            "  \"int4_scheme\": \"symmetric_per_block_zp0\",\n  \"codes\": \"[-7, 7], -8 unused\",\n"
            "  \"scale\": \"absmax / 7 in f64, stored F32; 1.0 for an all-zero block\",\n"
            "  \"rounding\": \"round half to even (rint)\",\n  \"n_linears\": %llu,\n  \"n_weights\": %llu,\n"
            "  \"linear_bytes\": %llu,\n  \"bits_per_weight\": %.4f,\n  \"max_err_over_scale\": %.6f,\n"
            "  \"rmse_vs_bf16\": %.6e,\n  \"clamped\": %llu\n}\n",
            INT4_PIN_SCHEMA, argv[1], out_path.c_str(), INT4_PIN_BLOCK, stats.n_linears, stats.n_weights,
            (unsigned long long)linear_bytes, 8.0 * linear_bytes / (double)stats.n_weights, stats.max_err_over_scale,
            rmse, stats.clamped);
    fclose(j);
    bool ok = bad == 0 && copied_bad == 0 && stats.clamped == 0 && stats.n_linears == 128;
    printf("VERDICT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
