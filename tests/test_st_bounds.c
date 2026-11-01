/* safetensors_c.h must grow past any fixed tensor count and refuse files whose
 * header or tensor byte ranges fall outside the file.
 * gcc -std=c11 -g -fsanitize=address,undefined -Iinclude tests/test_st_bounds.c -o test_st_bounds
 */
#define _DEFAULT_SOURCE
#include "safetensors_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char g_dir[512];
static int g_failed = 0;

static void check(int ok, const char *what) {
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) g_failed++;
}

static const char *write_file(const char *name, const void *bytes, size_t n) {
    static char path[1024];
    snprintf(path, sizeof(path), "%s/%s", g_dir, name);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(bytes, 1, n, f) != n) {
        perror(path);
        exit(2);
    }
    fclose(f);
    return path;
}

static size_t build(uint8_t *out, uint64_t header_len_field, const char *json, const void *data, size_t data_len) {
    size_t jl = strlen(json);
    memcpy(out, &header_len_field, 8);
    memcpy(out + 8, json, jl);
    memcpy(out + 8 + jl, data, data_len);
    return 8 + jl + data_len;
}

static int open_rc(const char *name, const void *bytes, size_t n) {
    SafeTensors st;
    int rc = safetensors_open(&st, write_file(name, bytes, n));
    safetensors_close(&st);
    return rc;
}

static void test_valid_file(void) {
    const char *json = "{\"__metadata__\":{\"format\":\"pt\"},"
                       "\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},"
                       "\"b\":{\"dtype\":\"BF16\",\"shape\":[1,2],\"data_offsets\":[8,12]}}";
    const uint8_t data[12] = {0, 0, 128, 63, 0, 0, 0, 64, 1, 2, 3, 4};
    uint8_t buf[512];
    size_t n = build(buf, strlen(json), json, data, sizeof(data));
    SafeTensors st;
    int rc = safetensors_open(&st, write_file("valid.safetensors", buf, n));
    check(rc == 0 && st.n_entries == 2, "valid file opens with 2 tensors");
    const TensorEntry *b = rc == 0 ? safetensors_find(&st, "b") : NULL;
    check(b && b->n_dims == 2 && b->shape[1] == 2 && memcmp(b->data, data + 8, 4) == 0,
          "tensor b points at its 4 data bytes");
    safetensors_close(&st);
}

static void test_many_tensors(void) {
    const int count = 3000;
    size_t cap = (size_t)count * 96 + 64;
    char *json = (char *)malloc(cap);
    size_t o = (size_t)snprintf(json, cap, "{");
    for (int i = 0; i < count; ++i)
        o += (size_t)snprintf(json + o, cap - o, "%s\"t%d\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[%d,%d]}",
                              i ? "," : "", i, i, i + 1);
    o += (size_t)snprintf(json + o, cap - o, "}");
    uint8_t *data = (uint8_t *)malloc((size_t)count);
    for (int i = 0; i < count; ++i) data[i] = (uint8_t)i;
    uint8_t *buf = (uint8_t *)malloc(8 + o + (size_t)count);
    size_t n = build(buf, o, json, data, (size_t)count);
    SafeTensors st;
    int rc = safetensors_open(&st, write_file("many.safetensors", buf, n));
    check(rc == 0 && st.n_entries == count, "3000 tensors open past the old 2048 cap");
    const TensorEntry *last = rc == 0 ? safetensors_find(&st, "t2999") : NULL;
    check(last && *(const uint8_t *)last->data == (uint8_t)2999, "last of 3000 tensors reads its byte");
    safetensors_close(&st);
    free(json);
    free(data);
    free(buf);
}

static void test_rejects(void) {
    const char *json = "{\"w\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":[0,16]}}";
    uint8_t data[16];
    memset(data, 7, sizeof(data));
    uint8_t buf[512];

    size_t n = build(buf, strlen(json), json, data, sizeof(data));
    check(open_rc("truncated.safetensors", buf, n - 5) < 0, "truncated data region is rejected");

    n = build(buf, (uint64_t)1 << 40, json, data, sizeof(data));
    check(open_rc("oversized_header.safetensors", buf, n) < 0, "header_len past end of file is rejected");

    n = build(buf, strlen(json) + sizeof(data) + 1, json, data, sizeof(data));
    check(open_rc("header_one_past.safetensors", buf, n) < 0, "header_len one byte past end of file is rejected");

    check(open_rc("short.safetensors", buf, 5) < 0, "file shorter than the 8-byte length prefix is rejected");

    const char *inverted = "{\"w\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":[12,4]}}";
    n = build(buf, strlen(inverted), inverted, data, sizeof(data));
    check(open_rc("inverted.safetensors", buf, n) < 0, "begin > end is rejected");

    const char *wrap = "{\"w\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":[0,18446744073709551632]}}";
    n = build(buf, strlen(wrap), wrap, data, sizeof(data));
    check(open_rc("wrap.safetensors", buf, n) < 0, "offset that overflows uint64 is rejected");

    const char *unterminated = "{\"w\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":[0,16]";
    uint8_t tail[16];
    memset(tail, 'x', sizeof(tail));
    n = build(buf, strlen(unterminated), unterminated, tail, sizeof(tail));
    check(open_rc("unterminated.safetensors", buf, n) < 0, "header that ends mid-object is rejected");
}

int main(void) {
    const char *tmp = getenv("TMPDIR");
    snprintf(g_dir, sizeof(g_dir), "%s/st_bounds_XXXXXX", tmp && *tmp ? tmp : "/tmp");
    if (!mkdtemp(g_dir)) {
        perror("mkdtemp");
        return 2;
    }
    test_valid_file();
    test_many_tensors();
    test_rejects();
    printf("%s: %d failed\n", g_failed ? "FAIL" : "PASS", g_failed);
    return g_failed ? 1 : 0;
}
