/* test_st_bounds.c — safetensors_c.h hardening: no 2048-tensor cap, and malformed
 * files are rejected (with the file closed) instead of read out of bounds.
 * Optional argv[1]: a real .safetensors file that must open and have every tensor in range.
 * Build: gcc -O1 -fsanitize=address,undefined -Iinclude tests/test_st_bounds.c -o test_st_bounds
 */
#include "safetensors_c.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

static const char *TMP = "/tmp/glint_st_bounds.safetensors";

/* Write [u64 header_len][header][data of data_bytes zero bytes]. header_len_field overrides the true length. */
static void write_file(const char *header, uint64_t header_len_field, size_t data_bytes) {
    FILE *f = fopen(TMP, "wb");
    assert(f);
    uint64_t hl = header_len_field;
    fwrite(&hl, 8, 1, f);
    fwrite(header, 1, strlen(header), f);
    for (size_t i = 0; i < data_bytes; i++) fputc(0, f);
    fclose(f);
}

static int open_case(const char *header, size_t data_bytes) {
    write_file(header, strlen(header), data_bytes);
    SafeTensors st;
    int rc = safetensors_open(&st, TMP);
    if (rc == 0) safetensors_close(&st);
    else CHECK(st.mmap_ptr == NULL && st.entries == NULL, "failed open must leave st zeroed (rc=%d)", rc);
    return rc;
}

int main(int argc, char **argv) {
    printf("=== TEST SAFETENSORS BOUNDS ===\n");

    /* valid, one tensor */
    CHECK(open_case("{\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}", 16) == 0, "valid file rejected");

    /* more tensors than the old 2048 cap */
    {
        const int n = 5000;
        size_t cap = (size_t)n * 90 + 16;
        char *h = (char *)malloc(cap), *w = h;
        w += sprintf(w, "{");
        for (int i = 0; i < n; i++)
            w += sprintf(w, "%s\"t%d\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[%d,%d]}", i ? "," : "", i, i, i + 1);
        w += sprintf(w, "}");
        write_file(h, strlen(h), (size_t)n);
        SafeTensors st;
        int rc = safetensors_open(&st, TMP);
        CHECK(rc == 0, "5000-tensor file failed rc=%d", rc);
        if (rc == 0) {
            CHECK(st.n_entries == n, "n_entries=%d want %d", st.n_entries, n);
            const TensorEntry *e = safetensors_find(&st, "t4999");
            CHECK(e && e->begin == 4999, "last tensor missing");
            safetensors_close(&st);
        }
        free(h);
    }

    /* offsets past end of file */
    CHECK(open_case("{\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}", 8) == -5, "end past EOF accepted");
    /* begin > end */
    CHECK(open_case("{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[8,4]}}", 16) == -5, "begin>end accepted");
    /* byte count disagrees with dtype*shape */
    CHECK(open_case("{\"a\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,4]}}", 16) == -5, "size mismatch accepted");
    /* shape overflow */
    CHECK(open_case("{\"a\":{\"dtype\":\"F32\",\"shape\":[4294967296,4294967296],\"data_offsets\":[0,0]}}", 16) == -5, "shape overflow accepted");
    /* header_len larger than file */
    write_file("{}", 1ull << 40, 0);
    { SafeTensors st; CHECK(safetensors_open(&st, TMP) == -3, "huge header_len accepted"); }
    /* file shorter than 8 bytes */
    { FILE *f = fopen(TMP, "wb"); fwrite("abc", 1, 3, f); fclose(f);
      SafeTensors st; CHECK(safetensors_open(&st, TMP) == -3, "3-byte file accepted"); }
    /* truncated JSON */
    CHECK(open_case("{\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]", 16) == -4, "truncated JSON accepted");
    /* missing offsets */
    CHECK(open_case("{\"a\":{\"dtype\":\"F32\",\"shape\":[2,2]}}", 16) == -4, "missing data_offsets accepted");
    /* header not an object */
    CHECK(open_case("[1,2]", 0) == -3, "non-object header accepted");
    /* nonexistent file */
    { SafeTensors st; CHECK(safetensors_open(&st, "/nonexistent/x.safetensors") == -1, "missing file not -1"); }

    /* real weights: every tensor must lie inside the file and match dtype*shape */
    if (argc > 1) {
        SafeTensors st;
        int rc = safetensors_open(&st, argv[1]);
        CHECK(rc == 0, "real file %s rc=%d", argv[1], rc);
        if (rc == 0) {
            printf("real file: %d tensors, %zu bytes\n", st.n_entries, st.file_size);
            CHECK(st.n_entries > 0, "no tensors in real file");
            safetensors_close(&st);
        }
    }

    remove(TMP);
    if (failures) { printf("SAFETENSORS BOUNDS: %d FAILURES\n", failures); return 1; }
    printf("SAFETENSORS BOUNDS PASSED.\n");
    return 0;
}
