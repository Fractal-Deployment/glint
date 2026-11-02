#ifndef P7_SAFETENSORS_C_H
#define P7_SAFETENSORS_C_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[256];
    char dtype[16];
    int64_t shape[4];
    int n_dims;
    uint64_t begin;
    uint64_t end;
    const void *data;
} TensorEntry;

typedef struct {
    void *mmap_ptr;
    size_t file_size;
    uint64_t header_len;
    TensorEntry *entries;
    int n_entries;
    int capacity;
} SafeTensors;

static inline const char* _st_skip_ws(const char* p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static inline const char* _st_parse_string(const char* p, char* out, size_t out_sz) {
    p = _st_skip_ws(p);
    if (*p != '"') return NULL;
    p++;
    size_t i = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) p++;
        if (i + 1 < out_sz) out[i++] = *p;
        p++;
    }
    if (*p != '"') return NULL;
    out[i] = '\0';
    return p + 1;
}

/* Bytes per element for the dtypes whose size is fixed; 0 when unknown. */
static inline int _st_dtype_bytes(const char *dtype) {
    if (!strcmp(dtype, "F64") || !strcmp(dtype, "I64") || !strcmp(dtype, "U64")) return 8;
    if (!strcmp(dtype, "F32") || !strcmp(dtype, "I32") || !strcmp(dtype, "U32")) return 4;
    if (!strcmp(dtype, "F16") || !strcmp(dtype, "BF16") || !strcmp(dtype, "I16") || !strcmp(dtype, "U16")) return 2;
    if (!strcmp(dtype, "I8") || !strcmp(dtype, "U8") || !strcmp(dtype, "BOOL")) return 1;
    return 0;
}

/* Returns 0 on success; on any failure the SafeTensors is closed and zeroed.
 *   -1 open/fstat failed        -2 mmap failed
 *   -3 header length invalid (file < 8 bytes, or header_len > file - 8) or header is not an object
 *   -4 malformed header JSON    -5 tensor offsets/size inconsistent with file or dtype/shape
 *   -6 out of memory */
static inline void safetensors_close(SafeTensors *st);

static inline int safetensors_open(SafeTensors *st, const char *path) {
    memset(st, 0, sizeof(*st));
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat sb;
    if (fstat(fd, &sb) < 0) { close(fd); return -1; }
    st->file_size = (size_t)sb.st_size;
    if (st->file_size < 8) { close(fd); memset(st, 0, sizeof(*st)); return -3; }
    st->mmap_ptr = mmap(NULL, st->file_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (st->mmap_ptr == MAP_FAILED) { memset(st, 0, sizeof(*st)); return -2; }

    memcpy(&st->header_len, st->mmap_ptr, sizeof(uint64_t));
    if (st->header_len > (uint64_t)st->file_size - 8) { safetensors_close(st); return -3; }
    const uint64_t data_len = (uint64_t)st->file_size - 8 - st->header_len;
    const uint8_t *data_base = (const uint8_t*)st->mmap_ptr + 8 + st->header_len;

    /* The header is not NUL-terminated inside the mapping (tensor data follows it). */
    char *header_json = (char*)malloc((size_t)st->header_len + 1);
    if (!header_json) { safetensors_close(st); return -6; }
    memcpy(header_json, (const char*)st->mmap_ptr + 8, (size_t)st->header_len);
    header_json[st->header_len] = '\0';

    st->capacity = 256;
    st->entries = (TensorEntry*)calloc(st->capacity, sizeof(TensorEntry));
    if (!st->entries) { free(header_json); safetensors_close(st); return -6; }
    st->n_entries = 0;

    int rc = 0;
    const char *p = _st_skip_ws(header_json);
    if (*p != '{') { free(header_json); safetensors_close(st); return -3; }
    p++;

#define ST_BAD(code) do { rc = (code); goto done; } while (0)
    while (*p && *p != '}') {
        p = _st_skip_ws(p);
        if (*p == '}') break;
        char key[256];
        p = _st_parse_string(p, key, sizeof(key));
        if (!p) ST_BAD(-4);
        p = _st_skip_ws(p);
        if (*p != ':') ST_BAD(-4);
        p++;
        p = _st_skip_ws(p);

        if (strcmp(key, "__metadata__") == 0) {
            int depth = 0;
            while (*p) {
                if (*p == '{') depth++;
                else if (*p == '}') { depth--; if (depth == 0) { p++; break; } }
                p++;
            }
            if (depth != 0) ST_BAD(-4);
        } else {
            if (*p != '{') ST_BAD(-4);
            p++;
            TensorEntry e;
            memset(&e, 0, sizeof(e));
            strncpy(e.name, key, sizeof(e.name) - 1);
            int have_offsets = 0, have_dtype = 0, have_shape = 0;

            while (*p && *p != '}') {
                p = _st_skip_ws(p);
                if (*p == '}') break;
                char field[64];
                p = _st_parse_string(p, field, sizeof(field));
                if (!p) ST_BAD(-4);
                p = _st_skip_ws(p);
                if (*p != ':') ST_BAD(-4);
                p++;
                p = _st_skip_ws(p);

                if (strcmp(field, "dtype") == 0) {
                    p = _st_parse_string(p, e.dtype, sizeof(e.dtype));
                    if (!p) ST_BAD(-4);
                    have_dtype = 1;
                } else if (strcmp(field, "shape") == 0) {
                    if (*p != '[') ST_BAD(-4);
                    p++;
                    while (*p && *p != ']') {
                        p = _st_skip_ws(p);
                        if (*p == ']') break;
                        if (*p < '0' || *p > '9') ST_BAD(-4);
                        int64_t v = 0;
                        while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
                        if (e.n_dims >= 4) ST_BAD(-5); /* TensorEntry holds at most 4 dims */
                        e.shape[e.n_dims++] = v;
                        p = _st_skip_ws(p);
                        if (*p == ',') p++;
                    }
                    if (*p != ']') ST_BAD(-4);
                    p++;
                    have_shape = 1;
                } else if (strcmp(field, "data_offsets") == 0) {
                    if (*p != '[') ST_BAD(-4);
                    p++;
                    p = _st_skip_ws(p);
                    uint64_t b = 0, en = 0;
                    if (*p < '0' || *p > '9') ST_BAD(-4);
                    while (*p >= '0' && *p <= '9') { b = b * 10 + (*p - '0'); p++; }
                    p = _st_skip_ws(p);
                    if (*p != ',') ST_BAD(-4);
                    p++;
                    p = _st_skip_ws(p);
                    if (*p < '0' || *p > '9') ST_BAD(-4);
                    while (*p >= '0' && *p <= '9') { en = en * 10 + (*p - '0'); p++; }
                    p = _st_skip_ws(p);
                    if (*p != ']') ST_BAD(-4);
                    p++;
                    e.begin = b;
                    e.end = en;
                    have_offsets = 1;
                } else {
                    int depth = 0;
                    while (*p) {
                        if (*p == '{' || *p == '[') depth++;
                        else if (*p == '}' || *p == ']') { if (depth == 0) break; depth--; }
                        else if (*p == ',' && depth == 0) break;
                        p++;
                    }
                }
                p = _st_skip_ws(p);
                if (*p == ',') p++;
            }
            if (*p != '}') ST_BAD(-4);
            p++;
            if (!have_offsets || !have_dtype || !have_shape) ST_BAD(-4);

            /* Offsets are relative to the data section and must lie inside it. */
            if (e.begin > e.end || e.end > data_len) ST_BAD(-5);
            int eb = _st_dtype_bytes(e.dtype);
            if (eb) {
                uint64_t n = 1;
                for (int d = 0; d < e.n_dims; d++) {
                    if (e.shape[d] < 0) ST_BAD(-5);
                    if (e.shape[d] != 0 && n > UINT64_MAX / (uint64_t)e.shape[d]) ST_BAD(-5);
                    n *= (uint64_t)e.shape[d];
                }
                if (n > UINT64_MAX / (uint64_t)eb || n * (uint64_t)eb != e.end - e.begin) ST_BAD(-5);
            }
            e.data = data_base + e.begin;

            if (st->n_entries == st->capacity) {
                int new_cap = st->capacity * 2;
                TensorEntry *grown = (TensorEntry*)realloc(st->entries, (size_t)new_cap * sizeof(TensorEntry));
                if (!grown) ST_BAD(-6);
                st->entries = grown;
                st->capacity = new_cap;
            }
            st->entries[st->n_entries++] = e;
        }
        p = _st_skip_ws(p);
        if (*p == ',') p++;
    }
    if (*p != '}') ST_BAD(-4); /* unterminated top-level object */
done:
#undef ST_BAD
    free(header_json);
    if (rc != 0) safetensors_close(st);
    return rc;
}

static inline const TensorEntry* safetensors_find(const SafeTensors *st, const char *name) {
    for (int i = 0; i < st->n_entries; i++) {
        if (strcmp(st->entries[i].name, name) == 0) return &st->entries[i];
    }
    return NULL;
}

static inline void safetensors_close(SafeTensors *st) {
    if (st->entries) free(st->entries);
    if (st->mmap_ptr && st->mmap_ptr != MAP_FAILED) munmap(st->mmap_ptr, st->file_size);
    memset(st, 0, sizeof(*st));
}

#ifdef __cplusplus
}
#endif

#endif /* P7_SAFETENSORS_C_H */
