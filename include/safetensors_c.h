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

static inline const char* _st_parse_u64(const char* p, uint64_t* out) {
    p = _st_skip_ws(p);
    if (*p < '0' || *p > '9') return NULL;
    uint64_t v = 0;
    while (*p >= '0' && *p <= '9') {
        uint64_t d = (uint64_t)(*p - '0');
        if (v > (UINT64_MAX - d) / 10) return NULL;
        v = v * 10 + d;
        p++;
    }
    *out = v;
    return p;
}

static inline const char* _st_skip_value(const char* p) {
    int depth = 0;
    while (*p) {
        if (*p == '"') {
            char scratch[2];
            p = _st_parse_string(p, scratch, sizeof(scratch));
            if (!p) return NULL;
            if (depth == 0) return p;
            continue;
        }
        if (*p == '{' || *p == '[') depth++;
        else if (*p == '}' || *p == ']') {
            if (depth == 0) return p;
            if (--depth == 0) return p + 1;
        } else if (*p == ',' && depth == 0) return p;
        p++;
    }
    return NULL;
}

static inline const char* _st_parse_entry(const char* p, TensorEntry* e) {
    if (*p != '{') return NULL;
    p++;
    int have_offsets = 0;
    for (;;) {
        p = _st_skip_ws(p);
        if (*p == '}') break;
        char field[64];
        p = _st_parse_string(p, field, sizeof(field));
        if (!p) return NULL;
        p = _st_skip_ws(p);
        if (*p != ':') return NULL;
        p = _st_skip_ws(p + 1);
        if (strcmp(field, "dtype") == 0) {
            p = _st_parse_string(p, e->dtype, sizeof(e->dtype));
        } else if (strcmp(field, "shape") == 0) {
            if (*p != '[') return NULL;
            p = _st_skip_ws(p + 1);
            while (p && *p != ']') {
                uint64_t v;
                p = _st_parse_u64(p, &v);
                if (!p || e->n_dims >= 4) return NULL;
                e->shape[e->n_dims++] = (int64_t)v;
                p = _st_skip_ws(p);
                if (*p == ',') p = _st_skip_ws(p + 1);
            }
            if (p) p++;
        } else if (strcmp(field, "data_offsets") == 0) {
            if (*p != '[') return NULL;
            p = _st_parse_u64(p + 1, &e->begin);
            if (!p) return NULL;
            p = _st_skip_ws(p);
            if (*p != ',') return NULL;
            p = _st_parse_u64(p + 1, &e->end);
            if (!p) return NULL;
            p = _st_skip_ws(p);
            if (*p != ']') return NULL;
            p++;
            have_offsets = 1;
        } else {
            p = _st_skip_value(p);
        }
        if (!p) return NULL;
        p = _st_skip_ws(p);
        if (*p == ',') p++;
        else if (*p != '}') return NULL;
    }
    return have_offsets ? p + 1 : NULL;
}

static inline int _st_push(SafeTensors *st, const TensorEntry *e) {
    if (st->n_entries == st->capacity) {
        int cap = st->capacity ? st->capacity * 2 : 256;
        TensorEntry *grown = (TensorEntry*)realloc(st->entries, (size_t)cap * sizeof(TensorEntry));
        if (!grown) return -1;
        st->entries = grown;
        st->capacity = cap;
    }
    st->entries[st->n_entries++] = *e;
    return 0;
}

/* Returns 0, or <0 with *st left closed: -1 open/stat, -2 mmap, -3 malformed
 * header, -4 header or a tensor's [begin,end) outside the file, -5 out of memory. */
static inline int _st_parse(SafeTensors *st) {
    if (st->file_size < 8) return -4;
    memcpy(&st->header_len, st->mmap_ptr, 8);
    if (st->header_len > st->file_size - 8) return -4;
    const uint8_t *data_base = (const uint8_t*)st->mmap_ptr + 8 + st->header_len;
    const uint64_t data_len = st->file_size - 8 - st->header_len;

    char *json = (char*)malloc((size_t)st->header_len + 1);
    if (!json) return -5;
    memcpy(json, (const char*)st->mmap_ptr + 8, (size_t)st->header_len);
    json[st->header_len] = '\0';

    int rc = -3;
    const char *p = _st_skip_ws(json);
    if (*p != '{') goto done;
    p++;
    for (;;) {
        p = _st_skip_ws(p);
        if (*p == '}') { rc = 0; break; }
        char key[256];
        p = _st_parse_string(p, key, sizeof(key));
        if (!p) break;
        p = _st_skip_ws(p);
        if (*p != ':') break;
        p = _st_skip_ws(p + 1);
        if (strcmp(key, "__metadata__") == 0) {
            p = _st_skip_value(p);
        } else {
            TensorEntry e;
            memset(&e, 0, sizeof(e));
            strncpy(e.name, key, sizeof(e.name) - 1);
            p = _st_parse_entry(p, &e);
            if (p && (e.begin > e.end || e.end > data_len)) { rc = -4; break; }
            if (p) {
                e.data = data_base + e.begin;
                if (_st_push(st, &e) != 0) { rc = -5; break; }
            }
        }
        if (!p) break;
        p = _st_skip_ws(p);
        if (*p == ',') p++;
        else if (*p != '}') break;
    }
done:
    free(json);
    return rc;
}

static inline void safetensors_close(SafeTensors *st);

static inline int safetensors_open(SafeTensors *st, const char *path) {
    memset(st, 0, sizeof(*st));
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat sb;
    if (fstat(fd, &sb) < 0) { close(fd); return -1; }
    st->file_size = (size_t)sb.st_size;
    if (st->file_size == 0) { close(fd); return -4; }
    st->mmap_ptr = mmap(NULL, st->file_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (st->mmap_ptr == MAP_FAILED) { st->mmap_ptr = NULL; return -2; }
    int rc = _st_parse(st);
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
    if (st->mmap_ptr) munmap(st->mmap_ptr, st->file_size);
    memset(st, 0, sizeof(*st));
}

#ifdef __cplusplus
}
#endif

#endif /* P7_SAFETENSORS_C_H */
