#include "safetensors_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/home/workspace/pins/debug_hf_smollm2/SmolLM2-135M/model.safetensors";
    printf("=== TEST SAFETENSORS C MMAP LOADER ===\n");
    printf("Testing file: %s\n", path);

    SafeTensors st;
    memset(&st, 0, sizeof(st));
    int rc = safetensors_open(&st, path);
    if (rc != 0) {
        printf("safetensors_open failed with rc=%d\n", rc);
        return 1;
    }

    printf("Open OK: n_entries=%d, file_size=%zu bytes, header_len=%lu\n",
           st.n_entries, st.file_size, (unsigned long)st.header_len);
    assert(st.n_entries > 0);

    const TensorEntry *e = safetensors_find(&st, "model.embed_tokens.weight");
    if (!e && st.n_entries > 0) {
        // Fall back to first tensor if embed_tokens has different prefix
        e = &st.entries[0];
    }
    assert(e != NULL);

    printf("Sample tensor '%s': dtype=%s shape=[%ld, %ld] data_bytes=%lu\n",
           e->name, e->dtype, e->shape[0], e->shape[1], (unsigned long)(e->end - e->begin));
    assert(e->data != NULL);

    safetensors_close(&st);
    printf("SAFETENSORS C MMAP LOADER PASSED GREEN.\n");
    return 0;
}
