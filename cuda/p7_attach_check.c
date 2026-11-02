/* p7_attach_check — loads libp7_kernels.so, requires the version and entry-point symbols, runs
 * p7_abi_selftest. Exit 0 pass, 4 hard block (missing library/symbol/version), 1 selftest failed. */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s libp7_kernels.so EXPECTED_ABI\n", argv[0]); return 2; }
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { fprintf(stderr, "HARD_BLOCK: dlopen: %s\n", dlerror()); return 4; }
    static const char *need[] = {"p7_abi_version", "p7_abi_selftest", "int8_mma_gemm", "glint_plug_streamer_create",
                                 "glint_plug_streamer_set_host_data", "glint_plug_streamer_start",
                                 "glint_plug_streamer_acquire", "glint_plug_streamer_release",
                                 "glint_plug_streamer_get_stats", "glint_plug_streamer_destroy", "p7_int8_gemm_flat", "p7_int8_gemm_plug_flat"};
    for (unsigned i = 0; i < sizeof(need) / sizeof(need[0]); i++)
        if (!dlsym(h, need[i])) { fprintf(stderr, "HARD_BLOCK: missing symbol %s\n", need[i]); return 4; }
    const char *(*ver)(void) = (const char *(*)(void))dlsym(h, "p7_abi_version");
    if (strcmp(ver(), argv[2]) != 0) { fprintf(stderr, "HARD_BLOCK: ABI %s != expected %s\n", ver(), argv[2]); return 4; }
    int (*selftest)(char *, size_t) = (int (*)(char *, size_t))dlsym(h, "p7_abi_selftest");
    char msg[512] = "";
    int rc = selftest(msg, sizeof(msg));
    printf("P7_SELFTEST rc=%d %s\n", rc, msg);
    return rc == 0 ? 0 : 1;
}
