# Project 7 attach contract (`p7_attach_abi_v1`)

The orchestrator attaches glint through `training_orchestrator/scripts/orch_p7_start_gate.sh`.

## Gate
1. `attach/runtime_attach.sh --phase --out-dir --mode 1..5 --pin` checks the pin schema for the mode, `lib/libp7_kernels.so`, and `attach/ABI_VERSION`, then writes `boards/p7_runtime_attach.env`. Exit 0, 2 (bad arguments) or 4 (HARD_BLOCK).
2. The gate then runs `lib/p7_attach_check LIB ABI` on the GPU under `with_gpu_flock.sh`. It `dlopen`s the library, requires every symbol below, requires `p7_abi_version()` to equal `ABI_VERSION`, and runs `p7_abi_selftest`.
3. The gate stamps `boards/p7_start_gate.json` (`attached` or `hard_block`) and exits 0 or 4. A file that exists is not an attach: only a passing self-test is.

## Exported symbols (extern "C")
- `p7_abi_version`, `p7_abi_selftest`
- `int8_mma_gemm` (W8A8 INT8 tensor-core GEMM over a `bf16_to_int8_pin_v1` linear; `include/int8_mma_gemm.h`)
- `int8_mma_gemm_plug` (same, plus a p7_plug_res4_v1 slice added by an accumulating second pass; additive, ABI unchanged)
- `p7_int8_gemm_flat` (flat-argument int8_mma_gemm for FFI callers such as Mojo; additive)
- `glint_plug_streamer_{create,set_host_data,start,acquire,release,get_stats,destroy}` (2-slot ring; `include/glint_plug_streamer.h`)

## Self-test (`p7_abi_selftest`)
A 4-layer INT8 stack quantized with the pin rule runs once resident and once streamed through the ring. It passes only if the outputs match bit for bit, are finite and mostly non-zero, match a CPU float reference of the kernel arithmetic (max relative error 1e-4), and the streamer moved exactly the layer bytes.

## Changing the ABI
Any change to a signature or to the symbol list bumps `ABI_VERSION` and `P7_ABI_VERSION_STR` in `cuda/p7_abi.c` together. Build: `scripts/build_p7_kernels.sh` (container `40-Runtime-Deps`, sm_86).
