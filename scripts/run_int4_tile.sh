#!/usr/bin/env bash
# Builds and runs the INT4 base receipts on the RTX 3060: the quantizer test, the BF16 -> INT4 pin conversion
# (skipped when PIN exists), the tile GEMM kernel test with the pin's matrices, then the Arm 0 forward with the
# pin's linears through the tile kernel and through dequant-to-BF16 cuBLAS.
# Usage: scripts/run_int4_tile.sh [PIN.safetensors]
# Exit: 0 all PASS or reported, 1 a test or the conversion check FAIL, 3 setup, build or CUDA error.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${P7_CONTAINER:-40-Runtime-Deps}"
MODEL_DIR="${P7_BF16_MODEL_DIR:-/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083}"
BUILD_DIR="${P7_BUILD_DIR:-/mnt/spill/compile_scratch/p7-int4}"
PIN="${1:-$BUILD_DIR/pin/model.safetensors}"
LEASE="${GPU_LEASE:-/home/workspace/fractal-deployment/training_orchestrator/scripts/with_gpu_flock.sh}"
NVCC="/usr/bin/nvcc -O2 -arch=sm_86"
AS_USER=(--user 1000 --group 1000)
in_box() { lxc exec "$CONTAINER" "${AS_USER[@]}" -- bash -c "$1"; }
gate() { local s=0; "$@" || s=$?; case "$s" in 0) ;; 1) exit 1 ;; *) exit 3 ;; esac; }

in_box "mkdir -p '$BUILD_DIR' \
  && g++ -O2 -o '$BUILD_DIR/test_int4_pin' '$ROOT/tests/test_int4_pin.cpp' \
  && g++ -O2 -o '$BUILD_DIR/int4_pin_convert' '$ROOT/cuda/int4_pin_convert.cpp' \
  && $NVCC -o '$BUILD_DIR/test_int4_tile_gemm' '$ROOT/tests/test_int4_tile_gemm.cu' '$ROOT/cuda/int4_tile_gemm.cu' -lcublas \
  && $NVCC -o '$BUILD_DIR/test_arm0_int4_tile' '$ROOT/tests/test_arm0_int4_tile.cu' '$ROOT/cuda/int4_tile_gemm.cu' -lcublas -lcrypto" || exit 3
gate in_box "'$BUILD_DIR/test_int4_pin'"
if ! in_box "test -f '$PIN'"; then
  gate in_box "mkdir -p '$(dirname "$PIN")' && '$BUILD_DIR/int4_pin_convert' '$MODEL_DIR' '$(dirname "$PIN")'"
fi
gate bash "$LEASE" -- lxc exec "$CONTAINER" "${AS_USER[@]}" -- "$BUILD_DIR/test_int4_tile_gemm" "$PIN"
for gemm in tile cublas; do
  bash "$LEASE" -- lxc exec "$CONTAINER" "${AS_USER[@]}" -- "$BUILD_DIR/test_arm0_int4_tile" --gemm "$gemm" "$PIN" \
    "$MODEL_DIR" "$ROOT/docs/p7_mode3_fold/artifacts" || exit 3
done
