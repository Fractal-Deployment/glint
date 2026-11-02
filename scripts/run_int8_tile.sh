#!/usr/bin/env bash
# Builds and runs the INT8 tile GEMM receipts on the RTX 3060: the kernel test, then the Arm 0 forward
# with the pin's linears through the tile kernel and through the anchor's dequant-to-BF16 cuBLAS path.
# Usage: scripts/run_int8_tile.sh [PIN.safetensors]
# Exit: 0 all PASS or reported, 1 kernel test FAIL, 3 setup, build or CUDA error.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${P7_CONTAINER:-40-Runtime-Deps}"
MODEL_DIR="${P7_BF16_MODEL_DIR:-/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083}"
PIN="${1:-/home/workspace/pins/Phi-4-mini-int8-from-bf16/model.safetensors}"
BUILD_DIR="${P7_BUILD_DIR:-/mnt/spill/compile_scratch/p7-int8-base}"
LEASE="${GPU_LEASE:-/home/workspace/fractal-deployment/training_orchestrator/scripts/with_gpu_flock.sh}"
NVCC="/usr/bin/nvcc -O2 -arch=sm_86"

lxc exec "$CONTAINER" -- bash -c "mkdir -p '$BUILD_DIR' \
  && $NVCC -o '$BUILD_DIR/test_int8_tile_gemm' '$ROOT/tests/test_int8_tile_gemm.cu' '$ROOT/cuda/int8_tile_gemm.cu' -lcublas \
  && $NVCC -o '$BUILD_DIR/test_arm0_int8_tile' '$ROOT/tests/test_arm0_int8_tile.cu' '$ROOT/cuda/int8_tile_gemm.cu' -lcublas -lcrypto" || exit 3
status=0
bash "$LEASE" -- lxc exec "$CONTAINER" -- "$BUILD_DIR/test_int8_tile_gemm" "$PIN" || status=$?
case "$status" in 0) ;; 1) exit 1 ;; *) exit 3 ;; esac
for gemm in tile cublas; do
  bash "$LEASE" -- lxc exec "$CONTAINER" -- "$BUILD_DIR/test_arm0_int8_tile" --gemm "$gemm" "$PIN" \
    "$MODEL_DIR" "$ROOT/docs/p7_mode3_fold/artifacts" || exit 3
done
