#!/usr/bin/env bash
# Builds and runs the resolution receipts on the RTX 3060: the residual kernel test, then the BF16 floor and every
# arm over the INT8 pin with the streamed-vs-resident check. Needs the HF logits from p7_resolution_reference.py.
# Usage: scripts/run_resolution.sh OUT_TSV [FRAC...]
# Exit: 0 all PASS or reported, 1 kernel test FAIL or streamed arm differs, 3 setup, build or CUDA error.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${P7_CONTAINER:-40-Runtime-Deps}"
MODEL_DIR="${P7_BF16_MODEL_DIR:-/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083}"
PIN="${P7_INT8_PIN:-/home/workspace/pins/Phi-4-mini-int8-from-bf16/model.safetensors}"
BUILD_DIR="${P7_BUILD_DIR:-/mnt/spill/compile_scratch/p7-resolution}"
LOGITS="${P7_HF_LOGITS:-$BUILD_DIR/ref/hf_logits_2048.bf16}"
LEASE="${GPU_LEASE:-/home/workspace/fractal-deployment/training_orchestrator/scripts/with_gpu_flock.sh}"
NVCC="/usr/bin/nvcc -O2 -arch=sm_86 -I$ROOT/include"
OUT="${1:?usage: $0 OUT_TSV [FRAC...]}"
shift

lxc exec "$CONTAINER" -- bash -c "mkdir -p '$BUILD_DIR' \
  && $NVCC -o '$BUILD_DIR/test_residual_correction' '$ROOT/tests/test_residual_correction.cu' '$ROOT/cuda/residual_correction.cu' -lcublas \
  && $NVCC -o '$BUILD_DIR/test_arm0_resolution' '$ROOT/tests/test_arm0_resolution.cu' '$ROOT/cuda/residual_correction.cu' \
     '$ROOT/cuda/int8_tile_gemm.cu' '$ROOT/cuda/glint_plug_streamer.cu' -lcublas -lcrypto" || exit 3
status=0
bash "$LEASE" -- lxc exec "$CONTAINER" -- "$BUILD_DIR/test_residual_correction" || status=$?
case "$status" in 0) ;; 1) exit 1 ;; *) exit 3 ;; esac
ARGS=("$PIN" "$MODEL_DIR" "$ROOT/docs/p7_mode3_fold/artifacts" "$ROOT/docs/p7_resolution/artifacts" "$LOGITS" "$OUT")
bash "$LEASE" -- lxc exec "$CONTAINER" -- "$BUILD_DIR/test_arm0_resolution" "${ARGS[@]}" "$@" || exit $?
