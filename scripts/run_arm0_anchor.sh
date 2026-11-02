#!/usr/bin/env bash
# Exit status is the gate verdict (GATE_RULES.md section 5.C.1): 0 PASS, 1 FAIL.
# 2 is a usage error and 3 a setup, I/O, build or CUDA error, never a verdict.
# Usage: scripts/run_arm0_anchor.sh [--dump DIR] [--int8 PIN.safetensors]
# With --int8 the linear weights come from the INT8 pin, dequantized to BF16, and the run only reports.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${P7_CONTAINER:-40-Runtime-Deps}"
MODEL_DIR="${P7_BF16_MODEL_DIR:-/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083}"
BUILD_DIR="${P7_BUILD_DIR:-/mnt/spill/compile_scratch/p7-bf16-anchor}"
LEASE="${GPU_LEASE:-/home/workspace/fractal-deployment/training_orchestrator/scripts/with_gpu_flock.sh}"

lxc exec "$CONTAINER" -- bash -c "mkdir -p '$BUILD_DIR' && /usr/bin/nvcc -O2 -arch=sm_86 \
  -o '$BUILD_DIR/test_arm0_bf16_full' '$ROOT/tests/test_arm0_bf16_full.cu' -lcublas -lcrypto" || exit 3
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
status=0
bash "$LEASE" -- lxc exec "$CONTAINER" -- "$BUILD_DIR/test_arm0_bf16_full" \
  "$@" "$MODEL_DIR" "$ROOT/docs/p7_mode3_fold/artifacts" | tee "$LOG" || status=$?
case "$status" in
  0|2|3) exit "$status" ;;
  1) grep -q '^VERDICT: FAIL' "$LOG" && exit 1 || exit 3 ;;
  *) exit 3 ;;
esac
