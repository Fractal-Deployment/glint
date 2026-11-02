#!/usr/bin/env bash
# Builds and runs the resolution receipts on the RTX 3060: the residual kernel test, then the BF16 floor and every
# arm over the base pin with the streamed-vs-resident check. Needs the HF logits from p7_resolution_reference.py.
# P7_BASE picks the base (int8, the default, or int4) and P7_RESID_BITS the residual codes (4 or 8; default 4
# over INT8, 8 over INT4). P7_EVAL_ARTIFACTS and P7_RES_ARTIFACTS point at another eval segment's ruler.
# Usage: scripts/run_resolution.sh OUT_TSV [FRAC...]
# Exit: 0 all PASS or reported, 1 kernel test FAIL or streamed arm differs, 3 setup, build or CUDA error.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${P7_CONTAINER:-40-Runtime-Deps}"
MODEL_DIR="${P7_BF16_MODEL_DIR:-/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083}"
BASE="${P7_BASE:-int8}"
BUILD_DIR="${P7_BUILD_DIR:-/mnt/spill/compile_scratch/p7-resolution}"
case "$BASE" in
  int8) PIN="${P7_PIN:-/home/workspace/pins/Phi-4-mini-int8-from-bf16/model.safetensors}"; BITS="${P7_RESID_BITS:-4}" ;;
  int4) PIN="${P7_PIN:-/mnt/spill/compile_scratch/p7-int4/pin/model.safetensors}"; BITS="${P7_RESID_BITS:-8}" ;;
  *) echo "P7_BASE must be int8 or int4"; exit 3 ;;
esac
EVAL_ART="${P7_EVAL_ARTIFACTS:-$ROOT/docs/p7_mode3_fold/artifacts}"
RES_ART="${P7_RES_ARTIFACTS:-$ROOT/docs/p7_resolution/artifacts}"
LOGITS="${P7_HF_LOGITS:-$BUILD_DIR/ref/hf_logits_2048.bf16}"
LEASE="${GPU_LEASE:-/home/workspace/fractal-deployment/training_orchestrator/scripts/with_gpu_flock.sh}"
NVCC="/usr/bin/nvcc -O2 -arch=sm_86 -I$ROOT/include"
AS_USER=(--user 1000 --group 1000)
OUT="${1:?usage: $0 OUT_TSV [FRAC...]}"
shift

lxc exec "$CONTAINER" "${AS_USER[@]}" -- bash -c "mkdir -p '$BUILD_DIR' \
  && $NVCC -o '$BUILD_DIR/test_residual_correction' '$ROOT/tests/test_residual_correction.cu' '$ROOT/cuda/residual_correction.cu' -lcublas \
  && $NVCC -o '$BUILD_DIR/test_arm0_resolution' '$ROOT/tests/test_arm0_resolution.cu' '$ROOT/cuda/residual_correction.cu' \
     '$ROOT/cuda/int8_tile_gemm.cu' '$ROOT/cuda/int4_tile_gemm.cu' '$ROOT/cuda/glint_plug_streamer.cu' -lcublas -lcrypto" || exit 3
status=0
bash "$LEASE" -- lxc exec "$CONTAINER" "${AS_USER[@]}" -- "$BUILD_DIR/test_residual_correction" || status=$?
case "$status" in 0) ;; 1) exit 1 ;; *) exit 3 ;; esac
ARGS=(--base "$BASE" --resid-bits "$BITS" "$PIN" "$MODEL_DIR" "$EVAL_ART" "$RES_ART" "$LOGITS" "$OUT")
bash "$LEASE" -- lxc exec "$CONTAINER" "${AS_USER[@]}" -- "$BUILD_DIR/test_arm0_resolution" "${ARGS[@]}" "$@" || exit $?
