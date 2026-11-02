#!/usr/bin/env bash
# Piece H: plug selection at equal plug bytes. Runs the resident W8A8 forward with res4 plugs on a set of
# layers and reports PPL, top-1 vs the HF oracle and KL(BF16 anchor || this). Arms at budget K layers:
#   static_uniform (evenly spaced), static_first, static_last, oracle (top-K by measured single-layer KL gain),
#   predicted (top-K by the weight map's base rel_l2 per layer, no forward pass).
# Needs the anchor logits (ARM0_LOGITS_OUT from test_arm0_bf16_full) and the plug file (tools/plug_encode.c).
# Usage: scripts/run_plug_selection.sh K OUT.tsv
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
K="${1:-8}"; OUT="${2:-plug_selection.tsv}"
W="${P7_BUILD_DIR:-/mnt/spill/compile_scratch/wdm}"
M="${P7_BF16_MODEL_DIR:-/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083}"
PIN="${P7_PIN:-/home/workspace/pins/Phi-4-mini-int8-from-bf16/model.safetensors}"
LEASE="${GPU_LEASE:-/home/workspace/fractal-deployment/training_orchestrator/scripts/with_gpu_flock.sh}"
WMAP="$ROOT/docs/p7_mode3_fold/artifacts/E_PLUG_RES4_phi4mini_error.json"
run() { # name layers
  local o; o="$(bash "$LEASE" -- lxc exec 40-Runtime-Deps -- env ARM0_LOGITS_REF="$W/anchor_logits.bin" ARM0_PLUG_FILE="$W/phi4_res4.plug.safetensors" \
      ARM0_PLUG_LAYERS="$2" "$W/arm0i" --gemm mma "$PIN" "$M" "$ROOT/docs/p7_mode3_fold/artifacts" 2>&1)"
  local ppl t1 klm klp
  ppl="$(sed -n 's/^Measured Perplexity: *//p' <<<"$o")"; t1="$(sed -n 's/^Top-1 Matches: *\([0-9]*\) .*/\1/p' <<<"$o")"
  klm="$(sed -n 's/.*per token: mean \([^ ]*\).*/\1/p' <<<"$o")"; klp="$(sed -n 's/.*p99 \([^ ]*\) .*/\1/p' <<<"$o")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$ppl" "$t1" "$klm" "$klp" | tee -a "$OUT"
}
: >"$OUT"; printf 'arm\tlayers\tppl\ttop1_of_2047\tkl_mean\tkl_p99\n' >>"$OUT"
run none ""
for l in $(seq 0 31); do run "single_$l" "$l"; done
topk() { sort -t$'\t' -k2,2 -g "$1" | head -"$K" | cut -f1 | sort -n | paste -sd,; }
# oracle: lowest KL mean among singles
ORACLE="$(grep '^single_' "$OUT" | sort -t$'\t' -k5,5 -g | head -"$K" | cut -f2 | sort -n | paste -sd,)"
# predicted: layers ranked by summed base rmse^2 weighted by size from the weight map
PRED="$(jq -r '[.linears[]|select(.name|test("layers"))|{l:(.name|capture("layers\\.(?<n>[0-9]+)").n|tonumber),v:.rel_l2_base}]|group_by(.l)|map({l:.[0].l,v:(map(.v)|add)})|sort_by(-.v)|.[0:'"$K"']|map(.l)|sort|join(",")' "$WMAP")"
STEP=$((32 / K)); UNI="$(seq 0 "$STEP" 31 | head -"$K" | paste -sd,)"
run static_uniform "$UNI"; run static_first "$(seq 0 $((K-1)) | paste -sd,)"; run static_last "$(seq $((32-K)) 31 | paste -sd,)"
run oracle "$ORACLE"; run predicted "$PRED"
