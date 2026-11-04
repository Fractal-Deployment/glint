#!/usr/bin/env bash
# Rerun every glint test and model arm on the current GPU/firmware state and write one summary table.
# Records driver, VBIOS, GSP firmware, boot time and kernel first. Builds from the checked-out sources into a fresh
# build dir in the CUDA container, then runs each test under the GPU lease. A failing step is recorded, not fatal.
# Usage: scripts/retest_all.sh OUT_DIR          (OUT_DIR must not exist)  Exit 0 only if every step passed.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?usage: retest_all.sh OUT_DIR}"
APPEND="${RETEST_APPEND:-0}"; ONLY="${RETEST_ONLY:-}"   # RETEST_APPEND=1 RETEST_ONLY=regex reruns chosen steps into an existing dir
[ -e "$OUT" ] && [ "$APPEND" != 1 ] && { echo "REFUSED: $OUT exists" >&2; exit 3; }; mkdir -p "$OUT"
C="${P7_CONTAINER:-40-Runtime-Deps}"; B="${P7_RETEST_BUILD:-/root/p7build-retest}"
LEASE="${GPU_LEASE:-/home/workspace/fractal-deployment/training_orchestrator/scripts/with_gpu_flock.sh}"
M=/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083
PINS=/home/workspace/pins; PIN="$PINS/Phi-4-mini-int8-from-bf16/model.safetensors"; GL="$PINS/Phi-4-mini-glint-from-bf16"
D=/home/workspace/lakes/p7_glint; ART="$ROOT/docs/p7_mode3_fold/artifacts"; PLUG="$D/phi4_res4.plug.safetensors"
MOJO="export MODULAR_HOME=/root/.pixi/envs/modular/share/max; export PATH=/root/.pixi/envs/modular/bin:\$PATH"
SUM="$OUT/summary.tsv"; [ "$APPEND" = 1 ] || printf 'step\trc\tseconds\tkey_output\n' > "$SUM"; FAILS=0

[ "$APPEND" = 1 ] || { echo "date_utc: $(date -u +%FT%TZ)"; echo "boot: $(uptime -s)"; echo "kernel: $(uname -r)"
  nvidia-smi --query-gpu=name,driver_version,vbios_version,memory.total --format=csv,noheader
  nvidia-smi -q | grep -iE "GSP Firmware|VBIOS|Driver Version|CUDA Version"
  echo "container: $(lxc exec "$C" -- nvidia-smi --query-gpu=driver_version,vbios_version --format=csv,noheader)"
  echo "glint_commit: $(git -C "$ROOT" rev-parse HEAD)"; } > "$OUT/firmware.txt"

lx()  { lxc exec "$C" -- bash -c "$1"; }
host() { bash -c "$1"; }
gpu() { bash "$LEASE" -- lxc exec "$C" -- bash -c "$1"; }
step() { # name, mode(lx|gpu), command, key-output extractor (reads the log on stdin)
  local name="$1" mode="$2" cmd="$3" ext="${4:-tail -n 1}" t0 rc key
  if [ -n "$ONLY" ] && ! [[ "$name" =~ $ONLY ]]; then return 0; fi
  t0=$(date +%s); $mode "$cmd" > "$OUT/$name.log" 2>&1; rc=$?
  key=$(bash -c "$ext" < "$OUT/$name.log" 2>/dev/null | tr '\t\n' '  ' | cut -c1-230)
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" "$key" | tee -a "$SUM"
  [ "$rc" -ne 0 ] && FAILS=$((FAILS + 1)); return 0
}
NV="/usr/bin/nvcc -O2 -arch=sm_86 -I$ROOT/include"
lx "mkdir -p $B" || exit 3

# ---------- builds
build() { step "build_$1" lx "cd $ROOT && $NV -o $B/$1 $2" "tail -n 2"; }
build int8_mma_gemm_test   "tests/test_int8_mma_gemm.cu cuda/int8_mma_gemm.cu"
build int8_tile_gemm_test  "tests/test_int8_tile_gemm.cu cuda/int8_tile_gemm.cu -lcublas"
build late_plug_test       "tests/test_late_plug.cu cuda/int8_mma_gemm.cu"
build offload_stream_test  "tests/test_int8_offload_stream.cu cuda/int8_mma_gemm.cu cuda/glint_plug_streamer.cu"
build plug_evict_test      "tests/test_plug_evict_real.cu cuda/glint_plug_streamer.cu"
build plug_streamer_test   "tests/test_glint_plug_streamer.cu cuda/glint_plug_streamer.cu"
build streamer_real_test   "tests/test_streamer_real_weights.cu cuda/glint_plug_streamer.cu cuda/glint_htile.cu cuda/glint_encode.cu"
build htile_test           "tests/test_glint_htile.cu cuda/glint_htile.cu"
build bf16full             "tests/test_arm0_bf16_full.cu cuda/glint_htile.cu -lcublas -lcrypto"
build arm0i                "tests/test_arm0_int8_tile.cu cuda/int8_tile_gemm.cu cuda/int8_mma_gemm.cu cuda/glint_plug_streamer.cu -lcublas -lcrypto"
step build_plug_encode lx "gcc -O2 -I$ROOT/include -o $B/pe $ROOT/tools/plug_encode.c -lm" "tail -n 2"
step build_p7_kernels host "cd $ROOT && bash scripts/build_p7_kernels.sh" "tail -n 1"
step build_mojo_forward lx "$MOJO; cd $ROOT && mojo build -I mojo -o $B/fwd mojo/p7_forward.mojo 2>&1 | grep -v warning | tail -3" "tail -n 2"

# ---------- CUDA kernel and transport tests
step attach_selftest gpu "$ROOT/lib/p7_attach_check $ROOT/lib/libp7_kernels.so p7_attach_abi_v1" "grep P7_SELFTEST"
step int8_mma_gemm_kernel gpu "$B/int8_mma_gemm_test" "tail -n 1"
step int8_tile_gemm_kernel gpu "$B/int8_tile_gemm_test --quick $PIN" "tail -n 2"
step late_plug_real_weights gpu "$B/late_plug_test $PIN $PLUG $M model.layers.5.mlp.down_proj.weight" "tail -n 3"
step mode2_offload_deepseek1p5b gpu "$B/offload_stream_test" "tail -n 4"
step plug_evict_smollm2 gpu "$B/plug_evict_test" "tail -n 2"
step plug_streamer_poison gpu "$B/plug_streamer_test" "tail -n 3"
step streamer_real_weights_phi4 gpu "$B/streamer_real_test $M/model-00001-of-00002.safetensors" "tail -n 3"
step htile_k_bounds gpu "$B/htile_test" "tail -n 2"

# ---------- Phi-4-mini model arms through the Arm 0 harness (2047 tokens)
RES='grep -E "^Measured Perplexity|^Top-1 Matches|VERDICT|per token" | sed "s/Measured Perplexity: */PPL /;s/Top-1 Matches: */top1 /" | tr "\n" " "'
step arm0_bf16_anchor        gpu "$B/bf16full $M $ART" "$RES"
step arm0_int8_dequant_cublas gpu "$B/bf16full --int8 $PIN $M $ART" "$RES"
step arm0_nf4                gpu "ARM0_NF4=1 $B/bf16full $M $ART" "$RES"
step arm0_glint_hole_plus_plug gpu "$B/bf16full --glint $GL $M $ART" "$RES"
KL="ARM0_LOGITS_REF=$D/anchor_logits.bin"
for g in tile cublas mma mma-stream; do step "arm0_int8_$g" gpu "env $KL $B/arm0i --gemm $g $PIN $M $ART" "$RES"; done
ALL=$(seq -s, 0 31)
step arm0_int8_mma_plugs          gpu "env $KL ARM0_PLUG_FILE=$PLUG ARM0_PLUG_LAYERS=$ALL $B/arm0i --gemm mma $PIN $M $ART" "$RES"
step arm0_int8_mma_resolve        gpu "env $KL ARM0_RESOLVE=1 $B/arm0i --gemm mma $PIN $M $ART" "$RES"
step arm0_int8_mma_plugs_resolve  gpu "env $KL ARM0_RESOLVE=1 ARM0_PLUG_FILE=$PLUG ARM0_PLUG_LAYERS=$ALL $B/arm0i --gemm mma $PIN $M $ART" "$RES"
for b in 3 2; do step "m4_lossy_b$b" gpu "env $KL ARM0_PLUG_FILE=$D/plug_b$b.safetensors ARM0_PLUG_LAYERS=$ALL $B/arm0i --gemm mma $PIN $M $ART" "$RES"; done
step plug_selection_k8 host "cd $ROOT && P7_BUILD_DIR=$B bash scripts/run_plug_selection.sh 8 $OUT/plug_selection_k8_host.tsv > /dev/null; awk -F'\t' '\$1!~/^single/' $OUT/plug_selection_k8_host.tsv" "tail -n 7 | cut -f1,3,4,5 | tr '\n' ';'"
step weight_rmse_reproduce lx "$B/pe $PIN $M $OUT/_plug_b4.tmp.safetensors 4 | python3 -c 'import json,sys;d=json.load(sys.stdin);print(\"rmse_base\",d[\"rmse_base\"],\"rmse_with_plug\",d[\"rmse_base_plus_plug\"])'; rm -f $OUT/_plug_b4.tmp.safetensors" "tail -n 1"

# ---------- Mojo / MAX
mj() { step "$1" gpu "$MOJO; cd $ROOT && $2 2>&1 | grep -v warning" "${3:-tail -n 2}"; }
mj mojo_int8_mma_vs_cuda      "mojo run mojo/p7_int8_mma_gemm.mojo $ROOT/lib/libp7_kernels.so" "grep -E 'BITEXACT|NEAR|slowdown' | tr '\n' ' '"
mj mojo_prefetch_overlap      "mojo run mojo/p7_prefetch.mojo" "grep -E 'overlapped|hidden|serial' | tr '\n' ' '"
mj mojo_ring_cuda_gemm        "mojo run mojo/p7_ring_cuda_gemm.mojo $ROOT/lib/libp7_kernels.so" "tail -n 2 | tr '\n' ' '"
mj mojo_ring_mojo_gemm        "mojo run -I mojo mojo/p7_ring_mojo_gemm.mojo" "tail -n 2 | tr '\n' ' '"
mj mojo_attach_check          "mojo run mojo/p7_attach_check.mojo $ROOT/lib/libp7_kernels.so" "tail -n 2 | tr '\n' ' '"
mj mojo_layer_ops             "mojo run mojo/p7_layer_ops.mojo" "tail -n 3 | tr '\n' ' '"
mj mojo_rope                  "mojo run mojo/p7_rope.mojo" "tail -n 2 | tr '\n' ' '"
mj mojo_attention             "mojo run mojo/p7_attention.mojo" "tail -n 2 | tr '\n' ' '"
FW='grep -E "^Perplexity|^Top-1|^layers done|Forward time" | tr "\n" " "'
step mojo_forward_int8           gpu "env MODULAR_HOME=/root/.pixi/envs/modular/share/max $B/fwd $PIN $M $ART" "$FW"
step mojo_forward_plugs          gpu "env MODULAR_HOME=/root/.pixi/envs/modular/share/max $B/fwd $PIN $M $ART $PLUG none" "$FW"
step mojo_forward_resolve        gpu "env MODULAR_HOME=/root/.pixi/envs/modular/share/max $B/fwd $PIN $M $ART none resolve" "$FW"
step mojo_forward_plugs_resolve  gpu "env MODULAR_HOME=/root/.pixi/envs/modular/share/max $B/fwd $PIN $M $ART $PLUG resolve" "$FW"

echo "steps with non-zero rc: $FAILS" | tee -a "$SUM"
[ "$FAILS" -eq 0 ]
