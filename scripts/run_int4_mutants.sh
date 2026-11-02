#!/usr/bin/env bash
# Mutation check for the INT4 tile GEMM test: each mutant is a one-line break in a copy of cuda/int4_tile_gemm.cu,
# built against tests/test_int4_tile_gemm.cu and run without the pin. The test must fail on every mutant, plainly
# or under racecheck for the one that only races. Then compute-sanitizer memcheck and racecheck run the unmutated
# kernel over the same cases.
# Usage: scripts/run_int4_mutants.sh
# Exit: 0 every mutant caught and the sanitizers clean, 1 a mutant survived or a sanitizer reported, 3 setup.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${P7_CONTAINER:-40-Runtime-Deps}"
BUILD_DIR="${P7_BUILD_DIR:-/mnt/spill/compile_scratch/p7-int4}/mutants"
LEASE="${GPU_LEASE:-/home/workspace/fractal-deployment/training_orchestrator/scripts/with_gpu_flock.sh}"
SANITIZER="${P7_SANITIZER:-/opt/generations/workshop-generation/.pixi/envs/kernel/bin/compute-sanitizer}"
AS_USER=(--user 1000 --group 1000)
in_box() { lxc exec "$CONTAINER" "${AS_USER[@]}" -- bash -c "$1"; }
on_gpu() { bash "$LEASE" -- lxc exec "$CONTAINER" "${AS_USER[@]}" -- "$@"; }

declare -A MUTANT=(
  [scale_block_ignores_half]='s|size_t blk = flat / INT4_PIN_BLOCK + half;|size_t blk = flat / INT4_PIN_BLOCK;|'
  [scale_switch_off_by_one]='s|(kk < hi_from ? s_lo|(kk <= hi_from ? s_lo|'
  [no_barrier_after_wait]='/^        __syncthreads();$/d'
  [load_next_before_wait]='s|^        cp_async_wait<STAGES - 2>();$|        if (kt + STAGES - 1 < ktiles) load_stage<ASYNC>(ring[(kt + STAGES - 1) % STAGES], x, w, m, m0, n0, (kt + STAGES - 1) * BK);\n        cp_async_wait<STAGES - 2>();|; s|^        if (next < ktiles) load_stage|        if (false) load_stage|'
  [skip_async_stage_load]='s|if (next < ktiles) load_stage|if (next < ktiles \&\& !ASYNC) load_stage|'
  [async_rows_need_only_k16]='s|w.k_in % 32 == 0|w.k_in % 16 == 0|'
  [nibbles_swapped]='s|float a = (float)code_lo(b)|float a = (float)code_hi(b)|; s|float c = (float)code_hi(b)|float c = (float)code_lo(b)|'
)

in_box "mkdir -p '$BUILD_DIR'" || exit 3
survived=0
for name in $(printf '%s\n' "${!MUTANT[@]}" | sort); do
  src="$BUILD_DIR/$name.cu"
  in_box "sed -e '${MUTANT[$name]}' '$ROOT/cuda/int4_tile_gemm.cu' > '$src' && ! cmp -s '$src' '$ROOT/cuda/int4_tile_gemm.cu' \
    && sed -i 's|#include \"../include/|#include \"$ROOT/include/|' '$src' \
    && /usr/bin/nvcc -O2 -arch=sm_86 -o '$BUILD_DIR/$name' '$ROOT/tests/test_int4_tile_gemm.cu' '$src' -lcublas 2>/dev/null" \
    || { echo "MUTANT $name: did not apply or build"; exit 3; }
  status=0
  out=$(on_gpu "$BUILD_DIR/$name" 2>&1) || status=$?
  failing=$(grep -c 'FAIL$' <<<"$out" || true)
  verdict="plain run exit $status, $failing failing cases"
  if [ "$status" -eq 0 ]; then
    race=$(on_gpu "$SANITIZER" --tool racecheck "$BUILD_DIR/$name" 2>&1 | grep 'RACECHECK SUMMARY' || true)
    verdict="$verdict; racecheck: ${race#========= }"
    grep -q ' 0 hazards' <<<"$race" && { survived=$((survived + 1)); verdict="$verdict -> SURVIVED"; }
  fi
  echo "MUTANT $name: $verdict"
  grep -m1 'KERNEL ERROR' <<<"$out" || true
done

in_box "/usr/bin/nvcc -O2 -arch=sm_86 -I'$ROOT/include' -o '$BUILD_DIR/unmutated' '$ROOT/tests/test_int4_tile_gemm.cu' '$ROOT/cuda/int4_tile_gemm.cu' -lcublas" || exit 3
dirty=0
for tool in memcheck racecheck; do
  summary=$(on_gpu "$SANITIZER" --tool "$tool" "$BUILD_DIR/unmutated" 2>&1 | grep -E 'ERROR SUMMARY|RACECHECK SUMMARY|^VERDICT' | tr '\n' ' ')
  echo "SANITIZER $tool on the unmutated kernel: $summary"
  case "$tool" in
    memcheck) grep -q 'ERROR SUMMARY: 2 errors' <<<"$summary" || dirty=1 ;;
    racecheck) grep -q ' 0 hazards' <<<"$summary" || dirty=1 ;;
  esac
done
echo "memcheck expects exactly 2 errors: the deliberate failed cudaMalloc in the pending-error case and its return."
[ "$survived" -eq 0 ] && [ "$dirty" -eq 0 ] && { echo "VERDICT: PASS"; exit 0; }
echo "VERDICT: FAIL ($survived mutants survived, sanitizer dirty $dirty)"
exit 1
