#!/usr/bin/env bash
set -uo pipefail
BIN="${1:?usage: $0 BIN MODEL_DIR ARTIFACTS_DIR}" MODEL="${2:?usage: $0 BIN MODEL_DIR ARTIFACTS_DIR}" ART="${3:?usage: $0 BIN MODEL_DIR ARTIFACTS_DIR}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/arm0_exits_XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
failed=0

expect() {
  local want="$1" what="$2"
  shift 2
  "$@" >"$TMP/out.log" 2>&1
  local got=$?
  if [ "$got" -eq "$want" ]; then echo "[PASS] $what (exit $got)"
  else echo "[FAIL] $what: exit $got, want $want"; tail -2 "$TMP/out.log"; failed=$((failed + 1)); fi
}

copy_art() { mkdir -p "$TMP/$1" && cp "$ART"/eval_tokens_2048.txt "$ART"/oracle_top1_2048.bin "$ART"/hf_meta.json "$TMP/$1/"; }

mkdir -p "$TMP/empty"
expect 3 "missing tokens, oracle and meta" "$BIN" "$MODEL" "$TMP/empty"

copy_art no_meta && rm "$TMP/no_meta/hf_meta.json"
expect 3 "missing hf_meta.json" "$BIN" "$MODEL" "$TMP/no_meta"

copy_art bad_oracle && printf '\x01' | dd of="$TMP/bad_oracle/oracle_top1_2048.bin" bs=1 seek=0 conv=notrunc status=none
expect 3 "oracle whose sha256 differs from hf_meta.json" "$BIN" "$MODEL" "$TMP/bad_oracle"

copy_art bad_tokens && sed -i '1s/^[0-9]/9/' "$TMP/bad_tokens/eval_tokens_2048.txt"
expect 3 "tokens whose sha256 differs from hf_meta.json" "$BIN" "$MODEL" "$TMP/bad_tokens"

copy_art good
expect 3 "missing safetensors shards" "$BIN" "$TMP/empty" "$TMP/good"
expect 3 "missing INT8 pin" "$BIN" --int8 "$TMP/empty/pin.safetensors" "$MODEL" "$TMP/good"
expect 3 "dump dir that cannot be created" "$BIN" --dump /proc/arm0/dump "$MODEL" "$TMP/good"
expect 2 "usage" "$BIN" "$MODEL"

echo "$([ "$failed" -eq 0 ] && echo PASS || echo FAIL): $failed failed"
[ "$failed" -eq 0 ]
