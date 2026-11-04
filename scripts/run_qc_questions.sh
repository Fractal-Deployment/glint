#!/usr/bin/env bash
# Builds the Stage 0 question list p7-s0-questions.list from the local QC pools (rules §0.5). No Python.
# Needs the node_modules + pinned tokenizer + reference fixture of the QC count kit (v3) and the pools under
# /home/workspace/p7-qc-pools/2026-10-30.v1. Writes a NEW dir (never overwrites).
# Usage: scripts/run_qc_questions.sh OUT_DIR [--seed 20261022]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?usage: run_qc_questions.sh OUT_DIR [--seed N]}"; shift || true
KIT="${P7_QC_KIT:-/home/workspace/supervisor/scratch/p7-qc-pools-code-20261030.v3}"
WORK="${P7_QC_WORK:-/home/workspace/lakes/p7_glint/qc_work}"
mkdir -p "$WORK"
ln -sfn "$KIT/node_modules" "$WORK/node_modules"
ln -sfn "$KIT/tokenizer-cfbefacb" "$WORK/tokenizer-cfbefacb"
cp --update=none "$KIT/reference_fixture.json" "$WORK/reference_fixture.json"
cp "$ROOT/tools/qc_questions/select_questions.mjs" "$WORK/select_questions.mjs"
cd "$WORK" && node select_questions.mjs --out-dir "$OUT" "$@"
