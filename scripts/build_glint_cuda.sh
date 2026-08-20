#!/usr/bin/env bash
# RTX 3060: sm_86. Run on the machine with nvcc, not the sandbox.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCH="${CUDA_ARCH:-sm_86}"
nvcc -O3 -std=c++17 -arch="$ARCH" -shared -Xcompiler -fPIC \
  -I"$ROOT/include" \
  -o "$ROOT/libglint_encode.so" \
  "$ROOT/cuda/glint_encode.cu"
echo "wrote $ROOT/libglint_encode.so"
