#!/usr/bin/env bash
# Builds lib/libp7_kernels.so and lib/p7_attach_check inside the CUDA container (no nvcc on the host).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${P7_CONTAINER:-40-Runtime-Deps}"
mkdir -p "$ROOT/lib"
lxc exec "$CONTAINER" -- bash -c "cd '$ROOT' && /usr/bin/nvcc -O2 -std=c++17 -arch=sm_86 -shared -Xcompiler -fPIC -Iinclude \
  -o lib/libp7_kernels.so cuda/p7_abi.c cuda/p7_abi_selftest.cu cuda/int8_mma_gemm.cu cuda/glint_plug_streamer.cu \
  && gcc -O2 -o lib/p7_attach_check cuda/p7_attach_check.c -ldl"
echo "wrote $ROOT/lib/libp7_kernels.so"
