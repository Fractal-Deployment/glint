"""Host H-TILE simulator for GLINT.
S1 dataflow: K-tiles, inflate GLINT_CELLS[h][p]*absmax, f32 MAC.
Optional bf16 round-trip on fragments (CUDA-like offset).
"""
from __future__ import annotations
from typing import List, Sequence
from dtype_io import pack_bf16, unpack_bf16
from glint_codec import CELLS
def _bf16(v: float) -> float:
    return unpack_bf16(pack_bf16([float(v)]))[0]
def gemm_ref_flat(x: Sequence[float], w: Sequence[float], M: int, N: int, K: int) -> List[float]:
    y = [0.0] * (M * N)
    for m in range(M):
        for n in range(N):
            s = 0.0
            xb = m * K
            wb = n * K
            for k in range(K):
                s += x[xb + k] * w[wb + k]
            y[m * N + n] = s
    return y
def htile_gemm_flat(
    x: Sequence[float],
    hole: Sequence[int],
    plug: Sequence[int],
    absmax: Sequence[float],
    M: int,
    N: int,
    K: int,
    blocksize: int,
    tile_k: int = 64,
    round_bf16: bool = False,
) -> List[float]:
    y = [0.0] * (M * N)
    for n in range(N):
        base = n * K
        for k0 in range(0, K, tile_k):
            k1 = min(K, k0 + tile_k)
            wtile = [0.0] * (k1 - k0)
            for t, k in enumerate(range(k0, k1)):
                w = CELLS[hole[base + k]][plug[base + k]] * absmax[(base + k) // blocksize]
                wtile[t] = _bf16(w) if round_bf16 else w
            for m in range(M):
                s = y[m * N + n]
                xb = m * K
                for t, k in enumerate(range(k0, k1)):
                    xv = _bf16(x[xb + k]) if round_bf16 else x[xb + k]
                    s += xv * wtile[t]
                y[m * N + n] = s
    return y
def max_abs_flat(a: Sequence[float], b: Sequence[float]) -> float:
    m = 0.0
    for i in range(min(len(a), len(b))):
        d = abs(float(a[i]) - float(b[i]))
        if d > m:
            m = d
    return m
