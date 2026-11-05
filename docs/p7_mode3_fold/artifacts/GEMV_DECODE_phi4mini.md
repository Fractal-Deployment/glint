# GLINT decode GEMV (M = 1) over the 4-bit hole + plug, Phi-4-mini (2026-11-04) — glint #89

`k_glint_gemv` in `tests/test_arm0_bf16_full.cu` (`ARM0_GEMV=1`, used for M = 1 linears; prefill still uses H-TILE): one warp per output row; each lane reads 16 B of hole and 16 B of plug (32 weights) per step so a warp's loads are contiguous; the 16 x 16 cell table is in shared memory (constant memory serializes on the divergent nibble index); the activation row is staged in shared memory as f32; f32 accumulation of `(GLINT_CELLS[hole][plug] * absmax) * x`; warp-shuffle reduction. `GLINT_PARENT_ONLY=1` swaps in the parent table for the fold-only arm. Runs through the decode path of `DECODE_PATH_phi4mini.md` (#88). Phi-4-mini, RTX 3060, 64-token prompt, 15 timed steps, one run each.

## Accuracy: checked against a double-precision reference on every linear of the decode run
12,160 linear calls (95 steps x 128 linears). Error of each kernel against a one-thread-per-row double-precision reference on the same hole/plug/absmax/x, relative to the RMS of the reference output:

| Arm | GEMV max / mean | H-TILE max / mean | GEMV vs H-TILE max / mean |
|---|---|---|---|
| hole + plug | 1.70e-5 / 1.30e-6 | 1.45e-4 / 1.35e-5 | 1.46e-4 / 1.36e-5 |
| fold only (parent table) | 1.59e-5 / 1.23e-6 | 1.13e-4 / 1.26e-5 | 1.23e-4 / 1.26e-5 |

The new kernel is about ten times closer to the exact value than H-TILE. The difference between the two kernels is therefore H-TILE's own f32 error (a sequential per-thread accumulation), not an error in the GEMV. Consequence for the decode-vs-prefill check: prefill uses H-TILE, so decode with this kernel differs from prefill at the level of H-TILE's rounding, which bf16 rounding points then amplify (top-1 equal at 76/80 and 74/80 positions; mean |dCE| 6.6e-2 and 4.1e-2, the same size as the cuBLAS arm in the decode note). The greedy generations are the anchor's and the fold-only rollout's, token for token.

## Speed (decode, per token)
| Arm | ms/token | ms/layer | tokens/s | generation |
|---|---|---|---|---|
| dense bf16 cuBLAS (reference, 7.6 GB of weights per token) | 31.0 | 0.97 | 32.3 | = anchor rollout |
| GLINT hole + plug, **GEMV**, resident | 39.6 | 1.24 | **25.3** | = anchor rollout (16/16) |
| fold only, GEMV | 37.6 | 1.18 | 26.6 | = fold-only rollout (16/16) |
| GLINT hole + plug, GEMV, **plug streamed whole through the ring** | 147.9 | 4.62 | **6.8** | = resident GEMV |
| GLINT hole + plug, H-TILE (previous decode kernel) | 455 | 14.2 | 2.2 | = anchor rollout |

- The decode kernel is 11.5x faster than H-TILE and within 28% of the time of dense bf16 cuBLAS while reading about half the bytes (about 4 GB per token against 7.6 GB). Read rate is about 100 GB/s of the card's roughly 360 GB/s, and fold only is only 5% faster than hole + plug, so the kernel is bound by the nibble decode and the shared-memory table, not by memory: there is room to improve, unmeasured here.
- **First measured fold + stream + plug decode speed:** streaming the whole plug costs 4.62 ms per layer against 1.24 ms resident. The extra 3.4 ms per layer is the part of the 4.35 ms fetch (`FOLD_BUDGET_FETCH_phi4mini.md`) that does not hide behind compute; at this kernel speed whole-plug streaming is transfer-bound and caps decode near 7 tokens/s, as the fetch arithmetic predicted. Streamed output equals resident.
- With the fetch costs already measured, a selected 5% fetch (0.31-0.35 ms per layer) is below the 1.24 ms layer compute, so it would hide; that selected-fetch decode is not run (needs the K-major layout and selected-fetch path, #86), and the rollout study (#82) says the activation-selected arms at 10-25% of the plug do not track the full model through 16 generated tokens.

## What this does not show
No KL along the generated path, 3 sequences of 16 tokens at most, 15 timed steps, one text, one run per arm. Timing includes launch overhead and a per-step host embedding copy. The GEMV is not tuned (no vectorized x reads, no tensor-core path, one warp per row with 4 rows per block). Not reviewed by the Test Designer.

## Reproduce
Build as in `FOLD_STREAM_phi4mini.md`. `ARM0_GEMV=1 ARM0_DECODE=64,16 [ARM0_GEMV_CHECK=1] bf16full [--glint PIN | --glint-stream PIN] MODEL ART`; fold only adds `GLINT_PARENT_ONLY=1`. Logs: `/home/workspace/lakes/p7_glint/gemv_20261104/`.
