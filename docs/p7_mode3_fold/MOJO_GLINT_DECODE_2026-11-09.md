# Mojo/MAX GLINT KV-cache decode, resident and streamed (2026-11-09)

**Project:** GitHub Project #6, issues #116 (P7-MOJO-D1, Phase B) and #117 (P7-MOJO-S1, first step). **Status:** EXECUTED teacher-forced reference results; PHASE B and PHASE C NOT COMPLETE. **M6 NOT ACCEPTED.** This document records executed runs only.

## What is committed

| File | Role |
|---|---|
| `mojo/p7_glint_linear.mojo` | pin-sourced single-linear GLINT GEMV probe (3072x3072 row check vs CPU reference, max abs diff 5.668e-7) |
| `mojo/p7_glint_decode.mojo` | full 32-layer single-token KV decode over `glint_pin_v1` (hole, plug, absmax, embedded `_glint_cells`), resident plug, warp-based M=1 GEMV |
| `mojo/p7_glint_stream_decode.mojo` | same decode with the whole per-layer plug streamed from host through a two-slot ring (no selection) |
| `mojo/p7_refine.mojo` | `RefineConfig` scaffold: transfer granularity, prefetch distance, selection rule, application rule, staging slots as separate fields. A typed protocol only, **not** a scheduler; only whole-layer transfer and "all" selection are exercised |
| `scripts/p7_mojo_glint_decode.sh` | build and run recipe (40-Runtime-Deps, GPU lease) |

`p7_forward.mojo` and `p7_prefetch.mojo` are unchanged.

## Provenance and reproduction

Mojo 1.0.0 ed45d567, `MODULAR_HOME=/root/.pixi/envs/modular/share/max`, RTX 3060, driver 595.99.02, GPU lease. Pin `/home/workspace/pins/Phi-4-mini-glint-from-bf16`; checkpoint `microsoft/Phi-4-mini-instruct@cfbefacb99257ffa30c83adab238a50856ac3083`; tokens and HF oracle from `artifacts/`. The sources above were written uncommitted during the original runs; before this commit both decoders were **rebuilt from the worktree files and re-run at 32 steps**: both builds exit 0, resident and streamed 32-step outputs are identical to each other and identical to the first 32 positions of the original 2047-position run (`/home/workspace/lakes/p7_glint/mojo_repro_20261109/`). Original logs: `/home/workspace/lakes/p7_glint/mojo_decode_preflight_20261109/`.

## Direct results (2047 teacher-forced positions)

| Path | PPL | Oracle top-1 | Decode time | Speed |
|---|---:|---:|---:|---:|
| Mojo GLINT resident | 5.4847669445 | 2007/2047 (98.05%) | 268.35 s | 7.63 tok/s |
| Mojo GLINT streamed (whole plug each token) | 5.4847669445 | 2007/2047 (98.05%) | 403.24 s | 5.08 tok/s |

- Streamed == resident at all 2047 positions (top-1 id and cross-entropy strings identical, 0 differences). This is the same-path transport check, at the granularity the log records.
- Both pass the recorded teacher-forced thresholds (top-1 >= 95%, PPL within 2% of 5.486957).
- The streamed run requests 1,610,612,736 plug bytes per token. That figure is **requested** bytes; independent H2D accounting, stall and overlap telemetry are **not** recorded and remain open for #117.
- Warp GEMV vs the earlier scalar GEMV: four different top-1 predictions in a 32-token comparison; unexplained numerical sensitivity, kept as a finding.

## Cross-backend comparison (same pin, same 2047 positions)

Files: `/home/workspace/lakes/p7_glint/xbackend_20261109/`.

| Pair | Top-1 agree | mean abs dCE | p99 dCE | max dCE |
|---|---|---:|---:|---:|
| CUDA H-TILE decode vs CUDA GEMV decode (same code, same pin) | 2015/2047 (98.44%) | 0.0271 | 0.231 | 0.351 |
| Mojo vs CUDA GEMV | 2015/2047 (98.44%) | 0.0256 | 0.225 | 0.380 |
| Mojo vs CUDA H-TILE | 2017/2047 (98.53%) | 0.0255 | 0.226 | 0.300 |

Top-1 vs oracle: Mojo 2007, CUDA H-TILE 2007, CUDA GEMV 2000. The equal Mojo/H-TILE count is a coincidence: of their 30 disagreeing positions each matches the oracle on 15. The Mojo-vs-CUDA gap is the same size as the gap between two CUDA kernels on one pin, so Mojo's GLINT decode is inside the numerical variation CUDA already shows against itself. It is **not** bitwise or intermediate-tensor parity, and it covers one text and one model.

## Timing (approximate, short context only)

- CUDA resident hole+plug GEMV decode, 47 steps after a 64-token prefix, 5 runs: 35.1-38.4 ms/token, 26.1-28.5 tok/s.
- Mojo resident decode, 32 steps: 98.8-102.3 ms/token, 9.77-10.12 tok/s (two runs). Positions differ (0-31 vs 64-111). CUDA is roughly 2.6-2.9x faster here; this is not a matched long-context ratio.
- The CUDA 2047-position gate took 267 s wall including load and per-position bookkeeping, so no per-token claim is made for it.

## What this does not establish

- M6, free-running tracking under Amendment 2, or any >= 10 tok/s claim with all mechanisms active.
- Activation-driven selection, prefetch timing, grouped transfer, last-consumer buffer reuse, independent measured H2D bytes, overlap, or peak VRAM in Mojo.
- Whether CUDA's `gemv_init` reads the pin's `_glint_cells` (it builds its table from compiled constants; the two tables are byte-identical for current pins, SHA256 `e774eceefa0d565e330f966d8ba1ea64f09208f4769bb6dd512bcd419bcb0365`).
- Cross-model or cross-text behaviour.

## Streamed decode: copy, compute and overlap (ablation, 64 steps, 3 repeats each)

`p7_glint_stream_decode.mojo` takes an optional 5th argument: 0 = normal, 1 = copies only, 2 = compute only (no plug copies). Modes 1 and 2 print `MOJO_GLINT_STREAM_ABLATION ... outputs invalid` and exist only for timing. Rule fixed before the runs: overlap = copy-only + compute-only - full. Default mode 0 outputs are unchanged: all 64 positions of all three runs are identical to the original resident run. Files: `/home/workspace/lakes/p7_glint/mojo_stream_ablation_20261109/`.

| Mode | ms/token, three runs | Median |
|---|---|---:|
| full (copy + compute) | 207.5, 182.9, 184.9 | 184.9 |
| copy only | 144.6, 140.3, 143.6 | 143.6 |
| compute only | 85.0, 86.1, 85.0 | 85.0 |

- Copy-only moves 1,610,612,736 requested bytes per token in 143.6 ms: **11.22 GB/s**, about 97% of the 11.6 GB/s PCIe Gen3 x16 figure used elsewhere in this project. Whole-plug streaming is bandwidth-bound.
- Overlap = 143.6 + 85.0 - 184.9 = **43.7 ms**; about 51% of the compute side is hidden. A perfect pipeline would cost max(143.6, 85.0) = 143.6 ms, so about 41 ms per token (about 22%) of the full-run time is not explained by the link. Cause not isolated: the two-slot ring allows only one layer of lookahead, and each layer issues four separate copies. This is a measurement, not a diagnosis.
- **Ceiling for whole-plug streaming: about 7.0 tok/s** at this link speed, regardless of how fast the compute gets. Reaching 10 tok/s (<= 100 ms/token) needs fewer bytes per token, roughly <= 1.1 GB at this bandwidth, so a selected fetch (or a smaller plug code) is required before overlap tuning can matter for the 10 tok/s target.
- Compute-only here (85.0 ms) is lower than the resident decode at 32 steps (98.8-102.3 ms); positions and run lengths differ and I have not reconciled the two.
- Short context only (positions 0-63); not a 2047-position figure. Not M6.
