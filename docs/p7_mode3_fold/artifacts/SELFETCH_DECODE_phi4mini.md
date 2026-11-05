# Selected-channel fetch decode, Phi-4-mini (2026-11-04) — glint #86

`ARM0_SELFETCH=f` in `tests/test_arm0_bf16_full.cu` (with `--glint-demand`, whose prefill is the fast cuBLAS path; decode is the path of `DECODE_PATH_phi4mini.md`). For each one-token linear: (1) `k_glint_gemv` with the parent table over the resident row-major 4-bit hole; (2) `k_sf_select`, a single block that picks the top fraction `f` of the token's input channels by |x| (bit-pattern binary search, then compaction to an index list); (3) `k_sf_corr`, a thread per output row that, for each selected channel, reads that channel's plug nibble **directly from a K-major copy of the plug in pinned host memory over PCIe (zero-copy)** and adds `(GLINT_CELLS[h][p] - PARENT[h]) * absmax * x[c]`. Nothing is masked: bytes that are not selected are never read. The K-major copy is built once at load (a nibble transpose of the pin's plug; 1.8 GB pinned).

## Correctness
Every one of 12,160 linear calls of a decode run (f = 0.10) was compared with a one-thread-per-row double-precision reference built from the same selection: largest error **2.2e-5** of the output RMS, mean **1.3e-6** (the same accuracy as the plain GEMV in `GEMV_DECODE_phi4mini.md`). Bytes fetched match the fraction: 81.5 MB per token at f = 0.05 (5.06% of the 1.61 GB plug), 162.7 MB at 0.10 (10.10%), 404.5 MB at 0.25 (25.11%).

## Results (64-token prompt, 80 positions teacher-forced against the full-model prefill of the same run, 15 free-running steps timed, one run each)
| f (plug fetched) | plug MB per token | top-1 equal to full prefill (of 80) | ms/token | ms/layer | tokens/s |
|---|---|---|---|---|---|
| 0 (fold only, parent GEMV) | 0 | 64 | 39.1 | 1.22 | 25.6 |
| 0.05 | 81.5 | 73 | 76.2 | 2.38 | 13.1 |
| 0.10 | 162.7 | 72 | 109.0 | 3.41 | 9.2 |
| 0.25 | 404.5 | 73 | 207.8* | 6.49 | 4.8* |
| 1.0 (full model, resident GEMV) | n/a | 74 | 40.3 | 1.26 | 24.8 |
| (whole plug streamed through the ring, `GEMV_DECODE_phi4mini.md`) | 1610.6 | 76 | 147.9 | 4.62 | 6.8 |
\* the f = 0.25 timing comes from a run that also copied the selected count to the host after every linear for byte counting, so it includes that synchronization; f = 0.05 and 0.10 were also run that way (68.4 ms at 0.05), which shows the run-to-run spread (about 10%).

## Reading
- *Measured:* real selected fetch works end to end. At 5% of the plug it gives 13.1 tokens/s, about 1.9x the whole-plug stream (6.8) and half the resident rate (24.8); at 10% 9.2 tokens/s.
- *Measured:* the fetch is not PCIe-bound here. Going from f = 0 to 0.05 adds 37 ms per token for 81.5 MB, an effective 2.2 GB/s (0.10: 70 ms for 163 MB, 2.3 GB/s) against the 7-11 GB/s the fetch benchmark reached. The cost is the correction kernel itself: one thread per output row walks the selected list and reads the row-major hole and the absmax at scattered addresses, one sector per element. The hole and absmax would need a K-major layout (or a tiled kernel) to approach the bus rate. At the benchmark's 7 GB/s the 5% fetch would take about 12 ms per token instead of 37.
- *Not conclusive:* agreement with the full prefill at 80 positions is 73/80 at 5% against 74/80 for the full-model decode itself, so on these 80 positions the selected arms sit within one position of the arithmetic-noise ceiling; this is a small sample from the first 80 tokens and a different metric from the 2047-position top-1 versus the oracle in `DEMAND_MAP_phi4mini.md` (channel f = 0.05: 95.16%).
- *Cross-check against the emulation:* the generation at f = 0.10 matches the masked-input rollout arm for the same prompt for its first 6 tokens (`220 16 4066 410 23287 15658`) and then splits at a near-tie (11121 vs 5424 in this run's exact f32 arithmetic versus the emulation's bf16 rounding). The rollout study (`DEMAND_MAP_ROLLOUT_phi4mini.md`) found activation-selected arms at 10-25% of the plug do not track the full model through 16 generated tokens; this path inherits that: speed is shown, free-running fidelity is not.

## What this does not show
- This harness keeps the full plug resident on the device too (prefill and the reference need it), so no VRAM saving is claimed for this arm; the decode itself never reads the device plug.
- A hole and absmax K-major layout, kernel tuning, overlap of fetch with the parent GEMV, and fetch for the plug of the next layer ahead of time are not done. The selection uses |x| of the current linear's input, so the set is known only after the previous op finishes (no prefetch of it yet).
- 80 teacher-forced positions and 15 timed steps of one text, one run per arm. Not reviewed by the Test Designer.

## Reproduce
Build as in `FOLD_STREAM_phi4mini.md`. `ARM0_DEMAND_RULE=act ARM0_DEMAND_F=1 ARM0_SELFETCH=0.05 [ARM0_SELFETCH_STATS=1] [ARM0_SELFETCH_CHECK=1] ARM0_DECODE=64,16 bf16full --glint-demand PIN MODEL ART`. Logs: `/home/workspace/lakes/p7_glint/selfetch_20261104/`; generated ids per arm: `selfetch_phi4mini/*.txt`.
