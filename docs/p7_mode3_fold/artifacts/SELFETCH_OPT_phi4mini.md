# Selected-channel fetch, optimized: K-major hole + absmax, staged gather, fetch overlapped with compute (2026-11-04) — glint #86, operator target 10+ tokens/s

Follows `SELFETCH_DECODE_phi4mini.md`, where the correction kernel was the bottleneck (effective 2.2 GB/s against a 7-11 GB/s bus). Three changes, enabled by `ARM0_SELFETCH_OPT=1`:
1. **K-major hole and absmax in VRAM** (hole `[K][N/2]`, absmax `[K/64][N]`) next to the existing row-major copies, so a warp reading one selected channel reads contiguous bytes. The parent GEMV still uses the row-major hole.
2. **Staged gather**: `k_sf_gather` copies the selected channels' plug from pinned host memory into a device staging buffer with 16-byte loads (the 7 GB/s pattern from `FOLD_BUDGET_FETCH_phi4mini.md`), then `k_sf_corr2` reads the staging buffer, the hole and the absmax coalesced; its channel list is split four ways with a deterministic partial-sum buffer.
3. **Overlap**: the gather runs on a second stream concurrently with the parent GEMV of the same linear (the selection depends only on `x`), joined by an event before the correction.

## Correctness
Against the same double-precision reference with the same selection, 12,160 linears at f = 0.10: largest error **1.9e-5** of the output RMS (mean 1.4e-6). Bytes per token unchanged (162.7 MB, 10.10%). The 16 generated tokens at f = 0.10 equal the masked-input emulation's for the same prompt exactly (`220 16 4066 410 23287 15658 11121 483 10664 142760 220 15 4066 410 23287 15658`), including the point where the first implementation split from it.

## Speed (Phi-4-mini decode, same protocol as before, 15 timed steps, one run each)
| plug fetched | before (ms/token, tokens/s) | **optimized** ms/token | **tokens/s** | extra over parent-only (39.1 ms) | effective extra-byte rate |
|---|---|---|---|---|---|
| 5% (81.5 MB) | 76.2, 13.1 | **55.9** | **17.9** | 16.8 ms | 4.9 GB/s |
| 10% (162.7 MB) | 109.0, 9.2 | **80.2** | **12.5** | 41.1 ms | 4.0 GB/s |
| 25% (404.5 MB) | 207.8*, 4.8* | **124.8** | **8.0** | 85.7 ms | 4.7 GB/s |
| 50% (~805 MB) | not run | **202.5** | **4.9** | 163.4 ms | 4.9 GB/s |
| (parent-only, no plug) | 39.1, 25.6 | | | | |
| (whole plug streamed through the ring) | 147.9, 6.8 | | | | |
\* included a host synchronization for byte counting.

- **The operator's target of 10+ tokens/s is met at 5% and 10% of the plug, not at 25%.** The kernel is now limited by the bus, not by itself: the extra time per byte is 4-5 GB/s, near the 7 GB/s the gather reached alone, and the overlap with the 1.0 ms/layer parent GEMV hides part of it (at 5% the extra cost is 16.8 ms against the 12 ms the bytes alone would take at 7 GB/s plus launch overheads).
- Free-running quality at these fractions is **not** established: the rollout study (`DEMAND_MAP_ROLLOUT_phi4mini.md`) found activation-selected arms at 10-25% of the plug diverge from the full model after 6-7 tokens; speed and fidelity are separate results here.
- Memory: this harness holds the row-major hole and plug on the device for prefill and the reference, plus the K-major hole (1.8 GB) and absmax (0.2 GB) for this path. A deployment would choose one layout; no VRAM saving is claimed for this arm.

## What this does not show
No prefetch of the selected set across ops (the set depends on the previous op's output), no tuning of the split factor, one text, 15 timed steps, one run per arm. Not reviewed by the Test Designer.

## Reproduce
`ARM0_DEMAND_RULE=act ARM0_DEMAND_F=1 ARM0_SELFETCH=0.1 ARM0_SELFETCH_OPT=1 [ARM0_SELFETCH_CHECK=1] ARM0_DECODE=64,16 bf16full --glint-demand PIN MODEL ART`. Logs `/home/workspace/lakes/p7_glint/selfetch_opt_20261104/`.
