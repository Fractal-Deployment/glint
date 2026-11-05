# End-to-end decode rate of the tiered plug in the real decode path, Phi-4-mini (2026-11-05) — glint #92, M6 acceptance

Answers the question `PACKED_TIER_BENCH_rtx3060.md` left open: that benchmark bounded the end-to-end rate between 8.6 and 12.9 tokens/s but had no parent GEMV, selection kernel or scaling. Here the real harness decode path (KV cache, glint GEMV for the parent, selected-channel fetch with the optimized staged-gather overlap, `--glint-demand`) is timed with the T-A configuration. `ARM0_DECODE=64,48` (64-token prompt, 48 decode steps), one run per arm, same build as the induction probe (branch `feat/p7-tiered-plug`). Raw: `/home/workspace/lakes/p7_glint/e2e_20261105/`. Not pre-registered.

| Arm | ms/token | tokens/s |
|---|---|---|
| parent only (about 0.01% fetched) | 38.0 | 26.3 |
| T-A: 10% 4-bit + 30% 3-bit-merged cells | **184.2** | **5.43** |
| 4-bit `act` 50% | 212.5 | 4.71 |

## Reading
- *Measured:* **the best tracking arm decodes at 5.4 tokens/s end to end in the current implementation. M6 acceptance (at least 10 tokens/s while tracking) is not met.**
- *Measured:* the extra over the parent-only path is 146 ms for T-A and 174 ms for `act` 50%, about twice the 77 ms and 100 ms the fused packed-store benchmark needs for the same bytes (`PACKED_TIER_BENCH_rtx3060.md`). The benchmark therefore shows what the fetch can cost; this path does not reach it.
- *Not measured, so not claimed:* where the extra 70-75 ms goes. Candidates are the channel selection per linear, the two-band staging (the band B table is applied from a 4-bit-sized staging buffer, not a packed one), the lack of overlap with the parent GEMV across layers, and 128 launches plus syncs per token. Closing the gap needs the fused kernel of the benchmark inside the decode path, with selection and absmax scaling.
- The code path here emulates the 3-bit band by merged cells (bytes are not actually reduced), so this is the rate of the emulation, not of a packed store. The benchmark shows packing would not change the time per byte read.

## What this does not show
One run per arm, 47 decode steps; the harness prints one mean. The 3-bit band is emulated. No accuracy is reported here (it is the rollout study's `TIERED_PLUG_phi4mini.md`).
