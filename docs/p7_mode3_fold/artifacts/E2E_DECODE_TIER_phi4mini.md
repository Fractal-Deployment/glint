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

## Follow-up: stop re-fetching band A, overlap the gathers with the parent GEMV (branch `fix/p7-tier-gather-overlap`)
Reading the tier branch of `selfetch_linear` explained part of the gap above: it was the sequential, unoverlapped path, and its band B gather read the **union** of both bands (`nAll` channels, band A included) and only skipped band A in the correction, so the PCIe traffic was 50% of the plug (about 805 MB), not the 40% of T-A. Change (test harness only, `tests/test_arm0_bf16_full.cu`): `k_sf_gather` takes an optional skip mask so band B fetches only its own channels, and both gathers run on the second stream concurrently with the parent GEMV (the same pattern the single-band path uses).

| build | ms/token over 2-3 runs | tokens/s |
|---|---|---|
| before (this document's table) | 184.2; 211.8, 218.9 in two later runs | 5.4; 4.7, 4.6 |
| after | 163.1, 160.4, 170.5 | 6.1, 6.2, 5.9 |

- *Measured:* about 20-25% faster; **still 6 tokens/s, M6 acceptance (10+) still not met.** The remaining excess over the benchmark's 77 ms + 38 ms is unexplained (selection, two correction passes, 128+ launches per token, the unpacked 3-bit emulation).
- *Caveat on equivalence:* `DECODE_CHECK` (top-1 equal 105-107 of 112, bitwise-equal losses 0-2) varies from run to run in the **old** build too, so the decode path with the tier is not run-to-run deterministic (cause not investigated: the split-K sums are deterministic per the earlier work, so a stream ordering effect or a near-tie in greedy decoding is suspected, not established). The change was therefore checked for consistency with that existing spread, not for bit equality. A full rollout study of the new build was not rerun.

## Where the 164 ms goes: timing ablation (`ARM0_SF_ABL`, same build, T-A, 32 decode steps, one run each)
The flag skips phases in the tier branch (output is then wrong; **timing only**): bit 2 skips both gathers, bit 4 skips both corrections and sums, bit 8 skips band B entirely. (A bit that would skip the selection was not used: the gathers read its index list.)

| skipped | what remains | ms/token |
|---|---|---|
| nothing | everything | 164.4 |
| gathers | base + selection + corrections | 73.3 |
| corrections | base + selection + gathers | 141.7 |
| gathers and corrections | base + selection (+ stream/launch overhead) | 40.9 |
| band B (gather and correction) | band A only (10%) | 69.6 |

- *Measured:* base + selection costs 41 ms (the parent alone was 38.0 ms), so **selection is about 3 ms and cheap**. The **gathers cost about 100 ms** (141.7 - 40.9) for 644 MB of staged bytes (the 3-bit band is emulated at 4-bit size), about 6.4 GB/s; the **two correction passes cost about 32 ms** (73.3 - 40.9). The sum 41 + 100 + 32 = 173 ms is close to the measured 164 ms, so **the gather is not hidden by the parent GEMV** to any large degree.
- *Inference (arithmetic, not tested):* the dependency is per linear: the channels to fetch are chosen from that linear's own input `x`, which exists only after the previous linear finished. The only work that can overlap a gather is therefore that linear's parent GEMV (about 0.3 ms per linear, 38 ms per token in all). So the floor for this design is the fetch time of the bytes at the PCIe rate plus the correction time, minus at most the base GEMV time: at 523 MB packed and 6.4-8 GB/s that is 65-82 ms of fetch + 32 ms of correction, about 100-115 ms, i.e. **9-10 tokens/s at best for T-A even with a perfect kernel**, with no margin. The 10+ tokens/s target at tracking quality needs either fewer bytes (no tracking arm exists below 523 MB) or to take the fetch off the critical path by choosing channels **before** the layer's input exists (prefetch from the previous layer's or previous token's activations, as DecDEC/Deja Vu style predictors do). Whether such a prediction tracks is untested.
