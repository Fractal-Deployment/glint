
## Follow-up: stop re-fetching band A, overlap the gathers with the parent GEMV (branch `fix/p7-tier-gather-overlap`)
Reading the tier branch of `selfetch_linear` explained part of the gap above: it was the sequential, unoverlapped path, and its band B gather read the **union** of both bands (`nAll` channels, band A included) and only skipped band A in the correction, so the PCIe traffic was 50% of the plug (about 805 MB), not the 40% of T-A. Change (test harness only, `tests/test_arm0_bf16_full.cu`): `k_sf_gather` takes an optional skip mask so band B fetches only its own channels, and both gathers run on the second stream concurrently with the parent GEMV (the same pattern the single-band path uses).

| build | ms/token over 2-3 runs | tokens/s |
|---|---|---|
| before (this document's table) | 184.2; 211.8, 218.9 in two later runs | 5.4; 4.7, 4.6 |
| after | 163.1, 160.4, 170.5 | 6.1, 6.2, 5.9 |

- *Measured:* about 20-25% faster; **still 6 tokens/s, M6 acceptance (10+) still not met.** The remaining excess over the benchmark's 77 ms + 38 ms is unexplained (selection, two correction passes, 128+ launches per token, the unpacked 3-bit emulation).
- *Caveat on equivalence:* `DECODE_CHECK` (top-1 equal 105-107 of 112, bitwise-equal losses 0-2) varies from run to run in the **old** build too, so the decode path with the tier is not run-to-run deterministic (cause not investigated: the split-K sums are deterministic per the earlier work, so a stream ordering effect or a near-tie in greedy decoding is suspected, not established). The change was therefore checked for consistency with that existing spread, not for bit equality. A full rollout study of the new build was not rerun.
