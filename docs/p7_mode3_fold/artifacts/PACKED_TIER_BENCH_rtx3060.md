# Packed mixed-width plug store: measured fetch + unpack time, RTX 3060 (2026-11-05) — spec Amendment 9, glint #92

Replaces one assumption in `TIERED_PLUG_phi4mini.md`: its speed figure ("523 MB at 4.0-4.9 GB/s, about 6-7 tokens/s, assuming free unpacking of a packed mixed-width store") was arithmetic. Here a real packed store and a fused fetch+unpack+accumulate kernel are timed. **Speed only**: data are random and the output is not checked against anything, so nothing here says whether the arms track the full model (that is the rollout study). Not pre-registered (no decision threshold was written before the data); reported as measurements.

Setup (`tests/bench_packed_tier.cu`): all 32 layers x 4 linears of Phi-4-mini (QKV, O, GATE_UP, DOWN) in pinned mapped host memory, separate regions per layer. Band A channel = n_out 4-bit codes (n_out/2 bytes); band B channel = n_out 3-bit codes packed 8 per 3 bytes (3 n_out/8 bytes). One token = 128 kernel launches on one stream, each reading the selected channels directly from host memory over PCIe, unpacking via a 16-entry table, accumulating `x[c]*cell` in fp32 (split-K, 8-way atomics). Selection is a fixed-seed random subset (counts and bytes are real, which channels is not). Raw: `/home/workspace/lakes/p7_glint/packed_20261105/`.

| Arm | bytes/token | ms/token (run 1 / run 2) | GB/s | plug-only tokens/s |
|---|---|---|---|---|
| 4-bit `act` 50% | 805 MB | 98.4 / 101.8 | 7.9-8.2 | 9.8-10.2 |
| 4-bit `act` 25% | 403 MB | 48.2 / 46.4 | 8.4-8.7 | 20.8-21.6 |
| **T-A** 10% 4-bit + 30% packed 3-bit | **523 MB** | **77.3 / 77.3** | 6.8 | **12.9** |
| T-A with band B stored as 4-bit (control) | 644 MB | 78.0 / 79.4 | 8.1-8.3 | 12.6-12.8 |
| F3 10% + 10% packed 3-bit | 282 MB | 35.8 / 36.3 | 7.8 | 27.5-27.9 |
| F4 5% + 10% packed 3-bit | 201 MB | 26.4 / 26.2 | 7.6-7.7 | 37.9-38.2 |
| T-C 25% + 25% packed 3-bit | 705 MB | 96.4 / 100.0 | 7.1-7.3 | 10.0-10.4 |

## Reading
- *Measured:* the fused fetch reaches 7-8.7 GB/s, well above the 4.0-4.9 GB/s of the earlier separate gather (`SELFETCH_OPT_phi4mini.md`; the PCIe Gen3 x16 ceiling measured earlier is 11.6 GB/s). The 523 MB of T-A costs **77 ms**, not the 107-131 ms of the arithmetic.
- *Measured:* packing to 3 bits cuts the bytes by 19% (644 to 523 MB) but **not the time**: the control with band B as 4-bit takes 78-79 ms. The 3-bit path reaches only 6.8 GB/s because each thread reads three unaligned bytes and unpacks 8 codes; at equal time, the byte saving is eaten by the unpack. A vectorized, aligned layout might recover it (untested).
- **Speed bound for the best tracking arm (T-A):** 12.9 tokens/s for the plug traffic alone. If the parent GEMV (39 ms in `GEMV_DECODE_phi4mini.md`) cannot be overlapped, the serial sum is 77 + 39 = 116 ms, about **8.6 tokens/s**; if the plug traffic is fully hidden under the parent GEMV, the bound is the 77 ms of fetch, 12.9 tokens/s. The truth is in between and is **not measured here**: this benchmark has no parent GEMV, no selection kernel, no absmax scaling, and no accuracy check.
- Arms F3 and F4 would exceed 25 tokens/s on bytes, but they do not track the full model (rollout study), so their speed does not count toward acceptance.

## What this does not show
Synthetic data, one random selection, two runs of the benchmark (spread 0-4% per arm); `act` selection and the parent GEMV are absent, so the end-to-end rate is a bound, not a measurement; the 8 split-K atomics, kernel launch gaps (128 launches per token) and the pinned-memory read path are all in the numbers, but their share is not separated. M6 acceptance (tracks AND at least 10 tokens/s, measured end to end) is still not demonstrated: T-A tracks and its plug traffic alone fits under 100 ms, but the combined end-to-end rate was not measured. Not reviewed by the Test Designer.

## Reproduce
`nvcc -O2 -arch=sm_86 -o bench tests/bench_packed_tier.cu && ./bench 20` inside `40-Runtime-Deps` under the GPU lock (branch `bench/p7-packed-tier`).
