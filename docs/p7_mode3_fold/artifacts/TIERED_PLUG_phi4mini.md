# Tiered plug (4-bit top band + 3-bit next band + base-only rest), Phi-4-mini (2026-11-05) — spec Amendments 5 and 6, glint #92

Executes Amendments 5 and 6 of `docs/p7_mode3_fold/DEMAND_MAP_SPEC_2026-11-04.md` (arms and rules fixed before the data; Amendment 6 was written after arm T-A's result was known, which it discloses). It tests the recommendation of `RESEARCH_BRIEF_2026-11-05.md` (SpAx's three-tier idea on top of a resident base). Per token and linear: the top `k1` input channels by |x| receive the 4-bit plug correction, the next `k2` channels receive a 3-bit-code correction (merged-bin emulation, `ARM0_TIER_BITS_B=3`, a second constant table), every other channel is base-only. Same decode-path rollout as Amendment 3 (12 prompts x 32 steps, reference = the 4-bit full model). **Bytes are analytic** `(k1 + 0.75 k2) x 1.61 GB`; no packed mixed-width store or kernel exists, and the arms are run sequentially, so there is no speed measurement. Raw: `/home/workspace/lakes/p7_glint/tiered_20261105/`; ids `tiered_phi4mini/*.txt`.

| Arm | k1 (4-bit) | k2 (3-bit) | bytes/token | matched prefix /32 | agreement | all-match prompts |
|---|---|---|---|---|---|---|
| BF16 anchor (noise floor, Amendment 3) | | | | 26.67 | 91.7% | 9 / 12 |
| 4-bit `act` 50% (Amendment 3) | 50% | | ~805 MB | 29.92 | 99.0% | 11 / 12 |
| 3-bit `act` 50% (Amendment 4) | | 50% | ~604 MB | 26.17 | 86.2% | 9 / 12 |
| 4-bit `act` 25% (Amendment 3) | 25% | | ~403 MB | 20.33 | 72.7% | 5 / 12 |
| invariance: tier path, both bands 4-bit | 25% | 25% | ~805 MB | 27.17 | 90.9% | 10 / 12 |
| **T-A** | 10% | 30% | **~523 MB** | **29.33** | **91.7%** | **10 / 12** |
| T-B | 15% | 30% | ~604 MB | 23.17 | 84.4% | 8 / 12 |
| T-C (informational) | 25% | 25% | ~705 MB | 30.08 | 94.0% | 11 / 12 |
| F1 | 5% | 20% | ~322 MB | 19.92 | 70.8% | 4 / 12 |
| F2 | 10% | 15% | ~342 MB | 16.67 | 61.7% | 3 / 12 |
| F3 | 10% | 10% | ~282 MB | 15.83 | 58.9% | 3 / 12 |
| F4 | 5% | 10% | ~201 MB | 18.17 | 66.7% | 4 / 12 |
| F5 (almost 3-bit only) | 1% | 30% | ~378 MB | 23.00 | 72.7% | 6 / 12 |

## Verdict under the pre-registered rules
- **Amendment 5: the tiered plug improves the frontier.** T-A tracks the full model (matched prefix 29.33 >= 20.0, agreement 91.7% >= 82.5%) at about 523 MB, fewer bytes than the best tracking coarse-code arm (3-bit `act` 50%, 604 MB) and 35% fewer than the 4-bit `act` 50% arm (805 MB). (The second criterion, T-B beating the 3-bit arm by 5 points at equal bytes, is not met: 84.4% against 86.2%.)
- **Amendment 6: the byte target for 10+ tokens/s is not met.** The byte-feasible threshold was <= 285 MB; the arms at or below it (F3 282 MB, F4 201 MB) do not track (58.9%, 66.7%), and neither does any arm below 523 MB (F1 322 MB 70.8%, F2 342 MB 61.7%, F5 378 MB 72.7%).
- **The 4-bit top band is credited:** T-A exceeds the near-3-bit-only control F5 by 19.0 percentage points (91.7% vs 72.7%), well over the 5-point rule, with 40% of channels touched in both.
- **M6 acceptance (tracks AND >= 10 tokens/s) is still not met.** With the effective fetch rate of 4.0-4.9 GB/s measured for the optimized kernel, 523 MB per token would cost about 107-131 ms of fetch on top of the 39 ms base: roughly 6-7 tokens/s (arithmetic, assuming free unpacking of a packed mixed-width store).

## How much to trust the ranking among tracking arms
The invariance arm shows the measurement noise: the tier path with both bands at 4 bits is the same selection as the plain 4-bit `act` 50% arm (99.0%), but it scores 90.9% because splitting the correction into two partial sums changes float rounding and greedy decoding diverges at near-ties; the BF16 anchor against the full model scores 91.7% for the same reason. So agreement here carries about 8 points of numerical noise. The conclusions that survive it: T-A and T-C track (at or above the noise level, 91.7% and 94.0%), arms F1-F5 do not (59-73%, well below the 82.5% threshold), and the 4-bit band matters (a 19-point gap). The differences among T-A, T-B and the 3-bit 50% arm (91.7%, 84.4%, 86.2%) are not resolved: T-B has more bytes than T-A and a lower score, which is within that noise.

## What this does not show
Emulated reconstruction only; the speed figures are arithmetic from a fetch model, with the decode cost of a coded tier unmodelled (SpAx reports 52-64 ms of GPU time per token for its trellis). One text, 12 prompts, 32 steps, one run per arm; token-exact agreement counts near-ties as divergence. The candidate set is small (two band widths and three splits), not a search. Not reviewed by the Test Designer.

## Reproduce
`ARM0_DECODE_ROLL=12,32,64,128 ARM0_DECODE_ROLL_OUT=out.txt ARM0_DEMAND_RULE=act ARM0_DEMAND_F=1 ARM0_SELFETCH=<k1> ARM0_SELFETCH_OPT=1 ARM0_TIER_FRAC_B=<k2> ARM0_TIER_BITS_B=3 bf16full --glint-demand PIN MODEL ART` (code on branch `feat/p7-tiered-plug`).
