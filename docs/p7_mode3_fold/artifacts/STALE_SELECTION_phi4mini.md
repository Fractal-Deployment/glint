# Can the channels be chosen before the layer runs? Stale (previous-token) selection, Phi-4-mini (2026-11-05) — glint #92

Follows `E2E_DECODE_TIER_phi4mini.md`: the per-linear dependency (channels chosen from this token's own input, so the fetch cannot start before the input exists) bounds the tiered design at about 9-10 tokens/s. A fetch that starts earlier needs the channels chosen from something already known. Test: `ARM0_SELFETCH_RULE=stale` ranks each linear's input channels by the |x| that the **same linear saw on the previous token** (kept per layer and linear), instead of by the current |x|. The correction still multiplies by the current x. This is the cheapest predictor (DecDEC / Deja-Vu style predictors from the previous step); it is not a learned predictor. Rule stated before the runs (in the session, not in the spec file): *stale selection tracks if token agreement with the full 4-bit model is at least 82.5% and the matched prefix at least 20, the same bar T-A had to meet in `TIERED_PLUG_phi4mini.md`.* Same rollout as there (12 prompts x 32 steps, reference = full 4-bit model, scored by the same prefix/agreement script). One run per arm. Raw: `/home/workspace/lakes/p7_glint/stale_20261105/`.

| Arm | bytes/token | matched prefix /32 | agreement | all-match |
|---|---|---|---|---|
| T-A, current |x| (earlier result) | ~523 MB | 29.33 | 91.7% | 10 / 12 |
| **T-A, stale** | ~523 MB | **14.08** | **51.0%** | 4 / 12 |
| 4-bit `act` 50%, current (earlier) | ~805 MB | 29.92 | 99.0% | 11 / 12 |
| **4-bit `act` 50%, stale** | ~805 MB | **19.00** | **71.6%** | 6 / 12 |
| 4-bit `act` 25%, current (earlier) | ~403 MB | 20.33 | 72.7% | 5 / 12 |
| 4-bit `act` 10%, current (earlier) | ~161 MB | 15.50 | 55.5% | 4 / 12 |
| random 25% (earlier) | ~403 MB | 11.42 | 38.8% | 2 / 12 |
| base only (fold, earlier) | 0 | 5.67 | 23.2% | 0 / 12 |

## Verdict
- **Stale selection does not track** under the stated rule: T-A 51.0% (< 82.5%) and 50% 71.6% (< 82.5%), prefix 14.1 and 19.0 (< 20).
- *Measured:* it is worth much more than random selection (T-A stale 51.0% against random 25% at 38.8% with fewer bytes), so the previous token's |x| carries information about this token's important channels. But it recovers roughly half of what the current-|x| selection does: stale T-A (523 MB) behaves like current-|x| `act` 10% (161 MB, 55.5%), and stale `act` 50% (805 MB) like current `act` 25% (403 MB, 72.7%).
- **Consequence for M6:** the way to take the fetch off the critical path with this predictor costs about the quality of the bytes saved, so no stale arm at or below 523 MB tracks. A predictor better than "same linear, previous token" (previous layer's activations, a learned one) is untested, as are bytes above 805 MB (which would already be below 10 tokens/s). M6 acceptance (tracks and at least 10 tokens/s) is not met by any arm measured.

## What this does not show
One run per arm; token-exact agreement carries about 8 points of numerical noise (the BF16 anchor against the full model scores 91.7%), which does not change the verdict (the gaps are 20-40 points). One text, 12 prompts. The stale buffer carries over between prompts (the first token of each prompt uses the last token of the previous prompt) and the first decode step ranks by zeros; both are minor against 64-token prompts but not measured. No speed measurement was taken with the stale rule. Not reviewed by the Test Designer.

## Reproduce
`ARM0_DECODE_ROLL=12,32,64,128 ARM0_DECODE_ROLL_OUT=out.txt ARM0_DEMAND_RULE=act ARM0_DEMAND_F=1 ARM0_SELFETCH=<k1> ARM0_SELFETCH_OPT=1 ARM0_SELFETCH_RULE=stale [ARM0_TIER_FRAC_B=<k2> ARM0_TIER_BITS_B=3] bf16full --glint-demand PIN MODEL ART` (branch `exp/p7-stale-selection`, stacked on `fix/p7-tier-overlap-stacked`).

## Second predictor: previous layer, same token (`ARM0_SELFETCH_RULE=prevlayer`)
Same rule and bar as above (agreement at least 82.5% and matched prefix at least 20), stated before the runs. Channels of a linear are ranked by this token's input to the **same linear one layer earlier** (layer 0 falls back to the previous token's input). The correction still uses the current x. One run per arm, same rollout.

| Arm | matched prefix /32 | agreement | all-match |
|---|---|---|---|
| T-A, previous layer | 11.08 | 49.0% | 3 / 12 |
| 4-bit `act` 50%, previous layer | 13.33 | 49.5% | 4 / 12 |

- **Does not track** (49.0% and 49.5% against 82.5%). It is also **worse than the previous-token predictor** for the 50% arm (49.5% against 71.6%) and about equal for T-A (49.0% against 51.0%); at 805 MB it sits near current-|x| `act` 10% (161 MB, 55.5%).
- *Measured:* the input of the same linear in the neighbouring layer predicts the important channels worse than the same linear's input on the previous token. *Not tested:* a learned predictor, a mix of both, or predictors keyed to the residual stream; the per-linear bound on speed (`E2E_DECODE_TIER_phi4mini.md`) stands, and no arm measured tracks and reaches 10 tokens/s.
