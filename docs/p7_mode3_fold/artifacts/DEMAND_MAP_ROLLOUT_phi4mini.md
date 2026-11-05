# Demand map, rollout: does the degraded model keep tracking the full model when it feeds on its own tokens? (Phi-4-mini, 2026-11-04)

Executes Amendment 2 of `docs/p7_mode3_fold/DEMAND_MAP_SPEC_2026-11-04.md` (rules fixed before any rollout data, glint #81). Greedy generation by repeated full forwards on a padded 2048-token buffer (causal attention makes the logits at the last real position independent of what follows; this is not a KV-cache decode path, which does not exist here). P = 3 prompts (the first 64 tokens of the evaluation text at offsets 0, 512, 1024), S = 16 greedy steps, 48 generated tokens per arm. Reference = the f = 1 arm (full plug). Harness: `ARM0_ROLLOUT=3,16,64` in `tests/test_arm0_bf16_full.cu`; generated ids per arm are in `rollout_phi4mini/*.txt`. Making repeated forwards possible needed a fix: `arm0_forward` never freed its ~1.5 GB of device buffers and ran out of memory on the fourth call.

| Arm (vs the f = 1 arm) | teacher-forced top-1 (PR #82) | matched prefix (of 16, mean) | agreement, all 48 | steps 1-8 | steps 9-16 | drop (pp) |
|---|---|---|---|---|---|---|
| BF16 CUDA anchor | 98.44% | 16.00 | 100.0% | 100.0% | 100.0% | 0.0 |
| fold only (f = 0) | 91.06% | 1.67 | 12.5% | 20.8% | 4.2% | 16.7 |
| `act` channel, f = 0.25 | 96.34% | 6.00 | 37.5% | 41.7% | 33.3% | 8.3 |
| `act` channel, f = 0.1 | 95.55% | 6.67 | 56.3% | 50.0% | 62.5% | -12.5 |
| `act` block, f = 0.25 | 94.19% | 7.00 | 58.3% | 54.2% | 62.5% | -8.3 |
| `random` channel, f = 0.1 (seed 42) | 91.55% | 6.67 | 43.8% | 50.0% | 37.5% | 12.5 |

## Verdict under the pre-registered rules
- *Tracks the full model through rollout* (matched prefix >= 0.75 of the anchor's, i.e. >= 12, and agreement >= 0.9 of the anchor's): **no arm other than the anchor meets it.** The activation-selected arms diverge after 6-7 tokens on average (agreement 38-58%), against 16 and 100% for the anchor.
- *Compounding* (drop more than 10 pp above the anchor's): flagged for fold only (16.7) and `random` channel f = 0.1 (12.5); not flagged for the three `act` arms (8.3, -12.5, -8.3).
- The anchor equals the full arm on all 48 tokens, so the noise floor from near-ties is zero on these prompts and the rules are strict.

## What this does and does not show
- *Measured:* teacher-forced top-1 overstates how closely a plug-reduced model follows the full model in free-running generation. Channel-selected 25% of the plug (96.34% teacher-forced) diverges from the full model after about 6 tokens, not the roughly 27 that independent 3.7% per-token errors would suggest. Fold only (91%) diverges almost immediately.
- *Not separable here:* `act` channel f = 0.1 (agreement 56%) and `random` channel f = 0.1 (44%) have the same mean matched prefix (6.67). With 3 prompts the matched prefix is three integers; the first divergence sits at tokens 1-2 for two of the three prompts in most arms, i.e. at positions where the full model's top-1 margin may be small. The teacher-forced gap between `act` and `random` (95.55% vs 91.55% top-1, KL 1.5e-2 vs 7.4e-2) is not visible in this rollout at this sample size, and this document does not claim it is or is not present.
- Generations degenerate into repeats (for example `19837 19837 19837 ...`, `1302 1302 ...`) on this text, which makes agreement easier to keep once a loop starts.
- Not shown: any KL along the generated path, a larger prompt set, longer rollouts, or a different text. A KV-cache decode path (needed for per-step activation-selected fetch) still does not exist.
