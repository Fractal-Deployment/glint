# Demand map, Phi-4-mini: how much plug does a step need? (2026-11-04)

Executes `docs/p7_mode3_fold/DEMAND_MAP_SPEC_2026-11-04.md` (glint PR #81) as written. Decision rules were fixed before any data. 59 arms, all exit 0. Phi-4-mini, `Phi-4-mini-glint-from-bf16`, 2047 tokens, oracle PPL 5.486957, teacher-forced, RTX 3060. Harness: `tests/test_arm0_bf16_full.cu --glint-demand` (PR stacked on #80). Per-arm table: `DEMAND_MAP_phi4mini_summary.tsv`. Raw logs, per-arm JSON, anchor logits and the static calibration file: `/home/workspace/lakes/p7_glint/demand_map_20261104/`.

Each linear is `y = x*Wparent^T + mask(x)*Wcorr^T` (both bf16, cuBLAS), `mask` keeping the top fraction `f` of 64-channel input blocks per token. KL is KL(BF16 CUDA anchor || arm); the anchor is `bf16full` on the same tokens (PPL 5.491616, top-1 2015/2047, reproduced).

## Sanity gates
- S1, `f = 1`: PPL 5.488540, top-1 2010/2047 (98.19%), KL mean 9.52e-4. Within 0.16% PPL of the H-TILE hole+plug run (5.479566). PASS (tolerance 0.5%).
- S2, `f = 0`: PPL 5.833352, top-1 1864/2047 (91.06%), KL mean 7.93e-2. Identical PPL and top-1 to the NF4 cuBLAS arm (PR #49), so the hole alone is exactly the NF4 result, and within 0.23% of the H-TILE fold-only run (5.846849). PASS.

Reference `A` = S1 (PPL 5.488540, top-1 98.19%, KL mean 9.52e-4).

## Results (block granularity, bytes = plug bytes fetched / all plug bytes; actual bytes round up to whole blocks)
| rule | f | bytes | PPL | top-1 | KL mean | KL p99 | KL max | Jaccard(prev token) | recall(prev token) |
|---|---|---|---|---|---|---|---|---|---|
| fold only | 0 | 0 | 5.8334 | 91.06% | 7.93e-2 | 0.603 | 14.18 | - | - |
| act | .01 | .0195 | 5.6461 | 93.26% | 4.33e-2 | 0.350 | 7.95 | .335 | .347 |
| act | .05 | .0605 | 5.6221 | 93.84% | 3.40e-2 | 0.283 | 0.70 | .307 | .403 |
| act | .1 | .1035 | 5.6146 | 93.94% | 3.01e-2 | 0.273 | 0.66 | .311 | .431 |
| act | .25 | .25 | 5.5938 | 94.19% | 2.27e-2 | 0.214 | 0.82 | .368 | .517 |
| act | .5 | .5 | 5.5624 | 95.36% | 1.39e-2 | 0.122 | 0.68 | .499 | .656 |
| oracle | .01 / .1 / .5 | | 5.6259 / 5.6187 / 5.5420 | 93.31 / 94.04 / 95.11% | 4.02e-2 / 2.99e-2 / 1.35e-2 | | 1.17 / 0.67 / 0.68 | | |
| static | .01 / .1 / .5 | | 5.6239 / 5.6269 / 5.5638 | 93.11 / 93.75 / 94.48% | 4.35e-2 / 3.59e-2 / 1.79e-2 | | 2.67 / 1.17 / 0.91 | 1 | 1 |
| random 42 | .01 / .1 / .5 | | 5.8323 / 5.7945 / 5.5116 | 91.30 / 91.55 / 95.02% | 7.80e-2 / 7.39e-2 / 2.46e-2 | | 14.3 / 14.3 / 6.55 | .018 / .059 / .336 | .02 / .104 / .5 |
| random 43 | .01 / .1 / .5 | | 5.8417 / 5.8071 / 5.4808 | 91.50 / 91.40 / 94.68% | 7.86e-2 / 7.48e-2 / 2.30e-2 | | 14.2 / 14.2 / 1.13 | | |
| full plug | 1 | 1 | 5.4885 | 98.19% | 9.52e-4 | 0.0078 | 0.091 | - | - |
(All rows for every f are in the summary TSV.)

## Verdict under the pre-registered rules
*Sufficient* = top-1 >= 95.0% and PPL within 2.0% of `A` (<= 5.598) and KL mean <= 2x `A` (<= 1.9e-3).
- `act` at f = 0.5: top-1 95.36% (pass), PPL +1.35% (pass), KL mean 1.39e-2 = 14.5x `A` (**fail**). At f <= 0.25 top-1 is below 95% as well. No f < 1 is sufficient for `act` at block granularity, the granularity the rule names; channel granularity is reported in its own section below.
- **Premise supported** (some f* <= 0.25 sufficient, act beats random): **not met.**
- **Premise falsified** ("act is not sufficient at f = 0.5", or act within noise of random at every f in [0.05, 0.5]): **the first condition is met; the second is not** (act separates from random at f <= 0.1 by far more than the seed spread).
- So, as written, the premise is not supported on this model for the plug alone, block-granular, teacher-forced, one text. The KL bar (2x `A`) is the agent-proposed part and is strict: it alone fails f = 0.5. By the operator's own top-1 bar (95.0%) f = 0.5 passes narrowly and every f <= 0.25 fails. The operator or Test Designer may strike or replace the bar; the numbers above do not change.

## Measured observations (what the data shows, separate from the verdict)
1. **Graded, not a cliff, and diffuse.** KL falls 7.93e-2 -> 4.33e-2 -> 3.01e-2 -> 2.27e-2 -> 1.39e-2 at 0 / 2 / 10 / 25 / 50% of plug bytes, then 9.5e-4 at 100%. The last half of the blocks removes more KL than the first half. *Inference:* the plug's value is spread across input channels rather than concentrated in a few.
2. **Activation selection removes the catastrophic tail.** Fold-only and `random` keep max token KL near 14; `act` at 6% of bytes cuts it to 0.70, `oracle` is similar. At 2% bytes it is still 7.9.
3. **Activation norm already captures the choice.** `oracle` (activation norm times the true correction's column norm) is within 7% KL, 0.4% PPL and 0.3 points of top-1 of `act` at every f tested: no meaningful gain from knowing the correction's size. (Only `random` was run with two seeds, so there is no seed spread for `act` or `oracle`; they are deterministic.)
4. **A fixed hot set matches `act` at <= 2% bytes** (top-1 93.11% vs 93.26%; KL 4.35e-2 vs 4.33e-2) and is 19-29% worse in KL at >= 10%, but never changes between tokens. The calibration tokens (first 256) are inside the evaluation text, so this row is in-sample.
5. **Weak temporal predictability.** The previous token's `act` set at the same linear overlaps the current set by Jaccard 0.31-0.50 and recalls 35-66% of it (random: 2-50%, equal to f). A prefetch on that prediction alone would miss 57% of the blocks at f = 0.1.
6. **Layers differ in sensitivity but errors add.** Restricting only one layer to f = 0.1 (others full): KL mean 1.4e-3 (layer 24) to 3.4e-3 (layer 30) and 6.2e-3 (layer 31, top-1 96.73%); layers 0-1 are also higher (2.5e-3, 2.2e-3). The 32 single-layer excesses over `A` sum to 3.16e-2; the all-layers f = 0.1 arm shows 2.91e-2, so the excess is additive to within 9%. *Inference:* a per-layer fraction budget is a sound way to allocate, and the last two layers need the most.
7. **PPL alone cannot rank rules at high f.** At f = 0.5 `random` seeds 42 and 43 give PPL 5.5116 and 5.4808, one below `A`; top-1 and KL were used.

## Channel granularity (the spec's deferred secondary; run after the block sweep, `act` and `random` only)
Same arms with the selection unit a single input channel (the DecDEC-comparable form): the top fraction `f` of a token's channels by |x| (`act`) or by a hash (`random`, seed 42). Thresholds are found by a bit-pattern binary search, so ties can keep a few extra channels (actual bytes in the table).

| rule | f | bytes | PPL | top-1 | KL mean | KL p99 | KL max |
|---|---|---|---|---|---|---|---|
| act | .5 | .501 | 5.4853 | 97.75% | 2.18e-3 (2.29x A) | 0.0183 | 0.066 |
| act | .25 | .251 | 5.4755 | 96.34% | 7.04e-3 | 0.0638 | 0.43 |
| act | .1 | .101 | 5.5102 | 95.55% | 1.52e-2 | 0.134 | 0.53 |
| act | .05 | .0506 | 5.5451 | 95.16% | 2.07e-2 | 0.176 | 1.06 |
| act | .01 | .0102 | 5.6037 | 94.09% | 3.04e-2 | 0.258 | 0.73 |
| random | .5 / .25 / .1 / .05 / .01 | same | 5.7234 / 5.7969 / 5.8030 / 5.8304 / 5.8420 | 92.53 / 91.70 / 91.55 / 91.60 / 91.16% | 5.61e-2 / 6.61e-2 / 7.39e-2 / 7.56e-2 / 7.85e-2 | | 14.2 at every f |

- Against the same sufficiency rule: `act` at f = 0.5 passes top-1 and PPL and misses the KL bar by 14% (2.29x vs 2x). At f = 0.1 it passes top-1 (95.55%) and PPL (+0.4%) and misses KL by 16x. The pre-registered premise verdict above is for block granularity and is unchanged; these rows are reported, not used to overturn it.
- Measured, descriptive: the same fraction of bytes buys much more at channel granularity than at block granularity (at 10% of bytes: top-1 95.55% vs 93.94%, KL 1.52e-2 vs 3.01e-2; at 50%: KL 2.18e-3 vs 1.39e-2). Dropping only the top-1 + PPL conditions, the smallest fraction passing is f = 0.05 (5% of plug bytes, top-1 95.16%).
- *Limit of the byte count:* it is idealized. The plug is stored row-major by output row, so one input channel's nibbles are scattered across every row. Fetching single channels over PCIe needs a column-major (K-major) plug layout that this pin does not have; block granularity (64 channels) is the contiguous-fetch case.

## Exploratory, not pre-registered: skipping inputs for the whole linear (hole included)
A TEAL-style test of "the model does not need all of itself": with `ARM0_DEMAND_MASK_PARENT=1` the parent GEMM also sees the masked input, so low-activation inputs are skipped for the 4-bit hole as well as for the plug. Run after the pre-registered results above were fixed; one run per arm; uniform fraction across all linears and layers.

| unit | rule | f | PPL | top-1 | KL mean |
|---|---|---|---|---|---|
| block (64 ch) | act | .9 / .75 / .6 / .5 | 6.80 / 51.5 / 9.1e5 / 3.1e9 | 83.83 / 47.48 / 0.15 / 0.05% | 0.25 / 2.4 / 12.1 / 20.1 |
| channel | act | .9 | 5.4792 | 97.12% | 2.48e-3 |
| channel | act | .75 | 5.5846 | 93.94% | 2.60e-2 |
| channel | act | .6 | 5.8093 | 88.76% | 0.111 |
| channel | act | .5 | 6.3103 | 84.56% | 0.228 |
| channel | random | .75 / .5 | 2.7e6 / 3.1e12 | 0.44 / 0.00% | 13.2 / 27.2 |

- Measured: skipping the lowest 10% of channels (by |x|, per token, per linear) costs little (top-1 97.12%, KL 2.6x the full arm); skipping 25% already drops top-1 below the 95% bar; skipping 40-50% fails clearly on this model with uniform fractions. Zeroing whole 64-channel blocks collapses the model even at 10%.
- This is not a replication of TEAL (arXiv 2408.14690), which reports 40-50% sparsity with little loss on Llama/Mistral models with a calibrated per-matrix allocation; here the fraction is uniform, the model is a 3.8B model on a 4-bit-hole base, and the text is one 2047-token passage. It says that on this model, with a uniform fraction, the hole cannot be skipped as freely as the plug can.

## What this does not show
- Teacher-forced prefill, one token's selection independent of the others; no rollout, so no compounding of errors through generated tokens.
- Fidelity against bytes, not time; no speed claim. One model, one text. `static` and `oracle` were not run at channel granularity (the spec limits it to `act` and `random`).
- `static` uses tokens inside the evaluation text for calibration.

## Reproduce
Build as in `FOLD_STREAM_phi4mini.md`. Anchor: `ARM0_LOGITS_OUT=anchor.bin bf16full MODEL ART`. An arm: `ARM0_DEMAND_RULE=act ARM0_DEMAND_F=0.1 ARM0_DEMAND_STATS=a.json ARM0_LOGITS_REF=anchor.bin bf16full --glint-demand PIN MODEL ART`. Static: first `ARM0_DEMAND_F=1 ARM0_DEMAND_STATIC_OUT=calib.bin`, then `ARM0_DEMAND_RULE=static ARM0_DEMAND_STATIC_IN=calib.bin`. Layer restriction: `ARM0_DEMAND_ONLY_LAYER=L`. Channel granularity: `ARM0_DEMAND_GRAN=channel`. Whole-linear skip: `ARM0_DEMAND_MASK_PARENT=1`.
