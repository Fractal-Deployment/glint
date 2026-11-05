# Free-running fidelity of the selected-fetch decode: 12 prompts x 32 steps (spec Amendment 3, 2026-11-04) — glint #92 (M6)

Executes Amendment 3 of `docs/p7_mode3_fold/DEMAND_MAP_SPEC_2026-11-04.md` (rules and arms fixed before any data, glint #81). Greedy generation through the KV-cache decode path (`ARM0_DECODE_ROLL=12,32,64,128`): 12 prompts (64 tokens of the evaluation text at offsets 0, 128, ..., 1408), 32 generated tokens each, 384 tokens per arm; the reference is the full model (the decode GEMV on hole + plug). Arms use the optimized selected-channel fetch of `SELFETCH_OPT_phi4mini.md` (`act` = top-f input channels by |x|; `rnd` = the same count chosen by a seeded hash); policy A = f = 0.50 in layers {0, 1, 2, 29, 30, 31} and f = 0.05 elsewhere (average 13.4% of the plug), chosen from the per-layer sensitivity study before this run. Phi-4-mini, `glint_pin_v1`, RTX 3060, one run per arm. Generated ids: `decode_rollout_phi4mini/*.txt`; table: `decode_rollout_phi4mini/summary.tsv`.

| Arm (vs the full model) | plug fetched | matched prefix (of 32, mean) | agreement, all 384 | steps 1-16 | steps 17-32 | prompts matching all 32 | decode tokens/s |
|---|---|---|---|---|---|---|---|
| BF16 CUDA anchor (noise floor) | n/a | 26.67 | 91.7% | 95.8% | 87.5% | 9 / 12 | 32.3 |
| fold only | 0 | 5.67 | 23.2% | 34.4% | 12.0% | 0 / 12 | 25.6 |
| `act` | 5% | 9.25 | 43.8% | 50.5% | 37.0% | 2 / 12 | 17.9 |
| `act` | 10% | 15.50 | 55.5% | 65.6% | 45.3% | 4 / 12 | 12.5 |
| `act` | 25% | 20.33 | 72.7% | 81.8% | 63.5% | 5 / 12 | 8.0 |
| **`act`** | **50%** | **29.92** | **99.0%** | 99.0% | 99.0% | **11 / 12** | 4.9 |
| `rnd` | 10% | 10.17 | 34.1% | 42.7% | 25.5% | 2 / 12 | 12.5 |
| `rnd` | 25% | 11.42 | 38.8% | 51.0% | 26.6% | 2 / 12 | 8.0 |
| per-layer policy A | 13.4% avg | 18.67 | 65.1% | 72.4% | 57.8% | 5 / 12 | **11.5** (87.0 ms/token) |
Speeds are from `SELFETCH_OPT_phi4mini.md` (same fractions, 15 timed steps); policy A was timed separately with the same protocol; `rnd` shares the speed of `act` at the same fraction (same bytes).

## Verdict under the pre-registered rules
- *Tracks the full model* (matched prefix >= 0.75 x the anchor's = 20.0 and agreement >= 0.9 x the anchor's = 82.5%): **only `act` at 50%** (29.92, 99.0%), which matches the full model better than the BF16 anchor does (91.7%). `act` at 25% passes the prefix test (20.33) and fails the agreement test (72.7%). Everything below 25%, policy A and fold only fail.
- *Activation selection separable from random* (>= 10 percentage points): **yes at both fractions tested**: 10%: 55.5% vs 34.1% (+21.4 points); 25%: 72.7% vs 38.8% (+33.9 points). This reverses the "not separable" reading of Amendment 2, which came from 3 prompts and 16 steps; with 12 x 32 the difference is large.
- *M6 acceptance* (glint #92: tracks the full model **and** decodes at >= 10 tokens/s): **not met.** The one arm that tracks decodes at 4.9 tokens/s; the arms at or above 10 tokens/s (5%, 10%, policy A) do not track.

## What the data says
1. **Free-running fidelity needs far more of the plug than teacher-forced top-1 suggested**: teacher-forced top-1 at 10% was 95.55% (`DEMAND_MAP_phi4mini.md`), yet only 4 of 12 prompts reproduce the full model for 32 tokens. Agreement climbs from 23% (no plug) to 99% (half the plug), with the steepest rise between 25% and 50%.
2. **Activation selection carries real information**: at equal bytes it beats a random choice of channels by 21-34 points, and fold-only (no plug) is the floor.
3. **A per-layer allocation helps modestly**: policy A (13.4% average bytes) reaches 65.1% against 55.5% for a uniform 10% and 72.7% for a uniform 25%; it sits above the line between those two points for its byte count, so the sensitivity ranking from the demand map is informative but does not move the 50% requirement.
4. **The gap to the target is bytes per token, i.e. bus time**: tracking needs about 50% of the plug (about 805 MB per token); at the 7 GB/s the gather reaches, that is about 115 ms, so even a perfectly overlapped fetch caps this arm near 6-9 tokens/s on this PCIe Gen3 x16 link (11.6 GB/s measured best case). The two levers that remain are fewer bytes per fetched channel (the M5 compression direction: a plug coded in fewer than 4 bits) and a smarter allocation than uniform or the one hand-chosen layer set; neither was run here.

## What this does not show
12 prompts of one text and 32 steps; the decode path's arithmetic (m = 1 kernels) differs from the padded-forward rollout of Amendment 2, so the numbers are compared within this study only. No KL along the generated path. Policy A is one hand-chosen allocation. Agreement is token-exact, which makes near-ties count as divergence (the anchor itself scores 91.7%). Not reviewed by the Test Designer.

## Reproduce
`ARM0_DECODE_ROLL=12,32,64,128 ARM0_DECODE_ROLL_OUT=out.txt ARM0_DEMAND_RULE=act ARM0_DEMAND_F=1 ARM0_SELFETCH=<f> ARM0_SELFETCH_OPT=1 [ARM0_SELFETCH_RULE=random] [ARM0_SELFETCH_HI_LAYERS=0,1,2,29,30,31 ARM0_SELFETCH_HI_F=0.5] bf16full --glint-demand PIN MODEL ART`; the anchor is `bf16full MODEL ART` with `ARM0_DECODE_ROLL`. Logs: `/home/workspace/lakes/p7_glint/decode_rollout_20261104/`.
