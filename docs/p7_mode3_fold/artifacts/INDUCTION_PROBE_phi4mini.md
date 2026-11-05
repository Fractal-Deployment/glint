# Induction (copy-retrieval) probe, Phi-4-mini (2026-11-05) — research brief recommendation 4

**Pre-registration status (disclosure):** the rule was not written into the spec before the data. The design was fixed in the code comment (`ARM0_INDUCT`) and in my plan before any arm ran, but no decision threshold was recorded. This document therefore reports the numbers and reads them against the controls only; it makes no pass/fail claim.

Why: the research brief notes that perplexity and short-prefix agreement can hide damage to retrieval (copying earlier context). Probe: 16 sequences of 48 pseudo-random token ids (fixed LCG, ids 1000-29999) are shown twice in a row through the KV-cache decode path; at each position of the second copy the greedy next token is compared with the true continuation (teacher forced). 752 predictions per arm. Raw: `/home/workspace/lakes/p7_glint/induct_20261105/`.

| Arm | accuracy | hits / 752 | note |
|---|---|---|---|
| full 4-bit model (reference) | 88.7% | 667 | |
| 4-bit `act` 50% (~805 MB) | 89.0% | 669 | tracks in the rollout study |
| 4-bit `act` 25% (~403 MB) | 89.2% | 671 | does not track in the rollout study |
| T-A, 10% + 30% 3-bit (~523 MB) | 89.0% | 669 | best tracking tier |
| F3, 10% + 10% 3-bit (~282 MB) | 89.0% | 669 | does not track |
| F4, 5% + 10% 3-bit (~201 MB) | 89.2% | 671 | does not track |
| random selection, 25% | 87.4% | 657 | control: same bytes as `act` 25% |
| base only (about 0.01% fetched) | 87.2% | 656 | control: no plug |

## Reading
- **The probe does not discriminate.** All eight arms lie within 87.2-89.2% (656-671 of 752). The base alone is within 1.5 points of the full model, and the arms that failed the rollout study (F3, F4, `act` 25%) score the same as the ones that passed. A binomial standard error at n=752 is about 1.2 points, so the 2-point spread is not resolved.
- *Measured:* on random-token copying, the GLINT hole alone preserves the induction behaviour; the plug adds nothing detectable here. Stated as inference, not shown: induction heads are not what the plug corrects, so copying is not the failure mode behind the rollout divergence.
- The probe cannot support the claim "retrieval survives in the cheap arms" for natural text, only for this synthetic copy task. The accuracy ceiling (about 89%) shows the task itself is not saturated, so a ceiling effect does not explain the flat result.
- No arm's acceptance status changes. The rollout study (agreement with the full model) remains the tracking criterion; M6 acceptance stays unmet.

## What this does not show
One synthetic task, one seed, 16 sequences, one run per arm; no packed store or speed measurement; the three arms that first failed (T-A, F3, act25) were rerun after an LXD restart interrupted them, and the surviving logs are from the rerun. Natural-text retrieval (needle in a document) is untested. Not reviewed by the Test Designer.

## Reproduce
`ARM0_INDUCT=16,48 ARM0_DEMAND_RULE=act ARM0_DEMAND_F=1 [ARM0_SELFETCH=<k1> ARM0_SELFETCH_OPT=1 ARM0_TIER_FRAC_B=<k2> ARM0_TIER_BITS_B=3] bf16full --glint-demand PIN MODEL ART`; base only: F=0.0001 and SELFETCH=0.0001 (code on `feat/p7-tiered-plug`; build adds `cuda/glint_plug_streamer.cu` to the link line).
