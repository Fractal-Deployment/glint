# Calibrated GLINT hole (GPTQ-style rounding onto the NF4 parent grid), Phi-4-mini — spec Amendment 11, glint #92 (pre-registered; results at the end)

Written and committed **before** any result exists. Follows `CALIBRATION_PROXY_llamacpp.md`, which tested calibration on a different codec; this builds the real thing for GLINT's hole. Tool: `scripts/glint_gptq_hole.py`, run in the registered container environment `/opt/venv` (torch 2.5.1+cu121, transformers 4.57.6), no new packages. Rationale for not using a GPTQ library: they target a uniform 4-bit grid; GLINT's hole is the NF4 code, so the rounding is implemented on that grid directly.

Method: per decoder linear (128 of them), GPTQ with H = 2/n X^T X from 30 x 512 tokens of `calib_docs.txt` (workspace markdown, disjoint from the evaluation text), damp 0.01, columns left to right in blocks of 128, group scale = absmax of the updated weights of each 64-column group, layers in sequence over hole-only outputs. The plug is the nearest child cell *inside the chosen parent cell* for the original weight, so a weight whose calibrated hole is not its nearest cell saturates at the cell edge. The embedding/lm_head and norms are copied from the source pin unchanged.

Arms (evaluation text: first 2047 tokens of `SCIENCE_JOURNAL.md`, reference = BF16 logits `anchor_logits.bin`): (a) fold-only (parent only) on the existing NF4 pin = baseline; (b) fold-only on the calibrated pin; (c) hole + plug full on both pins.

**Decision rule (fixed now):**
- *Worth pursuing* if (b) mean KL <= 70% of (a) (a reduction of at least 30%, the bar used for the imatrix proxy) **and** (c) on the calibrated pin has PPL <= 1.01 x the existing pin's full PPL (5.4885, i.e. <= 5.543).
- *Calibration helps the base but costs the full arm* if (b) meets the 30% bar but (c) exceeds 1.01 x.
- *No help* if (b) is above 70% of (a).
One run, one calibration text, one seed; no noise estimate beyond that.


---
# RESULTS (2026-11-06; the rule above was committed before this run, commit 2997128)
Build: `scripts/glint_gptq_hole.py`, 30 x 512 calibration tokens (16,711 tokens in the file), 128 of 128 linears replaced, 462 s on the RTX 3060 in `/opt/venv`; the pin is at `/home/workspace/lakes/p7_glint/gptq_hole_20261106/pin`. Evaluation: first 2047 tokens of the journal against the BF16 logits, one run each. Raw logs next to the pin.

| Arm | mean KL vs BF16 | PPL (BF16 oracle 5.487) | top-1 agreement |
|---|---|---|---|
| (a) fold-only, existing NF4 pin | 7.90e-2 | 5.847 | 91.30% |
| **(b) fold-only, calibrated hole** | **3.03e-2** | **5.598** | **93.99%** |
| (c) hole + plug, existing pin | 8.7e-4 | 5.480 | 98.05% |
| **(c) hole + plug, calibrated pin** | **5.41e-2** | **5.742** | **92.82%** |

## Verdict under the pre-registered rule
- (b) / (a) = 0.384, a **61.6% reduction** in KL (bar: at least 30%): **met**. The calibrated base alone (PPL 5.598) is better than the data-free NF4 base by 0.25 PPL, a bigger gain than any llama.cpp proxy in `CALIBRATION_PROXY_llamacpp.md`.
- (c) on the calibrated pin: PPL 5.742 = 1.048 x the existing pin's 5.480 (bar: at most 1.01 x, 5.543): **not met**. The full reconstruction is worse than before by 4.8% PPL and its KL is 60 times higher (5.4e-2 against 8.7e-4).
- Outcome: **"calibration helps the base but costs the full arm."** This is the second branch of the rule.

## Reading
- *Measured:* a calibrated hole and the existing plug do not compose. The plug can only choose a child cell inside the parent cell the hole picked. GPTQ's error feedback sometimes picks a hole that is not the weight's nearest cell; the plug then saturates at that cell's edge, so hole + plug is worse than the nearest-cell pin. (The fraction of weights where the calibrated hole differs from the nearest cell was not counted in this run.)
- *Measured:* this matters for the fetch design, not only for fold-only. In any tracking arm the fetched channels use hole + plug and the rest use the hole alone, so a calibrated pin helps the unfetched 60-90% and hurts the fetched 10-40%.
- *Inference, untested:* the plug needs to be built to correct the calibrated hole, not to refine the nearest cell. Candidates: pick the plug against the GPTQ-updated weight; allow the plug to address the neighbouring parent cell; calibrate the hole with the full 256-level grid in the error model; or give the plug its own table. Which one keeps the 62% base gain and the 98% full-arm fidelity is the next experiment.

## What this does not show
One calibration text of 16.7k tokens, one run, 2047 evaluation tokens, one seed; the calibration and evaluation texts are different (workspace docs against the journal) but both are English technical prose. No rollout, tracking or speed measurement was made on the calibrated pin yet. Not reviewed by the Test Designer.
