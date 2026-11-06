# Calibrated GLINT hole (GPTQ-style rounding onto the NF4 parent grid), Phi-4-mini — spec Amendment 11, glint #92 (PRE-REGISTRATION, results below once run)

Written and committed **before** any result exists. Follows `CALIBRATION_PROXY_llamacpp.md`, which tested calibration on a different codec; this builds the real thing for GLINT's hole. Tool: `scripts/glint_gptq_hole.py`, run in the registered container environment `/opt/venv` (torch 2.5.1+cu121, transformers 4.57.6), no new packages. Rationale for not using a GPTQ library: they target a uniform 4-bit grid; GLINT's hole is the NF4 code, so the rounding is implemented on that grid directly.

Method: per decoder linear (128 of them), GPTQ with H = 2/n X^T X from 30 x 512 tokens of `calib_docs.txt` (workspace markdown, disjoint from the evaluation text), damp 0.01, columns left to right in blocks of 128, group scale = absmax of the updated weights of each 64-column group, layers in sequence over hole-only outputs. The plug is the nearest child cell *inside the chosen parent cell* for the original weight, so a weight whose calibrated hole is not its nearest cell saturates at the cell edge. The embedding/lm_head and norms are copied from the source pin unchanged.

Arms (evaluation text: first 2047 tokens of `SCIENCE_JOURNAL.md`, reference = BF16 logits `anchor_logits.bin`): (a) fold-only (parent only) on the existing NF4 pin = baseline; (b) fold-only on the calibrated pin; (c) hole + plug full on both pins.

**Decision rule (fixed now):**
- *Worth pursuing* if (b) mean KL <= 70% of (a) (a reduction of at least 30%, the bar used for the imatrix proxy) **and** (c) on the calibrated pin has PPL <= 1.01 x the existing pin's full PPL (5.4885, i.e. <= 5.543).
- *Calibration helps the base but costs the full arm* if (b) meets the 30% bar but (c) exceeds 1.01 x.
- *No help* if (b) is above 70% of (a).
One run, one calibration text, one seed; no noise estimate beyond that.
