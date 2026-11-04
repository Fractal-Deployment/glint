# M4-LOSSY sweep: residual plug bit width (Phi-4-mini, 2026-11-03)

Knob and grid: `M4_LOSSY_AMENDMENT_2026-11-03.md` (committed before this data). Every level is MEASURED; none is PASS/FAIL.
Harness: CUDA INT8 W8A8 forward with the plug on all 32 layers, 2047 tokens, HF oracle PPL 5.486957; KL is vs the BF16 CUDA anchor.

| level | b | weight RMSE vs BF16 (base alone 1.983e-4) | PPL | top-1 /2047 | KL mean | KL p99 | ideal packed plug bytes |
|---|---|---|---|---|---|---|---|
| no plug | - | 1.983e-04 | 5.496700 | 2003 | 1.748e-03 | 1.518e-02 | 0 |
| L0 (reference) | 4 | 1.385e-05 | 5.483878 | 2004 | 1.240e-03 | 9.733e-03 | 1,811,939,328 |
| L1 | 3 | 3.235e-05 | 5.483677 | 2011 | 1.195e-03 | 8.207e-03 | 1,409,286,144 |
| L2 | 2 | 9.696e-05 | 5.501449 | 2010 | 1.144e-03 | 9.746e-03 | 1,006,632,960 |

Per-level files: `M4_LOSSY_b{4,3,2}_weights.json` (per-linear errors), `M4_LOSSY_sweep.tsv` (whole-model readouts). The harness stores every level in 4-bit fields, so the stored file is 1,811,970,608 bytes at all levels; the last column is the ideal packed size (b bits per weight plus one F32 scale per 64 weights), not a size anyone has written.

## Reading (no threshold was set, none is claimed)
- **Weight space is monotone and steep:** plug RMSE goes 1.39e-5 -> 3.24e-5 -> 9.70e-5 as b falls 4 -> 3 -> 2 (7x), still half of the base's 1.98e-4 at b = 2.
- **The whole-model readouts are not monotone and barely move:** KL mean falls (1.240e-3 -> 1.195e-3 -> 1.144e-3) while weight error rises. PPL and top-1 wobble by a few tokens (b = 2 PPL 5.5014 is above the no-plug 5.4967; top-1 2004 -> 2011 -> 2010).
- Most likely cause, consistent with the resolution result (PR #46): with 8-bit activations the activation quantization error is the floor, so shrinking an already small weight residual shows up only as noise in these readouts. This is an interpretation; it was not isolated here (a W8A16 / resolution-on run would test it).
- NON_MONOTONIC count (adjacent level pairs where a metric improves as b falls): weight RMSE 0, KL mean 2, KL p99 1, PPL 1, top-1 1.
- Seed spread: zero by construction (deterministic codec).
- Bytes: b = 2 would be 1.007e9 bytes against 1.812e9 for b = 4 (-44%) with no visible whole-model cost in these readouts; that is the practical finding, subject to the substitution below.

## Substitution and limits
The rules ask for the Part A metric e at layers 3/16/29 on the job-25 captures; those are not available here, so e is not measured. Weight RMSE and whole-model KL/PPL/top-1 stand in. One model (Phi-4-mini), one text, one run per level.
