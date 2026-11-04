# Amendment: M4-LOSSY knob and level grid (written before any lossy data)

Date: 2026-11-03. Author: Claude (coding agent) from the rules text, not a Test Designer pick. Status: agent-proposed;
the knob choice and grid are open to strike by Jadon / Test Designer before they are used for any claim.
Rule it implements: `p7-round3-fold-rules-DRAFT.md` M4-LOSSY ("Level grid: set by the knob(s) the lossy codec exposes ...
a bit-width knob gets one level per integer width from (raw width - 1) down to 2 ... written as a dated amendment
before any lossy data"). No lossy measurement exists in this repository at the time this file is committed.

## The codec and its knob
Codec under test: the `p7_plug_res_b` residual plug (tools/plug_encode.c). For each linear, r = W_bf16 - dequant(INT8 base),
quantized per 64-weight block with scale sp = max|r| / (2^(b-1) - 1), p = clamp(rint(r / sp), -(2^(b-1)-1), 2^(b-1)-1),
and the corrected weight is dequant(base) + p * sp. This is a lossy codec of the residual, and the knob is
**b, the residual bit width per weight**. Raw width = 4 (the format measured in glint PRs #42-#46 and #69); the b = 4 run
is the codec's own reference level, not a lossless one against BF16.

## Level grid
| level | b | meaning |
|---|---|---|
| L0 (reference) | 4 | the shipped res4 plug |
| L1 | 3 | one bit less |
| L2 | 2 | down to the rules' floor of 2 (values in {-1, 0, +1}) |

No b = 1 level (a 1-bit signed residual with a zero-excluded range would change the codec, not just the knob).
Order L1 = least loss is reversed from the rules' "L1 = least loss" only in naming: L1 (b = 3) is the least lossy lossy level.

## What is measured per level (all readouts; every level MEASURED or INVALID, never PASS/FAIL)
1. Reconstructed-weight RMSE and max|error| vs BF16, over all 128 Phi-4-mini linears (tool output).
2. Whole-model readouts through the CUDA W8A8 harness with the plug on all 32 layers: perplexity, top-1 vs the HF oracle,
   and per-token KL(BF16 CUDA anchor || level) mean / p99 / max.
3. Plug bytes: ideal packed b bits per weight plus the F32 scale per 64 weights (the harness stores 4-bit fields so it can reuse the
   kernel; the stored file size is NOT the compressed size, the ideal packed size is reported separately and labeled).
4. Seeds: the codec is deterministic, so spread is zero by construction and is reported as zero.
5. Report-only: NON_MONOTONIC count (adjacent level pairs where an error metric improves as b falls).

## Substitution (stated, not hidden)
The rules ask for the Part A metric e at layers 3/16/29 on the job-25 captures. Those captures are not available here, so e is
NOT measured. The weight-space RMSE/max and the whole-model KL/PPL above stand in as readouts and are labeled as a substitution.
