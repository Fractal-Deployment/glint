# Demand map: how much plug does a step need? (spec and decision rules, written before any data)

Date: 2026-11-04. Author: Claude (coding agent) from the operator's statements of 2026-11-04 (session 168e7895) and glint #79. Status: **agent-proposed; open to strike by the operator and the Test Designer before any result is used for a claim.** No demand-map measurement exists in this repository at the time this file is committed.

## Question (operator's premise, stated as a test)
The model does not need the whole model to keep producing the answer. For Phi-4-mini, how much of the plug (the part that lifts the folded model from the NF4 point to BF16 fidelity) must be present at each step for the output to stay close to the full model, and can activations choose it?

## Method: split each linear into parent plus correction
A GLINT weight is `GLINT_CELLS[hole][plug] * absmax`. Write it as `PARENT[hole]*absmax + D`, with `D = (GLINT_CELLS[hole][plug] - PARENT[hole]) * absmax`. Linearity gives, for a linear with input `x`:

`y_sel = H_parent(x) + H_delta(mask(x))`

`H_parent` is the existing H-TILE kernel with the parent table (`GLINT_PARENT_ONLY`, PR #80). `H_delta` is the same kernel with a delta table (cell minus parent). `mask(x)` zeros the input channels whose plug correction is not fetched. For the full mask the result equals the full decode up to f32 summation order; for the empty mask it equals the fold-only arm. Selection is per token (row of `x`) and per linear.

Arithmetic (amended before any data, same day): the H-TILE kernel needs about 135 s per forward and the sweep has about 50 arms (about 4 hours), so the sweep harness decodes parent and correction to bf16 per linear on the fly (transient, one linear at a time) and multiplies with cuBLAS, as the NF4 and INT8 arms do. `H_parent` and `H_delta` above then mean "parent weights" and "correction weights" in bf16, each rounded separately. Consequences, fixed now: the `f = 1` and `f = 0` arms are re-measured in this harness and are the references for the sweep; S1 and S2 below compare against those re-measured arms exactly (an all-ones mask and an all-zeros mask are exact), and against the H-TILE numbers within 0.5% PPL, which is a cross-implementation tolerance sized from the 0.23% gap already seen between the NF4 cuBLAS arm and the fold-only H-TILE arm.

Unit of selection (block granularity is the primary result; channel granularity is run for `act` and `random` only and only after the block sweep is reported):
- **block**: 64 consecutive input channels (a K-block), the realistic fetch unit (all rows' nibbles for those channels).
- **channel**: single input channels, comparable to DecDEC (arXiv 2412.20185).

Scores (a selection rule picks the top fraction `f` of units per token per linear):
- **act**: L2 norm of the token's activations over the unit (the proposal under test).
- **random**: uniformly random units at the same `f`, fixed seed 42 (control).
- **static**: units ranked by their mean activation norm over the first 256 tokens, the same set for every later token (a PowerInfer-style hot set; calibration tokens are inside the evaluation text, so this is report-only).
- **oracle**: units ranked by `norm(x_unit) * norm(D columns)` using the true correction (upper reference, uses information a runtime does not have).

## Arms and fractions
`f` in {1, 0.5, 0.25, 0.1, 0.05, 0.01, 0}, for rules act, random, static, oracle, at both granularities, all 32 layers, all four linears. Model Phi-4-mini, the 2047-token text, oracle PPL 5.486957, glint pin `Phi-4-mini-glint-from-bf16`, teacher-forced.

Sanity gates (an arm that fails one is INVALID, not a result):
- S1: `f = 1` equals the harness's own all-ones-mask arm exactly, and is within 0.5% PPL of the H-TILE hole+plug run (PPL 5.479566, top-1 2007/2047).
- S2: `f = 0` equals the harness's own all-zeros-mask arm exactly, and is within 0.5% PPL of the H-TILE fold-only run (PPL 5.846849, top-1 1869/2047).

## Readouts (every arm MEASURED or INVALID, never PASS/FAIL by itself)
1. PPL and its delta to the oracle; top-1 vs the HF oracle.
2. KL(BF16 CUDA anchor || arm) per token: mean, p99, max. The anchor logits come from `bf16full` with `ARM0_LOGITS_OUT`, run once on the same tokens. (Known: the anchor is itself about 0.0015 KL from HF; reported, not corrected.)
3. Bytes fetched: selected units' plug nibbles over total plug nibbles, per layer and overall. absmax and hole stay resident and are not counted.
4. **Churn**: for consecutive tokens, the Jaccard overlap of the selected sets at the same linear, per layer.
5. **Predictability**: recall of the true `act` set at fraction `f` when the set is replaced by the previous token's set at the same linear.
6. Per-layer sensitivity: for `f` = 0.1, the PPL change when only one layer is restricted (others full), to locate where demand concentrates.

## Decision rules (fixed now)
Let `A` = the full hole+plug arm (PPL 5.479566, top-1 2007/2047; its KL against the anchor is measured in the same sweep). Call fraction `f` *sufficient for rule R* when, at that `f`: top-1 >= 95.0% (the operator-set bar, GATE_RULES C.1) **and** PPL within 2.0% of `A` **and** KL mean <= 2x the KL mean of `A`.
- **Premise supported on this model**: some `f* <= 0.25` is sufficient for `act` at block granularity, **and** at `f*` the `act` rule beats `random` by at least 1.0 percentage point of top-1 or 0.5% of PPL.
- **Premise falsified on this model (this granularity, this text)**: `act` is not sufficient at `f = 0.5`, or `act` is within noise of `random` at every `f` in [0.05, 0.5]. (Noise: two `random` seeds, 42 and 43, define the spread.)
- **Decode-hiding feasible**: reported only as the smallest sufficient `f` and its bytes; the bandwidth argument (a layer's decode compute ~0.2 ms against ~4.4 ms for a whole 48 MiB plug at the measured 11.28 GB/s) is arithmetic, not a measurement, and sets the target of about 5% bytes fetched.
- No outcome here alters the codec, the pin, or a ledger row without the operator's lock.

## What this does not measure (stated, not hidden)
- Teacher-forced prefill selects per token independently, so it does not include the compounding of errors when the degraded model feeds its own tokens back. A rollout (greedy generation from a prompt, comparing token sequences and KL along the generated path against the full arm) is Phase 2b and needs its own spec amendment.
- It does not test the streamed weight-residual (resolution) mechanism, only the plug. Streaming selected rows is a speed question for Phase 3.
- It measures fidelity against bytes, not time. No speed claim follows.
- One model, one text. Qwen3 and the others come after the arm set exists for them.

## Implementation notes (not results)
The delta table is `GLINT_CELLS[h][p] - PARENT[h]`. The kernel keeps its table in `__constant__` memory, so the harness swaps tables between the two launches of a linear. The mask is applied to `x` on the device before the delta launch. Per-token top-f selection is done on the device per linear. Activations for the oracle rule come from the true decoded `D`.

## Amendment 2 (rollout, written before any rollout data; 2026-11-04)
Teacher-forced prefill hides compounding. Rollout measures it with greedy generation by repeated full forwards (causal attention makes the logits at the last real position independent of what follows, so a padded 2048-token buffer is exact; this is not a KV-cache decode path, which does not exist in this repository).
- **Setup:** P = 3 prompts, each the first PL = 64 tokens of the evaluation text at offsets 0, 512 and 1024 placed at position 0; S = 16 greedy steps. 48 generated tokens per arm.
- **Arms:** BF16 CUDA anchor (reference noise floor), full plug (f = 1), fold only (f = 0), `act` channel f = 0.25, `act` channel f = 0.1, `act` block f = 0.25, `random` channel f = 0.1 (seed 42).
- **Metrics:** for each arm against the f = 1 arm: mean matched-prefix length (tokens before the first difference, per prompt) and same-position agreement over all 48 tokens; and agreement in steps 9-16 minus steps 1-8 (the drop). The anchor against f = 1 gives the noise floor: two near-identical models still diverge at near-ties.
- **Rules (fixed now):** an arm *tracks the full model through rollout* when its mean matched-prefix is at least 0.75 of the anchor's and its same-position agreement is at least 0.9 of the anchor's. An arm shows *compounding* when its drop exceeds the anchor's drop by more than 10 percentage points. Fold only and `random` are controls expected to fail both.
- **Limits:** 3 prompts of one text, 16 steps; matched-prefix lengths are integers with large variance; no claim beyond Phi-4-mini on this text.
