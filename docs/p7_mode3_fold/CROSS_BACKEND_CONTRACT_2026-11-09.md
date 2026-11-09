# Project 7 — Cross-backend representation, arithmetic and lifetime contract (2026-11-09)

**Status:** PROPOSED compatibility contract for R0/R1 (#109/#110), not operator-ratified code or a successful Mojo parity test. GLINT Project #6. Keep CUDA reference and Mojo/MAX implementation simultaneously. Do not promote an experimental format to native standard by default.

## Why this contract exists

The two source-backed execution arms are currently not the same:
- Mojo p7_forward.mojo: bf16_to_int8_pin_v1 (I8 and F32 scale per 64), optional separately supplied p7_plug_res4_v1, optional activation-residual second pass. It runs prefill/teacher-forced fixed T=2048; no token-by-token KV decode.
- CUDA tests/test_arm0_bf16_full.cu: glint_pin_v1 with parent-hole nibble and child-plug nibble, absolute scale per flattened 64, GLINT 16x16 cells, single-token GEMV / KV-cache decode / selected fetch.
A measurement comparing these without pin and arithmetic alignment is a comparison of different representations, not a CUDA-vs-Mojo implementation parity claim.

## Compatibility objective and permitted interim choice

Use glint_pin_v1 as the *interim loaded format* for Mojo hole/plug decode because the accepted CUDA harness already uses it. Retain the INT8 pin as an independent M1 reference. Record nested_nf8_pin_v1 and p7_plug_res4_v1 as separate layouts.

**No definitive codec winner is selected.** R1 must compare candidates using identical source bytes, reference tokens, total host+device storage, transfer bytes and reconstruction cost. Include independently encoded corrections for calibrated folds to avoid the failed calibrated-hole/original-child composition.

## Required glint_pin_v1 invariants (verify in source before implementation)

1. Header/tensor identity and source SHA: validate schema, tensor names, dimensions, dtype, and source checkpoint revision.
2. Nibble packing: low nibble for flattened element 2j, high nibble for 2j+1 (CUDA k_gemv_ref decodes j&1 accordingly). Check all odd/even row offsets.
3. Scale block: index floor(j/64), where j=row*K+k in flattened row-major indexing; do not silently reblock by row if this changes indexing.
4. Reconstruction candidate: w_j = cells[hole_j][plug_j] * absmax[floor(j/64)]. Fold-only uses parent[hole_j]*absmax; full and fold-only must be validated separately.
5. Table custody: _glint_cells metadata or tensor is part of the proposed pin contract. **Open discrepancy:** CUDA gemv_init currently copies cells_for_bits derived from built-in GLINT_CELLS, so loaded-cell authority must be verified/fixed on BOTH implementations. Do not claim the current CUDA code consumes the pin's table.
6. Kernel shape/bounds: checked element count, block64 compatibility, valid nibble sizes and address alignment; K multiple-of-32 for existing vectorized CUDA GEMV path, otherwise explicit guarded fallback. No out-of-bounds reading at partial tiles.
7. Format migration: schema/version change, conversion provenance, independent decoder test, and compatibility gates before adoption.

## Numerical reference and when an increment is valid

For one linear with canonical input X and W approximately F+Delta, mathematical real-arithmetic decomposition Y = XF + XDelta is valid. In actual floating-point execution, operation order and BF16 rounding sites can change numerical results.

Compare explicitly:
- Direct full-cell reconstruction followed by GEMV: acc += (cells[h][p] * scale) * x.
- Fold computation then correction accumulation: acc_fold += (parent[h] * scale)*x; acc_corr += ((cells[h][p]-parent[h])*scale)*x.
These need not be bit-identical after BF16/f32 rounding, reordering, and warp accumulation. Document expected numerical tolerance from matched references, and use identical kernel arithmetic for a real transport-bit-identical claim.

Rounding and numerical dependencies to audit in each backend:
- input embeddings and RMSNorm (variance, epsilon, scaling and BF16 casts);
- Q/K/V projection, RoPE (partial rotary, longrope factor and position);
- attention score scale, softmax, attention output and output projection;
- residual addition, MLP normalization, SwiGLU gate, down projection;
- final norm, tied output projection and logits;
- KV-cache representation and updates: per layer, 2048 positions, 8 K/V heads of dimension 128, f32 in current CUDA path. Preserve canonical position/order/initialization and separately account for actual KV bytes.

**Canonical state rule:** a speculative or coarse preview may drive prefetch, but must never overwrite canonical residuals, KV, or downstream inputs without proof of correctness or recomputing affected operations. Correction of a linear after a dependent nonlinearity was evaluated is not generally interchangeable with canonical evaluation.

## Separate operational interfaces

| Interface | Input | Output | Rule |
|---|---|---|---|
| Transfer granularity | future consumers, contiguous memory groups, byte budget | whole layer / tile / channel-group fetch | transfer may exceed final consumed set |
| Prefetch timing | earliest causal signal, transfer estimate, arithmetic deadline | start/end events and priority | no future leakage; misses/stalls counted |
| Refinement selection | available activation information and candidate bytes | region/precision list | log oracle only as offline diagnostic, not early prediction |
| Refinement application | canonical input, selected correction, compute dependency | updated accumulation | before dependent nonlinearity or explicit recompute |
| Lifetime manager | consumer count, events, residency budget | recycle/cache/evict action | no reuse before last consumer |
| Measurement | same model/pin/token set/oracle | KL/top1/PPL, latency, H2D bytes, peak VRAM | compare prefill and decode separately |

Use independent config settings for these rather than a single overloaded percentage. Changing a selected-fetch rule does not establish changed prefetch timing.

## Resident memory and transport

Use a bounded moving VRAM envelope: resident fold + canonical KV + active accumulators + runtime workspaces + staging pool + optional correction cache. Categories can share physical allocations; account for actual peak, not a double-counted theoretical total. Prefer larger batches only where deadlines and new transferred bytes justify them. Count speculative bytes unused.

The existing 2-slot Mojo/MAX stream/event ring is a **reference primitive**, not already integrated with p7_forward.mojo. Extend it after successful initial whole-layer plug streaming and last-consumer safety verification.

## Test hierarchy without changing operator gates

1. Single linear, same glint_pin_v1 and same input: compare unpacked values, fold-only and full-cell outputs, intermediate x/scale; verify table handling and numerical ordering.
2. KV decode on existing Mojo INT8 arm: canonical token positions, attention and KV update versus CUDA/reference, matched oracle/PPL/top-1. Measure decode per token, not forward throughput.
3. GLINT resident pin through Mojo KV decode: verify reference tolerance on the same 2047-token teacher-forced input.
4. Whole-layer plug streamed through Mojo ring: bit-identical **within the same Mojo arithmetic path** to resident mode, independently measured H2D activity.
5. Independent transport/selection/timing/application knobs: compare identical correction sets before alternative policies.
6. R5 integrated M6: preserve issue #92's free-running tracking plus >=10 tok/s, report fraction, per-layer budget, bytes/token, VRAM. No quality claims based solely on teacher-forced PASS.

**Recorded gates:** GATE_RULES.md (operator-set top1 >=95%, PPL within 2%; KL diagnostic), transport integrity, and distinct Amendment-2 tracking rule (agent-proposed numerical bars, pre-registered and referenced by #92). Do not substitute one for another.

## Pending questions

- Does current pin construction store its 16x16 table as planned? Verify exact metadata and loader behavior; the compile-time table currently requires investigation.
- What is the exact set of BF16 rounding sites, warp reduction order and acceptable cross-backend difference? Establish from matched instrumentation.
- Which R1 native representation minimizes total transferred bytes plus compute while maintaining fidelity? No format chosen yet.
- Can the host-buffer weight load use current supported MAX API without changing ownership and pinned-buffer lifetime? Consult the installed version.
- What happens on a late prefetch? Preserve canonical correctness, document stall/recompute.
- Which historical source artifacts are missing? R0 remains partial until reconciled.

**Implemented by this document:** none. It is a testable compatibility and execution specification, not a runnable decoder.
