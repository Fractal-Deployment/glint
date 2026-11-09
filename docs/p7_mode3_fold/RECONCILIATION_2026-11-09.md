# Project 7 — Source reconciliation and implementation reality (2026-11-09)

**Scope:** GitHub Project #6, GLINT; documentation-only reconciliation for R0 / issue #109.
**Evidence boundary:** code read from GLINT branch exp/p7-gptq-hole (observed at 78b09bf), on-disk CUDA gate summaries from /home/workspace/lakes/p7_glint/gate_20261106, GitHub Project #6 issues #79/#92/#109-#115, and the operator's 2026-11-09 architectural directive. This is a source review, NOT a new GPU experiment or a review signature. See missing sources below.

## Authority and definitions

- **Operator-approved architectural direction (2026-11-09):** retain Mojo/MAX and MLIR scheduling direction. CUDA may serve as a reference and controlled experimental harness; no implicit CUDA-only pivot. Fold/unfold, correction/plug, streaming, scheduling and VRAM lifetimes are separate but co-designed mechanisms.
- **Operator mode ruling (2026-11-04; Project #6 / #92):** M1 plain INT8; M2 INT8 + CPU offload; M3 folded representation with host plugs; M4 resolution (historically called M5); M5 compression (historically called M4); M6 routing with fold + streaming + plugs simultaneously.
- **Historical conflict:** the Oct-20 plan, round-3 historical rules and STATUS_2026-11-04.md contain old numbering or a different M6 definition. They remain useful historical evidence, not the current mode authority.
- **Plug ambiguity (unresolved):** (a) GLINT child nibble of hole+plug format; (b) bitplane supplement supporting BF16 reconstruction; (c) nested NF8 sub-index; (d) residual plug p7_plug_res4_v1 for INT8. Never equate these automatically.
- **Resolution ambiguity (unresolved):** activation-residual second pass; offloaded weight-residual correction; or a historical mode name. Specify the mechanism and operator-approved mode separately.
- **Source custody:** dense BF16/F16 source weights are required for constructing genuine refinement; an already lossy NF4 index cannot reveal discarded source information.
- **Decision not yet made:** the final native fold/plug representation is not preselected. glint_pin_v1 is a working compatibility format for Mojo/CUDA comparison, NOT the mandated long-term codec.

## Reconciliation by mechanism

| Mechanism | Original intent / status | Current source-backed implementation | Evidence / consequence | Missing capability / next action |
|---|---|---|---|---|
| BF16 numerical reference | Verified-reference requirement | CUDA arm0 matches HF to a documented numerical tolerance; different arithmetic is known | BF16 CUDA PPL 5.491616 and 98.44% vs HF; not bit-identical | Preserve same checkpoint/tokenizer and reference logits; record rounding sites and tolerance |
| M1 resident INT8 | Executed | Existing CUDA and Mojo W8A8 forwards, differing numerical paths | Mojo forward prefill numbers are not decode numbers | Matched decode harness |
| M2 streamed INT8 | Executed CUDA foundation | INT8 resident/streamed equivalence previously demonstrated | No inference that Mojo forward streams its weights | Mojo decode and actual ring integration |
| M3 folded + plug | Executed CUDA experiments | CUDA glint_pin_v1 hole+child, GEMV decode, selected-fetch; Mojo uses different INT8+res4 design | Full CUDA hole+plug close to BF16; base-only gate fails | Same pin and numerical contract in Mojo, then streamed == resident gate |
| M4 resolution | Partially implemented; terminology conflicts | Activation-residual second pass in Mojo; experimental weight-residual paths elsewhere | Mechanisms not interchangeable | Name exactly which resolution path is tested |
| M5 compression | Experimental primitives | Mixed 3/4-bit tier, compression and transfer benches on draft stack | T-A throughput target not met end-to-end | Real packed integration with measured totals |
| M6 active routing | NOT ACCEPTED | CUDA activation-selected fetch, one-token KV decode, rollout and timing controls | Fast selected arms pass teacher-forced but fail separate free-running tracking | Integrated fidelity AND >=10 tok/s under existing #92 criteria; no substituted threshold |
| Mojo/MAX | Retained operator direction; not replaced by CUDA | mojo/p7_forward.mojo has fixed-T prefill; mojo/p7_prefetch.mojo has separate 2-slot ring with events | Neither a Mojo KV decode nor actual model-wide ring integration | New Mojo decoder on existing INT8 format; shared math; then glint_pin_v1 |
| MLIR scheduling | Research direction, not yet executed | MAX low-level DeviceContext and Mojo kernels are present | No source-backed MAX Graph/MLIR implementation or measured MLIR benefit | Hypothesis + falsifyIf + before/after metrics before compiler claims |
| Selective prefetch | Proposed closed-loop refinement | CUDA selection, gather, correction and timing coupled in selfetch_linear | Previous-token stale selection failed tracked rollout | Separate control of transfer unit, timing, selection, application; causal availability graph |
| Ephemeral plugs / persistent accumulators | Architectural hypothesis, issue #25 | Separate residual arithmetic and ring work exist | Does not establish fused in-GEMV staging or safe last-consumer eviction | Lifetime contract and linear/nonlinear dependency proof |
| GPU residency | Operator: bounded moving VRAM envelope | GPU ring and resident/streamed CUDA paths | 1.3 GiB-class memory savings reported for full plug stream | Dynamic staging/cache budget, peak VRAM, speculative overfetch, exact release events |
| Cross-model / 20B | Long-term hypothesis | Other-model pins and some forwards; full selective decode not generalized | No measured full 20B inference | Correct physical bandwidth and capacity envelope; independent held-out tests |

## Distinct tests and gate provenance

1. **Recorded teacher-forced quality gate (GATE_RULES.md and Amendment 12):** oracle top-1 >=95%, absolute PPL difference <=2% for PASS, KL reported without accepted threshold. Existing 2047-position Phi-4-mini decode runs are appended to artifacts/GATE_TEACHER_FORCED_phi4mini.md from summary.txt and summary2.txt. Act 10% PASS (95.46%, 95.60%); act 5% PASS narrowly (95.16%, 95.16%); fold-only FAIL (93.11%). 47-step timing act10 = 14.16 tok/s; act05 = 17.72 tok/s. These are NOT 2047-step timing estimates.
2. **Free-running selection tracking:** Amendment 2 rollout criterion as recorded in DEMAND_MAP_ROLLOUT_phi4mini.md / DECODE_ROLLOUT_phi4mini.md: >=82.5% agreement and mean matched-prefix >=20/32 against the full GLINT decode, based on BF16 anchor normalization. It was **agent-proposed, pre-registered, referenced by an operator-set #92 issue but its numeric bars were not separately ratified**. Preserve it while explicitly surfacing its authority.
3. **M6 acceptance (issue #92):** all three mechanisms active, Phi-4-mini decode >=10 tok/s, selected free-running result tracks under the referenced Amendment 2 rule; record fraction, bytes/token and per-layer allocation. No current executed arm satisfies BOTH conditions.
4. **Transport fidelity:** the explicitly identical transport of the *same arithmetic implementation and selections* must be bit-identical when required by GATE_RULES. Do not demand bit-identical results between distinct Mojo/CUDA arithmetic algorithms; use recorded numerical tolerances, intermediate comparisons, and KL/top-1.

## Concrete source checks made on 2026-11-09

- mojo/p7_forward.mojo: fixed T=2048, NT=2047, 32 layers, INT8 pin (name.weight and name.int8_scale), separate optional res4 plugs and activation residual pass. It has no KV-cache token step.
- mojo/p7_prefetch.mojo: 32-layer synthetic 48 MiB-per-layer work, 2-slot GPU pool, stream/events; not imported into p7_forward.mojo.
- tests/test_arm0_bf16_full.cu: GLINT decode GEMV, base-only toggle, KV decode, and ARM0_DECODE_GATE are present. Its k_glint_gemv reconstructs each element as (table[h][p]*absmax)*x and accumulates in f32.
- **Compatibility gap:** gemv_init currently calls cells_for_bits to initialize c_cells from compiled GLINT_CELLS; it does not visibly read a pin-provided _glint_cells table at that call. Therefore the pin-table-as-authority rule is a required compatibility test/fix, not a claim about current CUDA parity.
- Both worktrees under /home/workspace/fractal-deployment are present: main checkout on feat/qwen3-4b-base-pins, second on exp/p7-gptq-hole; preserve unrelated untracked directory.
- NVIDIA RTX 3060, driver 595.99.02 observed. This does not establish reproducibility across devices.

## Missing or unresolved inputs (not silently reconstructed)

- The historical complete September-and-earlier Drive research and original raw planning text were not exhaustively re-reviewed in this pass. Current Drive discussion is a reference, not proof of that historical review.
- DEMAND_MAP_SPEC_2026-11-04.md and the loader issue #100 specification were not available as local files during initial inspection; cross-check GitHub before declaring a complete source reconciliation.
- libp7_streamer.so source for the legacy streamer benchmark is absent from the reviewed GLINT source tree; preserve the Mojo ring as the reproducible reference.
- The Nov 2 ladder paste and certain older files require source recovery.
- Corrected cross-backend pin-table loading, KV decode, actual ring integration, and matched performance/fidelity measurements remain unimplemented in Mojo.
- Stray branch fix/p7-tier-gather-overlap is operator-controlled; do not delete it.

## Dependency order and downstream meaning

R0 source truth and causality -> R1 representation/codec compatibility -> R2 causal demand signals -> R3 grouped transport and bounded staging -> R4 safe arithmetic/refinement lifetime -> R5 integrated M6 -> R6 cross-model feasibility. R3 batching controls can begin without an R2 predictive winner. Mojo INT8 KV decode and glint_pin_v1 decoding are enabling milestones feeding R3/R4; they are not M6 acceptance by themselves.

**Status:** R0 evidence reconciliation PARTIAL pending missing historical sources; R1 definition PARTIAL, not a format selection. Nothing in this document claims an executed Mojo decode or a ratified new performance gate.
