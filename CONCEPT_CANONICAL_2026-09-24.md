# GLINT — conceptual architecture (not “4+4 quantization”)
# Court: /home/workspace/glint/ (implementation) · concept SSOT this file · card: lookup/cards/glint-concept.md
# Source: /home/jfox/Documents/The canonicalization.md (operator, 2026-09-24). Not meaning-court.

## Canonical concept (designed)

\[
\boxed{\text{fold/compress}\rightarrow\text{unfold}\rightarrow\text{conditional relational refinement}}
\]

with a **bounded, moving** high-fidelity portion of the model.

- The **folded** representation remains computationally useful (coarse computation / navigation), not a mere placeholder for missing bits.
- **Unfolding** is first-class: storage representation and computational representation are deliberately different. Low-bit stored structure can unfold into what the active operation needs.
- The **plug** is refinement information needed to resolve a relevant region of pretrained relational structure more faithfully — not definitionally “the remaining bits.” A bit-complete residual is one possible implementation.
- **Activation determines refinement demand.** Attention softmax is one candidate signal, not the definition of routing and not proof of parameter importance. Attention, MLP gates, residual activity, feature activity, temporal locality, and others remain empirical candidates.
- Goal is **not** minimum activation. Maximize active relational fidelity subject to VRAM ≤ B (moving unfold budget).
- Example envelopes such as “8 GB folded + ~2 GB unfolded” are **architectural envelopes**, not established measurements.
- **Mojo/MAX/MLIR** belong to runtime realization (routing, async transfer, unfold, stage, compute jointly schedulable), not conceptual proof.
- **“No slowdown”** is a hypothesis: latency hiding if routing + transfer + unfolding fit in available overlap.
- Exact BF16/FP32 recovery requires preserving enough information. Current GLINT 4+4 Gaussian refinement is still **lossy**. Do not conflate lossy 4+4 with a residual/bit-complete exact scheme.

## Central falsification question

Can activation-derived routing identify a bounded high-fidelity subset that tracks the full-precision model closely enough, while transfer/unfolding stays inside the latency and memory envelope?

## Truth states

- **Designed:** selective fold → unfold → conditional refinement under a moving budget (this file).
- **Implemented:** whatever `/home/workspace/glint/` currently builds (often narrower; historically contaminated with “unusual 8-bit quantization” readings).
- **Demonstrated:** only what lakes / measured runs show. Latency-neutral selective unfolding is not demonstrated by design alone.

## Jurisdiction

GLINT is adjacent to LLMVE / HSVE (hardware working-set) but is **not** an LLMVE Ω variable and must not be merged into LLMVE findings by analogy. Prefer distinction-first (`lookup_canonicality_precedence`).
