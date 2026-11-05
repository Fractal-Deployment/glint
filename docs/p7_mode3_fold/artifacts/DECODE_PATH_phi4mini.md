# Decode path with a KV cache, Phi-4-mini (2026-11-04) — glint #84

`ARM0_DECODE=PL,S` in `tests/test_arm0_bf16_full.cu` steps one token at a time through the 32 layers with a per-layer KV cache (536 MB of float K/V for 2048 positions): `k_dec_rope` and `k_dec_attn` are single-row copies of the prefill RoPE and online-softmax attention kernels (same arithmetic, position passed explicitly); norm, SwiGLU, residual and the linear callback are the prefill ones with m = 1, so the same decode runs with cuBLAS, with H-TILE, with the fold-only table, and with the plug streamed through the ring (the streamer is restarted once per token). Logits are a cuBLAS m = 1 GEMM against the tied embedding.

Two checks per arm, PL = 64 prompt tokens, 80 positions teacher-forced then 15 free-running steps, one run each:
1. **Decode vs prefill, same process**: top-1 and CE loss at each of the 80 positions against the prefill results of the same run.
2. **Greedy generation** of 16 tokens from the 64-token prompt (the first prompt of the rollout study) against the rollout's padded-forward generation for the same arm.

| Arm | top-1 equal to prefill | loss bitwise equal | max abs dCE | mean abs dCE | generation = rollout prompt 0 | ms/token | ms/layer | tokens/s |
|---|---|---|---|---|---|---|---|---|
| BF16 CUDA anchor (cuBLAS dense) | 77 / 80 | 2 / 80 | 3.53e-1 | 6.60e-2 | yes, 16/16 | 31.0 | 0.97 | 32.3 |
| GLINT hole + plug, H-TILE resident | 80 / 80 | 37 / 80 | 3.92e-3 | 7.55e-5 | yes, 16/16 | 455 | 14.2 | 2.20 |
| GLINT hole + plug, H-TILE, plug streamed through the ring | 80 / 80 | 37 / 80 | 3.92e-3 | 7.55e-5 | yes (= resident) | 465 | 14.5 | 2.15 |
| fold only (hole = NF4), H-TILE | 80 / 80 | 42 / 80 | 1.56e-2 | 2.53e-4 | yes, 16/16 | 459 | 14.4 | 2.18 |

## Reading
- **The decode path is correct.** With the deterministic H-TILE linears it agrees with prefill on every top-1 and to a mean 7.6e-5 in CE; the remaining difference is the attention reduction order and the m = 1 versus chunked logits GEMM. With cuBLAS the m-dependent algorithm choice moves the mean CE by 0.066, which is the bf16 noise floor already seen between two correct implementations (KL about 1e-3), and 77 of 80 top-1 agree. The greedy generations are token-for-token the ones from the padded-forward rollout, for the anchor, the full GLINT arm and the fold-only arm.
- **Streamed == resident in decode** (same CE differences, same tokens); the ring costs about 10 ms per token here (465 vs 455 ms).
- **Decode compute per layer, measured for the first time.** Dense bf16 cuBLAS: 0.97 ms per layer (31 ms per token, 32 tokens/s; it reads 7.6 GB of weights per token). The current GLINT H-TILE kernel: 14.2 ms per layer (455 ms per token, 2.2 tokens/s) for hole + plug and 14.4 ms for fold only. H-TILE was built for tiled batches, not one row, and its time does not depend on whether the plug is read, so it is not a decode kernel. A GLINT decode kernel is the missing piece for any speed claim.
- **What that implies for streaming (arithmetic, using the 0.97 ms cuBLAS figure as a stand-in for what a good kernel could reach; not a measurement of any GLINT kernel):** fetching a whole layer's plug takes 4.35 ms (`FOLD_BUDGET_FETCH_phi4mini.md`), about 4.5x a 1 ms layer, so streaming the whole plug would cap decode near 7 tokens/s; fetching 5% (0.31-0.35 ms per layer) would fit under 1 ms per layer. Under the current H-TILE (14 ms per layer) any of these is hidden, which is why the stall in the streamed arm is near zero.

## What this does not show
- No GLINT decode kernel and so no GLINT decode speed; the cuBLAS arm is dense bf16, not 4-bit. Timing includes launch overhead and a host-side embedding copy per step, 15 steps, one run each.
- 80 positions of one text. No activation-selected fetch in decode yet (needs the column-major layout, #86, and a faster kernel).
- Not reviewed by the Test Designer.

## Reproduce
Build as in `FOLD_STREAM_phi4mini.md`. `ARM0_DECODE=64,16 ARM0_DECODE_OUT=g.txt bf16full [--glint PIN | --glint-stream PIN] MODEL ART`; fold only adds `GLINT_PARENT_ONLY=1`. Generated ids per arm: `decode_phi4mini/*.txt`. Raw logs: `/home/workspace/lakes/p7_glint/decode_20261104/`.
