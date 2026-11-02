# Full Phi-4-mini forward in Mojo / MAX (2026-11-02)

`mojo/p7_forward.mojo`: weights read by the Mojo safetensors reader, INT8 pin linears through the Mojo W8A8 tensor-core GEMM,
embedding, RMSNorm, RoPE (host double-precision cos/sin table), attention, SwiGLU, residuals, tied-embedding bf16 logits
and cross-entropy as Mojo kernels. No CUDA C, no Python. 2047 tokens, RTX 3060, Mojo 1.0.

| Path | PPL (HF oracle 5.486957) | top-1 vs HF oracle | forward |
|---|---|---|---|
| Mojo (this) | 5.481900 (-0.09%) | 2001/2047 = 97.75% | 8.05 s layers, 34.0 s total |
| CUDA INT8 W8A8 mma | 5.496700 (+0.18%) | 2003/2047 = 97.85% | 3.37 s |
| BF16 CUDA anchor | 5.491616 (+0.085%) | 2015/2047 = 98.44% | 7.7 s |

Passes the 95.0% Gate 1 top-1 threshold and the 1.0% PPL window. It is not bit-identical to the CUDA INT8 run: float32 `exp`,
the host RoPE table and different reduction orders in RMSNorm/logits all differ, which is why PPL differs in the third decimal.
Do not read the lower PPL as better fidelity; top-1 is 2 tokens lower.

Time: the layers take 8.05 s (CUDA INT8: 3.37 s, about 2.4x); the other ~26 s is the naive bf16 logits kernel (one thread per
logit, no tiling) and the one-thread-per-row cross-entropy. Not optimized.
Not checked: the oracle/token sha256 files (hf_meta.json) are read but not hash-verified in Mojo, and no KL against the BF16 anchor.
