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

## Plugs and resolution in the Mojo forward (same day)

`mojo/p7_forward.mojo PIN MODEL_DIR ARTIFACTS [PLUG.safetensors|none] [resolve]`. Plug = the res4 file from `tools/plug_encode.c`
on every layer, applied by the Mojo fused-plug kernel; resolution = the activation residual re-quantized and multiplied as a second
accumulating pass (Mojo `quant_resid` + the GEMM `accum` flag). 2047 tokens, HF oracle PPL 5.486957.

| Mojo arm | PPL | top-1 vs HF oracle | layers time |
|---|---|---|---|
| INT8 only | 5.481900 | 2001 (97.75%) | 8.05 s |
| + plugs | 5.488495 | 2006 (97.997%) | 11.2 s |
| + resolution | 5.484129 | 2013 (98.34%) | 10.1 s |
| + plugs + resolution | 5.477230 | 2014 (98.39%) | 13.9 s |
| CUDA INT8 + plugs + resolution (PR #46) | 5.476566 | 2016 (98.49%) | - |
| BF16 CUDA anchor | 5.491616 | 2015 (98.44%) | - |

Top-1 moves in the same direction and by about the same amount as the CUDA arms (plugs and resolution each add agreement; both together are
within 0.1 point of the CUDA best and of the BF16 anchor). Single-run differences of a few tokens are inside the noise between implementations.
