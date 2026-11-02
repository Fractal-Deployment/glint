# Mode 3 (fold) forward on the Arm 0 harness, Phi-4-mini, 2047 tokens (2026-11-02)

Pin: `glint_pin.py` on the BF16 shards -> `/home/workspace/pins/Phi-4-mini-glint-from-bf16` (CUDA encoder `libglint_encode.so`, blocksize 64, hole + plug nibbles + F32 absmax per block; 129 linears incl. the embedding, which the harness keeps BF16).
Run: `test_arm0_bf16_full --glint PIN MODEL_DIR ARTIFACTS` (linears through `launch_glint_htile_s3_f32`).

| Arm | PPL (oracle 5.486957) | PPL delta | top-1 vs HF oracle | forward |
|---|---|---|---|---|
| BF16 CUDA anchor | 5.491616 | +0.085% | 2015 (98.44%) | 7.7 s |
| INT8 W8A8 (mma) | 5.496700 | +0.18% | 2003 (97.85%) | 3.4 s |
| GLINT hole + plug (H-TILE, resident) | 5.479566 | +0.135% | 2007 (98.05%) | 136 s |

Reading: hole + plug reproduces the model at BF16 fidelity (PPL within 0.14%, top-1 98.05%, above the 95.0% gate), with a pin of 8 index bits + 1/16 F32 per weight, the same size class as INT8. The H-TILE kernel is 40x slower than the INT8 path (136 s vs 3.4 s), so the speed goal is not met by the current kernel.

Not valid and not claimed: the "hole-only, plug zeroed" run (PPL 1.7e14, top-1 0%). Plug index 0 is a specific child cell, not the parent centroid, so zeroing the plug nibble is not a hole-only decode; a real hole-only arm needs the parent-cell value. Also not run: plug-eviction/streaming on the glint pin, a lossy fold (M4), the fold vs INT8 equal-bytes test.
