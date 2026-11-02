# Piece G: resolution (Phi-4-mini, 2047 tokens, resident W8A8, KL vs BF16 CUDA anchor)

Measured 2026-11-02 with `tests/test_arm0_int8_tile.cu --gemm mma` (`ARM0_RESOLVE=1`, `ARM0_PLUG_FILE`, `ARM0_PLUG_LAYERS`).

| Arm | PPL (oracle 5.486957, anchor 5.491616) | top-1 vs HF oracle | KL mean | KL p99 | KL max | forward |
|---|---|---|---|---|---|---|
| INT8 W8A8 | 5.496700 | 2003 (97.85%) | 1.748e-3 | 1.518e-2 | - | 3.37 s |
| + res4 plugs, all 32 layers | 5.483878 | 2004 (97.90%) | 1.240e-3 | 9.73e-3 | 0.128 | - |
| + resolution (activation residual) | 5.495493 | 2008 (98.09%) | 1.204e-3 | 8.51e-3 | 0.333 | 4.80 s |
| + plugs + resolution | 5.476566 | 2016 (98.49%) | 4.174e-4 | 3.40e-3 | 0.029 | 6.03 s |

Reading: the weight residual (plugs) and the activation residual (resolution) each remove about the same
~30% of the KL; the floor with only one of them is ~1.2e-3 because the other error source remains. Both
together give 4.2x lower KL than plain W8A8, and top-1 agreement with the HF oracle (98.49%) equals the
BF16 CUDA anchor's own (98.44%). Cost: roughly 1.8x forward time, plug file 1.81 GB on top of the 3.42 GB base.

Residual form chosen from this harness: the activation residual is re-quantized with the pin rule per
(row, 64-block) and multiplied against the same weights (`int8_mma_gemm_resolve`). No extra file or stream
is needed for it. Not measured: a weight-side residual file beyond res4 (res4 alone already hits the
activation-limited floor), and a resolution pass only on selected layers.
