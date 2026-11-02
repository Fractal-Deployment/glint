# P7 Product 2 "resolution" at the INT8 base: first measurements (2026-11-02)

The goal is a compressed model on the GPU, unfolded one tile at a time, that lands closer to the BF16 original than existing quantizations of the same size. Product 2 keeps the residual R = W_bf16 - dequant(base) in CPU RAM and adds it back into the multiply where it matters most. This bench tests that mechanism at the INT8 base. INT4 comes later.

Rerun with `scripts/run_resolution.sh OUT_TSV 0.01 0.03 0.10 0.30`, which takes about two minutes on the RTX 3060. It first needs the HF logits, which `scripts/p7_resolution_reference.py` writes. Run that script as uid 1000 (`lxc exec --user 1000 --group 1000`) so the files it writes into the repo stay owned by the operator. Raw output is in `artifacts/run_2026-11-02.log` and `artifacts/results_2026-11-02.tsv`.

## Design

**Residual format.** R is stored at 4 bits per weight, as a code in [-7, 7] times the pin's own INT8 block scale times 1/7. No extra scale is stored, because the base scales are already resident. Rounding to the INT8 grid bounds |R| by scale/2, and the bf16 dequant adds up to 127/256 of a scale. The measured maximum over the model is |R/scale| = 0.9338, and no value was clamped. Codes are stored channel-major, so one input channel k of a linear is one contiguous row of n_out/2 bytes. The 128 linears total 1,610,612,736 bytes in pinned host RAM. The INT8 base linears are 3,422,552,064 bytes resident.

**Joining the GEMM.** Each arm builds xg, a copy of the activations with every unselected channel set to zero (dynamic arms) or gathered down to the selected columns (static arms). It dequantizes the selected residual columns to bf16 and runs one cuBLAS GEMM, xg times R_hat transposed, with fp32 accumulate. The result is added to the INT8 tile kernel's output. Fusing this into `unfold_pair` is not done yet. Correctness came first.

**Selection rules.** Here k is a fraction of each linear's input channels (K), and every arm at a given fraction moves the same bytes to within 0.6%.
- `decdec` follows DecDEC's rule. Per token and per linear, it keeps the top-k input channels by |x|. Selection is an exact per-row radix select, with ties going to the lowest index.
- `decdec_norm` keeps the top-k channels by |x| times the norm of that channel's residual column.
- `static` uses a fixed top-k per linear, ranked by mean|x| times the residual column norm. The ranking comes from a calibration pass over corpus tokens 2048..4095, which are disjoint from the eval tokens 0..2047.
- `trace` (exploratory) applies DecDEC's rule within each linear but splits the budget across layers. Each layer's share is proportional to the growth in relative hidden-state error between the base and full-residual forwards on the calibration text, capped at 100% of the layer.

**Streaming.** The selected data goes through the two-slot plug streamer that #31 fixed. Each layer is one plug, acquired at QKV and released after DOWN. For `static`, the plug is just the selected columns, so a 2048-token prefill moves exactly the bytes a decode token needs. For `decdec`, rows pick different channels. At 10%, the union of channels picked across the 2048 rows covers 98.7% of all channels (79% at 1%). So prefill streams the whole layer residual and selects on the device. DecDEC's decode path instead reads only the selected rows from mapped host memory (zero-copy). That decode path is not built here.

## Rulers

Every arm is scored on the ruler's 2048 tokens against HF BF16 logits over the full vocabulary. The script refuses to write those logits unless their argmax reproduces `oracle_top1_2048.bin`. A second KL is taken against this C/CUDA forward run with the BF16 weights. `bf16_splitk` is that same BF16 forward with K summed in two halves, a change of fp32 summation order and nothing else.

`resid_left` is measured inside each linear, on the arm's own inputs. It is the fraction of the full residual correction's output energy, ||X R_hat^T||^2, that the arm leaves out. Base is 1 and full is 0. Unlike final-logit KL, it is not swamped by rounding noise downstream.

Bytes/token is what one decode token must fetch. It is exact from the selection counts, including 2 bytes of index per channel for the dynamic arms. VRAM is `cudaMemGetInfo` usage during the run. The resident rows include the 1.61 GB residual held in VRAM so the arms can be compared directly. Readings move by about 0.05 GiB between runs.

## Results

| arm | frac | ppl | top-1 /2047 | KL vs HF | KL vs C-BF16 | resid_left | bytes/token |
|---|---|---|---|---|---|---|---|
| bf16 (C/CUDA) | | 5.491616 | 2015 | 0.001534 | 0 | | 0 |
| bf16_splitk | | 5.483657 | 2019 | 0.001489 | 0.001493 | | 0 |
| base INT8 | | 5.501980 | 2003 | 0.002584 | 0.002498 | 1 | 0 |
| full residual | 1.00 | 5.477725 | 2016 | 0.001570 | 0.001497 | 0 | 1,610,612,736 |
| decdec | 0.01 | 5.483785 | 2016 | 0.001737 | 0.001823 | 0.6511 | 16,231,360 |
| decdec_norm | 0.01 | 5.480124 | 2010 | 0.001719 | 0.001768 | 0.6493 | 16,231,360 |
| static | 0.01 | 5.474907 | 2007 | 0.001748 | 0.001808 | 0.7617 | 16,220,160 |
| trace | 0.01 | 5.484585 | 2011 | 0.001718 | 0.001802 | 0.7311 | 16,131,470 |
| decdec | 0.03 | 5.480847 | 2014 | 0.001697 | 0.001742 | 0.5312 | 48,300,672 |
| decdec_norm | 0.03 | 5.488156 | 2011 | 0.001716 | 0.001682 | 0.5305 | 48,300,672 |
| static | 0.03 | 5.475198 | 2010 | 0.001731 | 0.001824 | 0.6913 | 48,267,264 |
| trace | 0.03 | 5.477491 | 2011 | 0.001643 | 0.001783 | 0.6598 | 48,320,636 |
| decdec | 0.10 | 5.489291 | 2016 | 0.001729 | 0.001708 | 0.3341 | 161,084,160 |
| decdec_norm | 0.10 | 5.483439 | 2011 | 0.001605 | 0.001634 | 0.3332 | 161,084,160 |
| static | 0.10 | 5.480714 | 2010 | 0.001801 | 0.001832 | 0.5416 | 160,972,800 |
| trace | 0.10 | 5.486704 | 2008 | 0.001668 | 0.001686 | 0.5511 | 161,145,640 |
| decdec | 0.30 | 5.479828 | 2015 | 0.001639 | 0.001762 | 0.1190 | 483,695,104 |
| decdec_norm | 0.30 | 5.489755 | 2010 | 0.001650 | 0.001603 | 0.1187 | 483,695,104 |
| static | 0.30 | 5.476735 | 2017 | 0.001616 | 0.001629 | 0.3610 | 483,360,768 |
| trace | 0.30 | 5.485832 | 2017 | 0.001691 | 0.001668 | 0.4043 | 483,521,440 |

The streamed runs used fraction 0.10, the middle of the sweep. `static` streamed 160,972,800 H2D bytes per pass with 5.60 GiB VRAM in use. `decdec` streamed 1,610,612,736 bytes with 5.70 GiB in use, at 0.1 ms of measured stall. Both runs were bitwise identical to their resident twins in losses, top-1, KL on both rulers, and hidden states at 49 positions across all 33 slots.

## What the numbers say

1. **Final-logit KL has a floor of about 0.0015 in this forward.** Changing only the fp32 summation order (`bf16_splitk`) gives KL 0.001493 against the unsplit forward. That matches the forward's own distance to HF (0.001534) and the full-residual arm's distance to C-BF16 (0.001497). The forward rounds to bf16 between ops, and tiny perturbations amplify through that rounding into a fixed noise level. Base INT8 sits 0.0010 above the floor. Every corrected arm lands between 0.0016 and 0.0018 on either ruler. Differences between arms on KL, perplexity or top-1 are smaller than this floor and do not rank them. Perplexity below the oracle, as in the full arm, is the same noise.
2. **The full residual reaches the floor.** At 4 bits per weight it removes all of base INT8's measurable excess KL (0.002498 to 0.001497 vs C-BF16, against the 0.001493 floor).
3. **DecDEC's rule recovers most of the gap at 1% of the bytes.** At 16.2 MB per token, `decdec` cuts KL vs C-BF16 from 0.002498 to 0.001823, which is 67% of the base-to-floor gap (81% on the HF ruler). This is a single run on one 2048-token text.
4. **No arm beats DecDEC's rule at equal bytes.** On `resid_left`, the one metric that can tell the arms apart, dynamic selection beats calibrated static channels at every fraction, for example 0.334 vs 0.542 at 10%. Weighting |x| by residual column norm (`decdec_norm`) improves on plain |x| by at most 0.3% relative, which is no meaningful gain. The growth-budgeted `trace` arm leaves more residual energy than uniform DecDEC, because it moves budget toward early layers by design. Whether that helps downstream cannot be seen through the 0.0015 KL floor.

## Open

- INT4 base. At INT8 the quantization error is near the forward's own noise, so this bench cannot separate selection rules on final logits. An INT4 base would put the error well above the floor, where the rules can be told apart.
- Fusing the correction into the tile kernel's `unfold_pair`, and a zero-copy decode path that fetches each token's selected rows.
- A lower-noise end-to-end ruler (fp32 between ops, or several texts) to see whether `trace`-style budgets matter downstream.
