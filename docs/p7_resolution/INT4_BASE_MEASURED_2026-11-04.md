# INT4 base: the preserved work (#70) built, run and measured (2026-11-04) — glint #79

Branch `codex/p7-int4-resolution-recovery-20261102` (PR #70) held an INT4 converter, a W4A16 tile GEMM, tests and a resolution runner that had never been compiled or run ("preservation for review"). Its scratch directory (`/mnt/spill/compile_scratch/p7-int4`) no longer existed, so both scripts were run with `P7_BUILD_DIR=/home/workspace/lakes/p7_glint/int4_run_20261104` and the pin path set to the pin they build there; nothing else was changed. The HF-logits reference (`p7_resolution_reference.py`, offline oracle, registered `/opt/venv` in the container) was regenerated for the resolution runs; the only tracked file it changed is the logits path in `hf_logits_meta.json`. Operator context: the recorded 2026-10-29 ruling names INT4 stored / INT8 compute as the intended base, and the 2026-11-04 answer asks for every Ampere-compatible size.

## What passed (`scripts/run_int4_tile.sh`, exit 0; log `artifacts/int4_tile_run_2026-11-04.log`)
- Quantizer test: 7 cases PASS (absmax 7 gives scale 1; halves round to even; negative absmax maps to -7; all-zero block; codes in [-7, 7] without clamping; every weight within half a step of its code; nibble order and two's-complement packing).
- Conversion `bf16_to_int4_pin_v1`: 128 linears, 3,221,225,472 weights, 1,811,939,328 bytes, **4.5000 bits per weight** (4-bit code + F32 scale per 64); read-back max |w - code x scale| / scale 0.500, rmse 3.60e-3, clamped 0, copy mismatches 0; PASS.
- Tile GEMM test on synthetic shapes and the pin's matrices: 1147 cases ok, 0 failing.
- The mutation script (`run_int4_mutants.sh`) was not run: it needs a `compute-sanitizer` path that this machine does not have.

## INT4 base through the Arm 0 forward (Phi-4-mini, 2047 tokens, oracle PPL 5.486957)
| path | PPL | PPL vs oracle | top-1 | forward | VRAM for linears |
|---|---|---|---|---|---|
| INT4 tile GEMM, packed resident (W4A16) | 5.996130 | +9.28% | 1821 / 2047 (88.96%) | **3.94 s** | **1.81 GiB** |
| INT4 dequantized to BF16 + cuBLAS | 5.996130 | +9.28% | 1821 / 2047 | 3.42 s | 6.00 GiB |
Both paths give the same PPL and top-1 (identical). The packed kernel prefills 2048 tokens in 3.9 s, against about 135 s for the GLINT H-TILE kernel, with the 4-bit weights resident.
For comparison on the same text (earlier results): NF4 / the GLINT hole alone 5.833352, 91.06%; INT8 W8A8 5.4967, 97.85%. **A uniform symmetric INT4 base at 4.5 bits is worse than the NF4 hole at the same size** (KL against the CUDA BF16 forward 0.108 against 0.079).

## Resolution over the INT4 base (`scripts/run_resolution.sh`; the runner's own design: DecDEC rule = top-k input channels by |x| per token and linear)
Mean KL against the CUDA BF16 forward (2047 positions), top-1 against the oracle, bytes of residual per token. Residual width 8 bits (default over INT4, 3.22 GB) and 4 bits (1.61 GB); files `artifacts/int4_base_resid8_2026-11-04.tsv` and `..._resid4_...tsv`.

| residual fetched | bytes/token (4-bit residual) | `decdec` KL | `decdec_norm` KL | `trace` KL | `static` KL | `decdec` top-1 |
|---|---|---|---|---|---|---|
| none (base) | 0 | 0.1084 | | | | 1821 |
| 1% | 16.2 MB | 0.0466 | 0.0469 | 0.0536 | 0.0674 | 1899 |
| 3% | 48.3 MB | 0.0359 | 0.0362 | 0.0460 | 0.0592 | 1917 |
| 10% | 161 MB | 0.0209 | 0.0208 | 0.0391 | 0.0558 | 1945 |
| 30% | 484 MB | 0.00833 | 0.00847 | 0.0319 | 0.0399 | 1965 |
| full | 1611 MB | 0.00268 | | | | 2001 |
With an 8-bit residual (twice the bytes at each fraction): `decdec` 0.0460 / 0.0350 / 0.0202 / 0.00747 at 1 / 3 / 10 / 30% and 0.00159 for the full residual (top-1 2015). BF16 floor in this forward: about 0.0015.
- *Measured:* over an INT4 base the selection rules separate on final-logit KL (over INT8 they could not): the two dynamic rules (`decdec`, `decdec_norm`) are within noise of each other and clearly beat `trace` and `static`; the streamed arms equal the resident ones bit for bit (the runner's own check, VERDICT REPORT).
- *Measured:* a full 8-bit residual reaches the forward's noise floor; a full 4-bit residual does not (0.00268), which is the cost of a coarse residual code on a coarse base.

## INT4 + 4-bit residual against GLINT at the same stored size (8.5 bits per weight, top-k by |x| in both)
| fraction fetched | INT4 base + 4-bit residual, KL | GLINT (NF4 hole + child nibble, channel granularity), KL | GLINT bytes at that fraction |
|---|---|---|---|
| 0 | 0.1084 | 0.0793 | 0 |
| ~1% | 0.0466 | 0.0304 | 16.4 MB |
| ~10% | 0.0209 | 0.0152 | 163 MB |
| ~25-30% | 0.00833 (30%, 484 MB) | 0.00704 (25%, 404 MB) | 404 MB |
| 100% | 0.00268 | 0.00095 | 1611 MB |
GLINT's nested codec reaches a given KL with fewer fetched bytes than the uniform INT4 base with a residual, at every fraction measured, and its full-plug KL is about 3x lower. Caveats: the two sides come from different harnesses (GLINT: `bf16full --glint-demand` channel-granular emulation, `DEMAND_MAP_phi4mini.md`; INT4: this runner's DecDEC arms, whose selection counts include the index bytes), both are scored against the same CUDA BF16 forward on the same 2047 tokens, one run each.

## What this does not show
No INT4 decode kernel or speed (the receipts are prefill-only), no free-running generation, and no INT4 rollout; not the "INT4 stored / INT8 compute" variant (the tile kernel here keeps activations in BF16, W4A16); the mutation script and sanitizers were not run. Not reviewed by the Test Designer.
