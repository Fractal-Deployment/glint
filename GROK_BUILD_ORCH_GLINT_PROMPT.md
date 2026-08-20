# Paste into Grok Build CLI on training_orchestrator

Load **GLINT** pins from https://github.com/Jadon-Fox/glint (`schema: glint_pin_v1`).

GLINT = **Gaussian Location INside Tile**. Not NF8. Not INT8.
`w = GLINT_CELLS[hole_nibble][plug_nibble] * absmax[i/64]`

Copy `include/glint_cells.h` + `include/glint_meta.h`. CUDA only.

## L0 (this prompt)

1. `pin.json` schema `glint_pin_v1` + `model.safetensors`. Fail-closed.
2. `{stem}.weight` hole U8, `{stem}.weight.glint_plug` U8, `{stem}.weight.absmax` F32, `_glint_cells` optional.
3. Expand at load to f32 registry (Mode A). Env `ORCH_GLINT_PIN=/path`. Unset = NF4 product path.
4. Board: `base_dtype=glint_expanded` `train_ok=false` `measured_omega=false` `G1=OPEN`
5. Tiny pin from `python3 glint_pin.py` for CI.

## L1 after L0

H-TILE S3: `cp.async` hole **and** plug. Lookup `GLINT_CELLS[h][p]`. Prefetch plug[L+1] stream 1. Then S5 `mma.sync`. Goldens vs host `glint_htile_sim.py` (expect bf16 fragment offset — see OFFSETS in that repo).

Do not: rename to NF8, merge with INT8 pin, train_ok=true, Mojo.
