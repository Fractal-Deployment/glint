# Grok Build CLI — training_orchestrator — GLINT on the GPU

You are on a machine **with an RTX 3060 and nvcc**. Do **not** write a simulator. Do **not** run Python H-TILE as the product. Wire **real CUDA** load + expand (then existing GEMM). Evidence on device.

Repo to copy from: https://github.com/Jadon-Fox/glint  
`schema: glint_pin_v1`  
`w = GLINT_CELLS[hole][plug] * absmax[i/64]`

## Sim already ran (do not redo)

Host board (sandbox, toy 8×64×64 Gaussian):

| metric | value |
|---|---|
| GLINT vs f32 weight RMSE | 1.29e-4 |
| parent-only (no plug) RMSE | 1.85e-3 |
| H-TILE f32 vs decoded GLINT max_abs | **0** |
| bf16 fragment vs f32 MAC max_abs | 8.2e-5 |
| Y GLINT vs true | 6.4e-5 |
| Y parent vs true | 9.0e-4 |

**Optimizations already applied in glint:** encode packs in-kernel (no unpacked n-byte temps). Expand kernel exists: `glint_expand_packed_to_f32_host`.

**Your goldens on GPU:** expand vs a tiny pin, max_abs vs host decode `< 1e-5` in f32. Do **not** chase bf16 MMA in this slice.

## Job (L0 — stop here)

1. Copy `include/glint_cells.h`, `include/glint_meta.h`, `include/glint_encode.h`, `cuda/glint_encode.cu` into orch. `bash scripts/build_glint_cuda.sh` or nvcc into orch `lib/`.
2. Env `ORCH_GLINT_PIN=/abs/path/to/glint-pin`. Unset = **bit-identical NF4 path**. Do not invent `ORCH_BASE_PACK`. Do not change product default.
3. Detect `pin.json` `schema=glint_pin_v1` + `model.safetensors`. Fail-closed.
4. Per linear: `weight` hole U8, `weight.glint_plug` U8, `weight.absmax` F32. Norms copy.
5. **GPU expand** `glint_expand_packed_to_f32_host` (or device-ptr variant you add — prefer **leave f32 on device**, no host round-trip of the full W). Upload into existing registry slots.
6. One product step on **existing f32/f16 GEMM**. Board:
   - `base_dtype=glint_expanded`
7. CI: tiny pin (`glint_pin.py --allow-cpu` only for the fixture; product encode is CUDA).
8. Evidence JSON: nsys optional; **required** max_abs expand vs host decode on the tiny pin.

## Do not

- Python tile GEMM in orch
- Merge with INT8 pin
- Call it NF8
- Mojo
- Inner-loop PCIe
- S5 MMA this slice (L1 after L0 green: H-TILE `cp.async` hole+plug, `GLINT_CELLS[h][p]`, then `mma.sync`)

## After L0 green — automatic train

Driver already in glint (do not re-simulate):

```bash
bash scripts/glint_auto.sh --src /path/to/bf16 --course tdc_v2 --steps 8 --orch $PWD
```

Wire `ORCH_GLINT_PIN` in `scripts/run_tdc_muon_course.sh` / `pin_weights_env.sh` so that when the env is set, the product runner loads the GLINT pin instead of the Unsloth NF4 pin. Course aliases: `tdc_v2` `tdc_v1` `a` `b` or a jsonl path.

Kernel is **not** generated per model. Build `libglint_encode.so` once.

