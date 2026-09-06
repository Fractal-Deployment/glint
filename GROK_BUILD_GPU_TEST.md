# Paste into Grok Build CLI (GPU machine, 3060 + nvcc)

You have **two separate products**. Do not mix them.

| Tree | Role | Hands off |
|---|---|---|
| `Jadon-Fox/nf4-to-int8` | BF16→**INT8** pin (Ampere integers). User is A/B vs old INT8 pin **right now**. | Do **not** change dest, schema, or orch INT8 loader. |
| `Jadon-Fox/training_orchestrator` | Product NF4 H-TILE + Muon. INT8 pin test may be in flight. | Do **not** edit `main` for GLINT. No boxing TUI. No `ORCH_BASE_PACK`. |
| `Jadon-Fox/glint` | **GLINT** hole+plug CUDA. This job. | Only place you write code. |

## 0. Hygiene (do this first)

```bash
# orch — should stay on whatever commit the INT8 test is using
cd /path/to/training_orchestrator
git status -sb
git stash push -u -m "wip-not-glint"   # only if dirty AND user isn't mid-run
# if a train is running, do not stash. Work in /tmp/glint only.

git -C /path/to/nf4-to-int8 status -sb   # leave it
```

Work in a **sibling dir**, not inside orch:

```bash
git clone https://github.com/Jadon-Fox/glint.git /tmp/glint
cd /tmp/glint
# tip should include c7a13f6 H-TILE S3 coalesced smem
```

If orch already has untracked `scripts/tui/*`, leave it. That is not GLINT.

## 1. Build CUDA (sm_86)

```bash
export CUDA_ARCH=sm_86
bash scripts/build_glint_cuda.sh
test -f libglint_encode.so
nvcc -O3 -std=c++17 -arch=sm_86 -Iinclude -c cuda/glint_htile.cu -o /tmp/glint_htile.o
```

Fail closed if nvcc missing. Do not `--allow-cpu` for the product encode.

## 2. Tiny pin + encode/expand evidence (required)

Create a 8×64 BF16 linear safetensors (or use `test_glint.py` after building the .so). Then:

```bash
python3 glint_pin.py --src TINY_BF16 --out /tmp/glint-pin-cuda
python3 -c "import json; p=json.load(open('/tmp/glint-pin-cuda/pin.json')); assert p['schema']=='glint_pin_v1'; assert p.get('backend')=='cuda_bf16'; print(p)"
```

Goldens:

- Host `--allow-cpu` pin on the **same** tiny tensor vs CUDA pin: hole/plug bytes. If they differ, board `kParent` vs Python `PARENT` (known risk) — **do not silent-pass**.
- `glint_expand_packed_to_f32_host` vs `glint_codec.decode` on the CUDA pin: **max_abs < 1e-5**.

## 3. H-TILE S3 smoke (device)

Link `glint_htile.o` + `libglint_encode.so` into a 20-line driver: expand-or-resident hole/plug, random X, `launch_glint_htile_s3_f32`, compare to host `glint_htile_sim.htile_gemm_flat` on the same tiny M,N,K.

Board max_abs. Expect ~0 vs f32 sim; ~1e-4 if you round X to bf16 (sim offset). **nsys** one pass: confirm `cp.async` in SASS, not uncoalesced N-stride.

## 4. Do **not** in this session

- Set `ORCH_GLINT_PIN` on the INT8 comparison run
- Commit into `training_orchestrator` or `nf4-to-int8`
- Rename GLINT to NF8
- Full Phi-4 hour Python encode

## Done when

`/tmp/glint` has `.so`, EVIDENCE.json, H-TILE smoke rc=0. Orch and nf4-to-int8 **git status unchanged** from when you started (except their own INT8 test files). Paste EVIDENCE.json back.

L0 orch wire is a **later** prompt (`GROK_BUILD_ORCH_GLINT_PROMPT.md`) after this GPU evidence is green and the INT8 bakeoff is parked.
