# GLINT

**Gaussian Location INside Tile.**

A 4-bit **plug** names which child lives inside a 16-cell Gaussian **parent tile**. Reconstruction is a lookup, then H-TILE GEMM. Not NF8. Not uniform INT8. Not the `nf4-to-int8` program.

```
Ŵ = GLINT_CELLS[hole][plug] × absmax
```

Hole = parent cell (VRAM, same pack as H-TILE qpack).  
Plug = location inside that cell (4 bits). Drop it in → the weight is recognized.

```bash
# on the 3060 (nvcc)
bash scripts/build_glint_cuda.sh
python3 glint_pin.py --src /path/to/bf16 --out ./glint-pin          # GPU encode
python3 glint_pin.py --src tiny --out ./t --allow-cpu               # tests only
python3 glint_offset.py
python3 test_glint.py
```

Encode is **CUDA**: raw BF16 bytes go to the GPU; no Python float walk. That was the hour-long path. `libglint_encode.so` required unless `--allow-cpu`.

INT8 pin stays in [nf4-to-int8](https://github.com/Jadon-Fox/nf4-to-int8) (`--to int8`). Both from BF16; different 256.

Sim is **done** (offset board). Product is CUDA encode + CUDA expand. Orch: paste `GROK_BUILD_ORCH_GLINT_PROMPT.md` into Grok Build on the 3060.

