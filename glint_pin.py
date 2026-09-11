#!/usr/bin/env python3
"""BF16/F16/F32 dense linear → GLINT pin (hole + plug). Separate from nf4-to-int8."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dtype_io import numel_of, unpack_dense
from glint_codec import cells_flat, decode, encode, pack_f32, pack_indices, rmse, unpack_f32, unpack_indices
from glint_cuda import encode_bf16_cuda, load as load_cuda
from safetensors_io import SafeTensorsFile, write_safetensors

PARENT_KEEP = ("norm", "bias", "rotary", "cos_cached", "sin_cached")


def _is_keep(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in PARENT_KEEP) or name.endswith(".bias")


def convert(
    src: Path,
    out: Path,
    blocksize: int = 64,
    dry_run: bool = False,
    allow_cpu: bool = False,
) -> dict:
    src = src.expanduser().resolve()
    if src.is_dir():
        wf = src / "model.safetensors"
        if not wf.is_file():
            shards = sorted(src.glob("*.safetensors"))
            if len(shards) != 1:
                raise FileNotFoundError("need one .safetensors")
            wf = shards[0]
    else:
        wf = src
    out = out.expanduser().resolve()
    with SafeTensorsFile(str(wf)) as st:
        linears = []
        keep = []
        for name, info in st.tensors.items():
            if info.dtype in ("BF16", "F16", "F32") and name.endswith(".weight") and len(info.shape) == 2:
                if _is_keep(name):
                    keep.append(name)
                else:
                    linears.append(name)
            else:
                keep.append(name)
        plan = {
            "schema": "glint_pin_v1",
            "n_linears": len(linears),
            "n_keep": len(keep),
            "blocksize": blocksize,
            "training_cleared": False,
        }
        cuda_lib = load_cuda()
        if dry_run:
            plan["linears"] = linears
            plan["cuda"] = cuda_lib is not None
            return plan
        if cuda_lib is None and not allow_cpu:
            raise RuntimeError(
                "GLINT encode is CUDA (libglint_encode.so). "
                "bash scripts/build_glint_cuda.sh on the 3060, or pass --allow-cpu for tiny tests."
            )
        out.mkdir(parents=True, exist_ok=True)
        tensors = []
        reports = []
        for name in linears:
            info = st.tensors[name]
            n = numel_of(info.shape)
            raw = st.read_bytes(name)
            if cuda_lib is not None and info.dtype == "BF16":
                hole_b, plug_b, am_b = encode_bf16_cuda(raw, n, blocksize)
                am = unpack_f32(am_b)
                backend = "cuda_bf16"
                recon_rmse = None
            else:
                f32 = unpack_dense(info.dtype, raw, info.shape)[:n]
                hole_i, plug_i, am = encode(f32, blocksize)
                hole_b = bytes(pack_indices(hole_i))
                plug_b = bytes(pack_indices(plug_i))
                recon_rmse = rmse(f32, decode(hole_i, plug_i, am, blocksize))
                backend = "cpu_python"
            stem = name[: -len(".weight")]
            tensors.append((stem + ".weight", "U8", (len(hole_b), 1), hole_b))
            tensors.append((stem + ".weight.glint_plug", "U8", (len(plug_b), 1), plug_b))
            tensors.append((stem + ".weight.absmax", "F32", (len(am),), pack_f32(am)))
            state = json.dumps(
                {"quant_type": "glint", "blocksize": blocksize, "shape": list(info.shape), "training_cleared": False},
                separators=(",", ":"),
            ).encode()
            tensors.append((stem + ".weight.glint_state", "U8", (len(state),), state))
            reports.append(
                {
                    "src": stem,
                    "rmse": recon_rmse,
                    "shape": list(info.shape),
                    "backend": backend,
                }
            )
        for name in keep:
            info = st.tensors[name]
            tensors.append((name, info.dtype, info.shape, st.read_bytes(name)))
        tensors.append(("_glint_cells", "F32", (16, 16), pack_f32(cells_flat())))
    write_safetensors(
        str(out / "model.safetensors"),
        tensors,
        metadata={"format": "pt", "quantization": "glint", "training_cleared": "false"},
    )
    pin = {
        "schema": "glint_pin_v1",
        "src": str(wf),
        "dest": "glint",
        "n_converted": len(reports),
        "blocksize": blocksize,
        "rmse_mean": (
            sum(r["rmse"] for r in reports if r["rmse"] is not None)
            / max(1, sum(1 for r in reports if r["rmse"] is not None))
            if any(r["rmse"] is not None for r in reports)
            else None
        ),
        "backend": reports[0]["backend"] if reports else None,
        "training_cleared": False,
        "note": "hole=parent Gaussian cell, plug=child index. H-TILE inflate GLINT_CELLS[h][p]*absmax",
    }
    (out / "pin.json").write_text(json.dumps(pin, indent=2) + "\n")
    (out / "CONVERT_REPORT.json").write_text(json.dumps({"pin": pin, "modules": reports}, indent=2) + "\n")
    return {"pin": pin, "modules": reports}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="GLINT: BF16 → hole+plug pin")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--blocksize", type=int, default=64)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-cpu", action="store_true", help="Python encode (tiny tests only)")
    args = p.parse_args(argv)
    got = convert(Path(args.src), Path(args.out), args.blocksize, args.dry_run, args.allow_cpu)
    print(json.dumps(got if args.dry_run else got["pin"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
