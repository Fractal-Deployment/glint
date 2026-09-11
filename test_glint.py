#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from dtype_io import pack_bf16
from glint_codec import decode, encode, pack_indices, rmse, unpack_indices
from glint_htile_sim import gemm_ref_flat, htile_gemm_flat, max_abs_flat
from glint_offset import run as offset_run
from glint_pin import convert
from safetensors_io import SafeTensorsFile, write_safetensors


def test_codec_plug_is_location():
    w = [0.02 * i / 64.0 for i in range(128)]
    hole, plug, am = encode(w, 64)
    rec = decode(hole, plug, am, 64)
    assert all(0 <= h < 16 for h in hole)
    assert all(0 <= p < 16 for p in plug)
    assert rmse(w, rec) < rmse(w, [0.0] * len(w))


def test_pin_and_htile():
    N, K, M = 8, 64, 4
    w = [((i % 17) - 8) * 0.01 for i in range(N * K)]
    x = [((i % 9) - 4) * 0.02 for i in range(M * K)]
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        write_safetensors(
            str(src / "model.safetensors"),
            [
                ("lin.weight", "BF16", (N, K), pack_bf16(w)),
                ("model.norm.weight", "BF16", (8,), pack_bf16([1.0] * 8)),
            ],
        )
        dst = Path(td) / "pin"
        got = convert(src, dst, allow_cpu=True)
        assert got["pin"]["schema"] == "glint_pin_v1"
        assert got["pin"]["training_cleared"] is False
        with SafeTensorsFile(str(dst / "model.safetensors")) as st:
            assert "lin.weight.glint_plug" in st.tensors
            assert "_glint_cells" in st.tensors
            hole = unpack_indices(st.read_bytes("lin.weight"), N * K)
            plug = unpack_indices(st.read_bytes("lin.weight.glint_plug"), N * K)
        hole2, plug2, am = encode(w, 64)
        rec = decode(hole2, plug2, am, 64)
        y_ref = gemm_ref_flat(x, rec, M, N, K)
        y_ht = htile_gemm_flat(x, hole2, plug2, am, M, N, K, 64, 64, False)
        assert max_abs_flat(y_ref, y_ht) < 1e-6


def test_offset_board():
    b = offset_run(M=4, N=32, K=64, seed=1)
    assert b["htile_f32_vs_ref_glint_max_abs"] < 1e-6
    assert b["y_glint_vs_true_max_abs"] < b["y_parent_vs_true_max_abs"]
    assert b["training_cleared"] is False


if __name__ == "__main__":
    test_codec_plug_is_location()
    test_pin_and_htile()
    test_offset_board()
    print("TEST_GLINT_GREEN codec pin htile_sim offset not_training_cleared")
