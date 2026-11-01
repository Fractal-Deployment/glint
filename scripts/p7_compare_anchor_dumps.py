"""Layer-by-layer comparison of the C/CUDA Arm 0 dumps against the HF reference dump.

Usage: python p7_compare_anchor_dumps.py HF_DIR CUDA_DIR
HF_DIR holds p7_hf_reference_dump.py output; CUDA_DIR holds test_arm0_bf16_full dump_dir output.
"""
import json
import sys

import numpy as np

hf_dir, cuda_dir = sys.argv[1], sys.argv[2]
positions = json.load(open(f"{hf_dir}/hf_meta.json"))["positions"]
n = len(positions)
hf = np.fromfile(f"{hf_dir}/hf_hidden.bin", np.float32).reshape(33, n, 3072)
cu = np.fromfile(f"{cuda_dir}/cuda_hidden.bin", np.float32).reshape(33, n, 3072)
stats = np.fromfile(f"{hf_dir}/hf_pos_stats.bin", np.float32).reshape(2047, 4)
raw = np.fromfile(f"{cuda_dir}/cuda_pos.bin", np.uint8)
cuda_loss = raw[:2047 * 4].view(np.float32)
cuda_top1 = raw[2047 * 4:].view(np.int32)

print("slot  rel_err_median  rel_err_max@pos  min_cosine   (slot 0 embed, 1..31 layer output, 32 final norm)")
for s in range(33):
    diff = np.linalg.norm(cu[s] - hf[s], axis=-1) / np.linalg.norm(hf[s], axis=-1)
    cos = (cu[s] * hf[s]).sum(-1) / np.linalg.norm(cu[s], axis=-1) / np.linalg.norm(hf[s], axis=-1)
    print(f"{s:4d}  {np.median(diff):14.5f}  {diff.max():8.5f}@{positions[diff.argmax()]:<5d}  {cos.min():.6f}")

mismatch = np.nonzero(cuda_top1 != stats[:, 1].astype(np.int32))[0]
margin = stats[mismatch, 2] - stats[mismatch, 3]
print(f"top1 mismatches {len(mismatch)} / 2047; HF top1-top2 logit margin at mismatches: "
      f"exact ties {int((margin == 0).sum())}, <=0.25 {int((margin <= 0.25).sum())}, max {margin.max() if len(margin) else 0:.4f}")
for i, m in zip(mismatch, margin):
    print(f"  pos {i:4d} hf_loss {stats[i, 0]:.4f} cuda_loss {cuda_loss[i]:.4f} hf_margin {m:.4f}")
print(f"mean |loss diff| {np.abs(cuda_loss - stats[:, 0]).mean():.5f}")
