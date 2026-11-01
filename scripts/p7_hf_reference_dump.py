"""HuggingFace BF16 reference dump for the Arm 0 anchor (offline oracle only).

Usage: python p7_hf_reference_dump.py TOKENS_TXT OUT_DIR [MODEL_DIR] [ATTN_IMPL]

Writes to OUT_DIR:
  hf_pos_stats.bin   float32 [2047, 4]: loss, top1 id, top1 logit, top2 logit
  hf_hidden.bin      float32 [33, len(POSITIONS), 3072]: embeddings, layers 0..31 output, final norm
  hf_meta.json       ppl, top1 list sha256, positions, dtype, attention implementation
"""
import hashlib
import json
import struct
import sys

import torch
import torch.nn.functional as F
from transformers import Phi3ForCausalLM

MODEL_DIR = "/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083"
POSITIONS = list(range(16)) + list(range(16, 2048, 64)) + [2047]

tokens_path, out_dir = sys.argv[1], sys.argv[2]
model_dir = sys.argv[3] if len(sys.argv) > 3 else MODEL_DIR
attn_impl = sys.argv[4] if len(sys.argv) > 4 else "sdpa"
tokens = [int(t) for t in open(tokens_path).read().split()]
assert len(tokens) == 2048, len(tokens)

model = Phi3ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn_impl)
model.eval()
ids = torch.tensor([tokens], device="cuda")
with torch.no_grad():
    out = model(ids, output_hidden_states=True)

logits = out.logits[0, :-1]
labels = ids[0, 1:]
stats = torch.empty(2047, 4, dtype=torch.float32)
top1 = []
for i in range(0, 2047, 256):
    chunk = logits[i:i + 256]
    loss = F.cross_entropy(chunk.float(), labels[i:i + 256], reduction="none")
    vals, idx = torch.topk(chunk.float(), 2, dim=-1)
    arg = torch.argmax(chunk, dim=-1)
    stats[i:i + 256, 0] = loss.cpu()
    stats[i:i + 256, 1] = arg.float().cpu()
    stats[i:i + 256, 2] = vals[:, 0].cpu()
    stats[i:i + 256, 3] = vals[:, 1].cpu()
    top1 += arg.cpu().tolist()

stats.numpy().tofile(f"{out_dir}/hf_pos_stats.bin")
hidden = torch.stack([h[0, POSITIONS].float().cpu() for h in out.hidden_states])
hidden.numpy().tofile(f"{out_dir}/hf_hidden.bin")

top1_bytes = b"".join(struct.pack("<I", t) for t in top1)
open(f"{out_dir}/hf_top1.bin", "wb").write(top1_bytes)
mean_loss = float(stats[:, 0].double().mean())
meta = {
    "model_dir": model_dir,
    "tokens_sha256": hashlib.sha256(open(tokens_path, "rb").read()).hexdigest(),
    "dtype": "bfloat16",
    "attn_implementation": model.config._attn_implementation,
    "transformers": __import__("transformers").__version__,
    "torch": torch.__version__,
    "mean_loss": mean_loss,
    "ppl": float(torch.exp(torch.tensor(mean_loss, dtype=torch.float64))),
    "top1_sha256": hashlib.sha256(top1_bytes).hexdigest(),
    "positions": POSITIONS,
    "n_hidden": len(out.hidden_states),
}
json.dump(meta, open(f"{out_dir}/hf_meta.json", "w"), indent=2)
print(json.dumps({k: v for k, v in meta.items() if k != "positions"}, indent=2))
