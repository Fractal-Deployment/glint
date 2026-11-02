"""HuggingFace BF16 references for the resolution arms (offline oracle only).

Usage: python p7_resolution_reference.py ARTIFACTS_DIR RESOLUTION_ARTIFACTS_DIR LOGITS_OUT [MODEL_DIR]
The corpus is P7_CORPUS_PATH, by default /opt/models/science_log/SCIENCE_JOURNAL.md.

Tokenizes the corpus the eval tokens came from and checks that its first 2048 tokens are
ARTIFACTS_DIR/eval_tokens_2048.txt. Writes the next 2048 tokens, disjoint from the eval tokens, to
RESOLUTION_ARTIFACTS_DIR/calib_tokens_2048.txt. Runs the BF16 model on the eval tokens and writes its
logits for positions 0..2046 as raw bf16 [2047, vocab] to LOGITS_OUT, refusing to write if their argmax
differs from the ruler's oracle_top1_2048.bin. Records the logits file's sha256 in
RESOLUTION_ARTIFACTS_DIR/hf_logits_meta.json.
"""
import hashlib
import json
import os
import sys

import torch
import transformers
from transformers import AutoTokenizer, Phi3ForCausalLM

CORPUS_PATH = os.environ.get("P7_CORPUS_PATH", "/opt/models/science_log/SCIENCE_JOURNAL.md")
MODEL_DIR = "/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083"

artifacts_dir, res_dir, logits_out = sys.argv[1], sys.argv[2], sys.argv[3]
model_dir = sys.argv[4] if len(sys.argv) > 4 else MODEL_DIR
os.makedirs(res_dir, exist_ok=True)

eval_tokens = [int(t) for t in open(f"{artifacts_dir}/eval_tokens_2048.txt").read().split()]
corpus_tokens = AutoTokenizer.from_pretrained(model_dir).encode(open(CORPUS_PATH, encoding="utf-8").read())
assert corpus_tokens[:2048] == eval_tokens, "corpus tokenization does not reproduce eval_tokens_2048.txt"
calib = corpus_tokens[2048:4096]
assert len(calib) == 2048, f"corpus has {len(corpus_tokens)} tokens, need 4096"
open(f"{res_dir}/calib_tokens_2048.txt", "w").write(" ".join(str(t) for t in calib))

model = Phi3ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
model.eval()
with torch.no_grad():
    logits = model(torch.tensor([eval_tokens], device="cuda")).logits[0, :-1].contiguous()
assert logits.dtype == torch.bfloat16

top1 = torch.argmax(logits, dim=-1).to(torch.int32).cpu().numpy().astype("<u4").tobytes()
oracle = open(f"{artifacts_dir}/oracle_top1_2048.bin", "rb").read()
agree = sum(a == b for a, b in zip(memoryview(top1).cast("I"), memoryview(oracle).cast("I")))
assert top1 == oracle, f"HF argmax agrees with oracle_top1_2048.bin on {agree}/2047 rows only"

raw = logits.view(torch.int16).cpu().numpy().tobytes()
open(logits_out, "wb").write(raw)
meta = {
    "logits_file": logits_out,
    "shape": list(logits.shape),
    "dtype": "bf16",
    "sha256": hashlib.sha256(raw).hexdigest(),
    "eval_tokens_sha256": hashlib.sha256(open(f"{artifacts_dir}/eval_tokens_2048.txt", "rb").read()).hexdigest(),
    "calib_tokens": "corpus tokens 2048..4095",
    "calib_tokens_sha256": hashlib.sha256(open(f"{res_dir}/calib_tokens_2048.txt", "rb").read()).hexdigest(),
    "attn_implementation": model.config._attn_implementation,
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "device": torch.cuda.get_device_name(),
}
json.dump(meta, open(f"{res_dir}/hf_logits_meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
