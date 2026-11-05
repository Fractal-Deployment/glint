# GLINT arms on Qwen3-4B-Base (CUDA harness), 2026-11-04 — glint #85

`tests/test_q3_arms.cu`: the Arm 0 method for Qwen3-4B-Base. 2047 tokens of `SCIENCE_JOURNAL.md`, oracle = HF bf16 sdpa (PPL 4.514359; oracle files from `generate_oracle_ref_hf.py`, in the Qwen3 Mojo-forward branch, PR #77). Same bf16 rounding points as the Phi-4 harness, cuBLAS GEMMs on bf16 weights, reusing the shared norm, residual, cross-entropy and KL kernels; new kernels for what Qwen3 does differently: separate q/k/v and gate/up projections, per-head QK RMSNorm before a full-width RoPE (theta 1e6, tables built in double then rounded to bf16), grouped-query attention 32:8, eps 1e-6, tied embedding (vocab 151936), three shards. Linears come from the BF16 model (`Q3_MODE=bf16`) or from `Qwen3-4B-Base-glint-from-bf16` decoded to bf16 one matrix at a time: `full` = `GLINT_CELLS[hole][plug] * absmax`, `fold` = `PARENT[hole] * absmax`. Embedding and norms come from the BF16 model in every mode. RTX 3060, one run per arm. Logs: `q3_arms/*.log`; raw under `/home/workspace/lakes/p7_glint/q3_arms_20261104/`.

| Arm | PPL | PPL vs oracle | top-1 vs oracle | mean KL vs the BF16 arm | forward |
|---|---|---|---|---|---|
| BF16 (CUDA anchor) | 4.512231 | +0.047% (it is below the oracle by 0.047%) | 2017 / 2047 (98.53%) | 0 | 6.2 s |
| GLINT hole + plug | 4.518957 | +0.102% | 2009 / 2047 (98.14%) | 3.78e-4 (p99 3.2e-3, max 2.7e-2) | 6.0 s |
| GLINT fold only (the hole alone, NF4) | 4.615653 | +2.244% | 1892 / 2047 (92.43%) | 2.87e-2 (p99 0.20, max 2.14) | 5.6 s |
Earlier Qwen3 arms for reference (Mojo forward, INT8 base, PR #77/#83): INT8 4.511863 / 97.90%; + res4 plugs + resolution 4.515624 / 98.29%; no KL anchor existed for them.

## Reading
- *Measured:* the forward is correct: the BF16 arm is within 0.05% PPL and 98.5% top-1 of the HF oracle, the same quality as the Phi-4 anchor (98.44%), on the first run.
- *Measured:* GLINT behaves on Qwen3 as on Phi-4. The hole alone loses 2.2% PPL and about 6 points of top-1; adding the plug recovers to 0.10% PPL and 98.14% top-1, with KL 3.8e-4 against the BF16 arm (Phi-4: 9.5e-4). Qwen3's fold-only KL (2.9e-2) is lower than Phi-4's (7.9e-2), so Qwen3 is less sensitive to the NF4 hole.
- The decode, selected-fetch and rollout studies were run on Phi-4-mini only.

## What this does not show
One text, one run per arm. The embedding is BF16 in all arms (the glint pin also encodes it; not used). No streaming, decode or activation-selected fetch on Qwen3; no INT4 or other precisions on Qwen3. DeepSeek-R1-Distill-1.5B (Qwen2 layout with q/k/v biases) and SmolLM2-135M (Llama layout) are not covered. Not reviewed by the Test Designer.

## Reproduce
`nvcc -O2 -arch=sm_86 -Iinclude -o q3arms tests/test_q3_arms.cu -lcublas`; `Q3_MODE=bf16 ARM0_LOGITS_OUT=a.bin q3arms /home/workspace/pins/Qwen3-4B-Base-BF16 /home/workspace/pins/Qwen3-4B-Base-glint-from-bf16 ARTIFACTS`; then `Q3_MODE=full|fold ARM0_LOGITS_REF=a.bin q3arms ...`.
