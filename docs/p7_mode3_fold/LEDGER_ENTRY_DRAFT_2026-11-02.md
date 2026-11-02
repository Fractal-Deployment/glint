# P7 results ledger entry (DRAFT for TD review, glint #22)

Phi-4-mini-instruct, RTX 3060 sm_86, 2047 evaluation tokens (`artifacts/eval_tokens_2048.txt`, sha in `hf_meta.json`), HF BF16 sdpa oracle PPL 5.486957. KL is KL(BF16 CUDA anchor || arm) per token. Every number below is from a run on this machine in glint `main`; commands are the harness invocations named in each PR.

| Claim | Evidence | Result |
|---|---|---|
| Gate 1 anchor (BF16 CUDA vs HF), threshold 95.0% top-1, PPL within 1.0% | `scripts/run_arm0_anchor.sh`, PR #45 | PPL 5.491616 (+0.085%), top-1 2015/2047 = 98.44%, PASS |
| INT8 pin, plain W8 tile GEMM | PR #36 | PPL 5.501980, top-1 97.85%, KL mean 1.25e-3 |
| INT8 W8A8 tensor-core GEMM | PR #36/#38 | PPL 5.496700, top-1 97.85%, KL mean 1.75e-3, 3.37 s |
| Mode 2 offload: streamed == resident | PR #35 (1.5B, bitwise over 23.9M floats), PR #37 (Phi-4, CE and top-1 hash identical) | linear weights in VRAM 3.188 -> 0.199 GiB |
| Attach gate | PRs glint #40, orch #367 | selftest rc=0, 0 mismatches; forced failure exits 4 |
| Weight maps | PR #41, #49 | INT8 RMSE 1.98e-4 / max 1.10e-2; NF4 RMSE 3.05e-3 / max 0.156 |
| Plug (res4) | PR #42, #43 | RMSE 1.98e-4 -> 1.39e-5; 1.81 GB |
| Plug selection, 8 layers | PR #44 | oracle-chosen KL 1.24e-3, best static 1.48e-3, none 1.75e-3 (oracle is in-sample) |
| Resolution + plugs | PR #46 | KL 4.17e-4, top-1 98.49%, PPL 5.476566, 6.03 s |
| NF4 comparison | PR #49 | PPL 5.833352, top-1 91.06% (fails 95.0%) |
| Activation maps | PR #47, #48, #50 | BF16 rerun byte-identical; INT8 vs BF16 rms mean 0.16%, NF4 2.13% |

Not claimed: Mode 3/4 (fold) fidelity, M4 lossy sweep, the equal-bytes comparison against a BF16 model of the same total size, any run on a model other than Phi-4-mini (Mode 2 equivalence also on DeepSeek-R1-Distill-Qwen-1.5B), the 100%-on-confident-positions check of the Gate 1 amendment.
