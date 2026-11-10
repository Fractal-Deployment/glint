# Mojo/MAX GLINT KV-cache decode, resident and streamed (2026-11-09)

**Project:** GitHub Project #6, issues #116 (P7-MOJO-D1, Phase B) and #117 (P7-MOJO-S1, first step). **Status:** EXECUTED teacher-forced reference results; PHASE B and PHASE C NOT COMPLETE. **M6 NOT ACCEPTED.** This document records executed runs only.

## What is committed

| File | Role |
|---|---|
| `mojo/p7_glint_linear.mojo` | pin-sourced single-linear GLINT GEMV probe (3072x3072 row check vs CPU reference, max abs diff 5.668e-7) |
| `mojo/p7_glint_decode.mojo` | full 32-layer single-token KV decode over `glint_pin_v1` (hole, plug, absmax, embedded `_glint_cells`), resident plug, warp-based M=1 GEMV |
| `mojo/p7_glint_stream_decode.mojo` | same decode with the whole per-layer plug streamed from host through a two-slot ring (no selection) |
| `mojo/p7_refine.mojo` | `RefineConfig` scaffold: transfer granularity, prefetch distance, selection rule, application rule, staging slots as separate fields. A typed protocol only, **not** a scheduler; only whole-layer transfer and "all" selection are exercised |
| `scripts/p7_mojo_glint_decode.sh` | build and run recipe (40-Runtime-Deps, GPU lease) |

`p7_forward.mojo` and `p7_prefetch.mojo` are unchanged.

## Provenance and reproduction

Mojo 1.0.0 ed45d567, `MODULAR_HOME=/root/.pixi/envs/modular/share/max`, RTX 3060, driver 595.99.02, GPU lease. Pin `/home/workspace/pins/Phi-4-mini-glint-from-bf16`; checkpoint `microsoft/Phi-4-mini-instruct@cfbefacb99257ffa30c83adab238a50856ac3083`; tokens and HF oracle from `artifacts/`. The sources above were written uncommitted during the original runs; before this commit both decoders were **rebuilt from the worktree files and re-run at 32 steps**: both builds exit 0, resident and streamed 32-step outputs are identical to each other and identical to the first 32 positions of the original 2047-position run (`/home/workspace/lakes/p7_glint/mojo_repro_20261109/`). Original logs: `/home/workspace/lakes/p7_glint/mojo_decode_preflight_20261109/`.

## Direct results (2047 teacher-forced positions)

| Path | PPL | Oracle top-1 | Decode time | Speed |
|---|---:|---:|---:|---:|
| Mojo GLINT resident | 5.4847669445 | 2007/2047 (98.05%) | 268.35 s | 7.63 tok/s |
| Mojo GLINT streamed (whole plug each token) | 5.4847669445 | 2007/2047 (98.05%) | 403.24 s | 5.08 tok/s |

- Streamed == resident at all 2047 positions (top-1 id and cross-entropy strings identical, 0 differences). This is the same-path transport check, at the granularity the log records.
- Both pass the recorded teacher-forced thresholds (top-1 >= 95%, PPL within 2% of 5.486957).
- The streamed run requests 1,610,612,736 plug bytes per token. That figure is **requested** bytes; independent H2D accounting, stall and overlap telemetry are **not** recorded and remain open for #117.
- Warp GEMV vs the earlier scalar GEMV: four different top-1 predictions in a 32-token comparison; unexplained numerical sensitivity, kept as a finding.

## Cross-backend comparison (same pin, same 2047 positions)

Files: `/home/workspace/lakes/p7_glint/xbackend_20261109/`.

| Pair | Top-1 agree | mean abs dCE | p99 dCE | max dCE |
|---|---|---:|---:|---:|
| CUDA H-TILE decode vs CUDA GEMV decode (same code, same pin) | 2015/2047 (98.44%) | 0.0271 | 0.231 | 0.351 |
| Mojo vs CUDA GEMV | 2015/2047 (98.44%) | 0.0256 | 0.225 | 0.380 |
| Mojo vs CUDA H-TILE | 2017/2047 (98.53%) | 0.0255 | 0.226 | 0.300 |

Top-1 vs oracle: Mojo 2007, CUDA H-TILE 2007, CUDA GEMV 2000. The equal Mojo/H-TILE count is a coincidence: of their 30 disagreeing positions each matches the oracle on 15. The Mojo-vs-CUDA gap is the same size as the gap between two CUDA kernels on one pin, so Mojo's GLINT decode is inside the numerical variation CUDA already shows against itself. It is **not** bitwise or intermediate-tensor parity, and it covers one text and one model.

## Timing (approximate, short context only)

- CUDA resident hole+plug GEMV decode, 47 steps after a 64-token prefix, 5 runs: 35.1-38.4 ms/token, 26.1-28.5 tok/s.
- Mojo resident decode, 32 steps: 98.8-102.3 ms/token, 9.77-10.12 tok/s (two runs). Positions differ (0-31 vs 64-111). CUDA is roughly 2.6-2.9x faster here; this is not a matched long-context ratio.
- The CUDA 2047-position gate took 267 s wall including load and per-position bookkeeping, so no per-token claim is made for it.

## What this does not establish

- M6, free-running tracking under Amendment 2, or any >= 10 tok/s claim with all mechanisms active.
- Activation-driven selection, prefetch timing, grouped transfer, last-consumer buffer reuse, independent measured H2D bytes, overlap, or peak VRAM in Mojo.
- Whether CUDA's `gemv_init` reads the pin's `_glint_cells` (it builds its table from compiled constants; the two tables are byte-identical for current pins, SHA256 `e774eceefa0d565e330f966d8ba1ea64f09208f4769bb6dd512bcd419bcb0365`).
- Cross-model or cross-text behaviour.

## Streamed decode: copy, compute and overlap (ablation, 64 steps, 3 repeats each)

`p7_glint_stream_decode.mojo` takes an optional 5th argument: 0 = normal, 1 = copies only, 2 = compute only (no plug copies). Modes 1 and 2 print `MOJO_GLINT_STREAM_ABLATION ... outputs invalid` and exist only for timing. Rule fixed before the runs: overlap = copy-only + compute-only - full. Default mode 0 outputs are unchanged: all 64 positions of all three runs are identical to the original resident run. Files: `/home/workspace/lakes/p7_glint/mojo_stream_ablation_20261109/`.

| Mode | ms/token, three runs | Median |
|---|---|---:|
| full (copy + compute) | 207.5, 182.9, 184.9 | 184.9 |
| copy only | 144.6, 140.3, 143.6 | 143.6 |
| compute only | 85.0, 86.1, 85.0 | 85.0 |

- Copy-only moves 1,610,612,736 requested bytes per token in 143.6 ms: **11.22 GB/s**, about 97% of the 11.6 GB/s PCIe Gen3 x16 figure used elsewhere in this project. Whole-plug streaming is bandwidth-bound.
- Overlap = 143.6 + 85.0 - 184.9 = **43.7 ms**; about 51% of the compute side is hidden. A perfect pipeline would cost max(143.6, 85.0) = 143.6 ms, so about 41 ms per token (about 22%) of the full-run time is not explained by the link. Cause not isolated: the two-slot ring allows only one layer of lookahead, and each layer issues four separate copies. This is a measurement, not a diagnosis.
- **Ceiling for whole-plug streaming: about 7.0 tok/s** at this link speed, regardless of how fast the compute gets. Reaching 10 tok/s (<= 100 ms/token) needs fewer bytes per token, roughly <= 1.1 GB at this bandwidth, so a selected fetch (or a smaller plug code) is required before overlap tuning can matter for the 10 tok/s target.
- Compute-only here (85.0 ms) is lower than the resident decode at 32 steps (98.8-102.3 ms); positions and run lengths differ and I have not reconciled the two.
- Short context only (positions 0-63); not a 2047-position figure. Not M6.

## Activation-selected plug decoder (executed and retested, 2047 teacher-forced positions)

`mojo/p7_glint_select_decode.mojo` (GLINT #117, selected gather) decodes with the fold (parent) GEMV and then applies, per linear, a correction from the plug rows of the input channels with the largest current |x|:

- Parent pass: the existing warp GEMV with a table whose rows are the NF4 parent values (`glint_codec.PARENT`), so the plug nibble is ignored.
- Selection: per linear, the linear's bf16 input is copied to the host and the channels whose |x| key is >= the n-th largest key are kept, n = ceil(percent*K/100) (ties all kept; same rule as the CUDA act rule). Optional 6th argument fixes the channel count per linear instead (1 = the single largest-|x| channel). This is a reference implementation with one blocking host round trip per linear, **not** the fast path.
- Gather: a kernel reads the selected K-major plug rows directly from pinned host memory through a non-owning view of the host pointer (zero-copy). A probe showed this works in Mojo 1.0/MAX 26.5 with 0 mismatches over 64 MiB at 11.27 GB/s sequential (`/home/workspace/lakes/p7_glint/mojo_zerocopy_probe_20261109/`).
- Application: `y += sum_i (cell[h][p] - parent[h]) * absmax * x[c_i]`, summed in ascending channel order (deterministic; the CUDA path sums partials, so results are not bit-identical to it).
- The K-major plug copy is built at load time by a device transpose kernel and kept in pinned host memory.

**Status.** Executed and retested. A first pass overlapped a VM start on the same GPU (qemu `usb-wiper`, process start 14:46:54); the runs that overlapped it (0% and the one-channel run) were rerun, and the 10% and 5% runs were rerun as well. Reruns used `scripts/p7_mojo_select_retest.sh`, whose idle guard refuses to run with other compute processes, a running qemu VM, or mean utilization above 40% over 10 s (the summary records ambient desktop graphics, 30-31% in the reruns; no compute process, no VM). The decoder is deterministic: every rerun equals its first pass at all 2047 positions. Files: `/home/workspace/lakes/p7_glint/mojo_select_20261109/` (first pass), `mojo_select_retest_20261109/`, `mojo_select_retest2_20261109/`, `mojo_select_retest3_20261109/` (reruns). One rerun attempt died on an LXD `snapctl` error after a host restart (not a decoder error) and was repeated.

### Results (same pin, same 2047 teacher-forced positions, recorded gate: top-1 >= 95.0%, PPL within 2% of 5.486957)

| Mojo arm | Gathered bytes/token | PPL | PPL vs oracle | Top-1 vs oracle | Gate | ms/token (tok/s) |
|---|---:|---:|---:|---:|---|---:|
| fold only (0%) | 0 | 5.847467 | +6.57% | 1870/2047 (91.35%) | FAIL | 109.8-111.4 (9.0-9.1) |
| fold + 1 channel per linear | 0.45 MB | 5.634820 | +2.70% | 1905/2047 (93.06%) | FAIL | 134.0 (7.46) |
| 5% of channels | 81.5 MB | 5.535055 | +0.88% | 1940/2047 (94.77%) | **FAIL (top-1 0.23 points under)** | 163.6-174.2 (5.7-6.1) |
| 10% of channels | 162.7 MB | 5.511664 | +0.45% | 1957/2047 (95.60%) | PASS | 199.8-219.7 (4.6-5.0) |
| resident full plug (earlier) | 0 (resident) | 5.484767 | -0.04% | 2007/2047 (98.05%) | PASS | 131.1 (7.63) |

Timing ranges are first pass to rerun (the rerun had about 30% ambient graphics load). Decode is deterministic, so fidelity columns are identical between passes.

**Against CUDA on the same pin and gate** (CUDA act arms from `GATE_TEACHER_FORCED_phi4mini.md`):

| Arm | Mojo top-1 / PPL | CUDA top-1 / PPL | Mojo vs CUDA per-position top-1 agree | mean abs dCE |
|---|---|---|---:|---:|
| 10% | 95.60% / 5.5117 | 95.46-95.60% / 5.5096-5.5135 | 2016/2047 (98.49%) | 0.0277 |
| 5% | 94.77% / 5.5351 | 95.16% / 5.5503-5.5520 | 2019/2047 (98.63%) | 0.0286 |
| fold + 1 channel | 93.06% / 5.6348 | 93.11% / 5.6332 | not computed | not computed |

- The selection semantics reproduce CUDA's: Mojo's 10% arm lands inside CUDA's run-to-run range on both PPL and top-1, and the fold-plus-one-channel arm matches CUDA's `base_only` to 0.03% PPL and one position. Per-position agreement with CUDA (98.5-98.6%) is the same size as CUDA-vs-CUDA kernel variation on this pin (98.44%).
- **The 5% arm fails the top-1 bar on Mojo (94.77%) where it passed narrowly on CUDA (95.16%).** The gap is 8 positions; PPL is better on Mojo (5.535 vs 5.550). Under the project's worse-run rule for arms near the threshold, 5% is not a passing arm on Mojo.
- **Hypothesis now reproduced on both backends** (not explained): one selected channel per linear (0.03% of the plug bytes) recovers PPL 5.847 -> 5.635 and top-1 91.35% -> 93.06%. A natural suspect is large-activation outlier channels; no test of that has been run.

**Speed.** This decoder selects on the host with one blocking device-to-host copy and synchronize per linear (128 per token), then gathers by zero-copy kernel. The cost of that round trip alone is visible in the one-channel run: 134.0 ms/token against 109.8 for fold only, for 0.45 MB gathered. The 10% arm runs at 4.6-5.0 tok/s, below the 10 tok/s M6 target, so this is a **correctness and semantics reference, not the fast path**. CUDA's act 10% ran at 14.2 tok/s in short-context timing.

**What this establishes and does not.** Established: activation-selected plug application with a zero-copy gather runs inside the real Mojo KV decode, reproduces the CUDA arms' fidelity within CUDA's own variation, and is deterministic. Not established: any M6 claim (no free-running tracking study was run on Mojo; throughput is under 10 tok/s), device-side selection, prefetch ahead of the consuming linear, overlap of the gather with compute, independent measured H2D bytes (the byte counts are gathered rows, not NIC-level counters), peak VRAM, other texts or models. #117 stays open.


**Label correction (CUDA gate study).** The CUDA gate run named `base_only` (`ARM0_SELFETCH=0.0001`) selects ceil(0.0001*K) = 1 channel per linear, so it is the fold plus one selected channel per linear, not a pure fold-only arm. Pure fold-only (hole alone, equal to the NF4 arm) is recorded in `STATUS_2026-11-04.md` (PPL 5.833352, top-1 1864/2047 = 91.06%). Whether the single largest-|x| channel alone explains the gap between those two numbers is a hypothesis not yet tested on either backend.

## Parallel kernels ("fast" flag), 2047-position gate, same pin, 2026-11-09 evening

Per-token cost breakdown of the reference decoder (short context): logits 32 ms, GEMV 37 ms, cross-entropy 11 ms, rms_norm 10 ms (single-thread kernels), attention 1.7 ms at position 100 rising to 55 ms at 2000. New kernels (`p7_decode_fast_kernels.mojo`, `p7_glint_gemv_fast.mojo`) compute the same math with a different summation order, so they are not bit-identical; they are selected with a 7th argument `fast` and the reference path is unchanged.

| Arm (Mojo, same pin) | kernels | top-1 vs oracle | PPL | tok/s |
|---|---|---|---|---|
| fold-only (0%) | reference | 1870/2047 = 91.35% | 5.8475 | 9.1 |
| fold-only (0%) | fast | 1868/2047 = 91.25% | 5.8418 | 17.1 |
| act 10% | reference | 1957/2047 = 95.60% | 5.5117 | 4.55 |
| act 10% | fast | 1954/2047 = 95.46% | 5.5075 | 7.57 |

Per-position top-1 agreement, fast vs reference, act 10%: 2016/2047 = 98.49% (recorded CUDA-vs-CUDA / Mojo-vs-CUDA noise floor 98.44-98.53%). act 10% still passes the recorded gate (>=95.0%, PPL within 2% of 5.487).

Not shown: M6 (10 tok/s is not reached by any gate-passing arm; free-running tracking under Amendment 2 not measured on Mojo). Remaining cost at 2047 positions is attention (sequential per-position loop) and 128 per-token host round trips in the selection step. Fold-only fails the gate as before. 32-step runs are not used for any claim.

### Position-parallel attention (`fast+attn`), 2047 positions, same pin, idle-guarded

`mojo/p7_attention_split.mojo`: positions of one head split over 8 blocks, 4 warps per block with their own online-softmax state, then a combine kernel. One-layer microbench (`p7_attention_split_bench.mojo`, synthetic K/V): position 100 0.075 -> 0.009 ms, 1000 0.99 -> 0.043 ms, 2046 1.98 -> 0.066 ms; max abs output difference <= 1.1e-3 on values of magnitude ~1 (bf16 rounding of the softmax weight happens against a different running max).

| Arm (Mojo, same pin) | kernels | top-1 vs oracle | PPL | tok/s |
|---|---|---|---|---|
| act 10% | reference | 95.60% | 5.5117 | 4.55 |
| act 10% | fast | 95.46% | 5.5075 | 7.57 |
| act 10% | fast+attn | 95.55% (1956/2047) | 5.5124 | 8.81 |
| fold-only | fast+attn | 91.40% (1871/2047) | 5.8322 | 29.8 |

Per-position agreement at act 10%: fast+attn vs reference 98.73%, fast+attn vs fast 98.39% (noise floor 98.44-98.53%). Passes the recorded gate. 8.81 tok/s is still below 10: fold-only runs at 33.6 ms/token, so ~80 ms/token of the act-10% arm is selection (128 host round trips per token plus 163 MB/token gather, which alone is >= 14.5 ms at 11.2 GB/s). M6 not met.
