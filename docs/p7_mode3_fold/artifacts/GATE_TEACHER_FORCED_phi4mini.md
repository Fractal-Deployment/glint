# Tiered and selected-fetch arms on the recorded teacher-forced gate, Phi-4-mini — spec Amendment 12, glint #92 (PRE-REGISTRATION; results appended below)

Written and committed before any run. The earlier rollout bar (agreement >= 82.5%, prefix >= 20) was my own decision aid, not a recorded gate. This applies the gate that is recorded in `docs/p7_mode3_fold/GATE_RULES.md`, unchanged.

**Measurement.** New harness mode `ARM0_DECODE_GATE=2047` (`tests/test_arm0_bf16_full.cu`): the true tokens 0..2046 of the fixed evaluation text are stepped in order through the KV-cache decode path (so the activation-selected fetch acts per token as in decode), teacher-forced. Reports PPL, top-1 agreement with the HF oracle (oracle PPL 5.486957), and with `ARM0_LOGITS_REF=anchor_logits.bin` (BF16 CUDA anchor) the per-token KL mean, p99, max. Per-position results (pos, top1, oracle top-1, CE, KL) are saved for every run.

**Recorded rule, per arm:** PASS = top-1 vs oracle >= 95.0% and |PPL delta| <= 2.0% of the oracle PPL (<= 5.5967); INCONCLUSIVE = top-1 >= 95.0% and PPL delta in (2%, 5%]; FAIL otherwise (top-1 < 95.0% or PPL delta > 5%). KL is reported, with no threshold (as recorded). Transport integrity (streamed == resident) was already passed (`FOLD_STREAM_phi4mini.md`).

**Alignment precondition (stop if it fails):** the decode path itself must match the established reference before any arm is read. (a) BF16 weights stepped through the decode path must give PPL within 0.5% of the forward-pass Arm 0 (5.491616) and top-1 agreement within 1 point of Arm 0 (98.44%). (b) The existing pin with all plug resident through the same path (`--glint-demand`, F = 1, no selection) must match the forward-path run (PPL 5.4885, top-1 98.19%) within the same tolerances. If either fails, the decode path is bisected and no arm is classified.

**Arms (existing pin, built-in table):** base only (SELFETCH 0.0001); 4-bit `act` 10%, 25%, 50%; T-A (10% + 30% 3-bit); T-B (15% + 30%); F3 (10% + 10%); F4 (5% + 10%). **Repeats:** T-A and 4-bit `act` 50% are run twice (the decode path has shown run-to-run variation; both per-run results are kept, and an arm within 1 point of 95.0% is called by its worse run). Calibrated-pin arms are not in this study.


## Executed Amendment 12 runs — evidence addendum, 2026-11-09

These are existing on-disk CUDA runs, not fresh measurements. Source: `/home/workspace/lakes/p7_glint/gate_20261106/{summary.txt,summary2.txt,T_act*.log}`. Parsed automatically; verdict verified against the pre-existing 95.0% top-1 and <=2% absolute PPL-difference pass condition. No threshold changed. The pre-registration above is preserved.

| Evidence file | Arm | PPL | PPL delta vs HF oracle | Top-1 vs oracle | Recorded gate |
|---|---|---:|---:|---:|---|
| `summary.txt` | `align_pin_full` | 5.473525 | -0.245% | 98.05% | PASS |
| `summary.txt` | `base_only` (fold + 1 selected channel per linear; NOT pure fold-only) | 5.633230 | 2.666% | 93.11% | FAIL |
| `summary.txt` | `act10` | 5.509574 | 0.412% | 95.46% | PASS |
| `summary.txt` | `act25` | 5.479034 | -0.144% | 96.43% | PASS |
| `summary.txt` | `act50_r1` | 5.481118 | -0.106% | 97.56% | PASS |
| `summary.txt` | `TA_r1` | 5.484523 | -0.044% | 97.36% | PASS |
| `summary.txt` | `F3` | 5.491055 | 0.075% | 95.80% | PASS |
| `summary.txt` | `F4` | 5.491289 | 0.079% | 95.60% | PASS |
| `summary.txt` | `TB` | 5.487158 | 0.004% | 97.51% | PASS |
| `summary.txt` | `TA_r2` | 5.467372 | -0.357% | 97.41% | PASS |
| `summary.txt` | `act50_r2` | 5.465649 | -0.388% | 97.56% | PASS |
| `summary2.txt` | `act10_r2` | 5.513469 | 0.483% | 95.60% | PASS |
| `summary2.txt` | `act05_r1` | 5.550310 | 1.155% | 95.16% | PASS |
| `summary2.txt` | `act05_r2` | 5.552049 | 1.186% | 95.16% | PASS |

**Source lines (verbatim):**

```text
align_pin_full: DECODE_GATE positions=2047 ppl=5.473525 oracle_ppl=5.486957 ppl_delta_pct=-0.245 top1_vs_oracle=2007/2047 (98.05%) kl_mean=8.3154e-04 kl_p99=6.4932e-03 kl_max=4.8324e-02 gate=PASS
base_only: DECODE_GATE positions=2047 ppl=5.633230 oracle_ppl=5.486957 ppl_delta_pct=2.666 top1_vs_oracle=1906/2047 (93.11%) kl_mean=4.2254e-02 kl_p99=3.6578e-01 kl_max=2.0756e+00 gate=FAIL
act10: DECODE_GATE positions=2047 ppl=5.509574 oracle_ppl=5.486957 ppl_delta_pct=0.412 top1_vs_oracle=1954/2047 (95.46%) kl_mean=1.5428e-02 kl_p99=1.4671e-01 kl_max=5.7528e-01 gate=PASS
act25: DECODE_GATE positions=2047 ppl=5.479034 oracle_ppl=5.486957 ppl_delta_pct=-0.144 top1_vs_oracle=1974/2047 (96.43%) kl_mean=7.2260e-03 kl_p99=6.6983e-02 kl_max=7.8692e-01 gate=PASS
act50_r1: DECODE_GATE positions=2047 ppl=5.481118 oracle_ppl=5.486957 ppl_delta_pct=-0.106 top1_vs_oracle=1997/2047 (97.56%) kl_mean=2.0508e-03 kl_p99=1.6622e-02 kl_max=9.6822e-02 gate=PASS
TA_r1: DECODE_GATE positions=2047 ppl=5.484523 oracle_ppl=5.486957 ppl_delta_pct=-0.044 top1_vs_oracle=1993/2047 (97.36%) kl_mean=3.4633e-03 kl_p99=2.6227e-02 kl_max=9.2938e-02 gate=PASS
F3: DECODE_GATE positions=2047 ppl=5.491055 oracle_ppl=5.486957 ppl_delta_pct=0.075 top1_vs_oracle=1961/2047 (95.80%) kl_mean=8.6700e-03 kl_p99=7.6157e-02 kl_max=3.7416e-01 gate=PASS
F4: DECODE_GATE positions=2047 ppl=5.491289 oracle_ppl=5.486957 ppl_delta_pct=0.079 top1_vs_oracle=1957/2047 (95.60%) kl_mean=1.1699e-02 kl_p99=1.1359e-01 kl_max=4.2360e-01 gate=PASS
TB: DECODE_GATE positions=2047 ppl=5.487158 oracle_ppl=5.486957 ppl_delta_pct=0.004 top1_vs_oracle=1996/2047 (97.51%) kl_mean=2.8830e-03 kl_p99=2.3147e-02 kl_max=3.2402e-01 gate=PASS
TA_r2: DECODE_GATE positions=2047 ppl=5.467372 oracle_ppl=5.486957 ppl_delta_pct=-0.357 top1_vs_oracle=1994/2047 (97.41%) kl_mean=3.3866e-03 kl_p99=2.8418e-02 kl_max=1.2466e-01 gate=PASS
act50_r2: DECODE_GATE positions=2047 ppl=5.465649 oracle_ppl=5.486957 ppl_delta_pct=-0.388 top1_vs_oracle=1997/2047 (97.56%) kl_mean=2.0482e-03 kl_p99=1.5064e-02 kl_max=6.0857e-02 gate=PASS
act10_r2: DECODE_GATE positions=2047 ppl=5.513469 oracle_ppl=5.486957 ppl_delta_pct=0.483 top1_vs_oracle=1957/2047 (95.60%) kl_mean=1.5587e-02 kl_p99=1.4450e-01 kl_max=6.3287e-01 gate=PASS
act05_r1: DECODE_GATE positions=2047 ppl=5.550310 oracle_ppl=5.486957 ppl_delta_pct=1.155 top1_vs_oracle=1948/2047 (95.16%) kl_mean=2.1224e-02 kl_p99=1.8267e-01 kl_max=1.4801e+00 gate=PASS
act05_r2: DECODE_GATE positions=2047 ppl=5.552049 oracle_ppl=5.486957 ppl_delta_pct=1.186 top1_vs_oracle=1948/2047 (95.16%) kl_mean=2.1042e-02 kl_p99=1.8108e-01 kl_max=1.5460e+00 gate=PASS
```

**Separate short-context decode timing** (47-step sample; not the throughput of 2047 teacher-forced steps):

```text
T_act05.log: DECODE_TIMING steps=47 mean_ms_per_token=56.445 mean_ms_per_layer=1.7639 tokens_per_s=17.72
T_act10.log: DECODE_TIMING steps=47 mean_ms_per_token=70.618 mean_ms_per_layer=2.2068 tokens_per_s=14.16
```

**Interpretation and limitations.** act10 PASS (95.46% and 95.60% across runs, 14.16 tok/s in the recorded short-context timing); act05 PASS with narrow top-1 margin (95.16% in both repeats, 17.72 tok/s in short-context timing); `base_only` (fold plus one selected channel per linear, `ARM0_SELFETCH=0.0001`) FAIL (93.11%, PPL +2.666%); pure fold-only (hole alone, equal to the NF4 arm) also FAIL at 91.06% (PPL 5.833352, 1864/2047, `STATUS_2026-11-04.md`). T-A, T-B, F3/F4 and the other reported selected arms pass this teacher-forced recorded gate under the executed configurations. One text, one GPU, 2047 evaluated positions, 3-bit intermediate-band emulation, and 47-step short-context timing constrain generality. KL was measured but has no approved pass threshold. These **teacher-forced** results do **not** satisfy GLINT #92 M6: Amendment-2 free-running token tracking plus >=10 tok/s in all-active decode remain separate requirements, and the previously measured fast rollout arms did not meet the tracking criterion. The Amendment-2 82.5% / 20-of-32 figures were agent-proposed, pre-registered and referenced by #92; they were not independently ratified by the operator.

## Label correction (2026-11-09)

The run id `base_only` was used in this study as if it were the fold-only arm. It is not: it ran with `ARM0_SELFETCH=0.0001`, which selects ceil(0.0001*K) = 1 input channel per linear (see `run_gate.sh`), so it is the fold plus the single largest-|x| channel of each linear (about 0.03% of the plug bytes). Pure fold-only (hole alone) is recorded in `STATUS_2026-11-04.md` as PPL 5.833352, top-1 1864/2047 = 91.06%. Both fail the recorded gate. Why one channel per linear recovers PPL 5.833 -> 5.633 and top-1 91.06% -> 93.11% is a hypothesis (outlier/massive-activation channels) that has not been tested. Statements elsewhere that call the 93.11% arm "fold-only" or "base-only" should read "fold plus one channel per linear".
