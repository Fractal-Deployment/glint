# Coarser plug code (M5 compression direction), Phi-4-mini (2026-11-04) — spec Amendment 4, glint #92

Executes Amendment 4 of `docs/p7_mode3_fold/DEMAND_MAP_SPEC_2026-11-04.md` (arms and rules fixed before any data). `GLINT_PLUG_BITS=b` replaces the 16 x 16 cell table by one where the 2^(4-b) adjacent child indices that would share a b-bit code all take the mean of their cell values (the merged bin's reconstruction level): an **emulation of reconstruction**. Bytes are counted analytically as `f x (b/4)` of the 4-bit plug (1.61 GB); this repository has no packed 2-/3-bit plug store or kernel that reads one. Same rollout protocol as `DECODE_ROLLOUT_phi4mini.md` (12 prompts x 32 steps, reference = the 4-bit full model). Raw: `/home/workspace/lakes/p7_glint/plugbits_20261104/`; generated ids `plugbits_phi4mini/*.txt`.

## Teacher-forced cost of the code alone (full plug, 2047 positions, KL against the BF16 CUDA anchor)
| plug code | bytes vs 4-bit | mean KL | top-1 vs oracle | PPL |
|---|---|---|---|---|
| 4 bits (reference) | 100% | 9.52e-4 | 2010 / 2047 | 5.4885 |
| 3 bits | 75% | 2.13e-3 | 1999 / 2047 | 5.4770 |
| 2 bits | 50% | 1.77e-2 | 1964 / 2047 | 5.5631 |
A 3-bit child index keeps INT8-level fidelity (INT8 W8A8: 1.75e-3); a 2-bit index lands near llama.cpp's Q4_K_M (2.1e-2).

## Free-running rollout (against the 4-bit full model)
| Arm | plug code | channels fetched | plug bytes per token | matched prefix /32 | agreement | all-match prompts |
|---|---|---|---|---|---|---|
| reference: `act` 50%, 4-bit (Amendment 3) | 4 | 50% | ~805 MB | 29.92 | 99.0% | 11 / 12 |
| `act` 25%, 4-bit (Amendment 3) | 4 | 25% | ~403 MB | 20.33 | 72.7% | 5 / 12 |
| full plug | 3 | 100% | ~1208 MB | 30.08 | 95.8% | 11 / 12 |
| `act` 50% | 3 | 50% | **~604 MB** | 26.17 | **86.2%** | 9 / 12 |
| full plug | 2 | 100% | ~805 MB | 16.00 | 60.2% | 4 / 12 |
| `act` 50% | 2 | 50% | ~402 MB | 15.17 | 56.8% | 4 / 12 |
| `act` 25% | 2 | 25% | ~201 MB | 14.25 | 54.7% | 3 / 12 |

## Verdict under the pre-registered rule
*Improves the frontier* = an arm **tracks** (matched prefix >= 20.0 and agreement >= 82.5%) while moving **at most 403 MB** per token (half the 4-bit tracking arm's 805 MB). **Not met.** The only coarser-code arms that track are the 3-bit full plug (1208 MB) and the 3-bit `act` 50% (86.2% agreement, 604 MB, 25% fewer bytes than the 4-bit tracking arm but 50% more than the limit); every 2-bit arm fails to track, including the 2-bit *full* plug (60.2%), so the 2-bit code itself is the limit, not the selection.

## What the data says
1. A 3-bit code is nearly free in fidelity (teacher-forced KL 2.1e-3; free-running 95.8% agreement for the full plug) and saves 25% of the bytes; selecting half the channels on top of it still tracks (86.2%).
2. A 2-bit code is too coarse for free-running fidelity (60.2% even with every channel), although its teacher-forced KL (1.77e-2) looks moderate: the same teacher-forced-versus-free-running gap seen for the selection fraction.
3. At equal bytes, finer code on fewer channels beats coarser code on more: 4-bit at 25% (403 MB) reaches 72.7% agreement against 56.8% for 2-bit at 50% (402 MB).
4. Speed (arithmetic, not measured, assuming a packed 3-bit store and an unpack that costs nothing): the 4-bit 50% arm needs 163 ms of extra time for 805 MB (`SELFETCH_OPT_phi4mini.md`); scaling that to 604 MB gives about 122 ms extra, roughly 160 ms per token, about 6 tokens/s. That stays below the operator's 10 tokens/s target.

## What this does not show
Emulated reconstruction, not a real b-bit codec: a real encoder could choose better bin edges than merging adjacent children, and a packed store plus an unpacking kernel would add their own cost. One text, 12 prompts, 32 steps, one run per arm, token-exact agreement. Not reviewed by the Test Designer.

## Reproduce
`GLINT_PLUG_BITS=3 ARM0_DECODE_ROLL=12,32,64,128 ARM0_DECODE_ROLL_OUT=out.txt ARM0_DEMAND_RULE=act ARM0_DEMAND_F=1 ARM0_SELFETCH=0.5 ARM0_SELFETCH_OPT=1 bf16full --glint-demand PIN MODEL ART`; teacher-forced rows use `GLINT_PLUG_BITS=b ARM0_DEMAND_F=1 ARM0_LOGITS_REF=<anchor logits> bf16full --glint-demand PIN MODEL ART`.
