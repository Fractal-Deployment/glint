# VRAM budget knob and plug fetch speed, Phi-4-mini (2026-11-04)

Two measurements that follow `FOLD_STREAM_phi4mini.md` (streamed plug, byte-identical to resident) and `DEMAND_MAP_phi4mini.md` (how much plug a step needs). `glint_pin_v1`, RTX 3060 under the GPU lease, one run each.

## 1. Budget knob: the first N layers keep their plug in VRAM, the rest stream
`ARM0_STREAM_RESIDENT_LAYERS=N` with `--glint-stream` (tests/test_arm0_bf16_full.cu). The ring is built over the streamed layers only. `VRAM_USED_MIB` is device memory used after the forward, measured against the free memory at program start (so the context is excluded; it includes the BF16 embedding and norms, the 4-bit hole and absmax for every layer, the activation buffers, the cuBLAS workspace, the plug-resident layers and the 2-slot ring).

| N (plug-resident layers) | VRAM used | saved vs N = 32 | plug bytes moved | total stall | PPL | top-1 | KL vs N = 32 |
|---|---|---|---|---|---|---|---|
| 32 (all resident) | 5298.2 MiB | - | 0 | - | 5.479566 | 2007 | - |
| 24 | 5189.8 MiB | 108 MiB | 402,653,184 | 0.030 ms | 5.479566 | 2007 | 0 |
| 16 | 4790.2 MiB | 508 MiB | 805,306,368 | 0.070 ms | 5.479566 | 2007 | 0 |
| 8 | 4392.0 MiB | 906 MiB | 1,207,959,552 | 0.111 ms | 5.479566 | 2007 | 0 |
| 0 (all streamed) | 3993.1 MiB | 1305 MiB (1.27 GiB) | 1,610,612,736 | 0.153 ms | 5.479566 | 2007 | 0 |

- The logits of every N are identical to the all-resident run (KL exactly 0), so the knob changes memory and nothing else.
- Saving per streamed layer is about 48 MiB less the ring. Expected saving for all-streamed is 1536 - 96 = 1440 MiB; measured 1305 MiB. The 135-180 MiB shortfall at each N is unexplained (allocator granularity or buffers the streamed path adds); it is reported, not accounted for.
- This replaces the stale Oct-21 "1430 MiB saved" figure with a measurement on the current pin and a valid forward. As before, the H-TILE kernel takes about 135 s per forward, so the near-zero stall does not show hiding at speed.

## 2. Fetch speed of a selected fraction of a layer's plug
`tests/bench_plug_fetch.cu`: one Phi-4-mini layer's plug (QKV, O, GATE_UP, DOWN; 50,331,648 bytes) in pinned host RAM in a K-major layout (a channel's plug nibbles contiguous), 20 repetitions, fixed-seed random subset of units, adjacent units merged into runs. "memcpy" = one `cudaMemcpyAsync` per run; "zerocopy" = a gather kernel reading mapped pinned memory over PCIe (the DecDEC approach). A unit is a channel or 64 adjacent channels (block).

| unit | f | bytes | runs | memcpy ms/layer | zerocopy ms/layer | best GB/s |
|---|---|---|---|---|---|---|
| any | 1.0 | 50.3 MB | 4 | 4.348 | 6.561 | 11.6 (memcpy) |
| channel | .5 | 25.2 MB | 4639 | 7.373 | 3.665 | 6.9 (zerocopy) |
| channel | .25 | 12.6 MB | 3426 | 5.032 | 1.788 | 7.0 |
| channel | .1 | 5.04 MB | 1627 | 2.387 | **0.734** | 6.9 |
| channel | .05 | 2.52 MB | 845 | 1.252 | **0.354** | 7.1 |
| channel | .01 | 0.51 MB | 175 | 0.267 | 0.059 | 8.6 |
| block | .5 | 25.2 MB | 70 | **2.269** | 3.836 | 11.1 (memcpy) |
| block | .25 | 12.6 MB | 57 | **1.166** | 1.852 | 10.8 |
| block | .1 | 5.21 MB | 28 | **0.492** | 0.565 | 10.6 |
| block | .05 | 3.05 MB | 16 | **0.305** | 0.326 | 10.0 |
| block | .01 | 0.98 MB | 5 | **0.093** | 0.140 | 10.6 |

- Measured: fetch time falls in proportion to the fraction fetched. Block units (512 KB to 4 MB contiguous) keep near-peak PCIe bandwidth (10-11 GB/s) with plain copies; single channels (1.5 to 8 KB) need the zero-copy gather and reach about 7 GB/s; per-run copies of single channels collapse to 2 GB/s.
- The 11.6 GB/s full-layer copy agrees with the earlier measured streamer rate (11.28 GB/s).
- Combined with the demand map (PR #82): at channel granularity 5% of the plug (top-1 95.16%, KL 2.1e-2) costs 0.354 ms per layer to fetch; 10% (top-1 95.55%) costs 0.734 ms; the whole plug costs 4.35 ms. At block granularity 5% (top-1 93.84%) costs 0.305 ms.
- *Arithmetic, not measured:* 32 layers at 0.354 ms is 11.3 ms per token of fetch, an upper bound of about 88 tokens/s from fetch alone; at 0.734 ms it is 23.5 ms (about 43 tokens/s); streaming the whole plug is 139 ms (about 7 tokens/s). These ignore compute and overlap. A layer's decode compute of about 0.2 ms (60 MB of hole at 360 GB/s) is an estimate; there is no decode path in this repository to measure it, so whether these fetches hide behind compute is not shown.

## What this does not show
No activation-selected fetch was run end to end: the selection pattern in the benchmark is random, which fixes unit count and size but not which units an activation would pick (contiguity of real selections could change the run count). The plug is not stored K-major in the pin; the benchmark assumes a re-layout. One layer shape set (Phi-4-mini); PCIe link and host are this machine's.
