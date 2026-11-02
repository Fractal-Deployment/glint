# mojo/p7_prefetch.mojo — Mojo / MAX plug prefetch (glint #21, the "MLIR prefetch").
# A 2-slot VRAM ring fed from pinned host memory on the context's own stream while kernels read the slots on a
# second DeviceStream (ctx.create_stream()), ordered with DeviceEvent record/wait. No CUDA C, no Python.
# Measures what hiding the per-layer transfers behind compute is worth: copy-only, compute-only, overlapped.
# Pattern from modular/modular max/kernels/test/gpu/device_context/test_device_stream.mojo.
# Run: mojo run mojo/p7_prefetch.mojo   (MODULAR_HOME and PATH as in MOJO_ENV_NOTE.md)
from std.math import ceildiv
from std.time import perf_counter_ns
from std.gpu import global_idx
from max.gpu.host import DeviceContext

comptime LAYERS = 32
comptime LAYER_BYTES = 48 * 1024 * 1024
comptime WORDS = LAYER_BYTES // 4
comptime REPS = 24            # kernel launches per layer: stands in for the layer's GEMMs
comptime BLOCK = 256

def layer_work[
    in_origin: ImmOrigin, out_origin: MutOrigin
](
    slot: ImmPointer[UInt32, in_origin],
    acc: MutPointer[UInt32, out_origin],
    n: Int32,
):
    var i = global_idx.x
    if i < Int(n):
        var v = slot[unsafe_offset=i]
        if v == 0xFFFFFFFF:        # reads every word; practically never taken
            acc[unsafe_offset=0] = v

def main() raises:
    var ctx = DeviceContext()
    var ks = ctx.create_stream()                      # compute stream; ctx's own stream carries the copies
    var host = List[type_of(ctx.enqueue_create_host_buffer[DType.uint32](1))]()
    for l in range(LAYERS):
        var h = ctx.enqueue_create_host_buffer[DType.uint32](WORDS)
        for j in range(0, WORDS, 4096):
            h[j] = UInt32(l * 7 + j)
        host.append(h^)
    var slot0 = ctx.enqueue_create_buffer[DType.uint32](WORDS)
    var slot1 = ctx.enqueue_create_buffer[DType.uint32](WORDS)
    var res = ctx.enqueue_create_buffer[DType.uint32](1)
    res.enqueue_fill(0)
    ctx.synchronize()
    var work = ctx.compile_function[layer_work[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    comptime grid = ceildiv(WORDS, BLOCK)

    # (a) copy only
    var t0 = perf_counter_ns()
    for l in range(LAYERS):
        if l % 2 == 0:
            ctx.enqueue_copy(dst_buf=slot0, src_buf=host[l])
        else:
            ctx.enqueue_copy(dst_buf=slot1, src_buf=host[l])
    ctx.synchronize()
    var copy_ms = Float64(perf_counter_ns() - t0) / 1e6

    # (b) compute only, on the compute stream
    t0 = perf_counter_ns()
    for _ in range(LAYERS):
        for _ in range(REPS):
            ks.enqueue_function(work, slot0, res, Int32(WORDS), grid_dim=grid, block_dim=BLOCK)
    ks.synchronize()
    var comp_ms = Float64(perf_counter_ns() - t0) / 1e6

    # (c) overlapped. Events order the streams without blocking the host:
    #   the copy of layer l+1 waits for the kernels of layer l-1 (its slot is free),
    #   the kernels of layer l wait for the copy of layer l.
    var copy_done = List[type_of(ctx.create_event())]()
    var comp_done = List[type_of(ctx.create_event())]()
    for _ in range(LAYERS):
        copy_done.append(ctx.create_event())
        comp_done.append(ctx.create_event())
    t0 = perf_counter_ns()
    ctx.enqueue_copy(dst_buf=slot0, src_buf=host[0])
    ctx.stream().record_event(copy_done[0])
    for l in range(LAYERS):
        if l + 1 < LAYERS:
            if l >= 1:
                ctx.stream().enqueue_wait_for(comp_done[l - 1])
            if (l + 1) % 2 == 0:
                ctx.enqueue_copy(dst_buf=slot0, src_buf=host[l + 1])
            else:
                ctx.enqueue_copy(dst_buf=slot1, src_buf=host[l + 1])
            ctx.stream().record_event(copy_done[l + 1])
        ks.enqueue_wait_for(copy_done[l])
        for _ in range(REPS):
            if l % 2 == 0:
                ks.enqueue_function(work, slot0, res, Int32(WORDS), grid_dim=grid, block_dim=BLOCK)
            else:
                ks.enqueue_function(work, slot1, res, Int32(WORDS), grid_dim=grid, block_dim=BLOCK)
        ks.record_event(comp_done[l])
    ks.synchronize()
    ctx.synchronize()
    var over_ms = Float64(perf_counter_ns() - t0) / 1e6

    # correctness: the slot holds the last layer's pattern
    with slot1.map_to_host() as m:
        var ok = True
        for j in range(0, WORDS, 4096 * 64):
            if m[j] != UInt32((LAYERS - 1) * 7 + j):
                ok = False
        print("slot1_holds_layer31", ok)
    var gb = Float64(LAYERS) * Float64(LAYER_BYTES) / 1e9
    print("layers", LAYERS, "layer_MiB", LAYER_BYTES // (1024 * 1024), "kernel_launches_per_layer", REPS)
    print("copy_only_ms     ", copy_ms, "  bandwidth_GBps", gb / (copy_ms / 1000.0))
    print("compute_only_ms  ", comp_ms)
    print("overlapped_ms    ", over_ms)
    print("serial_sum_ms    ", copy_ms + comp_ms)
    print("hidden_fraction  ", (copy_ms + comp_ms - over_ms) / copy_ms)
    print("MOJO_PREFETCH_OVERLAP_DONE")
