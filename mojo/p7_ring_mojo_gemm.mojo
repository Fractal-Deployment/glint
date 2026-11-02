# mojo/p7_ring_mojo_gemm.mojo — all Mojo: the prefetch ring feeds the pure-Mojo INT8 tensor-core GEMM.
# No libp7_kernels.so, no CUDA C. Layers (INT8 q + F32 scales) stream from pinned host memory into a 2-slot ring on
# the context stream while the Mojo GEMM runs on a second stream; events order them. Each layer's output hash must
# equal the same GEMM with every layer resident.
# Run: mojo run -I mojo mojo/p7_ring_mojo_gemm.mojo
from std.math import ceildiv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from p7_int8_mma_gemm import quant_act, int8_mma_gemm, M, K, N, NKB

comptime LAYERS = 8
comptime QB = N * K
comptime SB = N * NKB * 4

def hash_out(h: type_of(DeviceContext().enqueue_create_host_buffer[DType.float32](1))) -> UInt64:
    var hsh = UInt64(1469598103934665603)
    for i in range(M * N):
        hsh = (hsh ^ UInt64(h[i].to_bits[DType.uint32]())) * 1099511628211
    return hsh

def main() raises:
    var ctx = DeviceContext()
    var ks = ctx.create_stream()
    var seed = UInt32(4242)
    var host = List[type_of(ctx.enqueue_create_host_buffer[DType.int8](1))]()
    var hosts = List[type_of(ctx.enqueue_create_host_buffer[DType.float32](1))]()
    for l in range(LAYERS):
        var h = ctx.enqueue_create_host_buffer[DType.int8](QB)
        for i in range(QB):
            seed = seed * 1664525 + 1013904223
            var v = Int((seed >> 24) & 0xFF) - 128
            if v < -127:
                v = -127
            h[i] = Int8(v)
        var sh = ctx.enqueue_create_host_buffer[DType.float32](SB // 4)
        for i in range(SB // 4):
            sh[i] = Float32(0.0004) * Float32(1 + (i + l) % 9)
        host.append(h^)
        hosts.append(sh^)
    var xh = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    for i in range(M * K):
        seed = seed * 1664525 + 1013904223
        xh[i] = (Float32(Int((seed >> 8) & 0xFFFF)) / 32768.0 - 1.0).cast[DType.bfloat16]()
    var x = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    ctx.enqueue_copy(dst_buf=x, src_buf=xh)
    var xq = ctx.enqueue_create_buffer[DType.int8](M * K)
    var sx = ctx.enqueue_create_buffer[DType.float32](M * NKB)
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var yh = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    var dummy_p = ctx.enqueue_create_buffer[DType.uint8](16)
    var dummy_s = ctx.enqueue_create_buffer[DType.float32](16)
    ctx.synchronize()
    var qk = ctx.compile_function[quant_act[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var gk = ctx.compile_function[int8_mma_gemm[False, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()

    ks.enqueue_function(qk, x, xq, sx, Int32(M * NKB), grid_dim=ceildiv(M * NKB, 64), block_dim=64)
    ks.synchronize()

    var res_hash = List[UInt64]()
    var str_hash = List[UInt64]()
    # resident
    var res_q = List[type_of(ctx.enqueue_create_buffer[DType.int8](1))]()
    var res_s = List[type_of(ctx.enqueue_create_buffer[DType.float32](1))]()
    for l in range(LAYERS):
        var d = ctx.enqueue_create_buffer[DType.int8](QB)
        var e = ctx.enqueue_create_buffer[DType.float32](SB // 4)
        ctx.enqueue_copy(dst_buf=d, src_buf=host[l])
        ctx.enqueue_copy(dst_buf=e, src_buf=hosts[l])
        res_q.append(d^)
        res_s.append(e^)
    ctx.synchronize()
    for l in range(LAYERS):
        ks.enqueue_function(gk, xq, sx, res_q[l], res_s[l], dummy_p, dummy_s, y, Int32(M), Int32(N),
                            grid_dim=(ceildiv(N, 64), ceildiv(M, 64)), block_dim=128)
        ks.synchronize()
        ctx.enqueue_copy(dst_buf=yh, src_buf=y)
        ctx.synchronize()
        res_hash.append(hash_out(yh))
    res_q.clear()
    res_s.clear()

    # streamed
    var q0 = ctx.enqueue_create_buffer[DType.int8](QB)
    var q1 = ctx.enqueue_create_buffer[DType.int8](QB)
    var s0 = ctx.enqueue_create_buffer[DType.float32](SB // 4)
    var s1 = ctx.enqueue_create_buffer[DType.float32](SB // 4)
    var copy_done = List[type_of(ctx.create_event())]()
    var comp_done = List[type_of(ctx.create_event())]()
    for _ in range(LAYERS):
        copy_done.append(ctx.create_event())
        comp_done.append(ctx.create_event())
    ctx.enqueue_copy(dst_buf=q0, src_buf=host[0])
    ctx.enqueue_copy(dst_buf=s0, src_buf=hosts[0])
    ctx.stream().record_event(copy_done[0])
    for l in range(LAYERS):
        if l + 1 < LAYERS:
            if l >= 1:
                ctx.stream().enqueue_wait_for(comp_done[l - 1])
            if (l + 1) % 2 == 0:
                ctx.enqueue_copy(dst_buf=q0, src_buf=host[l + 1])
                ctx.enqueue_copy(dst_buf=s0, src_buf=hosts[l + 1])
            else:
                ctx.enqueue_copy(dst_buf=q1, src_buf=host[l + 1])
                ctx.enqueue_copy(dst_buf=s1, src_buf=hosts[l + 1])
            ctx.stream().record_event(copy_done[l + 1])
        ks.enqueue_wait_for(copy_done[l])
        if l % 2 == 0:
            ks.enqueue_function(gk, xq, sx, q0, s0, dummy_p, dummy_s, y, Int32(M), Int32(N),
                                grid_dim=(ceildiv(N, 64), ceildiv(M, 64)), block_dim=128)
        else:
            ks.enqueue_function(gk, xq, sx, q1, s1, dummy_p, dummy_s, y, Int32(M), Int32(N),
                                grid_dim=(ceildiv(N, 64), ceildiv(M, 64)), block_dim=128)
        ks.record_event(comp_done[l])
        ks.synchronize()
        ctx.enqueue_copy(dst_buf=yh, src_buf=y)
        ctx.synchronize()
        str_hash.append(hash_out(yh))

    var bad = 0
    for l in range(LAYERS):
        if res_hash[l] != str_hash[l]:
            bad += 1
        print("layer", l, "resident", res_hash[l], "streamed", str_hash[l])
    print("layers", LAYERS, "mismatched_layers", bad)
    if bad != 0:
        raise Error("streamed != resident")
    print("PURE_MOJO_RING_GEMM_OK")
