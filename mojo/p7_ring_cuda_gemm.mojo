# mojo/p7_ring_cuda_gemm.mojo — option 1 wiring: the Mojo prefetch ring feeds the CUDA INT8 GEMM.
# Layer blobs (INT8 q, F32 scales, res4 plug nibbles, plug F32 scales; pin + plug layout) stream from pinned host memory into a 2-slot ring on the Mojo
# context stream while int8_mma_gemm (libp7_kernels.so via p7_int8_gemm_flat) runs on a second Mojo stream through
# its native CUDA handle. Each layer's output must hash equal to the same GEMM with every layer resident.
# Run: mojo run mojo/p7_ring_cuda_gemm.mojo LIB.so
from std.ffi import OwnedDLHandle
from std.sys import argv
from std.memory.unsafe_pointer import unsafe_cast
from max.gpu.host import DeviceContext
from max.gpu.host._nvidia_cuda import CUDA

comptime LAYERS = 8
comptime M = 70
comptime K = 3072
comptime N = 8192
comptime QB = N * K
comptime SB = N * (K // 64) * 4
comptime PQ = QB // 2             # res4 plug nibbles
comptime BLOB = QB + SB + PQ + SB   # q, scales, plug nibbles, plug scales


def main() raises:
    var args = argv()
    var lib = OwnedDLHandle(String(args[1]) if len(args) > 1 else String("/home/workspace/fractal-deployment/glint/lib/libp7_kernels.so"))
    var gemm_base = lib.get_function[Int32]("p7_int8_gemm_flat")
    var gemm = lib.get_function[Int32]("p7_int8_gemm_plug_flat")
    var ctx = DeviceContext()
    var ks = ctx.create_stream()
    var ks_native = unsafe_cast[Type=NoneType](CUDA(ks))

    var host = List[type_of(ctx.enqueue_create_host_buffer[DType.uint8](1))]()
    var seed = UInt32(12345)
    for l in range(LAYERS):
        var h = ctx.enqueue_create_host_buffer[DType.uint8](BLOB)
        for i in range(QB):
            seed = seed * 1664525 + 1013904223
            h[i] = UInt8((seed >> 24) & 0xFF)
        var sh = ctx.enqueue_create_host_buffer[DType.float32](SB // 4)
        for i in range(SB // 4):
            sh[i] = Float32(0.0005) * Float32(1 + (i + l) % 7)
        ctx.synchronize()
        for i in range(SB):
            h[QB + i] = sh.unsafe_ptr().unsafe_bitcast[UInt8]()[unsafe_offset=i]
        for i in range(PQ):
            seed = seed * 1664525 + 1013904223
            h[QB + SB + i] = UInt8((seed >> 24) & 0xFF)
        var ph = ctx.enqueue_create_host_buffer[DType.float32](SB // 4)
        for i in range(SB // 4):
            ph[i] = Float32(0.00003) * Float32(1 + (i + l) % 5)
        ctx.synchronize()
        for i in range(SB):
            h[QB + SB + PQ + i] = ph.unsafe_ptr().unsafe_bitcast[UInt8]()[unsafe_offset=i]
        host.append(h^)
    var xh = ctx.enqueue_create_host_buffer[DType.uint16](M * K)
    for i in range(M * K):
        seed = seed * 1664525 + 1013904223
        xh[i] = UInt16(0x3C00 + ((seed >> 20) & 0x3FF))
    var x = ctx.enqueue_create_buffer[DType.uint16](M * K)
    ctx.enqueue_copy(dst_buf=x, src_buf=xh)
    var xq = ctx.enqueue_create_buffer[DType.uint8](M * K)
    var sx = ctx.enqueue_create_buffer[DType.float32](M * (K // 64))
    var ydev = ctx.enqueue_create_buffer[DType.float32](M * N)
    var yhost = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.synchronize()

    var res_hash = List[UInt64]()
    var str_hash = List[UInt64]()

    # resident: all layers on the device first
    var res_blobs = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    for l in range(LAYERS):
        var d = ctx.enqueue_create_buffer[DType.uint8](BLOB)
        ctx.enqueue_copy(dst_buf=d, src_buf=host[l])
        res_blobs.append(d^)
    ctx.synchronize()
    for l in range(LAYERS):
        var q = Int(res_blobs[l].unsafe_ptr())
        var rc = gemm(Int(x.unsafe_ptr()), q, q + QB, q + QB + SB, q + QB + SB + PQ, Int32(N), Int32(K),
                      Int(ydev.unsafe_ptr()), Int32(M), Int(xq.unsafe_ptr()), Int(sx.unsafe_ptr()), Int(ks_native.value()))
        if rc != 0:
            raise Error("resident gemm rc=" + String(rc))
        ks.synchronize()
        ctx.enqueue_copy(dst_buf=yhost, src_buf=ydev)
        ctx.synchronize()
        var hsh = UInt64(1469598103934665603)
        for i in range(M * N):
            hsh = (hsh ^ UInt64(yhost[i].to_bits[DType.uint32]())) * 1099511628211
        res_hash.append(hsh)
    # sanity: the plug really changes the output (layer 0 with the base-only entry point)
    var q0 = Int(res_blobs[0].unsafe_ptr())
    var rc0 = gemm_base(Int(x.unsafe_ptr()), q0, q0 + QB, Int32(N), Int32(K), Int(ydev.unsafe_ptr()), Int32(M),
                        Int(xq.unsafe_ptr()), Int(sx.unsafe_ptr()), Int(ks_native.value()))
    if rc0 != 0:
        raise Error("base gemm rc=" + String(rc0))
    ks.synchronize()
    ctx.enqueue_copy(dst_buf=yhost, src_buf=ydev)
    ctx.synchronize()
    var bh = UInt64(1469598103934665603)
    for i in range(M * N):
        bh = (bh ^ UInt64(yhost[i].to_bits[DType.uint32]())) * 1099511628211
    print("layer0_base_only_hash", bh, "plug_changes_output", bh != res_hash[0])
    res_blobs.clear()

    # streamed: 2-slot ring; copies on ctx's stream, GEMM on ks, events between
    var slot0 = ctx.enqueue_create_buffer[DType.uint8](BLOB)
    var slot1 = ctx.enqueue_create_buffer[DType.uint8](BLOB)
    var copy_done = List[type_of(ctx.create_event())]()
    var comp_done = List[type_of(ctx.create_event())]()
    for _ in range(LAYERS):
        copy_done.append(ctx.create_event())
        comp_done.append(ctx.create_event())
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
        var q = Int(slot0.unsafe_ptr()) if l % 2 == 0 else Int(slot1.unsafe_ptr())
        var rc = gemm(Int(x.unsafe_ptr()), q, q + QB, q + QB + SB, q + QB + SB + PQ, Int32(N), Int32(K),
                      Int(ydev.unsafe_ptr()), Int32(M), Int(xq.unsafe_ptr()), Int(sx.unsafe_ptr()), Int(ks_native.value()))
        if rc != 0:
            raise Error("streamed gemm rc=" + String(rc))
        ks.record_event(comp_done[l])
        ks.synchronize()
        ctx.enqueue_copy(dst_buf=yhost, src_buf=ydev)
        ctx.synchronize()
        var hsh = UInt64(1469598103934665603)
        for i in range(M * N):
            hsh = (hsh ^ UInt64(yhost[i].to_bits[DType.uint32]())) * 1099511628211
        str_hash.append(hsh)

    var bad = 0
    for l in range(LAYERS):
        if res_hash[l] != str_hash[l]:
            bad += 1
        print("layer", l, "resident", res_hash[l], "streamed", str_hash[l])
    print("layers", LAYERS, "mismatched_layers", bad)
    if bad != 0:
        raise Error("streamed != resident")
    print("MOJO_RING_CUDA_GEMM_OK")
