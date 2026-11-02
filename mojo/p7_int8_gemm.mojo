# mojo/p7_int8_gemm.mojo — pure Mojo / MAX INT8 W8A8 GEMM over a bf16_to_int8_pin_v1 linear (glint option 2).
# Same arithmetic as cuda/int8_mma_gemm.cu: the activation is quantized per (row, 64-block) with the pin rule
# (scale = amax/127, round-half-even, clamp +-127), each 64-wide K block is an exact int32 dot product, and the
# result accumulates per block as (float(ic) * sx) * sw. No CUDA C. Checked against libp7_kernels.so on the same
# inputs, through the same Mojo harness.
# Run: mojo run mojo/p7_int8_gemm.mojo LIB.so
from std.ffi import OwnedDLHandle
from std.sys import argv
from std.math import ceildiv
from std.time import perf_counter_ns
from std.gpu import global_idx
from std.memory.unsafe_pointer import unsafe_cast
from max.gpu.host import DeviceContext
from max.gpu.host._nvidia_cuda import CUDA

comptime M = 70
comptime K = 3072
comptime N = 4096
comptime NKB = K // 64

def quant_act[
    xo: ImmOrigin, qo: MutOrigin, so: MutOrigin
](
    x: ImmPointer[BFloat16, xo],
    xq: MutPointer[Int8, qo],
    sx: MutPointer[Float32, so],
    nblocks: Int32,
):
    var b = global_idx.x                      # one thread per (row, 64-block)
    if b >= Int(nblocks):
        return
    var base = b * 64
    var am = Float32(0)
    for i in range(64):
        var a = abs(x[unsafe_offset=base + i].cast[DType.float32]())
        if a > am:
            am = a
    var s = am / Float32(127) if am > Float32(0) else Float32(1)
    sx[unsafe_offset=b] = s
    for i in range(64):
        var v = round(x[unsafe_offset=base + i].cast[DType.float32]() / s)   # round-half-even
        var c = v
        if c > Float32(127):
            c = Float32(127)
        if c < Float32(-127):
            c = Float32(-127)
        xq[unsafe_offset=base + i] = Int8(Int(c))

def int8_gemm[
    xqo: ImmOrigin, sxo: ImmOrigin, wo: ImmOrigin, swo: ImmOrigin, yo: MutOrigin
](
    xq: ImmPointer[Int8, xqo],
    sx: ImmPointer[Float32, sxo],
    wq: ImmPointer[Int8, wo],
    sw: ImmPointer[Float32, swo],
    y: MutPointer[Float32, yo],
    m: Int32,
    n: Int32,
):
    var idx = global_idx.x                    # one thread per output element
    var mm = Int(m)
    var nn = Int(n)
    if idx >= mm * nn:
        return
    var row = idx // nn
    var col = idx % nn
    var acc = Float32(0)
    for kb in range(NKB):
        var ic = Int32(0)
        for j in range(64):
            ic += Int32(xq[unsafe_offset=row * K + kb * 64 + j]) * Int32(wq[unsafe_offset=col * K + kb * 64 + j])
        acc += (Float32(ic) * sx[unsafe_offset=row * NKB + kb]) * sw[unsafe_offset=col * NKB + kb]
    y[unsafe_offset=idx] = acc

def main() raises:
    var args = argv()
    var lib = OwnedDLHandle(String(args[1]) if len(args) > 1 else String("/home/workspace/fractal-deployment/glint/lib/libp7_kernels.so"))
    var cuda_gemm = lib.get_function[Int32]("p7_int8_gemm_flat")
    var ctx = DeviceContext()
    var ks = ctx.create_stream()
    var ks_native = unsafe_cast[Type=NoneType](CUDA(ks))

    var seed = UInt32(777)
    var xh = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    for i in range(M * K):
        seed = seed * 1664525 + 1013904223
        xh[i] = (Float32(Int((seed >> 8) & 0xFFFF)) / 32768.0 - 1.0).cast[DType.bfloat16]()
    var qh = ctx.enqueue_create_host_buffer[DType.int8](N * K)
    for i in range(N * K):
        seed = seed * 1664525 + 1013904223
        var v = Int((seed >> 24) & 0xFF) - 128
        if v < -127:
            v = -127
        qh[i] = Int8(v)
    var sh = ctx.enqueue_create_host_buffer[DType.float32](N * NKB)
    for i in range(N * NKB):
        sh[i] = Float32(0.0004) * Float32(1 + i % 9)
    var x = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var wq = ctx.enqueue_create_buffer[DType.int8](N * K)
    var sw = ctx.enqueue_create_buffer[DType.float32](N * NKB)
    ctx.enqueue_copy(dst_buf=x, src_buf=xh)
    ctx.enqueue_copy(dst_buf=wq, src_buf=qh)
    ctx.enqueue_copy(dst_buf=sw, src_buf=sh)
    var xq_c = ctx.enqueue_create_buffer[DType.int8](M * K)
    var sx_c = ctx.enqueue_create_buffer[DType.float32](M * NKB)
    var y_c = ctx.enqueue_create_buffer[DType.float32](M * N)
    var xq_m = ctx.enqueue_create_buffer[DType.int8](M * K)
    var sx_m = ctx.enqueue_create_buffer[DType.float32](M * NKB)
    var y_m = ctx.enqueue_create_buffer[DType.float32](M * N)
    var yc_h = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    var ym_h = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.synchronize()

    var rc = cuda_gemm(Int(x.unsafe_ptr()), Int(wq.unsafe_ptr()), Int(sw.unsafe_ptr()), Int32(N), Int32(K),
                       Int(y_c.unsafe_ptr()), Int32(M), Int(xq_c.unsafe_ptr()), Int(sx_c.unsafe_ptr()), Int(ks_native.value()))
    if rc != 0:
        raise Error("cuda gemm rc=" + String(rc))
    ks.synchronize()

    var qk = ctx.compile_function[quant_act[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var gk = ctx.compile_function[int8_gemm[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    ctx.enqueue_function(qk, x, xq_m, sx_m, Int32(M * NKB), grid_dim=ceildiv(M * NKB, 64), block_dim=64)
    ctx.enqueue_function(gk, xq_m, sx_m, wq, sw, y_m, Int32(M), Int32(N), grid_dim=ceildiv(M * N, 128), block_dim=128)
    ctx.synchronize()

    ctx.enqueue_copy(dst_buf=yc_h, src_buf=y_c)
    ctx.enqueue_copy(dst_buf=ym_h, src_buf=y_m)
    ctx.synchronize()
    var bitdiff = 0
    var maxd = Float64(0)
    var maxref = Float64(0)
    for i in range(M * N):
        var a = yc_h[i]
        var b = ym_h[i]
        if a.to_bits[DType.uint32]() != b.to_bits[DType.uint32]():
            bitdiff += 1
        var d = abs(Float64(a) - Float64(b))
        if d > maxd:
            maxd = d
        if abs(Float64(a)) > maxref:
            maxref = abs(Float64(a))
    print("outputs", M * N, "bitwise_different", bitdiff, "max_abs_diff", maxd, "max_abs_output", maxref)
    # timing: 20 runs each, synchronized
    var t0 = perf_counter_ns()
    for _ in range(20):
        _ = cuda_gemm(Int(x.unsafe_ptr()), Int(wq.unsafe_ptr()), Int(sw.unsafe_ptr()), Int32(N), Int32(K),
                      Int(y_c.unsafe_ptr()), Int32(M), Int(xq_c.unsafe_ptr()), Int(sx_c.unsafe_ptr()), Int(ks_native.value()))
    ks.synchronize()
    var cuda_ms = Float64(perf_counter_ns() - t0) / 1e6 / 20.0
    t0 = perf_counter_ns()
    for _ in range(20):
        ctx.enqueue_function(qk, x, xq_m, sx_m, Int32(M * NKB), grid_dim=ceildiv(M * NKB, 64), block_dim=64)
        ctx.enqueue_function(gk, xq_m, sx_m, wq, sw, y_m, Int32(M), Int32(N), grid_dim=ceildiv(M * N, 128), block_dim=128)
    ctx.synchronize()
    var mojo_ms = Float64(perf_counter_ns() - t0) / 1e6 / 20.0
    print("cuda_tensorcore_ms_per_gemm", cuda_ms, "pure_mojo_naive_ms_per_gemm", mojo_ms, "slowdown", mojo_ms / cuda_ms)
    print("PURE_MOJO_INT8_GEMM_" + ("BITEXACT" if bitdiff == 0 else "NEAR") )
