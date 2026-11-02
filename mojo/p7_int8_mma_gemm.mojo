# mojo/p7_int8_mma_gemm.mojo — pure Mojo / MAX INT8 W8A8 tensor-core GEMM (inline PTX mma.m16n8k32.s8) over a bf16_to_int8_pin_v1 linear (glint option 2).
# Same arithmetic as cuda/int8_mma_gemm.cu: the activation is quantized per (row, 64-block) with the pin rule
# (scale = amax/127, round-half-even, clamp +-127), each 64-wide K block is an exact int32 dot product, and the
# result accumulates per block as (float(ic) * sx) * sw. No CUDA C. Checked against libp7_kernels.so on the same
# inputs, through the same Mojo harness.
# Run: mojo run mojo/p7_int8_gemm.mojo LIB.so
from std.ffi import OwnedDLHandle
from std.sys import argv
from std.math import ceildiv
from std.time import perf_counter_ns
from std.gpu import global_idx, thread_idx, block_idx
def barrier():
    inlined_assembly["bar.sync 0;", NoneType, constraints=""]()

from max.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.sys import inlined_assembly, _RegisterPackType
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

comptime BM = 64
comptime BN = 64
comptime SPAD = 80

def mma_s8(
    a0: UInt32, a1: UInt32, a2: UInt32, a3: UInt32, b0: UInt32, b1: UInt32,
    c: SIMD[DType.int32, 4],
) -> SIMD[DType.int32, 4]:
    var r = inlined_assembly[
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$0,$1,$2,$3};",
        _RegisterPackType[Int32, Int32, Int32, Int32],
        constraints="=r,=r,=r,=r,r,r,r,r,r,r,0,1,2,3",
    ](a0, a1, a2, a3, b0, b1, c[0], c[1], c[2], c[3])
    return SIMD[DType.int32, 4](r[0], r[1], r[2], r[3])

def pack4(p: Pointer[UInt32, MutUntrackedOrigin, address_space=AddressSpace.SHARED], off: Int) -> UInt32:
    return p[unsafe_offset=off >> 2]

def int8_mma_gemm[
    PLUG: Bool, xqo: ImmOrigin, sxo: ImmOrigin, wo: ImmOrigin, swo: ImmOrigin, pqo: ImmOrigin, pso: ImmOrigin, yo: MutOrigin
](
    xq: ImmPointer[Int8, xqo],
    sx: ImmPointer[Float32, sxo],
    wq: ImmPointer[Int8, wo],
    sw: ImmPointer[Float32, swo],
    pq: ImmPointer[UInt8, pqo],
    ps: ImmPointer[Float32, pso],
    y: MutPointer[Float32, yo],
    m: Int32,
    n: Int32,
):
    var sAB = stack_allocation[(BM + BN) * SPAD, Int8, address_space=AddressSpace.SHARED]()
    var sA = sAB
    var sB = sAB.unsafe_offset(BM * SPAD)
    var sP = stack_allocation[BN * SPAD, Int8, address_space=AddressSpace.SHARED]()
    var sSf = stack_allocation[BM + BN + BN, Float32, address_space=AddressSpace.SHARED]()
    var sSx = sSf
    var sSw = sSf.unsafe_offset(BM)
    var sSp = sSf.unsafe_offset(BM + BN)
    var tid = thread_idx.x
    var warp = tid >> 5
    var lane = tid & 31
    var g = lane >> 2
    var t = lane & 3
    var mm = Int(m)
    var nn = Int(n)
    var m0 = block_idx.y * BM
    var n0 = block_idx.x * BN
    var rb = warp * 16
    var acc = InlineArray[SIMD[DType.float32, 4], 8](fill=SIMD[DType.float32, 4](0))
    for kb in range(NKB):
        comptime for i in range(2):
            var idx = tid + i * 128
            var r = idx >> 2
            var c = (idx & 3) * 16
            var sA32 = sA.unsafe_bitcast[UInt32]()
            var sB32 = sB.unsafe_bitcast[UInt32]()
            var xq32 = xq.unsafe_bitcast[UInt32]()
            var wq32 = wq.unsafe_bitcast[UInt32]()
            comptime for w in range(4):
                var va = UInt32(0)
                var vb = UInt32(0)
                if m0 + r < mm:
                    va = xq32[unsafe_offset=(((m0 + r) * K + kb * 64 + c) >> 2) + w]
                if n0 + r < nn:
                    vb = wq32[unsafe_offset=(((n0 + r) * K + kb * 64 + c) >> 2) + w]
                sA32[unsafe_offset=((r * SPAD + c) >> 2) + w] = va
                sB32[unsafe_offset=((r * SPAD + c) >> 2) + w] = vb
        if tid < BM:
            sSx[unsafe_offset=tid] = sx[unsafe_offset=(m0 + tid) * NKB + kb] if m0 + tid < mm else Float32(0)
        else:
            sSw[unsafe_offset=tid - BM] = sw[unsafe_offset=(n0 + tid - BM) * NKB + kb] if n0 + tid - BM < nn else Float32(0)
        comptime if PLUG:
            for i in range(tid, BN * 32, 128):
                var r = i >> 5
                var b = i & 31
                var v = UInt8(0x88)
                if n0 + r < nn:
                    v = pq[unsafe_offset=(((n0 + r) * K + kb * 64) >> 1) + b]
                sP[unsafe_offset=r * SPAD + 2 * b] = Int8(Int(v & 15) - 8)
                sP[unsafe_offset=r * SPAD + 2 * b + 1] = Int8(Int(v >> 4) - 8)
            if tid < BN:
                sSp[unsafe_offset=tid] = ps[unsafe_offset=(n0 + tid) * NKB + kb] if n0 + tid < nn else Float32(0)
        barrier()
        var sA32f = sA.unsafe_bitcast[UInt32]()
        var sB32f = sB.unsafe_bitcast[UInt32]()
        var sP32f = sP.unsafe_bitcast[UInt32]()
        var ip = InlineArray[SIMD[DType.int32, 4], 8](fill=SIMD[DType.int32, 4](0))
        var ic = InlineArray[SIMD[DType.int32, 4], 8](fill=SIMD[DType.int32, 4](0))
        comptime for ks in range(2):
            var a0 = pack4(sA32f, (rb + g) * SPAD + ks * 32 + t * 4)
            var a1 = pack4(sA32f, (rb + g + 8) * SPAD + ks * 32 + t * 4)
            var a2 = pack4(sA32f, (rb + g) * SPAD + ks * 32 + 16 + t * 4)
            var a3 = pack4(sA32f, (rb + g + 8) * SPAD + ks * 32 + 16 + t * 4)
            comptime for nt in range(8):
                var b0 = pack4(sB32f, (nt * 8 + g) * SPAD + ks * 32 + t * 4)
                var b1 = pack4(sB32f, (nt * 8 + g) * SPAD + ks * 32 + 16 + t * 4)
                ic[nt] = mma_s8(a0, a1, a2, a3, b0, b1, ic[nt])
                comptime if PLUG:
                    var p0 = pack4(sP32f, (nt * 8 + g) * SPAD + ks * 32 + t * 4)
                    var p1 = pack4(sP32f, (nt * 8 + g) * SPAD + ks * 32 + 16 + t * 4)
                    ip[nt] = mma_s8(a0, a1, a2, a3, p0, p1, ip[nt])
        var x0 = sSx[unsafe_offset=rb + g]
        var x1 = sSx[unsafe_offset=rb + g + 8]
        comptime for nt in range(8):
            var w0 = sSw[unsafe_offset=nt * 8 + t * 2]
            var w1 = sSw[unsafe_offset=nt * 8 + t * 2 + 1]
            var cur = acc[nt]
            cur[0] += (Float32(ic[nt][0]) * x0) * w0
            cur[1] += (Float32(ic[nt][1]) * x0) * w1
            cur[2] += (Float32(ic[nt][2]) * x1) * w0
            cur[3] += (Float32(ic[nt][3]) * x1) * w1
            comptime if PLUG:
                var q0 = sSp[unsafe_offset=nt * 8 + t * 2]
                var q1 = sSp[unsafe_offset=nt * 8 + t * 2 + 1]
                cur[0] += (Float32(ip[nt][0]) * x0) * q0
                cur[1] += (Float32(ip[nt][1]) * x0) * q1
                cur[2] += (Float32(ip[nt][2]) * x1) * q0
                cur[3] += (Float32(ip[nt][3]) * x1) * q1
            acc[nt] = cur
        barrier()
    comptime for nt in range(8):
        var col = n0 + nt * 8 + t * 2
        var r0 = m0 + rb + g
        var r1 = r0 + 8
        if r0 < mm:
            if col < nn:
                y[unsafe_offset=r0 * nn + col] = acc[nt][0]
            if col + 1 < nn:
                y[unsafe_offset=r0 * nn + col + 1] = acc[nt][1]
        if r1 < mm:
            if col < nn:
                y[unsafe_offset=r1 * nn + col] = acc[nt][2]
            if col + 1 < nn:
                y[unsafe_offset=r1 * nn + col + 1] = acc[nt][3]

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
    var pqh = ctx.enqueue_create_host_buffer[DType.uint8](N * K // 2)
    for i in range(N * K // 2):
        seed = seed * 1664525 + 1013904223
        pqh[i] = UInt8((seed >> 24) & 0xFF)
    var psh = ctx.enqueue_create_host_buffer[DType.float32](N * NKB)
    for i in range(N * NKB):
        psh[i] = Float32(0.00002) * Float32(1 + i % 5)
    var pq = ctx.enqueue_create_buffer[DType.uint8](N * K // 2)
    var ps = ctx.enqueue_create_buffer[DType.float32](N * NKB)
    ctx.enqueue_copy(dst_buf=pq, src_buf=pqh)
    ctx.enqueue_copy(dst_buf=ps, src_buf=psh)
    var y_cp = ctx.enqueue_create_buffer[DType.float32](M * N)
    var y_mp = ctx.enqueue_create_buffer[DType.float32](M * N)
    var ycp_h = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    var ymp_h = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.synchronize()

    var rc = cuda_gemm(Int(x.unsafe_ptr()), Int(wq.unsafe_ptr()), Int(sw.unsafe_ptr()), Int32(N), Int32(K),
                       Int(y_c.unsafe_ptr()), Int32(M), Int(xq_c.unsafe_ptr()), Int(sx_c.unsafe_ptr()), Int(ks_native.value()))
    if rc != 0:
        raise Error("cuda gemm rc=" + String(rc))
    ks.synchronize()

    var qk = ctx.compile_function[quant_act[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var gk = ctx.compile_function[int8_mma_gemm[False, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    ctx.enqueue_function(qk, x, xq_m, sx_m, Int32(M * NKB), grid_dim=ceildiv(M * NKB, 64), block_dim=64)
    ctx.enqueue_function(gk, xq_m, sx_m, wq, sw, pq, ps, y_m, Int32(M), Int32(N), grid_dim=(ceildiv(N, 64), ceildiv(M, 64)), block_dim=128)
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
    # plug pass: CUDA fused plug vs Mojo fused plug, same inputs
    var cuda_plug = lib.get_function[Int32]("p7_int8_gemm_plug_flat")
    var rcp = cuda_plug(Int(x.unsafe_ptr()), Int(wq.unsafe_ptr()), Int(sw.unsafe_ptr()), Int(pq.unsafe_ptr()), Int(ps.unsafe_ptr()),
                        Int32(N), Int32(K), Int(y_cp.unsafe_ptr()), Int32(M), Int(xq_c.unsafe_ptr()), Int(sx_c.unsafe_ptr()), Int(ks_native.value()))
    if rcp != 0:
        raise Error("cuda plug rc=" + String(rcp))
    ks.synchronize()
    var gp = ctx.compile_function[int8_mma_gemm[True, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    ctx.enqueue_function(gp, xq_m, sx_m, wq, sw, pq, ps, y_mp, Int32(M), Int32(N), grid_dim=(ceildiv(N, 64), ceildiv(M, 64)), block_dim=128)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=ycp_h, src_buf=y_cp)
    ctx.enqueue_copy(dst_buf=ymp_h, src_buf=y_mp)
    ctx.synchronize()
    var pdiff = 0
    var pmax = Float64(0)
    var pchg = 0
    for i in range(M * N):
        if ycp_h[i].to_bits[DType.uint32]() != ymp_h[i].to_bits[DType.uint32]():
            pdiff += 1
        var d = abs(Float64(ycp_h[i]) - Float64(ymp_h[i]))
        if d > pmax:
            pmax = d
        if ymp_h[i].to_bits[DType.uint32]() != ym_h[i].to_bits[DType.uint32]():
            pchg += 1
    print("plug: outputs", M * N, "bitwise_different_vs_cuda", pdiff, "max_abs_diff", pmax, "outputs_changed_by_plug", pchg)
    print("PURE_MOJO_PLUG_" + ("BITEXACT" if pdiff == 0 else "NEAR"))
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
        ctx.enqueue_function(gk, xq_m, sx_m, wq, sw, pq, ps, y_m, Int32(M), Int32(N), grid_dim=(ceildiv(N, 64), ceildiv(M, 64)), block_dim=128)
    ctx.synchronize()
    var mojo_ms = Float64(perf_counter_ns() - t0) / 1e6 / 20.0
    print("cuda_tensorcore_ms_per_gemm", cuda_ms, "pure_mojo_mma_ms_per_gemm", mojo_ms, "slowdown", mojo_ms / cuda_ms)
    print("PURE_MOJO_INT8_MMA_" + ("BITEXACT" if bitdiff == 0 else "NEAR") )
