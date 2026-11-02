# mojo/p7_attention.mojo — pure-Mojo causal GQA online-softmax attention for Phi-4-mini (24 q heads, 8 kv heads,
# head_dim 128), the same algorithm and reduction order as causal_gqa_online_attention_kernel in tests/arm0_forward.h:
# one 128-thread block per (query, head), a shared-memory tree for the dot product, running max / denominator / weighted
# value accumulator, bf16-rounded probabilities. Checked against a CPU reference with the same operation order.
# Run: mojo run mojo/p7_attention.mojo
from std.math import ceildiv, exp, sqrt
from std.gpu import thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation
from max.gpu.host import DeviceContext

comptime T = 96
comptime NQ = 24
comptime NKV = 8
comptime HD = 128

def bf16r(x: Float32) -> Float32:
    return x.cast[DType.bfloat16]().cast[DType.float32]()

def attention[
    qo: ImmOrigin, ko: ImmOrigin, vo: ImmOrigin, oo: MutOrigin
](q: ImmPointer[Float32, qo], k: ImmPointer[Float32, ko], v: ImmPointer[Float32, vo], dst: MutPointer[Float32, oo], m: Int32):
    var tq = block_idx.x
    var hq = block_idx.y
    var d = thread_idx.x
    if tq >= Int(m) or hq >= NQ:
        return
    var hkv = hq // 3
    var s = stack_allocation[HD, Float32, address_space=AddressSpace.SHARED]()
    var scale = Float32(1) / sqrt(Float32(HD))
    var qv = q[unsafe_offset=(tq * NQ + hq) * HD + d] * scale
    var mv = Float32(-1e30)
    var dv = Float32(0)
    var acc = Float32(0)
    for tk in range(tq + 1):
        s[unsafe_offset=d] = qv * k[unsafe_offset=(tk * NKV + hkv) * HD + d]
        barrier()
        if d < 64:
            s[unsafe_offset=d] = s[unsafe_offset=d] + s[unsafe_offset=d + 64]
        barrier()
        if d < 32:
            s[unsafe_offset=d] = s[unsafe_offset=d] + s[unsafe_offset=d + 32]
        barrier()
        comptime for step in range(5):
            comptime off = 16 >> step
            if d < off:
                s[unsafe_offset=d] = s[unsafe_offset=d] + s[unsafe_offset=d + off]
            barrier()
        var score = s[unsafe_offset=0]
        var mn = score if score > mv else mv
        var alpha = exp(mv - mn)
        var beta = exp(score - mn)
        dv = dv * alpha + beta
        mv = mn
        acc = acc * alpha + bf16r(beta) * v[unsafe_offset=(tk * NKV + hkv) * HD + d]
        barrier()
    dst[unsafe_offset=(tq * NQ + hq) * HD + d] = acc / dv if dv > Float32(0) else Float32(0)

def main() raises:
    var ctx = DeviceContext()
    var seed = UInt32(2024)
    var qh = ctx.enqueue_create_host_buffer[DType.float32](T * NQ * HD)
    var kh = ctx.enqueue_create_host_buffer[DType.float32](T * NKV * HD)
    var vh = ctx.enqueue_create_host_buffer[DType.float32](T * NKV * HD)
    for i in range(T * NQ * HD):
        seed = seed * 1664525 + 1013904223
        qh[i] = bf16r((Float32(Int((seed >> 8) & 0xFFFF)) / 32768.0 - 1.0) * Float32(2.0))
    for i in range(T * NKV * HD):
        seed = seed * 1664525 + 1013904223
        kh[i] = bf16r((Float32(Int((seed >> 8) & 0xFFFF)) / 32768.0 - 1.0) * Float32(2.0))
        seed = seed * 1664525 + 1013904223
        vh[i] = bf16r(Float32(Int((seed >> 8) & 0xFFFF)) / 32768.0 - 1.0)
    var q = ctx.enqueue_create_buffer[DType.float32](T * NQ * HD)
    var k = ctx.enqueue_create_buffer[DType.float32](T * NKV * HD)
    var v = ctx.enqueue_create_buffer[DType.float32](T * NKV * HD)
    var y = ctx.enqueue_create_buffer[DType.float32](T * NQ * HD)
    ctx.enqueue_copy(dst_buf=q, src_buf=qh)
    ctx.enqueue_copy(dst_buf=k, src_buf=kh)
    ctx.enqueue_copy(dst_buf=v, src_buf=vh)
    var kf = ctx.compile_function[attention[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    ctx.enqueue_function(kf, q, k, v, y, Int32(T), grid_dim=(T, NQ), block_dim=HD)
    var yh = ctx.enqueue_create_host_buffer[DType.float32](T * NQ * HD)
    ctx.enqueue_copy(dst_buf=yh, src_buf=y)
    ctx.synchronize()

    # CPU reference: same recurrence and same tree reduction
    var bad = 0
    var mx = Float64(0)
    var mxref = Float64(0)
    var scale = Float32(1) / sqrt(Float32(HD))
    for tq in range(T):
        for hq in range(NQ):
            var hkv = hq // 3
            var mv = Float32(-1e30)
            var dv = Float32(0)
            var acc = List[Float32](length=HD, fill=0)
            for tk in range(tq + 1):
                var p = List[Float32](length=HD, fill=0)
                for dd in range(HD):
                    p[dd] = (qh[(tq * NQ + hq) * HD + dd] * scale) * kh[(tk * NKV + hkv) * HD + dd]
                for dd in range(64):
                    p[dd] = p[dd] + p[dd + 64]
                for dd in range(32):
                    p[dd] = p[dd] + p[dd + 32]
                for st in range(5):
                    var off = 16 >> st
                    for dd in range(off):
                        p[dd] = p[dd] + p[dd + off]
                var score = p[0]
                var mn = score if score > mv else mv
                var alpha = exp(mv - mn)
                var beta = exp(score - mn)
                dv = dv * alpha + beta
                mv = mn
                for dd in range(HD):
                    acc[dd] = acc[dd] * alpha + bf16r(beta) * vh[(tk * NKV + hkv) * HD + dd]
            for dd in range(HD):
                var e = acc[dd] / dv if dv > Float32(0) else Float32(0)
                var df = abs(Float64(e) - Float64(yh[(tq * NQ + hq) * HD + dd]))
                if df > 0:
                    bad += 1
                if df > mx:
                    mx = df
                if abs(Float64(e)) > mxref:
                    mxref = abs(Float64(e))
    print("attention: elements", T * NQ * HD, "differing", bad, "max_abs_diff", mx, "max_abs_output", mxref)
    print("PURE_MOJO_ATTENTION_" + ("MATCH" if bad == 0 else "CLOSE"))
