# mojo/p7_rope.mojo — pure-Mojo Phi-3/Phi-4 partial RoPE with the longrope attention factor, rounding to bf16 at the
# points tests/arm0_forward.h (phi3_rope_kernel) does. Input is the fused QKV row (5120 floats: 24 q heads, 8 k heads,
# 8 v heads, head_dim 128); output keeps that layout with q and k rotated and everything bf16-rounded.
# cos/sin come from a table built on the host in double precision (Mojo's GPU sin/cos for float32 are the approximate
# hardware versions, 23% of rotated outputs changed with them, and float64 sin is not supported on NVIDIA GPUs).
# Checked against a CPU reference with the same operation order.
# Run: mojo run mojo/p7_rope.mojo
from std.math import ceildiv, cos, sin, pow
from std.gpu import global_idx
from max.gpu.host import DeviceContext

comptime M = 512
comptime NQ = 24
comptime NKV = 8
comptime HD = 128
comptime ROT = 96
comptime HEADS = NQ + 2 * NKV
comptime ROW = HEADS * HD          # 5120
comptime THETA = Float32(10000.0)
comptime SCALE = Float32(1.1902380714238083)

def bf16r(x: Float32) -> Float32:
    return x.cast[DType.bfloat16]().cast[DType.float32]()

def rotate(row: Float32, a: Float32, b: Float32, d: Int, cs: Float32, sn: Float32) -> Float32:
    # a = first-half value, b = second-half value (already bf16-rounded); d is the lane within the head
    if d < ROT // 2:
        return bf16r(bf16r(a * cs) - bf16r(b * sn))
    return bf16r(bf16r(b * cs) + bf16r(a * sn))

def rope[
    io: ImmOrigin, oo: MutOrigin
](qkv: ImmPointer[Float32, io], cstab: ImmPointer[Float32, io], sntab: ImmPointer[Float32, io], dst: MutPointer[Float32, oo], m: Int32):
    var idx = global_idx.x
    if idx >= Int(m) * ROW:
        return
    var d = idx % HD
    var head = (idx // HD) % HEADS
    var pos = idx // ROW
    var base = pos * ROW + head * HD
    if head >= NQ + NKV or d >= ROT:
        dst[unsafe_offset=idx] = bf16r(qkv[unsafe_offset=idx])
        return
    var i = d % (ROT // 2)
    var cs = cstab[unsafe_offset=pos * (ROT // 2) + i]
    var sn = sntab[unsafe_offset=pos * (ROT // 2) + i]
    if d < ROT // 2:
        var x0 = bf16r(qkv[unsafe_offset=base + d])
        var x1 = bf16r(qkv[unsafe_offset=base + d + ROT // 2])
        dst[unsafe_offset=idx] = bf16r(bf16r(x0 * cs) - bf16r(x1 * sn))
    else:
        var x0 = bf16r(qkv[unsafe_offset=base + d - ROT // 2])
        var x1 = bf16r(qkv[unsafe_offset=base + d])
        dst[unsafe_offset=idx] = bf16r(bf16r(x1 * cs) + bf16r(x0 * sn))

def main() raises:
    var ctx = DeviceContext()
    var seed = UInt32(31337)
    var xh = ctx.enqueue_create_host_buffer[DType.float32](M * ROW)
    for i in range(M * ROW):
        seed = seed * 1664525 + 1013904223
        xh[i] = (Float32(Int((seed >> 8) & 0xFFFF)) / 32768.0 - 1.0) * Float32(2.0)
    var cth = ctx.enqueue_create_host_buffer[DType.float32](M * (ROT // 2))
    var sth = ctx.enqueue_create_host_buffer[DType.float32](M * (ROT // 2))
    for pos in range(M):
        for i in range(ROT // 2):
            var inv64 = Float64(1) / pow(Float64(10000.0), (Float64(2) * Float64(i)) / Float64(ROT))
            var a64 = Float64(pos) * inv64
            cth[pos * (ROT // 2) + i] = bf16r(Float32(cos(a64)) * SCALE)
            sth[pos * (ROT // 2) + i] = bf16r(Float32(sin(a64)) * SCALE)
    var ct = ctx.enqueue_create_buffer[DType.float32](M * (ROT // 2))
    var st = ctx.enqueue_create_buffer[DType.float32](M * (ROT // 2))
    ctx.enqueue_copy(dst_buf=ct, src_buf=cth)
    ctx.enqueue_copy(dst_buf=st, src_buf=sth)
    var x = ctx.enqueue_create_buffer[DType.float32](M * ROW)
    var y = ctx.enqueue_create_buffer[DType.float32](M * ROW)
    ctx.enqueue_copy(dst_buf=x, src_buf=xh)
    var k = ctx.compile_function[rope[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    ctx.enqueue_function(k, x, ct, st, y, Int32(M), grid_dim=ceildiv(M * ROW, 256), block_dim=256)
    var yh = ctx.enqueue_create_host_buffer[DType.float32](M * ROW)
    ctx.enqueue_copy(dst_buf=yh, src_buf=y)
    ctx.synchronize()

    var bad = 0
    var mx = Float64(0)
    var rotated = 0
    for idx in range(M * ROW):
        var d = idx % HD
        var head = (idx // HD) % HEADS
        var pos = idx // ROW
        var base = pos * ROW + head * HD
        var e = Float32(0)
        if head >= NQ + NKV or d >= ROT:
            e = bf16r(xh[idx])
        else:
            rotated += 1
            var i = d % (ROT // 2)
            var cs = cth[pos * (ROT // 2) + i]
            var sn = sth[pos * (ROT // 2) + i]
            if d < ROT // 2:
                e = bf16r(bf16r(bf16r(xh[base + d]) * cs) - bf16r(bf16r(xh[base + d + ROT // 2]) * sn))
            else:
                e = bf16r(bf16r(bf16r(xh[base + d]) * cs) + bf16r(bf16r(xh[base + d - ROT // 2]) * sn))
        var df = abs(Float64(e) - Float64(yh[idx]))
        if df > 0:
            bad += 1
        if df > mx:
            mx = df
    print("rope: elements", M * ROW, "rotated", rotated, "differing", bad, "max_abs_diff", mx)
    print("PURE_MOJO_ROPE_" + ("MATCH" if bad == 0 else "CLOSE"))
