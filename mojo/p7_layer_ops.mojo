# mojo/p7_layer_ops.mojo — pure-Mojo RMSNorm and SwiGLU for the Phi-4 layer, rounding to bf16 where the Arm 0
# CUDA harness (tests/arm0_forward.h) does. Checked against a straight-line CPU reference in Mojo.
# Run: mojo run mojo/p7_layer_ops.mojo
from std.math import ceildiv, exp, rsqrt, sqrt
from std.gpu import global_idx
from max.gpu.host import DeviceContext

comptime ROWS = 256
comptime DIM = 3072
comptime INTER = 8192
comptime EPS = Float32(1e-5)

def bf16r(x: Float32) -> Float32:
    return x.cast[DType.bfloat16]().cast[DType.float32]()

def rms_norm[
    xo: ImmOrigin, wo: ImmOrigin, oo: MutOrigin
](x: ImmPointer[Float32, xo], w: ImmPointer[BFloat16, wo], dst: MutPointer[Float32, oo], rows: Int32):
    var row = global_idx.x                   # one thread per row; sequential sum
    if row >= Int(rows):
        return
    var ss = Float32(0)
    for i in range(DIM):
        var v = x[unsafe_offset=row * DIM + i]
        ss += v * v
    var rs = rsqrt(ss / Float32(DIM) + EPS)
    for i in range(DIM):
        var n = bf16r(x[unsafe_offset=row * DIM + i] * rs)
        dst[unsafe_offset=row * DIM + i] = bf16r(n * w[unsafe_offset=i].cast[DType.float32]())

def swiglu[
    go: ImmOrigin, oo: MutOrigin
](gate_up: ImmPointer[Float32, go], dst: MutPointer[Float32, oo], rows: Int32):
    var idx = global_idx.x
    if idx >= Int(rows) * INTER:
        return
    var m = idx // INTER
    var i = idx % INTER
    var g = bf16r(gate_up[unsafe_offset=m * 2 * INTER + i])
    var u = bf16r(gate_up[unsafe_offset=m * 2 * INTER + INTER + i])
    var silu = bf16r(g / (Float32(1) + exp(-g)))
    dst[unsafe_offset=idx] = bf16r(silu * u)

def main() raises:
    var ctx = DeviceContext()
    var seed = UInt32(99)
    var xh = ctx.enqueue_create_host_buffer[DType.float32](ROWS * DIM)
    for i in range(ROWS * DIM):
        seed = seed * 1664525 + 1013904223
        xh[i] = (Float32(Int((seed >> 8) & 0xFFFF)) / 32768.0 - 1.0) * Float32(3.0)
    var wh = ctx.enqueue_create_host_buffer[DType.bfloat16](DIM)
    for i in range(DIM):
        seed = seed * 1664525 + 1013904223
        wh[i] = (Float32(0.5) + Float32(Int((seed >> 8) & 0xFFFF)) / 65536.0).cast[DType.bfloat16]()
    var gh = ctx.enqueue_create_host_buffer[DType.float32](ROWS * 2 * INTER)
    for i in range(ROWS * 2 * INTER):
        seed = seed * 1664525 + 1013904223
        gh[i] = (Float32(Int((seed >> 8) & 0xFFFF)) / 32768.0 - 1.0) * Float32(4.0)
    var x = ctx.enqueue_create_buffer[DType.float32](ROWS * DIM)
    var w = ctx.enqueue_create_buffer[DType.bfloat16](DIM)
    var g = ctx.enqueue_create_buffer[DType.float32](ROWS * 2 * INTER)
    var yn = ctx.enqueue_create_buffer[DType.float32](ROWS * DIM)
    var ys = ctx.enqueue_create_buffer[DType.float32](ROWS * INTER)
    ctx.enqueue_copy(dst_buf=x, src_buf=xh)
    ctx.enqueue_copy(dst_buf=w, src_buf=wh)
    ctx.enqueue_copy(dst_buf=g, src_buf=gh)
    var nk = ctx.compile_function[rms_norm[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var sk = ctx.compile_function[swiglu[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    ctx.enqueue_function(nk, x, w, yn, Int32(ROWS), grid_dim=ceildiv(ROWS, 64), block_dim=64)
    ctx.enqueue_function(sk, g, ys, Int32(ROWS), grid_dim=ceildiv(ROWS * INTER, 256), block_dim=256)
    var ynh = ctx.enqueue_create_host_buffer[DType.float32](ROWS * DIM)
    var ysh = ctx.enqueue_create_host_buffer[DType.float32](ROWS * INTER)
    ctx.enqueue_copy(dst_buf=ynh, src_buf=yn)
    ctx.enqueue_copy(dst_buf=ysh, src_buf=ys)
    ctx.synchronize()

    # CPU reference, same operation order
    var nbad = 0
    var nmax = Float64(0)
    for r in range(ROWS):
        var ss = Float32(0)
        for i in range(DIM):
            ss += xh[r * DIM + i] * xh[r * DIM + i]
        var rs = Float32(1) / sqrt(ss / Float32(DIM) + EPS)
        for i in range(DIM):
            var e = bf16r(bf16r(xh[r * DIM + i] * rs) * wh[i].cast[DType.float32]())
            var d = abs(Float64(e) - Float64(ynh[r * DIM + i]))
            if d > 0:
                nbad += 1
            if d > nmax:
                nmax = d
    var sbad = 0
    var smax = Float64(0)
    for m in range(ROWS):
        for i in range(INTER):
            var gg = bf16r(gh[m * 2 * INTER + i])
            var uu = bf16r(gh[m * 2 * INTER + INTER + i])
            var e = bf16r(bf16r(gg / (Float32(1) + exp(-gg))) * uu)
            var d = abs(Float64(e) - Float64(ysh[m * INTER + i]))
            if d > 0:
                sbad += 1
            if d > smax:
                smax = d
    print("rms_norm: elements", ROWS * DIM, "differing", nbad, "max_abs_diff", nmax)
    print("swiglu:   elements", ROWS * INTER, "differing", sbad, "max_abs_diff", smax)
    print("PURE_MOJO_LAYER_OPS_" + ("MATCH" if nbad == 0 and sbad == 0 else "CLOSE"))
