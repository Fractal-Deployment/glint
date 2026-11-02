# mojo/p7_forward.mojo — Phi-4-mini 2047-token forward pass entirely in Mojo / MAX (no CUDA C, no Python).
# Linears: INT8 pin through the Mojo W8A8 tensor-core GEMM (p7_int8_mma_gemm). Embedding, RMSNorm, RoPE, attention,
# SwiGLU, residuals, tied-embedding logits and cross-entropy are Mojo kernels with the bf16 rounding points of
# tests/arm0_forward.h. Compare with the CUDA harness arm: INT8 mma, PPL 5.496700, top-1 2003/2047.
# Run: mojo run -I mojo mojo/p7_forward.mojo PIN.safetensors MODEL_DIR ARTIFACTS_DIR
from std.math import ceildiv, exp, log, sqrt, rsqrt, cos, sin, pow
from std.sys import argv
from std.time import perf_counter_ns
from std.gpu import global_idx, thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation
from max.gpu.host import DeviceContext, DeviceBuffer
from p7_safetensors import parse_header, STEntry
from p7_int8_mma_gemm import quant_act, int8_mma_gemm

comptime T = 2048
comptime NT = 2047
comptime H = 3072
comptime INTER = 8192
comptime NQ = 24
comptime NKV = 8
comptime HD = 128
comptime ROT = 96
comptime QKV = 5120
comptime V = 200064
comptime LAYERS = 32
comptime EPS = Float32(1e-5)
comptime SCALE = Float32(1.1902380714238083)

def bf16r(x: Float32) -> Float32:
    return x.cast[DType.bfloat16]().cast[DType.float32]()

def embed_gather[
    eo: ImmOrigin, to: ImmOrigin, xo: MutOrigin
](emb: ImmPointer[BFloat16, eo], tok: ImmPointer[Int32, to], x: MutPointer[Float32, xo]):
    var i = global_idx.x
    if i >= T * H:
        return
    var t = i // H
    var d = i % H
    x[unsafe_offset=i] = emb[unsafe_offset=Int(tok[unsafe_offset=t]) * H + d].cast[DType.float32]()

def rms_norm_bf16[
    xo: ImmOrigin, wo: ImmOrigin, oo: MutOrigin
](x: ImmPointer[Float32, xo], w: ImmPointer[BFloat16, wo], dst: MutPointer[BFloat16, oo]):
    var row = global_idx.x
    if row >= T:
        return
    var ss = Float32(0)
    for i in range(H):
        var v = x[unsafe_offset=row * H + i]
        ss += v * v
    var rs = rsqrt(ss / Float32(H) + EPS)
    for i in range(H):
        var n = bf16r(x[unsafe_offset=row * H + i] * rs)
        dst[unsafe_offset=row * H + i] = (n * w[unsafe_offset=i].cast[DType.float32]()).cast[DType.bfloat16]()

def rope[
    io: ImmOrigin, co: ImmOrigin, so: ImmOrigin, oo: MutOrigin
](qkv: ImmPointer[Float32, io], cst: ImmPointer[Float32, co], snt: ImmPointer[Float32, so], dst: MutPointer[Float32, oo]):
    var idx = global_idx.x
    if idx >= T * QKV:
        return
    var d = idx % HD
    var head = (idx // HD) % (NQ + 2 * NKV)
    var pos = idx // QKV
    var base = pos * QKV + head * HD
    if head >= NQ + NKV or d >= ROT:
        dst[unsafe_offset=idx] = bf16r(qkv[unsafe_offset=idx])
        return
    var i = d % (ROT // 2)
    var cs = cst[unsafe_offset=pos * (ROT // 2) + i]
    var sn = snt[unsafe_offset=pos * (ROT // 2) + i]
    if d < ROT // 2:
        var x0 = bf16r(qkv[unsafe_offset=base + d])
        var x1 = bf16r(qkv[unsafe_offset=base + d + ROT // 2])
        dst[unsafe_offset=idx] = bf16r(bf16r(x0 * cs) - bf16r(x1 * sn))
    else:
        var x0 = bf16r(qkv[unsafe_offset=base + d - ROT // 2])
        var x1 = bf16r(qkv[unsafe_offset=base + d])
        dst[unsafe_offset=idx] = bf16r(bf16r(x1 * cs) + bf16r(x0 * sn))

def attention[
    ro: ImmOrigin, fo: MutOrigin, bo: MutOrigin
](r: ImmPointer[Float32, ro], dst: MutPointer[Float32, fo], dstb: MutPointer[BFloat16, bo]):
    var tq = block_idx.x
    var hq = block_idx.y
    var d = thread_idx.x
    var hkv = hq // 3
    var s = stack_allocation[HD, Float32, address_space=AddressSpace.SHARED]()
    var scale = Float32(1) / sqrt(Float32(HD))
    var qv = r[unsafe_offset=tq * QKV + hq * HD + d] * scale
    var mv = Float32(-1e30)
    var dv = Float32(0)
    var acc = Float32(0)
    for tk in range(tq + 1):
        s[unsafe_offset=d] = qv * r[unsafe_offset=tk * QKV + (NQ + hkv) * HD + d]
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
        acc = acc * alpha + bf16r(beta) * r[unsafe_offset=tk * QKV + (NQ + NKV + hkv) * HD + d]
        barrier()
    var o = acc / dv if dv > Float32(0) else Float32(0)
    dst[unsafe_offset=(tq * NQ + hq) * HD + d] = o
    dstb[unsafe_offset=(tq * NQ + hq) * HD + d] = o.cast[DType.bfloat16]()

def residual_add[
    ro: ImmOrigin, xo: MutOrigin
](res: ImmPointer[Float32, ro], x: MutPointer[Float32, xo]):
    var i = global_idx.x
    if i >= T * H:
        return
    x[unsafe_offset=i] = bf16r(x[unsafe_offset=i] + bf16r(res[unsafe_offset=i]))

def swiglu[
    go: ImmOrigin, fo: MutOrigin, bo: MutOrigin
](gu: ImmPointer[Float32, go], dst: MutPointer[Float32, fo], dstb: MutPointer[BFloat16, bo]):
    var idx = global_idx.x
    if idx >= T * INTER:
        return
    var m = idx // INTER
    var i = idx % INTER
    var g = bf16r(gu[unsafe_offset=m * 2 * INTER + i])
    var u = bf16r(gu[unsafe_offset=m * 2 * INTER + INTER + i])
    var silu = bf16r(g / (Float32(1) + exp(-g)))
    var val = bf16r(silu * u)
    dst[unsafe_offset=idx] = val
    dstb[unsafe_offset=idx] = val.cast[DType.bfloat16]()

comptime CH = 256

def logits_gemm[
    xo: ImmOrigin, eo: ImmOrigin, lo: MutOrigin
](xf: ImmPointer[BFloat16, xo], emb: ImmPointer[BFloat16, eo], lg: MutPointer[Float32, lo], rows: Int32):
    var idx = global_idx.x                    # c fastest: warp shares one embedding row
    var c = idx % CH
    var v = idx // CH
    if v >= V or c >= Int(rows):
        return
    var acc = Float32(0)
    for k in range(H):
        acc += xf[unsafe_offset=c * H + k].cast[DType.float32]() * emb[unsafe_offset=v * H + k].cast[DType.float32]()
    lg[unsafe_offset=c * V + v] = acc

def ce_argmax[
    lo: ImmOrigin, to: ImmOrigin, so: MutOrigin, io: MutOrigin
](lg: ImmPointer[Float32, lo], tgt: ImmPointer[Int32, to], loss: MutPointer[Float32, so], top1: MutPointer[Int32, io], rows: Int32):
    var r = global_idx.x
    if r >= Int(rows):
        return
    var mx = Float32(-1e30)
    var mi = 200064
    for v in range(V):
        var x = bf16r(lg[unsafe_offset=r * V + v])
        if x > mx or (x == mx and v < mi):
            mx = x
            mi = v
    var sum = Float32(0)
    for v in range(V):
        sum += exp(bf16r(lg[unsafe_offset=r * V + v]) - mx)
    var t = Int(tgt[unsafe_offset=r])
    loss[unsafe_offset=r] = log(sum) + mx - bf16r(lg[unsafe_offset=r * V + t])
    top1[unsafe_offset=r] = Int32(mi)

def read_range(path: String, begin: Int, n: Int) raises -> List[UInt8]:
    var f = open(path, "r")
    _ = f.seek(begin)
    return f.read_bytes(n)

def main() raises:
    var args = argv()
    var pin_path = String(args[1])
    var model_dir = String(args[2])
    var art_dir = String(args[3])
    var ctx = DeviceContext()
    var ks = ctx.create_stream()

    # tokens and oracle
    var tf = open(art_dir + "/eval_tokens_2048.txt", "r").read()
    var toks = List[Int32]()
    var cur = 0
    var have = False
    for b in tf.as_bytes():
        if b >= 48 and b <= 57:
            cur = cur * 10 + Int(b) - 48
            have = True
        elif have:
            toks.append(Int32(cur))
            cur = 0
            have = False
    if have:
        toks.append(Int32(cur))
    var ob = open(art_dir + "/oracle_top1_2048.bin", "r").read_bytes()
    print("tokens", len(toks), "oracle_bytes", len(ob))

    # pin header + BF16 shard headers
    def read_header(path: String) raises -> Tuple[List[STEntry], Int]:
        var f = open(path, "r")
        var h8 = f.read_bytes(8)
        var hl = 0
        for k in range(8):
            hl |= Int(h8[k]) << (8 * k)
        var hb = f.read_bytes(hl)
        return (parse_header(Span(hb)), 8 + hl)
    var pin_h = read_header(pin_path)
    var s1_path = model_dir + "/model-00001-of-00002.safetensors"
    var s2_path = model_dir + "/model-00002-of-00002.safetensors"
    var s1_h = read_header(s1_path)
    var s2_h = read_header(s2_path)

    def find_entry(hd: Tuple[List[STEntry], Int], name: String) -> Int:
        for i in range(len(hd[0])):
            if hd[0][i].name == name:
                return i
        return -1

    def load_u8(ctx: DeviceContext, path: String, hd: Tuple[List[STEntry], Int], name: String) raises -> type_of(ctx.enqueue_create_buffer[DType.uint8](1)):
        var ix = find_entry(hd, name)
        if ix < 0:
            raise Error("missing tensor " + name + " in " + path)
        var e = hd[0][ix].copy()
        var n = e.end - e.begin
        var bytes = read_range(path, hd[1] + e.begin, n)
        var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
        for i in range(n):
            host[i] = bytes[i]
        var dev = ctx.enqueue_create_buffer[DType.uint8](n)
        ctx.enqueue_copy(dst_buf=dev, src_buf=host)
        ctx.synchronize()
        return dev^

    print("loading weights ...")
    var t_load = perf_counter_ns()
    def bf16_from(ctx: DeviceContext, name: String, h1: Tuple[List[STEntry], Int], h2: Tuple[List[STEntry], Int], p1: String, p2: String) raises -> type_of(ctx.enqueue_create_buffer[DType.uint8](1)):
        if find_entry(h1, name) >= 0:
            return load_u8(ctx, p1, h1, name)
        return load_u8(ctx, p2, h2, name)
    var emb_u8 = bf16_from(ctx, "model.embed_tokens.weight", s1_h, s2_h, s1_path, s2_path)
    var fnorm_u8 = bf16_from(ctx, "model.norm.weight", s1_h, s2_h, s1_path, s2_path)
    var in_norm = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var post_norm = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var lq = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var ls = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var names = List[String]()
    names.append("self_attn.qkv_proj")
    names.append("self_attn.o_proj")
    names.append("mlp.gate_up_proj")
    names.append("mlp.down_proj")
    for l in range(LAYERS):
        in_norm.append(bf16_from(ctx, "model.layers." + String(l) + ".input_layernorm.weight", s1_h, s2_h, s1_path, s2_path))
        post_norm.append(bf16_from(ctx, "model.layers." + String(l) + ".post_attention_layernorm.weight", s1_h, s2_h, s1_path, s2_path))
        for j in range(4):
            var nm = "model.layers." + String(l) + "." + names[j] + ".weight"
            lq.append(load_u8(ctx, pin_path, pin_h, nm))
            ls.append(load_u8(ctx, pin_path, pin_h, nm + ".int8_scale"))
    print("weights loaded in", Float64(perf_counter_ns() - t_load) / 1e9, "s")
    print("MOJO_FORWARD_LOADED")

    # typed non-owning views over the uint8 weight buffers
    var emb = DeviceBuffer[DType.bfloat16](ctx, emb_u8.unsafe_ptr().unsafe_bitcast[BFloat16](), V * H, owning=False)
    var fnorm = DeviceBuffer[DType.bfloat16](ctx, fnorm_u8.unsafe_ptr().unsafe_bitcast[BFloat16](), H, owning=False)
    var wq_v = List[DeviceBuffer[DType.int8]]()
    var sw_v = List[DeviceBuffer[DType.float32]]()
    var innorm_v = List[DeviceBuffer[DType.bfloat16]]()
    var postnorm_v = List[DeviceBuffer[DType.bfloat16]]()
    var kin = List[Int]()
    var nout = List[Int]()
    kin.append(H)
    kin.append(H)
    kin.append(H)
    kin.append(INTER)
    nout.append(QKV)
    nout.append(H)
    nout.append(2 * INTER)
    nout.append(H)
    for l in range(LAYERS):
        innorm_v.append(DeviceBuffer[DType.bfloat16](ctx, in_norm[l].unsafe_ptr().unsafe_bitcast[BFloat16](), H, owning=False))
        postnorm_v.append(DeviceBuffer[DType.bfloat16](ctx, post_norm[l].unsafe_ptr().unsafe_bitcast[BFloat16](), H, owning=False))
        for j in range(4):
            wq_v.append(DeviceBuffer[DType.int8](ctx, lq[l * 4 + j].unsafe_ptr().unsafe_bitcast[Int8](), nout[j] * kin[j], owning=False))
            sw_v.append(DeviceBuffer[DType.float32](ctx, ls[l * 4 + j].unsafe_ptr().unsafe_bitcast[Float32](), nout[j] * kin[j] // 64, owning=False))

    # RoPE tables (double precision on the host)
    var cth = ctx.enqueue_create_host_buffer[DType.float32](T * (ROT // 2))
    var sth = ctx.enqueue_create_host_buffer[DType.float32](T * (ROT // 2))
    for pos in range(T):
        for i in range(ROT // 2):
            var inv64 = Float64(1) / pow(Float64(10000.0), (Float64(2) * Float64(i)) / Float64(ROT))
            var a64 = Float64(pos) * inv64
            cth[pos * (ROT // 2) + i] = bf16r(Float32(cos(a64)) * SCALE)
            sth[pos * (ROT // 2) + i] = bf16r(Float32(sin(a64)) * SCALE)
    var ct = ctx.enqueue_create_buffer[DType.float32](T * (ROT // 2))
    var st = ctx.enqueue_create_buffer[DType.float32](T * (ROT // 2))
    ctx.enqueue_copy(dst_buf=ct, src_buf=cth)
    ctx.enqueue_copy(dst_buf=st, src_buf=sth)

    var tokh = ctx.enqueue_create_host_buffer[DType.int32](T)
    for i in range(T):
        tokh[i] = toks[i]
    var tokd = ctx.enqueue_create_buffer[DType.int32](T)
    ctx.enqueue_copy(dst_buf=tokd, src_buf=tokh)
    var tgt = ctx.enqueue_create_buffer[DType.int32](NT)
    var tgth = ctx.enqueue_create_host_buffer[DType.int32](NT)
    for i in range(NT):
        tgth[i] = toks[i + 1]
    ctx.enqueue_copy(dst_buf=tgt, src_buf=tgth)

    var x = ctx.enqueue_create_buffer[DType.float32](T * H)
    var hbf = ctx.enqueue_create_buffer[DType.bfloat16](T * H)
    var qkv = ctx.enqueue_create_buffer[DType.float32](T * QKV)
    var rp = ctx.enqueue_create_buffer[DType.float32](T * QKV)
    var attn = ctx.enqueue_create_buffer[DType.float32](T * H)
    var attnb = ctx.enqueue_create_buffer[DType.bfloat16](T * H)
    var proj = ctx.enqueue_create_buffer[DType.float32](T * H)
    var gu = ctx.enqueue_create_buffer[DType.float32](T * 2 * INTER)
    var act = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var actb = ctx.enqueue_create_buffer[DType.bfloat16](T * INTER)
    var down = ctx.enqueue_create_buffer[DType.float32](T * H)
    var xq = ctx.enqueue_create_buffer[DType.int8](T * INTER)
    var sx = ctx.enqueue_create_buffer[DType.float32](T * INTER // 64)
    var lg = ctx.enqueue_create_buffer[DType.float32](CH * V)
    var loss = ctx.enqueue_create_buffer[DType.float32](NT)
    var top1 = ctx.enqueue_create_buffer[DType.int32](NT)
    var dummy_p = ctx.enqueue_create_buffer[DType.uint8](16)
    var dummy_s = ctx.enqueue_create_buffer[DType.float32](16)
    ctx.synchronize()

    var k_embed = ctx.compile_function[embed_gather[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_norm = ctx.compile_function[rms_norm_bf16[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_rope = ctx.compile_function[rope[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_attn = ctx.compile_function[attention[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_res = ctx.compile_function[residual_add[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_swi = ctx.compile_function[swiglu[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_quant = ctx.compile_function[quant_act[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gemm = ctx.compile_function[int8_mma_gemm[False, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_lg = ctx.compile_function[logits_gemm[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_ce = ctx.compile_function[ce_argmax[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()

    print("forward ...")
    var t0 = perf_counter_ns()
    ctx.enqueue_function(k_embed, emb, tokd, x, grid_dim=ceildiv(T * H, 256), block_dim=256)
    for l in range(LAYERS):
        ctx.enqueue_function(k_norm, x, innorm_v[l], hbf, grid_dim=ceildiv(T, 64), block_dim=64)
        # qkv
        ctx.enqueue_function(k_quant, hbf, xq, sx, Int32(T * (H // 64)), grid_dim=ceildiv(T * (H // 64), 64), block_dim=64)
        ctx.enqueue_function(k_gemm, xq, sx, wq_v[l * 4], sw_v[l * 4], dummy_p, dummy_s, qkv, Int32(T), Int32(QKV), Int32(H),
                             grid_dim=(ceildiv(QKV, 64), ceildiv(T, 64)), block_dim=128)
        ctx.enqueue_function(k_rope, qkv, ct, st, rp, grid_dim=ceildiv(T * QKV, 256), block_dim=256)
        ctx.enqueue_function(k_attn, rp, attn, attnb, grid_dim=(T, NQ), block_dim=HD)
        # o proj
        ctx.enqueue_function(k_quant, attnb, xq, sx, Int32(T * (H // 64)), grid_dim=ceildiv(T * (H // 64), 64), block_dim=64)
        ctx.enqueue_function(k_gemm, xq, sx, wq_v[l * 4 + 1], sw_v[l * 4 + 1], dummy_p, dummy_s, proj, Int32(T), Int32(H), Int32(H),
                             grid_dim=(ceildiv(H, 64), ceildiv(T, 64)), block_dim=128)
        ctx.enqueue_function(k_res, proj, x, grid_dim=ceildiv(T * H, 256), block_dim=256)
        ctx.enqueue_function(k_norm, x, postnorm_v[l], hbf, grid_dim=ceildiv(T, 64), block_dim=64)
        # gate_up
        ctx.enqueue_function(k_quant, hbf, xq, sx, Int32(T * (H // 64)), grid_dim=ceildiv(T * (H // 64), 64), block_dim=64)
        ctx.enqueue_function(k_gemm, xq, sx, wq_v[l * 4 + 2], sw_v[l * 4 + 2], dummy_p, dummy_s, gu, Int32(T), Int32(2 * INTER), Int32(H),
                             grid_dim=(ceildiv(2 * INTER, 64), ceildiv(T, 64)), block_dim=128)
        ctx.enqueue_function(k_swi, gu, act, actb, grid_dim=ceildiv(T * INTER, 256), block_dim=256)
        # down
        ctx.enqueue_function(k_quant, actb, xq, sx, Int32(T * (INTER // 64)), grid_dim=ceildiv(T * (INTER // 64), 64), block_dim=64)
        ctx.enqueue_function(k_gemm, xq, sx, wq_v[l * 4 + 3], sw_v[l * 4 + 3], dummy_p, dummy_s, down, Int32(T), Int32(H), Int32(INTER),
                             grid_dim=(ceildiv(H, 64), ceildiv(T, 64)), block_dim=128)
        ctx.enqueue_function(k_res, down, x, grid_dim=ceildiv(T * H, 256), block_dim=256)
    ctx.enqueue_function(k_norm, x, fnorm, hbf, grid_dim=ceildiv(T, 64), block_dim=64)
    ctx.synchronize()
    var t_layers = Float64(perf_counter_ns() - t0) / 1e9
    print("layers done in", t_layers, "s")

    var off = 0
    while off < NT:
        var cur = CH if off + CH <= NT else NT - off
        var hv = DeviceBuffer[DType.bfloat16](ctx, hbf.unsafe_ptr().unsafe_offset(off * H), cur * H, owning=False)
        ctx.enqueue_function(k_lg, hv, emb, lg, Int32(cur), grid_dim=ceildiv(V * CH, 256), block_dim=256)
        var tv = DeviceBuffer[DType.int32](ctx, tgt.unsafe_ptr().unsafe_offset(off), cur, owning=False)
        var lv = DeviceBuffer[DType.float32](ctx, loss.unsafe_ptr().unsafe_offset(off), cur, owning=False)
        var t1v = DeviceBuffer[DType.int32](ctx, top1.unsafe_ptr().unsafe_offset(off), cur, owning=False)
        ctx.enqueue_function(k_ce, lg, tv, lv, t1v, Int32(cur), grid_dim=ceildiv(cur, 64), block_dim=64)
        ctx.synchronize()
        off += cur
    var total = Float64(perf_counter_ns() - t0) / 1e9

    var lh = ctx.enqueue_create_host_buffer[DType.float32](NT)
    var th = ctx.enqueue_create_host_buffer[DType.int32](NT)
    ctx.enqueue_copy(dst_buf=lh, src_buf=loss)
    ctx.enqueue_copy(dst_buf=th, src_buf=top1)
    ctx.synchronize()
    var tot = Float64(0)
    var nmatch = 0
    for i in range(NT):
        tot += Float64(lh[i])
        var o = Int(ob[4 * i]) | (Int(ob[4 * i + 1]) << 8) | (Int(ob[4 * i + 2]) << 16) | (Int(ob[4 * i + 3]) << 24)
        if Int(th[i]) == o:
            nmatch += 1
    var ce = tot / Float64(NT)
    print("Forward time:", total, "s")
    print("Mean cross entropy:", ce)
    print("Perplexity:", exp(ce))
    print("Top-1 matches vs HF oracle:", nmatch, "/", NT, "=", Float64(nmatch) / Float64(NT) * 100.0, "%")
    print("MOJO_FORWARD_DONE")
    # the typed views above are non-owning: keep the backing buffers alive until here
    _ = emb_u8.unsafe_ptr()
    _ = fnorm_u8.unsafe_ptr()
    _ = len(in_norm) + len(post_norm) + len(lq) + len(ls)
