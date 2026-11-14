# Project 7 / GLINT: recorded step-through tracer for SmolLM2-135M (Llama architecture) on a glint_pin_v1 pin. Pure Mojo, CPU, f32.
# Runs the prompt token by token (KV cache) and records EVERY intermediate of the LAST prompt token to a JSON file that
# mojo/../scripts/p7_trace_viewer_build.sh turns into a self-contained viewer:
#   embedding decode, RMSNorm (sum of squares, mean, rsqrt, normalised, weight, output), q/k/v/o/gate/up/down/logits matmuls,
#   RoPE (cos, sin, before, after), KV cache, attention (scores, softmax, weighted V), SiLU gate, residual adds, final logits.
# For every matmul the input vector, output vector, and PROBE ROWS are recorded: for a probe row the hole nibble h, plug nibble p
# and block absmax of every weight element, so a viewer recomputes w = cells[h][p] * absmax, product = w * x and the running f32
# sum, and checks it against the traced output. Arithmetic is f32 with one fixed summation order (k = 0..K-1); no bf16 rounding of
# activations (the production Mojo/CUDA paths round some activations to bf16), so this is the f32 reference of the pin's math.
# Usage: mojo build -I mojo -o p7_trace_smol mojo/p7_trace_smol.mojo
#        ./p7_trace_smol PIN_DIR OUT.json ID,ID,...      (comma-separated SmolLM2 token ids; the last one is traced)
from std.math import exp, sqrt, cos, sin, pow
from std.memory import bitcast
from std.sys import argv
from p7_safetensors import parse_header, STEntry

comptime H = 576
comptime NH = 9
comptime NKV = 3
comptime HD = 64
comptime KVD = 192
comptime INTER = 1536
comptime NL = 30
comptime V = 49152
comptime EPS = Float32(1e-5)
comptime THETA = 100000.0

def u32_to_f32(v: UInt32) -> Float32:
    return bitcast[DType.float32](SIMD[DType.uint32, 1](v))

def f32_at(data: List[UInt8], off: Int) -> Float32:
    var v = UInt32(data[off]) | (UInt32(data[off + 1]) << 8) | (UInt32(data[off + 2]) << 16) | (UInt32(data[off + 3]) << 24)
    return u32_to_f32(v)

def bf16_at(data: List[UInt8], off: Int) -> Float32:
    var v = (UInt32(data[off]) << 16) | (UInt32(data[off + 1]) << 24)
    return u32_to_f32(v)

def entry_off(ents: List[STEntry], base: Int, name: String) raises -> Int:
    for i in range(len(ents)):
        if ents[i].name == name:
            return base + ents[i].begin
    raise Error("missing tensor " + name)

def entry_len(ents: List[STEntry], name: String) raises -> Int:
    for i in range(len(ents)):
        if ents[i].name == name:
            return ents[i].end - ents[i].begin
    raise Error("missing tensor " + name)

def load_f32s(data: List[UInt8], off: Int, n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(f32_at(data, off + 4 * i))
    return out^

def load_bf16s(data: List[UInt8], off: Int, n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(bf16_at(data, off + 2 * i))
    return out^

# nibble j of a packed tensor (low nibble first)
def nib(data: List[UInt8], off: Int, j: Int) -> Int:
    return Int((data[off + (j >> 1)] >> UInt8(4 * (j & 1))) & UInt8(15))

# One linear projection of the pin: hole nibbles, plug nibbles, block-64 absmax over the flattened row-major index.
struct Lin(Copyable, Movable):
    var name: String
    var ho: Int
    var po: Int
    var scales: List[Float32]
    var n: Int
    var k: Int

    def __init__(out self, name: String):
        self.name = name
        self.ho = 0
        self.po = 0
        self.scales = List[Float32]()
        self.n = 0
        self.k = 0

def make_lin(data: List[UInt8], ents: List[STEntry], base: Int, name: String, n: Int, k: Int) raises -> Lin:
    var l = Lin(name)
    l.ho = entry_off(ents, base, name)
    l.po = entry_off(ents, base, name + ".glint_plug")
    l.scales = load_f32s(data, entry_off(ents, base, name + ".absmax"), n * k // 64)
    l.n = n
    l.k = k
    return l^

def weight_at(data: List[UInt8], l: Lin, cells: List[Float32], j: Int) -> Float32:
    var h = nib(data, l.ho, j)
    var p = nib(data, l.po, j)
    return cells[h * 16 + p] * l.scales[j // 64]

# Product rounded to f32 as its own step: a non-inlined call stops the compiler fusing acc + w*x into one FMA, so the recorded
# arithmetic is exactly "round(w*x), then round(acc + product)", which the viewer reproduces bit for bit.
@no_inline
def mul32(a: Float32, b: Float32) -> Float32:
    return a * b

# Same idea for the reduction itself: a non-inlined add keeps the sum strictly left to right (the compiler otherwise reassociates
# simple reductions such as the sum of squares).
@no_inline
def add32(a: Float32, b: Float32) -> Float32:
    return a + b

def matvec(data: List[UInt8], l: Lin, cells: List[Float32], x: List[Float32]) -> List[Float32]:
    var y = List[Float32]()
    for r in range(l.n):
        var acc = Float32(0)
        var b = r * l.k
        for kk in range(l.k):
            var w = weight_at(data, l, cells, b + kk)
            var prod = mul32(w, x[kk])
            acc = acc + prod
        y.append(acc)
    return y^

# Floats are written as their raw IEEE-754 binary32 bit pattern (uint32): decimal printing is not guaranteed to round-trip.
def f2u(v: Float32) -> UInt32:
    return bitcast[DType.uint32](SIMD[DType.float32, 1](v))[0]

def jf(v: Float32) -> String:
    return String(f2u(v))

def arr_f(v: List[Float32]) -> String:
    var s = String("[")
    for i in range(len(v)):
        if i > 0:
            s += ","
        s += jf(v[i])
    s += "]"
    return s^

def arr_i(v: List[Int]) -> String:
    var s = String("[")
    for i in range(len(v)):
        if i > 0:
            s += ","
        s += String(v[i])
    s += "]"
    return s^

def probe_json(data: List[UInt8], l: Lin, row: Int) -> String:
    var hs = List[Int]()
    var ps = List[Int]()
    var ss = List[Float32]()
    var b = row * l.k
    for kk in range(l.k):
        hs.append(nib(data, l.ho, b + kk))
        ps.append(nib(data, l.po, b + kk))
    for blk in range(l.k // 64):
        ss.append(l.scales[(b + blk * 64) // 64])
    return String('{"row":') + String(row) + ',"h":' + arr_i(hs) + ',"p":' + arr_i(ps) + ',"s":' + arr_f(ss) + "}"

def pick_rows(y: List[Float32]) -> List[Int]:
    var rows = List[Int]()
    rows.append(0)
    var best = 0
    var bv = Float32(-1.0)
    for i in range(len(y)):
        var a = y[i]
        if a < 0:
            a = -a
        if a > bv:
            bv = a
            best = i
    if best != 0:
        rows.append(best)
    var mid = len(y) // 2
    if mid != 0 and mid != best:
        rows.append(mid)
    return rows^

def top_rows(y: List[Float32], k: Int) -> List[Int]:
    var rows = List[Int]()
    var used = List[Bool]()
    for _ in range(len(y)):
        used.append(False)
    for _ in range(k):
        var best = -1
        var bv = Float32(-1e30)
        for i in range(len(y)):
            if not used[i] and y[i] > bv:
                bv = y[i]
                best = i
        used[best] = True
        rows.append(best)
    return rows^

def add_tensor(mut tj: String, name: String, v: List[Float32]):
    tj += String('"') + name + '":' + arr_f(v) + ","

def rms_norm(x: List[Float32], w: List[Float32]) -> List[Float32]:
    var ss = Float32(0)
    for i in range(H):
        ss = add32(ss, mul32(x[i], x[i]))
    var mean = ss / Float32(H)
    var rs = Float32(1) / sqrt(mean + EPS)
    var out = List[Float32]()
    for i in range(H):
        var n = x[i] * rs
        out.append(w[i] * n)
    return out^

def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error("usage: p7_trace_smol PIN_DIR OUT.json ID,ID,...")
    var pin_dir = String(args[1])
    var out_path = String(args[2])
    var ids = List[Int]()
    var cur = 0
    var have = False
    for b in String(args[3]).as_bytes():
        if b == 44:
            ids.append(cur)
            cur = 0
            have = False
        else:
            cur = cur * 10 + Int(b) - 48
            have = True
    if have:
        ids.append(cur)
    var T = len(ids)
    var path = pin_dir + "/model.safetensors"
    var f = open(path, "r")
    var data = f.read_bytes()
    var hl = 0
    for j in range(8):
        hl |= Int(data[j]) << (8 * j)
    var hdr = List[UInt8]()
    for j in range(hl):
        hdr.append(data[8 + j])
    var ents = parse_header(Span(hdr))
    var base = 8 + hl
    var cells = load_f32s(data, entry_off(ents, base, "_glint_cells"), 256)
    var emb = make_lin(data, ents, base, "model.embed_tokens.weight", V, H)
    var fnorm = load_bf16s(data, entry_off(ents, base, "model.norm.weight"), H)

    var lq = List[Lin]()
    var lk = List[Lin]()
    var lv = List[Lin]()
    var lo = List[Lin]()
    var lg = List[Lin]()
    var lu = List[Lin]()
    var ld = List[Lin]()
    var in_w = List[List[Float32]]()
    var post_w = List[List[Float32]]()
    for l in range(NL):
        var p = String("model.layers.") + String(l) + "."
        lq.append(make_lin(data, ents, base, p + "self_attn.q_proj.weight", H, H))
        lk.append(make_lin(data, ents, base, p + "self_attn.k_proj.weight", KVD, H))
        lv.append(make_lin(data, ents, base, p + "self_attn.v_proj.weight", KVD, H))
        lo.append(make_lin(data, ents, base, p + "self_attn.o_proj.weight", H, H))
        lg.append(make_lin(data, ents, base, p + "mlp.gate_proj.weight", INTER, H))
        lu.append(make_lin(data, ents, base, p + "mlp.up_proj.weight", INTER, H))
        ld.append(make_lin(data, ents, base, p + "mlp.down_proj.weight", H, INTER))
        in_w.append(load_bf16s(data, entry_off(ents, base, p + "input_layernorm.weight"), H))
        post_w.append(load_bf16s(data, entry_off(ents, base, p + "post_attention_layernorm.weight"), H))
    print("SMOL_TRACE loaded pin, tokens", T)

    var kc = List[List[Float32]]()
    var vc = List[List[Float32]]()
    for _ in range(NL):
        kc.append(List[Float32]())
        vc.append(List[Float32]())

    var tj = String("")
    var oj = String("")
    var opn = 0
    var tpos = T - 1

    for pos in range(T):
        var trace = pos == tpos
        var x = List[Float32]()
        for kk in range(H):
            x.append(weight_at(data, emb, cells, ids[pos] * H + kk))
        if trace:
            add_tensor(tj, "embed", x)
            oj += String('{"id":') + String(opn) + ',"layer":-1,"kind":"embed","name":"token embedding (GLINT decode)","token":' + String(ids[pos]) + ',"y":"embed","probe":' + probe_json(data, emb, ids[pos]) + "},"
            opn += 1
        for l in range(NL):
            var ln = String("L") + String(l) + "."
            if trace:
                add_tensor(tj, ln + "x_in", x)
            # ---- input RMSNorm ----
            var h1 = rms_norm(x, in_w[l])
            if trace:
                var ss = Float32(0)
                for i in range(H):
                    ss = add32(ss, mul32(x[i], x[i]))
                var mean = ss / Float32(H)
                var rs = Float32(1) / sqrt(mean + EPS)
                add_tensor(tj, ln + "in_ln_w", in_w[l])
                add_tensor(tj, ln + "norm1", h1)
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"rmsnorm","name":"input_layernorm","x":"' + ln + 'x_in","w":"' + ln + 'in_ln_w","y":"' + ln + 'norm1","sumsq":' + jf(ss) + ',"mean":' + jf(mean) + ',"rsqrt":' + jf(rs) + ',"eps":' + jf(EPS) + "},"
                opn += 1
            # ---- q, k, v projections ----
            var q = matvec(data, lq[l], cells, h1)
            var k = matvec(data, lk[l], cells, h1)
            var v = matvec(data, lv[l], cells, h1)
            if trace:
                add_tensor(tj, ln + "q", q)
                add_tensor(tj, ln + "k", k)
                add_tensor(tj, ln + "v", v)
                var names = List[String]()
                names.append("q_proj")
                names.append("k_proj")
                names.append("v_proj")
                for mi in range(3):
                    var yv = q.copy()
                    var lin = lq[l].copy()
                    var yn = String("q")
                    if mi == 1:
                        yv = k.copy()
                        lin = lk[l].copy()
                        yn = String("k")
                    if mi == 2:
                        yv = v.copy()
                        lin = lv[l].copy()
                        yn = String("v")
                    var rows = pick_rows(yv)
                    var pj = String("[")
                    for ri in range(len(rows)):
                        if ri > 0:
                            pj += ","
                        pj += probe_json(data, lin, rows[ri])
                    pj += "]"
                    oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"matmul","name":"' + names[mi] + '","weight":"' + lin.name + '","N":' + String(lin.n) + ',"K":' + String(lin.k) + ',"x":"' + ln + 'norm1","y":"' + ln + yn + '","probes":' + pj + "},"
                    opn += 1
            # ---- RoPE (non-interleaved: pair i with i + HD/2) ----
            var cs = List[Float32]()
            var sn = List[Float32]()
            for i in range(HD // 2):
                var inv = Float64(1) / pow(Float64(THETA), (Float64(2) * Float64(i)) / Float64(HD))
                var a = Float64(pos) * inv
                cs.append(Float32(cos(a)))
                sn.append(Float32(sin(a)))
            var qr = q.copy()
            var kr = k.copy()
            for hh in range(NH):
                for i in range(HD // 2):
                    var a1 = q[hh * HD + i]
                    var a2 = q[hh * HD + i + HD // 2]
                    qr[hh * HD + i] = mul32(a1, cs[i]) - mul32(a2, sn[i])
                    qr[hh * HD + i + HD // 2] = mul32(a2, cs[i]) + mul32(a1, sn[i])
            for hh in range(NKV):
                for i in range(HD // 2):
                    var a1 = k[hh * HD + i]
                    var a2 = k[hh * HD + i + HD // 2]
                    kr[hh * HD + i] = mul32(a1, cs[i]) - mul32(a2, sn[i])
                    kr[hh * HD + i + HD // 2] = mul32(a2, cs[i]) + mul32(a1, sn[i])
            if trace:
                add_tensor(tj, ln + "cos", cs)
                add_tensor(tj, ln + "sin", sn)
                add_tensor(tj, ln + "q_rope", qr)
                add_tensor(tj, ln + "k_rope", kr)
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"rope","name":"rotary position embedding","position":' + String(pos) + ',"theta":' + jf(Float32(THETA)) + ',"q":"' + ln + 'q","k":"' + ln + 'k","cos":"' + ln + 'cos","sin":"' + ln + 'sin","q_out":"' + ln + 'q_rope","k_out":"' + ln + 'k_rope","heads_q":' + String(NH) + ',"heads_kv":' + String(NKV) + ',"head_dim":' + String(HD) + "},"
                opn += 1
            # ---- KV write ----
            for i in range(KVD):
                kc[l].append(kr[i])
                vc[l].append(v[i])
            # ---- attention (grouped-query: q head hq reads kv head hq // 3) ----
            var T1 = pos + 1
            var attn = List[Float32]()
            for _ in range(H):
                attn.append(Float32(0))
            var scores = List[Float32]()
            var probs = List[Float32]()
            for hq in range(NH):
                var kvh = hq // (NH // NKV)
                var sc = List[Float32]()
                var mx = Float32(-1e30)
                for t in range(T1):
                    var dot = Float32(0)
                    for d in range(HD):
                        dot = dot + mul32(qr[hq * HD + d], kc[l][t * KVD + kvh * HD + d])
                    var s = dot * Float32(0.125)
                    sc.append(s)
                    if s > mx:
                        mx = s
                var den = Float32(0)
                var ex = List[Float32]()
                for t in range(T1):
                    var e = exp(sc[t] - mx)
                    ex.append(e)
                    den = den + e
                for t in range(T1):
                    var pr = ex[t] / den
                    scores.append(sc[t])
                    probs.append(pr)
                for d in range(HD):
                    var acc = Float32(0)
                    for t in range(T1):
                        acc = acc + (ex[t] / den) * vc[l][t * KVD + kvh * HD + d]
                    attn[hq * HD + d] = acc
            if trace:
                add_tensor(tj, ln + "kc", kc[l])
                add_tensor(tj, ln + "vc", vc[l])
                add_tensor(tj, ln + "scores", scores)
                add_tensor(tj, ln + "probs", probs)
                add_tensor(tj, ln + "attn", attn)
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"attention","name":"attention (softmax over cached keys)","T":' + String(T1) + ',"q":"' + ln + 'q_rope","kc":"' + ln + 'kc","vc":"' + ln + 'vc","scores":"' + ln + 'scores","probs":"' + ln + 'probs","y":"' + ln + 'attn","heads":' + String(NH) + ',"kv_heads":' + String(NKV) + ',"head_dim":' + String(HD) + ',"scale":0.125},'
                opn += 1
            # ---- o_proj + residual ----
            var o = matvec(data, lo[l], cells, attn)
            var xm = List[Float32]()
            for i in range(H):
                xm.append(x[i] + o[i])
            if trace:
                add_tensor(tj, ln + "o", o)
                add_tensor(tj, ln + "x_mid", xm)
                var rows = pick_rows(o)
                var pj = String("[")
                for ri in range(len(rows)):
                    if ri > 0:
                        pj += ","
                    pj += probe_json(data, lo[l], rows[ri])
                pj += "]"
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"matmul","name":"o_proj","weight":"' + lo[l].name + '","N":' + String(H) + ',"K":' + String(H) + ',"x":"' + ln + 'attn","y":"' + ln + 'o","probes":' + pj + "},"
                opn += 1
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"residual","name":"residual add (attention)","a":"' + ln + 'x_in","b":"' + ln + 'o","y":"' + ln + 'x_mid"},'
                opn += 1
            # ---- post-attention RMSNorm ----
            var h2 = rms_norm(xm, post_w[l])
            if trace:
                var ss2 = Float32(0)
                for i in range(H):
                    ss2 = add32(ss2, mul32(xm[i], xm[i]))
                var mean2 = ss2 / Float32(H)
                var rs2 = Float32(1) / sqrt(mean2 + EPS)
                add_tensor(tj, ln + "post_ln_w", post_w[l])
                add_tensor(tj, ln + "norm2", h2)
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"rmsnorm","name":"post_attention_layernorm","x":"' + ln + 'x_mid","w":"' + ln + 'post_ln_w","y":"' + ln + 'norm2","sumsq":' + jf(ss2) + ',"mean":' + jf(mean2) + ',"rsqrt":' + jf(rs2) + ',"eps":' + jf(EPS) + "},"
                opn += 1
            # ---- MLP: gate, up, SiLU gate, down, residual ----
            var gt = matvec(data, lg[l], cells, h2)
            var up = matvec(data, lu[l], cells, h2)
            var sg = List[Float32]()
            var act = List[Float32]()
            for i in range(INTER):
                var s = Float32(1) / (Float32(1) + exp(-gt[i]))
                sg.append(s)
                act.append((gt[i] * s) * up[i])
            var dn = matvec(data, ld[l], cells, act)
            var xo = List[Float32]()
            for i in range(H):
                xo.append(xm[i] + dn[i])
            if trace:
                add_tensor(tj, ln + "gate", gt)
                add_tensor(tj, ln + "up", up)
                add_tensor(tj, ln + "sigmoid", sg)
                add_tensor(tj, ln + "act", act)
                add_tensor(tj, ln + "down", dn)
                add_tensor(tj, ln + "x_out", xo)
                for mi in range(2):
                    var yv = gt.copy()
                    var lin = lg[l].copy()
                    var nm = String("gate_proj")
                    var yn = String("gate")
                    if mi == 1:
                        yv = up.copy()
                        lin = lu[l].copy()
                        nm = String("up_proj")
                        yn = String("up")
                    var rows = pick_rows(yv)
                    var pj = String("[")
                    for ri in range(len(rows)):
                        if ri > 0:
                            pj += ","
                        pj += probe_json(data, lin, rows[ri])
                    pj += "]"
                    oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"matmul","name":"' + nm + '","weight":"' + lin.name + '","N":' + String(lin.n) + ',"K":' + String(lin.k) + ',"x":"' + ln + 'norm2","y":"' + ln + yn + '","probes":' + pj + "},"
                    opn += 1
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"silu_gate","name":"SiLU(gate) * up","gate":"' + ln + 'gate","up":"' + ln + 'up","sigmoid":"' + ln + 'sigmoid","y":"' + ln + 'act"},'
                opn += 1
                var rows3 = pick_rows(dn)
                var pj3 = String("[")
                for ri in range(len(rows3)):
                    if ri > 0:
                        pj3 += ","
                    pj3 += probe_json(data, ld[l], rows3[ri])
                pj3 += "]"
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"matmul","name":"down_proj","weight":"' + ld[l].name + '","N":' + String(H) + ',"K":' + String(INTER) + ',"x":"' + ln + 'act","y":"' + ln + 'down","probes":' + pj3 + "},"
                opn += 1
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"residual","name":"residual add (MLP)","a":"' + ln + 'x_mid","b":"' + ln + 'down","y":"' + ln + 'x_out"},'
                opn += 1
            x = xo.copy()
        if pos == T - 1:
            # ---- final norm and tied-embedding logits ----
            var hf = rms_norm(x, fnorm)
            var ssf = Float32(0)
            for i in range(H):
                ssf = add32(ssf, mul32(x[i], x[i]))
            var meanf = ssf / Float32(H)
            var rsf = Float32(1) / sqrt(meanf + EPS)
            add_tensor(tj, "final_x", x)
            add_tensor(tj, "final_ln_w", fnorm)
            add_tensor(tj, "final_norm", hf)
            oj += String('{"id":') + String(opn) + ',"layer":-1,"kind":"rmsnorm","name":"final norm","x":"final_x","w":"final_ln_w","y":"final_norm","sumsq":' + jf(ssf) + ',"mean":' + jf(meanf) + ',"rsqrt":' + jf(rsf) + ',"eps":' + jf(EPS) + "},"
            opn += 1
            var logits = matvec(data, emb, cells, hf)
            add_tensor(tj, "logits", logits)
            var tops = top_rows(logits, 3)
            var pjl = String("[")
            for ri in range(len(tops)):
                if ri > 0:
                    pjl += ","
                pjl += probe_json(data, emb, tops[ri])
            pjl += "]"
            oj += String('{"id":') + String(opn) + ',"layer":-1,"kind":"matmul","name":"logits (tied embedding head)","weight":"' + emb.name + '","N":' + String(V) + ',"K":' + String(H) + ',"x":"final_norm","y":"logits","probes":' + pjl + "},"
            opn += 1
            var top10 = top_rows(logits, 10)
            var tl = List[Float32]()
            var ti = List[Int]()
            for i in range(len(top10)):
                ti.append(top10[i])
                tl.append(logits[top10[i]])
            oj += String('{"id":') + String(opn) + ',"layer":-1,"kind":"output","name":"next-token prediction (greedy)","top_ids":' + arr_i(ti) + ',"top_logits":' + arr_f(tl) + "}"
            opn += 1
            print("SMOL_TRACE top1 token id", top10[0], "logit", logits[top10[0]])
    var js = String('{"meta":{"model":"SmolLM2-135M","pin":"glint_pin_v1","float_encoding":"every float is its binary32 bit pattern as a uint32","arithmetic":"f32, sequential k=0..K-1, product rounded before the add (no FMA), no activation rounding","hidden":') + String(H) + ',"layers":' + String(NL) + ',"heads":' + String(NH) + ',"kv_heads":' + String(NKV) + ',"head_dim":' + String(HD) + ',"inter":' + String(INTER) + ',"vocab":' + String(V) + '},"tokens":' + arr_i(ids) + ',"trace_pos":' + String(tpos) + ',"cells":' + arr_f(cells) + ',"tensors":{' + tj
    # drop the trailing comma of the tensor list
    var jb = js.as_bytes()
    var s2 = String(unsafe_from_utf8=jb[0 : len(jb) - 1]) + '},"ops":[' + oj + "]}"
    var of = open(out_path, "w")
    of.write(s2)
    print("SMOL_TRACE wrote", out_path, "ops", opn, "bytes", s2.byte_length())
