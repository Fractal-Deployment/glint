# GLINT Project 7: single-linear Mojo/MAX loader + pin-sourced codebook GEMV.
# Initial Phase B fixture for #116, not the full GLINT decode path.
# Args: SHARD.safetensors; tests layer0 self_attn.o_proj (N=K=3072).
from std.math import ceildiv
from std.sys import argv
from std.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from p7_safetensors import parse_header, STEntry
from p7_forward import read_range

comptime N = 3072
comptime K = 3072
comptime BLOCK = 64
comptime TABLE_ELEMS = 256

# GLINT full decode: (cells[h][p] * absmax[(row*K+k)//64]) * BF16(x).
# One GPU work item per row; deliberately sequential K reductions so this
# is a legible reference kernel, not an optimized production GLINT decoder.
def glint_gemv_pinned_table[
    ho: ImmOrigin, po: ImmOrigin, so: ImmOrigin, to: ImmOrigin, xo: ImmOrigin, yo: MutOrigin
](
    hole: ImmPointer[UInt8, ho],
    plug: ImmPointer[UInt8, po],
    scales: ImmPointer[Float32, so],
    cells: ImmPointer[Float32, to],
    x: ImmPointer[BFloat16, xo],
    y: MutPointer[Float32, yo],
    n: Int32, kdim: Int32,
):
    var row = global_idx.x
    if row >= Int(n):
        return
    var kmax = Int(kdim)
    var acc = Float32(0)
    for k in range(kmax):
        var idx = row * kmax + k
        var a = hole[unsafe_offset=idx >> 1]
        var b = plug[unsafe_offset=idx >> 1]
        var shift = UInt8(4 * (idx & 1))
        var h = Int((a >> shift) & UInt8(15))
        var p = Int((b >> shift) & UInt8(15))
        var w = cells[unsafe_offset=h * 16 + p] * scales[unsafe_offset=idx // BLOCK]
        acc += w * x[unsafe_offset=k].cast[DType.float32]()
    y[unsafe_offset=row] = acc

def main() raises:
    var args=argv()
    if len(args)<2:
        raise Error("usage: p7_glint_linear PIN_SHARD.safetensors")
    var path=String(args[1])
    def read_header(path: String) raises -> Tuple[List[STEntry], Int]:
        var f=open(path,"r")
        var h8=f.read_bytes(8)
        if len(h8)!=8:
            raise Error("invalid safetensors header")
        var hl=0
        for j in range(8):
            hl |= Int(h8[j]) << (8*j)
        if hl<=0 or hl>16*1024*1024:
            raise Error("unreasonably large header")
        var header=f.read_bytes(hl)
        return (parse_header(Span(header)),8+hl)

    var hd=read_header(path)

    def entry(hd: Tuple[List[STEntry], Int], name: String) raises -> STEntry:
        for i in range(len(hd[0])):
            if hd[0][i].name==name:
                return hd[0][i].copy()
        raise Error("missing tensor "+name)

    var wname=String("model.layers.0.self_attn.o_proj.weight")
    var hc=entry(hd,"_glint_cells")
    var hw=entry(hd,wname)
    var hp=entry(hd,wname+".glint_plug")
    var hs=entry(hd,wname+".absmax")
    if hc.dtype!="F32" or len(hc.shape)!=2 or hc.shape[0]!=16 or hc.shape[1]!=16 or hc.end-hc.begin!=1024:
        raise Error("_glint_cells must be [16,16] F32")
    if hw.dtype!="U8" or hp.dtype!="U8" or hw.end-hw.begin!=N*K//2 or hp.end-hp.begin!=N*K//2:
        raise Error("hole/plug nibble length mismatch")
    if hs.dtype!="F32" or hs.end-hs.begin!=N*K//BLOCK*4:
        raise Error("absmax block-64 flattened size mismatch")

    var ctx=DeviceContext()
    def load_u8(ctx: DeviceContext, path: String, hd: Tuple[List[STEntry], Int], name: String, n: Int) raises -> type_of(ctx.enqueue_create_buffer[DType.uint8](1)):
        var e=entry(hd,name)
        if e.end-e.begin!=n:
            raise Error("wrong tensor length "+name)
        var data=read_range(path,hd[1]+e.begin,n)
        var hb=ctx.enqueue_create_host_buffer[DType.uint8](n)
        for i in range(n):
            hb[i]=data[i]
        var dev=ctx.enqueue_create_buffer[DType.uint8](n)
        ctx.enqueue_copy(dst_buf=dev,src_buf=hb)
        ctx.synchronize()
        return dev^

    var hole_u8=load_u8(ctx,path,hd,wname,N*K//2)
    var plug_u8=load_u8(ctx,path,hd,wname+".glint_plug",N*K//2)
    var scale_u8=load_u8(ctx,path,hd,wname+".absmax",N*K//BLOCK*4)
    var cells_u8=load_u8(ctx,path,hd,"_glint_cells",1024)
    var scales=DeviceBuffer[DType.float32](ctx,scale_u8.unsafe_ptr().unsafe_bitcast[Float32](),N*K//BLOCK,owning=False)
    var cells=DeviceBuffer[DType.float32](ctx,cells_u8.unsafe_ptr().unsafe_bitcast[Float32](),TABLE_ELEMS,owning=False)
    var xh=ctx.enqueue_create_host_buffer[DType.bfloat16](K)
    for k in range(K):
        xh[k]=(Float32((k % 17)-8)/Float32(64)).cast[DType.bfloat16]()
    var x=ctx.enqueue_create_buffer[DType.bfloat16](K)
    var y=ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_buf=x,src_buf=xh)
    ctx.synchronize()
    var kernel=ctx.compile_function[glint_gemv_pinned_table[ImmUntrackedOrigin,ImmUntrackedOrigin,ImmUntrackedOrigin,ImmUntrackedOrigin,ImmUntrackedOrigin,MutUntrackedOrigin]]()
    ctx.enqueue_function(kernel,hole_u8,plug_u8,scales,cells,x,y,Int32(N),Int32(K),grid_dim=ceildiv(N,128),block_dim=128)
    ctx.synchronize()
    var yh=ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_buf=yh,src_buf=y)
    ctx.synchronize()
    for i in range(N):
        print("MOJO_GLINT_ROW",i,Float64(yh[i]))
    print("MOJO_GLINT_LINEAR_DONE",N,K)
    _=hole_u8.unsafe_ptr()
    _=plug_u8.unsafe_ptr()
    _=scale_u8.unsafe_ptr()
    _=cells_u8.unsafe_ptr()
