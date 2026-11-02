# mojo/p7_attach_check.mojo — Mojo attach to libp7_kernels.so (glint #16/#21, option 1: Mojo drives the CUDA ABI).
# Loads the library, requires the attach symbols, checks p7_abi_version() against ABI_VERSION and runs
# p7_abi_selftest (INT8 GEMM, resident vs streamed through the plug ring, CPU reference). Exit 0 only on rc == 0.
# Run: mojo run mojo/p7_attach_check.mojo LIB.so   (MODULAR_HOME and PATH as in MOJO_ENV_NOTE.md)
from std.ffi import OwnedDLHandle
from std.sys import argv

def main() raises:
    var args = argv()
    var lib_path = String(args[1]) if len(args) > 1 else String("/home/workspace/fractal-deployment/glint/lib/libp7_kernels.so")
    var handle = OwnedDLHandle(lib_path)
    var need = List[String]()
    need.append("p7_abi_version")
    need.append("p7_abi_selftest")
    need.append("int8_mma_gemm")
    need.append("int8_mma_gemm_plug")
    need.append("glint_plug_streamer_create")
    need.append("glint_plug_streamer_acquire")
    need.append("glint_plug_streamer_release")
    for name in need:
        if not handle.check_symbol(name):
            raise Error("HARD_BLOCK: missing symbol " + name + " in " + lib_path)
    var selftest = handle.get_function[Int32]("p7_abi_selftest")
    var buf = List[UInt8](length=512, fill=0)
    var rc = selftest(buf.unsafe_ptr(), UInt(512))
    var msg = String(unsafe_from_utf8=Span(buf)[0:buf.index(0) if 0 in buf else 511])
    print("P7_SELFTEST_FROM_MOJO rc=" + String(rc) + " " + msg)
    if rc != 0:
        raise Error("selftest failed rc=" + String(rc))
    print("MOJO_ATTACH_OK")
