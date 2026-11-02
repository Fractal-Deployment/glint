# mojo/p7_safetensors.mojo — pure-Mojo safetensors header reader (no Python). Parses the JSON header of a
# .safetensors file into tensor entries (name, dtype, shape, byte range) and exposes the file bytes. A header parser,
# not a general JSON parser: it handles the safetensors subset (string keys, "dtype", "shape", "data_offsets",
# "__metadata__" skipped).
# Test: mojo run mojo/p7_safetensors.mojo FILE.safetensors TENSOR_NAME   (prints the entry for TENSOR_NAME)
from std.sys import argv

struct STEntry(Copyable, Movable):
    var name: String
    var dtype: String
    var shape: List[Int]
    var begin: Int
    var end: Int

    def __init__(out self, name: String):
        self.name = name
        self.dtype = String("")
        self.shape = List[Int]()
        self.begin = 0
        self.end = 0

def skip_ws(b: Span[UInt8, _], mut i: Int):
    while i < len(b) and (b[i] == 32 or b[i] == 9 or b[i] == 10 or b[i] == 13):
        i += 1

def parse_string(b: Span[UInt8, _], mut i: Int) raises -> String:
    skip_ws(b, i)
    if b[i] != 34:
        raise Error("expected string at " + String(i))
    i += 1
    var start = i
    while b[i] != 34:
        if b[i] == 92:
            i += 1
        i += 1
    var s = String(unsafe_from_utf8=b[start:i])
    i += 1
    return s^

def parse_int(b: Span[UInt8, _], mut i: Int) raises -> Int:
    skip_ws(b, i)
    var v = 0
    var n = 0
    while i < len(b) and b[i] >= 48 and b[i] <= 57:
        v = v * 10 + Int(b[i]) - 48
        i += 1
        n += 1
    if n == 0:
        raise Error("expected integer at " + String(i))
    return v

def skip_value(b: Span[UInt8, _], mut i: Int) raises:
    skip_ws(b, i)
    if b[i] == 34:
        _ = parse_string(b, i)
        return
    var depth = 0
    while i < len(b):
        var c = b[i]
        if c == 34:
            _ = parse_string(b, i)
            if depth == 0:
                return
            continue
        if c == 123 or c == 91:
            depth += 1
        elif c == 125 or c == 93:
            if depth == 0:
                return
            depth -= 1
            if depth == 0:
                i += 1
                return
        elif c == 44 and depth == 0:
            return
        i += 1

def parse_header(b: Span[UInt8, _]) raises -> List[STEntry]:
    var out = List[STEntry]()
    var i = 0
    skip_ws(b, i)
    if b[i] != 123:
        raise Error("header does not start with {")
    i += 1
    while True:
        skip_ws(b, i)
        if b[i] == 125:
            break
        var key = parse_string(b, i)
        skip_ws(b, i)
        i += 1                                   # ':'
        if key == "__metadata__":
            skip_value(b, i)
        else:
            var e = STEntry(key)
            skip_ws(b, i)
            i += 1                               # '{'
            while True:
                skip_ws(b, i)
                if b[i] == 125:
                    i += 1
                    break
                var f = parse_string(b, i)
                skip_ws(b, i)
                i += 1                           # ':'
                if f == "dtype":
                    e.dtype = parse_string(b, i)
                elif f == "shape":
                    skip_ws(b, i)
                    i += 1                       # '['
                    while True:
                        skip_ws(b, i)
                        if b[i] == 93:
                            i += 1
                            break
                        e.shape.append(parse_int(b, i))
                        skip_ws(b, i)
                        if b[i] == 44:
                            i += 1
                elif f == "data_offsets":
                    skip_ws(b, i)
                    i += 1                       # '['
                    e.begin = parse_int(b, i)
                    skip_ws(b, i)
                    i += 1                       # ','
                    e.end = parse_int(b, i)
                    skip_ws(b, i)
                    i += 1                       # ']'
                else:
                    skip_value(b, i)
                skip_ws(b, i)
                if b[i] == 44:
                    i += 1
            out.append(e^)
        skip_ws(b, i)
        if b[i] == 44:
            i += 1
    return out^

def main() raises:
    var args = argv()
    var path = String(args[1])
    var want = String(args[2]) if len(args) > 2 else String("")
    var f = open(path, "r")
    var head = f.read_bytes(8)
    var hlen = 0
    for k in range(8):
        hlen |= Int(head[k]) << (8 * k)
    var hb = f.read_bytes(hlen)
    var entries = parse_header(Span(hb))
    print("tensors", len(entries), "header_bytes", hlen)
    for e in entries:
        if e.name == want:
            var sh = String("")
            for d in e.shape:
                sh += String(d) + " "
            print(e.name, e.dtype, "shape", sh, "begin", e.begin, "end", e.end, "data_start", 8 + hlen + e.begin)
