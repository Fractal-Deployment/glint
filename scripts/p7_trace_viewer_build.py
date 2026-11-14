#!/usr/bin/env python3
"""Embed a p7_trace_smol trace and the tokenizer vocabulary into one self-contained HTML viewer (build tooling only).
Usage (container python with `tokenizers`): p7_trace_viewer_build.py TRACE.json TOKENIZER.json TEMPLATE.html OUT.html"""
import json, sys
from tokenizers import Tokenizer

trace_p, tok_p, tpl_p, out_p = sys.argv[1:5]
tok = Tokenizer.from_file(tok_p)
n = tok.get_vocab_size()
vocab = {i: tok.decode([i]) for i in range(n)}
trace = open(trace_p).read()
json.loads(trace)  # must be valid JSON
tpl = open(tpl_p).read()
assert "/*TRACE_JSON*/" in tpl and "/*VOCAB_JSON*/" in tpl
vj = json.dumps(vocab, ensure_ascii=False).replace("</", "<\\/")
html = tpl.replace("/*TRACE_JSON*/", trace.replace("</", "<\\/")).replace("/*VOCAB_JSON*/", vj)
open(out_p, "w").write(html)
print("wrote", out_p, len(html), "bytes;", n, "vocab entries")
