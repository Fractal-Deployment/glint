# M4-LOSSLESS on the Phi-4-mini glint pin (2026-11-03)

`tools/glint_huff.c`: canonical Huffman per tensor and per stream (hole nibbles and plug nibbles separately), 16 code-length bytes
plus a flag byte per stream, falling back to raw when a stream would not shrink. The decode is run inside the tool and compared
symbol by symbol with the original; any mismatch makes the tool exit 1.

| | value |
|---|---|
| streams (129 linears x hole/plug) | 258 |
| round-trip failures | 0 (bit-identical) |
| raw index bytes | 3,835,822,080 |
| compressed index bytes | 3,795,757,061 |
| saved | 40,065,019 bytes = 1.044% |

Rules done-check for M4-LOSSLESS: "bit-identical to M3 + smaller bytes". Both hold for the index streams of this pin. The saving is small because
the nibble histograms are close to uniform: the order-0 joint entropy bound from `M4_LOSSLESS_ENTROPY_phi4mini.json` is 7.838 of 8 bits (2.0%), and a
per-stream Huffman code with integer code lengths recovers about half of that. Absmax scales (F32, 1 per 64 weights) are not compressed. This is the
pin's own index bytes; it says nothing about decode speed on the GPU (the decoder here is a C reference, not a kernel).
Limit: "bit-identical to M3" is shown as a lossless decode of the stored indices, not as bit-identical logits through a forward pass.
