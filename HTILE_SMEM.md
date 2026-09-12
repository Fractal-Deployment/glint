# GLINT H-TILE — coalesced K loads + smem
Packed hole is **sequential in K**, not in N. `i = n*K + k`, byte `i>>1`.
| Map | GMEM | 32-thread warp |
|---|---|---|
| **Wrong (old)** | `threadIdx.x` = N | neighbors read `hole` `K/2` bytes apart → many 32B sectors |
| **Right** | `threadIdx` walks **bytes along K** for one row | 16B `cp.async` of `hole[(n*K+k0)/2 + off]` — one sector |
Smem per stage (16×16×64 tile): hole 512 B + plug 512 B + X 2 KiB. D=2 doubles it. Fits 3060.
`cp.async.wait_group 1` while the next K-tile is in flight (`wait_group 0` on the last tile).
Absmax still gathered from GMEM (`i/blocksize`). Next: one float per row when `K % 64 == 0`.
