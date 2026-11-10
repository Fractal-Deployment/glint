# Project #6 / P7 fold — MIT Source Ledger custody packet (2026-11-10)

This is a mechanically verified **source-custody packet**, not a validated P7 theory, a semantic relationship graph, or a production MIT/SSRL/HANID service.

- Download/open `project6-p7-2026-11-10-ledger.zip` and read `README_HANDOFF.md`.
- To verify offline using the installed `mit-source-ledger` skill, extract the ZIP and run `python3 skills/mit-source-ledger/scripts/verify_ledger.py PATH_TO_EXTRACTED_LEDGER` from the skill's plugin root.
- The included `receipt.json` reports 49 sources, 48 unique text blobs, 12,689 verbatim lines, and status PASS with no integrity errors.
- Scope: 43 UTF-8 documents/metadata artifacts from the clean `glint` checkout at `ecf84013796452f8e10b09e41c07c81efa2abf16`, plus six separately versioned PR #107 documents at `eecbcfd37f15f022308461dc259a966373802e40`.
- This packet **does not** cover all P7 research, source code, binaries, GitHub board items, external repositories (including `bf16-to-int8`), or private sources.
- `SRC-*`, `SEG-*`, and other packet identifiers are transport-local, **not** real HANIDs, SSRLs, or authoritative references.
- No relationships were asserted or adjudicated; lexical matches cannot establish equivalence, verification, or authorization.
- The evidence is captured from two separate branch contexts without merging their claims. Any interpretation must cite exact source/line evidence, resolve conflicting versions explicitly, and label unsupported relationships OPEN.

No experimental code or training orchestration interfaces were modified.
