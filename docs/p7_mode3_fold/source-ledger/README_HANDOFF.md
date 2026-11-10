# MIT Source Ledger — handoff packet

**Project:** GitHub Project #6 — P7 fold — glint + bf16-to-int8 milestones

**Scope:** Jadon-Fox/glint clean local checkout at ecf84013796452f8e10b09e41c07c81efa2abf16: UTF-8 text files under docs/p7_mode3_fold plus README, concept canonical and project sync script; separate PR #107 text evidence at eecbcfd37f15f022308461dc259a966373802e40.

## Custody, not interpretation

This packet preserves supplied source data, exact utterances, and mechanical indices for a separate interpreter (such as Astra). No source-status, alleged operator ruling, title, or lexical similarity is independently verified by this packet. No HANID/SSRL/MERCS identifier is minted.

Read `manifest.json` and `operator_turns.jsonl` first, then `sources.jsonl`. Recover exact statements from `verbatim_lines.jsonl` and `text_blobs/`. Compare versions and contradictions only after reading source context. `relationships_asserted.jsonl` contains **attributed claims**, not settled facts; `relationships_pending.jsonl` contains incomplete relationship records.

## Reconstruction

Concatenate `verbatim_line` by source and line number, UTF-8 encode, check against `sources.jsonl:content_sha256`; or use `scripts/verify_ledger.py` from the reusable skill.

## Counts

- source_entries: 49
- unique_text_blobs: 48
- unique_raw_blobs: 0
- verbatim_line_segments: 12689
- operator_turns: 1
- exact_quote_pointers: 0
- lexical_occurrences: 3420
- asserted_relation_records: 0
- open_relation_records: 0

## Known exclusions

[
  "No bf16-to-int8 repository ingestion; P7 references it, but this pass only covers glint",
  "No training_orchestrator ingestion or modification",
  "No binary CUDA/PT/safetensors/weights, GitHub issue bodies, private services, or Google Drive",
  "No model-generated semantic relation assertions; all unresolved"
]

## Boundaries

Only staged sources are covered; extracted text may omit graphics/equations/tables. Blob hashes apply to staged bytes, not necessarily original native objects. No connected source is modified by the offline builder.
