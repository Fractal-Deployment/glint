#!/usr/bin/env node
// select_questions.mjs — builds p7-s0-questions.list (P7 round-3 rules §0.5), no Python.
// Per category QC1..QC6: a seeded draw of 24 items from the named local pool(s), first 16 = MAP, next 8 = CONFIRM.
// Draw: counter-based SplitMix64. key(i) = mix64(seed_u64 XOR cat_u64 + i) for every row i of the category's pooled
// universe (picked files in sorted path order, rows in file order); candidates are walked in ascending key order and
// accepted if tokens are inside the pool's inclusive bounds, they share < 50% of 8-grams with any accepted item (§0.5
// dedup), and they share < 50% of 8-grams with GSM8K test problems 0-80 (§0.5 / R10). cat_u64 = first 8 bytes of
// sha256(category id). Tokenizer: Phi-4-mini-instruct @ cfbefacb, no chat template, add_special_tokens=false, with the
// same 23-item reference check as count_tokens.mjs (any mismatch = INCONCLUSIVE, exit 4).
// Known gap: the cal16k problem list is not on disk, so the "drop any cal16k problem" step is NOT applied (recorded).
// Usage: node select_questions.mjs --out-dir DIR [--seed 20261022] [--n 24] [--map 16]
import { createHash } from "node:crypto";
import { createReadStream, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { Tokenizer } from "@huggingface/tokenizers";
import { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects, parquetSchema } from "hyparquet";

const HERE = dirname(fileURLToPath(import.meta.url));
const PIN = { model: "microsoft/Phi-4-mini-instruct", revision: "cfbefacb99257ffa30c83adab238a50856ac3083",
  tokenizer_json_sha256: "382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea",
  tokenizer_config_json_sha256: "9c9b6bc0c94d95f69f826c41069a3e8b387ac3ced89601d201886e99240ac9db" };
const POOLS = "/home/workspace/p7-qc-pools/2026-10-30.v1";
const CATS = [
  { id: "QC1", name: "factual recall", pools: [{ name: "qc1-nq_open", dir: `${POOLS}/qc1-nq_open`, files: /^nq_open\/train-/, fields: ["question"], arrays: [], lo: 8, hi: 64 }] },
  { id: "QC2", name: "arithmetic/multi-step math", pools: [{ name: "qc2-gsm8k-train", dir: `${POOLS}/qc2-gsm8k-train`, files: /^train-00000-of-00001\.parquet$/, fields: ["question"], arrays: [], lo: 32, hi: 256 }] },
  { id: "QC3", name: "code", pools: [
    { name: "qc3-mbpp", dir: `${POOLS}/qc3-mbpp`, files: /^full\//, fields: ["text", "test_list"], arrays: ["test_list"], lo: 32, hi: 512 },
    { name: "qc3-openai_humaneval", dir: `${POOLS}/qc3-openai_humaneval`, files: /^openai_humaneval\/test-/, fields: ["prompt"], arrays: [], lo: 32, hi: 512 }] },
  { id: "QC4", name: "long-context", pools: [{ name: "qc4-fineweb_edu_c1", dir: "/home/workspace/lakes/fineweb_edu_sample10bt_20261017", files: /^000_00000\.parquet$/, fields: ["text"], arrays: [], lo: 1500, hi: 3000 }] },
  { id: "QC5", name: "rare-token/multilingual", pools: [{ name: "qc5-belebele", dir: `${POOLS}/qc5-belebele`, files: /^data\/[a-z]{3}_[A-Za-z]{4}\.jsonl$/, fields: ["flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"], arrays: [], lo: 64, hi: 512 }] },
  { id: "QC6", name: "instruction following", pools: [{ name: "qc6-IFEval", dir: `${POOLS}/qc6-IFEval`, files: /^ifeval_input_data\.jsonl$/, fields: ["prompt"], arrays: [], lo: 16, hi: 256 }] },
];
const GSM8K_TEST = "/home/workspace/lakes/metamathqa_20261019/gsm8k_test.jsonl";
const sha256 = (b) => createHash("sha256").update(b).digest("hex");
const sha256File = (f) => new Promise((res, rej) => { const h = createHash("sha256"); createReadStream(f).on("error", rej).on("data", (c) => h.update(c)).on("end", () => res(h.digest("hex"))); });
const die = (code, o) => { console.log(JSON.stringify(o, null, 1)); process.exit(code); };

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const OUT = opt("--out-dir"); const SEED = BigInt(opt("--seed", "20261022")); const N = Number(opt("--n", "24")); const NMAP = Number(opt("--map", "16"));
if (!OUT) die(2, { error: "--out-dir required" });
if (existsSync(OUT)) die(3, { error: "out dir exists (never overwrite)", out: OUT });

const tj = readFileSync(join(HERE, "tokenizer-cfbefacb/tokenizer.json")), tc = readFileSync(join(HERE, "tokenizer-cfbefacb/tokenizer_config.json"));
if (sha256(tj) !== PIN.tokenizer_json_sha256 || sha256(tc) !== PIN.tokenizer_config_json_sha256) die(4, { status: "INCONCLUSIVE", reason: "tokenizer files do not match pinned snapshot" });
const tok = new Tokenizer(JSON.parse(tj.toString("utf8")), JSON.parse(tc.toString("utf8")));
const enc = (t) => Array.from(tok.encode(t, { add_special_tokens: false }).ids, Number);
const fixture = JSON.parse(readFileSync(join(HERE, "reference_fixture.json"), "utf8"));
let matched = 0; for (const it of fixture.items) { const ids = enc(it.text); if (ids.length === it.ids.length && ids.every((x, j) => x === it.ids[j])) matched++; }
if (matched !== fixture.items.length) die(4, { status: "INCONCLUSIVE", reason: "reference check mismatch", matched, total: fixture.items.length });

// --- generator
const M64 = (1n << 64n) - 1n;
const mix64 = (z) => { z = (z + 0x9E3779B97F4A7C15n) & M64; z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & M64; z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & M64; return (z ^ (z >> 31n)) & M64; };
const catU64 = (id) => BigInt("0x" + sha256(Buffer.from(id)).slice(0, 16));

// --- text helpers
const words = (t) => t.toLowerCase().split(/\s+/).filter(Boolean);
const grams = (t) => { const w = words(t), s = new Set(); for (let i = 0; i + 8 <= w.length; i++) s.add(w.slice(i, i + 8).join(" ")); return s; };
const norm = (t) => words(t).join(" ");
const shared = (a, b) => { if (!a.size || !b.size) return 0; let c = 0; const [s, l] = a.size <= b.size ? [a, b] : [b, a]; for (const g of s) if (l.has(g)) c++; return c / s.size; };
const gsm = readFileSync(GSM8K_TEST, "utf8").split("\n").filter(Boolean).slice(0, 81).map((l) => JSON.parse(l).question);
const gsmGrams = gsm.map(grams), gsmNorm = new Set(gsm.map(norm));

// --- pool readers
function walk(dir) { const o = []; for (const e of readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) { if (e.name.startsWith(".")) continue; const p = join(dir, e.name); if (e.isDirectory()) o.push(...walk(p)); else if (/\.(parquet|jsonl)$/.test(e.name)) o.push(p); } return o; }
const jsonlCache = new Map(), pqMeta = new Map(), rgCache = new Map();
async function segmentsOf(pool, startAt = 0) {
  const segs = []; let start = startAt;
  for (const f of walk(pool.dir)) {
    const rel = relative(pool.dir, f); if (!pool.files.test(rel)) continue;
    let rows;
    if (f.endsWith(".jsonl")) { const lines = readFileSync(f, "utf8").split("\n").filter((l) => l.trim() !== ""); jsonlCache.set(f, lines); rows = lines.length; }
    else { const buf = await asyncBufferFromFile(f); const md = await parquetMetadataAsync(buf); pqMeta.set(f, { buf, md, top: new Set(parquetSchema(md).children.map((c) => c.element.name)) }); rows = Number(md.num_rows); }
    segs.push({ pool, file: f, rel, rows, start }); start += rows;
  }
  return segs;
}
async function rowAt(seg, local) {
  if (seg.file.endsWith(".jsonl")) return JSON.parse(jsonlCache.get(seg.file)[local]);
  const { buf, md, top } = pqMeta.get(seg.file); let rs = 0;
  for (let g = 0; g < md.row_groups.length; g++) {
    const n = Number(md.row_groups[g].num_rows);
    if (local < rs + n) {
      const k = seg.file + "#" + g;
      if (!rgCache.has(k)) { if (rgCache.size > 6) rgCache.delete(rgCache.keys().next().value); rgCache.set(k, await parquetReadObjects({ file: buf, metadata: md, columns: seg.pool.fields.filter((f) => top.has(f)), rowStart: rs, rowEnd: rs + n })); }
      return rgCache.get(k)[local - rs];
    }
    rs += n;
  }
  throw new Error("row index past file");
}
const textOf = (pool, row) => { const parts = pool.fields.map((k) => { const v = row[k]; if (v == null) return null; if (pool.arrays.includes(k)) return v.join("\n"); return String(v); }); return parts.some((p) => p === null) ? null : parts.join("\n"); };

const i32sha = (ids) => { const b = Buffer.alloc(ids.length * 4); ids.forEach((v, i) => b.writeInt32LE(v, i * 4)); return sha256(b); };

mkdirSync(OUT, { recursive: true });
const listLines = ["category\tsplit\tsource_id\ttoken_count\ttoken_ids_sha256"], texts = [], receipt = {};
for (const cat of CATS) {
  const segs = []; for (const p of cat.pools) segs.push(...(await segmentsOf(p, segs.reduce((a, s) => a + s.rows, 0))));
  const total = segs.reduce((a, s) => a + s.rows, 0);
  const base = SEED ^ catU64(cat.id);
  const keys = new BigUint64Array(total); for (let i = 0; i < total; i++) keys[i] = mix64((base + BigInt(i)) & M64);
  const order = Uint32Array.from({ length: total }, (_, i) => i); order.sort((a, b) => (keys[a] < keys[b] ? -1 : keys[a] > keys[b] ? 1 : 0));
  const acc = [], accGrams = [], accNorm = new Set(); const rej = { missing_field: 0, out_of_bounds: 0, duplicate: 0, gsm8k_test_0_80: 0 }; let walked = 0;
  for (const gi of order) {
    if (acc.length >= N) break; walked++;
    let si = segs.length - 1; while (segs[si].start > gi) si--; const seg = segs[si], local = gi - seg.start;
    const text = textOf(seg.pool, await rowAt(seg, local)); if (text === null) { rej.missing_field++; continue; }
    const ids = enc(text); if (ids.length < seg.pool.lo || ids.length > seg.pool.hi) { rej.out_of_bounds++; continue; }
    const g = grams(text), nt = norm(text);
    if (accNorm.has(nt) || accGrams.some((a) => shared(g, a) >= 0.5)) { rej.duplicate++; continue; }
    if (gsmNorm.has(nt) || gsmGrams.some((a) => shared(g, a) >= 0.5)) { rej.gsm8k_test_0_80++; continue; }
    acc.push({ id: `${seg.pool.name}:${seg.rel}:${local}`, tokens: ids.length, sha: i32sha(ids), text }); accGrams.push(g); accNorm.add(nt);
  }
  if (acc.length < N) die(4, { status: "INCONCLUSIVE", reason: `${cat.id}: only ${acc.length} of ${N} items found`, rej });
  acc.forEach((it, k) => { const split = k < NMAP ? "MAP" : "CONFIRM"; listLines.push([cat.id, split, it.id, it.tokens, it.sha].join("\t")); texts.push(JSON.stringify({ category: cat.id, split, source_id: it.id, text: it.text })); });
  const poolShas = []; for (const s of segs) poolShas.push({ file: `${s.pool.name}/${s.rel}`, rows: s.rows, sha256: await sha256File(s.file) });
  receipt[cat.id] = { name: cat.name, universe_rows: total, candidates_walked: walked, rejected: rej, pool_files: poolShas, bounds: cat.pools.map((p) => ({ pool: p.name, lo: p.lo, hi: p.hi })) };
  console.error(`${cat.id}: ${acc.length} items from ${walked} candidates`, JSON.stringify(rej));
}
const listText = listLines.join("\n") + "\n", textsText = texts.join("\n") + "\n";
writeFileSync(join(OUT, "p7-s0-questions.list"), listText, { flag: "wx" });
writeFileSync(join(OUT, "p7-s0-questions.texts.jsonl"), textsText, { flag: "wx" });
const rc = { tool: "select_questions.mjs", tool_sha256: sha256(readFileSync(fileURLToPath(import.meta.url))), node: process.version, rules: "p7-round3-fold-rules-DRAFT.md §0.5 (sha f1f75980 on the branch copy)",
  seed: String(SEED), generator: "SplitMix64 counter-based: key(i)=mix64((seed XOR first8(sha256(category id)))+i); ascending key order", per_category: N, map: NMAP, confirm: N - NMAP,
  tokenizer: { ...PIN, chat_template: "excluded", add_special_tokens: false }, reference_check: { matched, total: fixture.items.length, fixture_sha256: sha256(readFileSync(join(HERE, "reference_fixture.json"))) },
  gsm8k_test_exclusion: { file: GSM8K_TEST, problems: "0-80", rule: ">= 50% shared 8-grams or identical normalized text" },
  not_applied: ["cal16k problem exclusion (cal16k list not found on disk)"], list_sha256: sha256(Buffer.from(listText)), texts_sha256: sha256(Buffer.from(textsText)),
  categories: receipt, finished_at: new Date().toISOString() };
writeFileSync(join(OUT, "selection_receipt.json"), JSON.stringify(rc, null, 1) + "\n", { flag: "wx" });
console.log(JSON.stringify({ status: "DRAWN", list_sha256: rc.list_sha256, texts_sha256: rc.texts_sha256, items: listLines.length - 1, out: OUT }, null, 1));
