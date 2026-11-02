# Spec — Exact-match / term-presence search

**Status:** awaiting approval (no code written yet)
**Date:** 2026-11-01
**Owner:** Dave Galloway
**Repo:** `/Volumes/CrucialX9/a Downloads/bowen_rag` (git-tracked)

## Goal

Answer the question "is this exact term or phrase used anywhere in the corpus, and
where?" with a deterministic lookup — a *filter*, not a ranker. Today none of the three
search modes can do this, and the gap has produced a demonstrably wrong answer from the
Chat/Report path (see Background). This spec adds an exact-match search to the index layer,
auto-unions its hits into retrieval, exposes it as a CLI/GUI "find" feature, and adds a
prompt guard so the LLM can no longer claim a term is absent from the corpus.

## Background — findings that motivate this spec

**The corpus and the index.** 459 source files (`source_files/`, mix of `.txt` and `.pdf`,
several OCR'd from scans) are indexed into `rag-document-search/references/` by
`build_index.py` (`chunk_metadata.json`, `tfidf_matrix.npz`, `vectorizer.json`) plus
`build_embeddings.py` (`embed_matrix.npy`). Every chunk already carries citation locators:
`doc_name`, `author`, `date`, `chapter`, `chapter_label`, `chapter_title`, `container`,
`paragraph_start`, `paragraph_end`, `page`, `section_title`, plus `ocr` and
`ocr_confidence`. The serving path is `streamlit_app.py` → `IndexManager` (imported by both
GUIs and by `bowen_ask.py`), which holds `self.chunks` (list of chunk dicts).

**The three search modes are all rankers, none is a presence test.**

- `semantic_search` (`streamlit_app.py:351`) — TF-IDF cosine similarity.
- `embedding_search` (`streamlit_app.py:506`) — `SentenceTransformer("all-MiniLM-L6-v2")`
  cosine similarity.
- `keyword_search` (`streamlit_app.py:404`) — token-frequency counter; its quoted-phrase
  path (`re.findall(r'"([^"]+)"', query)` → `tl.count(p)`) is the *only* literal matching in
  the app, and it is unusable as a presence test: substring (no word boundary), collapses to
  one chunk per document (`doc_best`), truncated to `top_k`, ranked, and mixed with word hits.

`hybrid_search`/`bm25_search`/`top_docs_search`/`combined_search` are rankers too.

**The failure this spec fixes (concrete).** A user asked about "emotional programming".
The report answered *"No source excerpt uses the term 'emotional programming' or a direct
equivalent with that exact phrase."* That is wrong. The exact phrase occurs verbatim,
case-insensitively, in **at least 15 documents**, including:

```
Bowen Theory Secrets_Chapter09_Emotional_Programming.txt : 11 occurrences
FSJ 4.2 Kerr Bowen Theory and Evolutionary Theory.txt      : 20 occurrences
Bowen Theory Secrets_Chapter11_The_Multigenerational...    :  4
FSJ 9.2 Bowen Subjectivity, Homo Sapiens, and Science.txt  :  2
FSJ 14.2 Galloway How Emotions Are Made.txt                :  2
+ ~10 more documents with 1 occurrence each (Introduction, Ch4/6/10/14/15/16/19/21/23,
  Epilogue, FSJ 9.1)
```

Root cause is retrieval, not the model: semantic/hybrid ranking surfaced
semantically-adjacent chunks and buried the literal matches, so the model never saw the
phrase. The Report prompt then instructed it — correctly, per the no-fabrication rule — to
say "the sources do not address this" when its window lacked it, converting *"not in my
retrieved window"* into *"not in the corpus."* A separate `keyword` search for "programmed"
*did* surface the chapter (literal token hit) and returned the exact quote, which is how the
contradiction was exposed.

**Caveats that bound the design.**

1. **`grep` on `source_files/` is not the ground truth.** `grep` reads `.pdf` as binary and
   returns 0 even when the text is present; only `.txt` greps are meaningful. The real
   presence test must run against the *indexed text* (`chunk_metadata.json` / `self.chunks`),
   which holds extracted text for both `.txt` and `.pdf`.
2. **OCR under-reports exact matches.** A phrase genuinely "in use" in a scanned PDF can be
   mangled in the text layer (misread letters, dropped spaces) and fail a literal match.
   Chunks already carry `ocr: true` + `ocr_confidence`; a presence report must surface that
   instead of a clean zero.
3. **Chunk seams.** Chunks are paragraph-aligned (never split a paragraph), but
   `split_long()` splits an overlong paragraph at sentence boundaries, so a multi-word phrase
   *could* straddle two chunks. A presence check may optionally scan the raw source text to
   close this, but the paragraph-aligned index is sufficient for the acceptance tests here.

## Core design principles

- **A filter is not a ranker.** Exact-match results are ordered by hit count only. No cosine
  score, no authority boost, no `top_k` truncation, no one-chunk-per-document collapse. If a
  phrase occurs in 60 chunks across 20 documents, all 60 are available to the caller.
- **Presence is computed deterministically, never by the LLM.** The model must never be the
  thing deciding whether a term "is used at all." The code answers that; the model only
  reports what the code found (see E8).
- **Word-boundary by default; substring and regex opt-in.** A bare term matches the word, not
  substrings (searching "self" must not match "yourself"; searching "differentiation" must
  match "self-differentiation"). Substring mode is deliberate and documented as
  no-boundary. Regex mode is raw user regex, and a malformed pattern is an explicit error,
  never an empty result.
- **No fabrication.** Presence results carry `doc_name` + the existing locator fields only.
  If a locator is unknown (no chapter/page), it is omitted, never invented.
- **One implementation in `IndexManager`.** Both GUIs and `bowen_ask.py` already import it,
  so a single method covers all three surfaces. No schema change to the index is required —
  the locator fields already exist at ingest.

## Requirements

| ID | Requirement | Test |
|---|---|---|
| E1 | `IndexManager.exact_search(needle, mode="word", case=False)` returns **every** chunk whose text contains the needle: a list of chunk dicts (the existing `self.chunks` entries) each with an added `"hits"` count, sorted by `hits` descending. No `top_k`, no per-doc dedup, no relevance/boost. `case=False` matches case-insensitively (lowercases both sides); `case=True` is literal. | `test_exact_match.py::test_e1_*` |
| E2 | `mode="word"` matches on word boundaries via regex lookaround `(?<![a-z0-9]){escaped}(?![a-z0-9])` (escaped with `re.escape`). "differentiation" matches "differentiation" and "self-differentiation" (hyphen is a boundary) but NOT "differentiated". "self" does not match "yourself" or "selfish". | `test_e2_*` |
| E3 | `mode="substring"` is literal `str.count`, case-insensitive, no boundaries: "fu" matches "fusion", "functional". Documented as such in the docstring. | `test_e3_*` |
| E4 | `mode="regex"` treats `needle` as a raw regex (`re.findall`). An invalid pattern raises `re.error` (caller reports it), never silently returns `[]`. | `test_e4_*` |
| E5 | `IndexManager.exact_presence(needle, mode="word", case=False)` returns an aggregate dict: `{"found": bool, "total_hits": int, "doc_count": int, "docs": [{doc_name, hits, ocr, ocr_confidence, locator…}]}` — the one-line answer to "is it in use at all". `docs` sorted by `hits` desc. Built on the same `_scan` core as `exact_search` (no duplicated logic). | `test_e5_*` |
| E6 | **Auto-union into retrieval.** When any quoted phrase (`"…"`) or (for `keyword`/`hybrid`/report paths) any significant query term appears exactly in the corpus, the exact-match chunks are **unioned** into the retrieved chunk set ahead of ranked chunks, up to a fixed budget (default `EXACT_UNION_BUDGET = 10`, configurable), each tagged `mode: "exact"` so the prompt can label them "exact matches". Acceptance: for query "emotional programming", the retrieved set MUST contain the Chapter 09 chunk(s). This is what prevents the demonstrated failure. | `test_e6_*` (fixture) + `test_exact_match.py::test_e6_emotional_programming_*` (corpus smoke, mounted-gated) |
| E7 | `bowen_ask.py` gains `--find "phrase" [--mode word|substring|regex] [--case]` which runs `exact_presence` and prints a deterministic presence report (per-doc hit counts + locators, or `0 occurrences in the indexed text`), **no LLM call**. Exit 0 when found, 1 when not found, 2 on bad input, consistent with the existing exit-code scheme. | `test_exact_match.py::test_e7_*` (subprocess or direct `main()` with monkeypatched argv/stdin) |
| E8 | **Prompt guard.** The Report/Chat prompt gains a rule: the model may state absence only in the form "does not appear *in the excerpts provided*", never "does not appear in the corpus"; and the retrieval path appends an `EXACT-MATCH VERIFICATION` block (from E5/E6) for quoted phrases in the query, which is the only basis on which the model may assert presence/absence. | integration-only (prompt string + presence block present in `build_prompt`/report path); the retrieved-excerpt phrasing is pinned by an assertion on the prompt text |
| E9 | GUI wiring (lower priority, separate tier): a "Find exact phrase/term" input + mode selector in both `streamlit_app.py` and `bowen_rag_gui.py` that renders `exact_presence` (doc list + locators + hit counts) and, on demand, the chunk text. | integration-only (UI; checked by hand) |

## Implementation notes

- Add `_scan(needle, mode, case)` (returns `[(chunk_idx, hits)]`) and the two public methods
  `exact_search` / `exact_presence` to `IndexManager` in `streamlit_app.py`. Both GUIs and
  `bowen_ask.py` get it for free.
- `exact_search` must return the chunk dicts **by value** (`{**c, "hits": n}`), never mutate
  `self.chunks`.
- Reuse the existing `re`, no new dependencies. Locator rendering can reuse
  `citations.passage_locator` / `format_passage_reference` so presence reports print the
  same locator strings the Report does.
- E6 union happens in the retrieval dispatch (the `mode` branch in `bowen_ask.py` and the
  corresponding `IndexManager` search entry points), not in `exact_search` itself: compute
  exact hits, prepend up to `EXACT_UNION_BUDGET` of them (by hit count), then append the
  ranked remainder, de-duplicating by chunk `id`.
- The `EXACT-MATCH VERIFICATION` block (E8) is produced by calling `exact_presence` on
  quoted phrases from the query; if none are quoted, the block is omitted (do not fabricate
  a check).

## Tests to be passed

New file `test_exact_match.py` at the repo root, stdlib `unittest` (matches the house style
in `test_index_consistency.py`, `test_citations.py`), run with:

```
cd "/Volumes/CrucialX9/a Downloads/bowen_rag" && python3 -m unittest test_exact_match -v
```

Fixtures are built offline exactly as `test_index_consistency.py` does — call
`build_index.build(src, out, config_dir=src)` on a small temp corpus, then load an
`IndexManager` against `out` — so the suite runs **without** the corpus volume or network.
The corpus-specific smoke test (`test_e6_emotional_programming_*`) is gated: `skipUnless` the
`references/chunk_metadata.json` exists and contains a chunk whose `doc_name` includes
`Emotional_Programming`; when the volume is mounted it asserts the acceptance condition
(≥1 hit and Chapter 09 named), otherwise it skips cleanly.

Concrete assertions (the "basis for tests"):

- `test_e1_returns_all_no_truncation`: phrase present in > `top_k` chunks across multiple
  docs → `len(result) == actual total`, each chunk once, `hits` equal to the true count.
- `test_e1_case_insensitive`: needle `"EMOTIONAL programming"` matches lowercase text with
  `case=False`, and does NOT with `case=True`.
- `test_e2_word_boundary`: pins the "differentiation" / "self-differentiation" /
  "differentiated" and "self" / "yourself" cases exactly as in E2.
- `test_e3_substring`: `"fu"` matches a doc containing "fusion".
- `test_e4_regex`: valid regex finds; `re.error` raised on `"("`.
- `test_e5_presence_aggregate`: `total_hits`, `doc_count`, per-doc `hits` and `ocr` fields
  correct; `found` false when absent.
- `test_e6_auto_union`: fixture query whose exact term is not the top semantic match →
  exact chunk still first in the returned set, tagged `mode == "exact"`.
- `test_e7_find_cli`: `--find` on a fixture index prints the presence report and exits
  0/1/2 correctly, with no LLM/API import executed (assert the Anthropic client is never
  constructed).
- `test_e6_emotional_programming_*` (mounted-gated): the real corpus contains ≥1 exact
  "emotional programming" hit naming `Bowen Theory Secrets_Chapter09_Emotional_Programming`.

## Not covered by this change

- Fuzzy/typo-tolerant matching (Levenshtein, n-gram): out of scope; that is what the
  semantic/embedding modes are for. Exact search deliberately does the opposite.
- OCR-specific re-matching against a corrected text layer.
- Any index rebuild or schema change: none is needed; locators already exist.
- Search across raw source text (chunk-seam closure) — deferred; paragraph-aligned chunks
  make the seam risk low. If wanted later, add a `source=chunks|source_files` option.

## Known limits

- Exact match is the **floor**, not ceiling, on OCR'd documents (`ocr: true`): a zero there
  means "not found in the extracted text", not "not in the source". The presence report
  surfaces `ocr`/`ocr_confidence` so this is visible, not silently wrong.
- Word-boundary mode treats apostrophes/hyphens as boundaries by the `[a-z0-9]` class; a
  phrase containing internal punctuation must use `substring` or `regex` mode.
- E6 auto-union adds a small fixed number of chunks; a query with a very common term
  (e.g. "family") will union the top hit-count chunks for that term, which is acceptable
  because they are literal matches and are labelled as such.

## Blocking inputs

None. The corpus and index already exist on the mounted volume; the unit suite builds its
own fixtures offline.
