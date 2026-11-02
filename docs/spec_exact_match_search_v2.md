# Spec v2 — Exact-match / term-presence search and provable negatives

**Status:** draft for approval — supersedes `docs/spec_exact_match_search.md` (v1, 2026-11-01)
**Date:** 2026-11-01
**Owner:** Dave Galloway
**Repo:** `/Volumes/CrucialX9/a Downloads/bowen_rag` (git-tracked; HEAD `7ab05aa` at the time of writing — see *Baseline* below)
**Companion:** `docs/implementation_plan_exact_match_2026-11-01.md` (tiers, order, files, acceptance checks)
**Evidence:** `REVIEW-bowen_rag-2026-11-01-literal-search.md` and `review_2026-11-01/*.output.txt` (currently in the Downloads copy of the repo); measurements below are from those runs against the shipped index.

## What changed from v1, and why

v1 is kept in substance: the design principles (a filter is not a ranker; presence is computed by code, never by the model; word-boundary default with substring and regex opt-in; no fabrication) and requirements E1–E9 stand. The measurements changed four things.

1. **The background was partly wrong about where the phrase is lost.** TF-IDF (Semantic and Top Docs) retrieves "emotional programming" well: 15 of 15 chunks contain it with the boost off, 12 of 15 on. What buries it is (a) the ×3 authority multiplier applied to raw scores — BM25 at the Chat default (12 chunks, boost on) returns 0 containing chunks and the best one ranks 25th; and (b) Keyword mode's OR-scoring of raw counts plus a punctuation bug ("programming?" never matches). Hybrid, the default mode, was measured after the model cache was made available: boost on, 0 of 12 containing chunks at the Chat default (0/15 Search, 2/30 Report); boost off, 12/12 (15/15, 27/30). Embedding alone: 0 with the boost at every k; 9–11 without. The Chat context actually sent (chunks plus ±1 neighbours) holds the phrase 0 times. So an exact-match union (E6) is necessary but not sufficient: without touching the boost and Keyword mode, every *unquoted* query keeps the defect. v2 adds E10 (Keyword) and E11 (boost) as requirements with their own tier.
2. **The v1 trigger for E6/E8 would not have fired on the demonstrated failure.** The user typed the term unquoted. v1 unions "any quoted phrase" and omits the verification block "if none are quoted". v2 defines a deterministic phrase-extraction rule (E6a) that covers short unquoted term queries and tells the user, in the UI and the prompt, when no check was run.
3. **Counts.** v1 says "at least 15 documents" from `grep` over `.txt`. In the indexed text: 93 occurrences, 66 chunks, 22 documents (substring, case-insensitive; 91 / 65 / 22 whole-word — one chunk reads "thatemotional programming", a lost space). The single implementation point is also not what v1 says: `bowen_rag_gui.py` defines its own `IndexManager` (line 260) and does not import `streamlit_app`; only `bowen_ask.py` does. E-requirements are therefore implemented once in a shared stdlib module and *delegated to* from both `IndexManager` classes (same pattern as `citations.py` and `index_fingerprint.py`).
4. **Text normalisation.** v1 scans raw lowercased chunk text. Chunk texts contain 84,388 newlines (PDF soft-wraps), so "emotional\nprogramming" would miss. v2 scans a whitespace-collapsed copy built once at load; the needle is normalised the same way.

## Baseline

Two working copies exist. `/Volumes/CrucialX9/a Downloads/bowen_rag` is at `7ab05aa` with untracked notes (`docs/spec_exact_match_search.md`, `HANDOVER.md`, …). `/Users/davemini2/Downloads/bowen_rag` is at `c5ac598`, two commits ahead (`68fd1a6`, `c5ac598`: Streamlit thinking-indicator CSS; the only tracked-file difference is `streamlit_app.py` after line 2035). Index files are byte-identical in both. The review, `literal_search.py` prototype and `test_literal_search.py` were written to the Downloads copy. **Before Tier 0 starts, bring CrucialX9 to `c5ac598`** (pull from GitHub `main` or fetch from the Downloads clone) so the two UI commits are not lost or re-conflicted; all line numbers below are valid at either commit because the diff is below the code in question.

## Goal

Answer "is this exact term or phrase used anywhere in the corpus, and where?" with a deterministic lookup, make sure the passages that contain it reach the model when any exist, and make every "absent" statement the user sees come from code that scanned the whole index, never from the model.

## Background (corrected)

The index is sound. `rag-document-search/references/chunk_metadata.json` holds 14,083 chunks from 401 documents (distinct `doc_name` values; v1's 459 counts files in `source_files/`, where a scanned PDF and its OCR `.txt` share one name); every chunk carries `text` plus the locator fields v1 lists. "emotional programming" is in 66 chunks and is one of the 8,000 TF-IDF bigram features in `vectorizer.json`.

The serving path is `IndexManager` — one in `streamlit_app.py:270` (used by the web app and `bowen_ask.py`) and a separate copy in `bowen_rag_gui.py:260`. All seven search methods are rankers; the only literal matching is the quoted-phrase branch of `keyword_search`, with the limits v1 states (one chunk per document, truncated to `top_k`, mixed with word hits). Measured on the shipped index, with the authority boost on, the number of returned chunks containing "emotional programming" is: BM25 k=12 **0/12** (off: 12/12); Keyword k=15 **0/15** (off: 4/15; for "What is emotional programming?" 0/15 either way because the `?` sticks to the last token); Semantic k=15 12/15 (off 15/15); Top Docs k=12 8/12 (off 11/12); Report top-docs k=30 for the question form 5/30 (off 20/30); **Hybrid k=12 0/12 (off 12/12)**; Embedding k=12 0/12 (off 9/12). Scaling the Tier-1 multiplier from 3.0 to 1.5 raises BM25 top-12 from 0 to 9 containing chunks; at 1.0 it is 12.

The report the user saw said *"No source excerpt uses the term 'emotional programming' …"*. That sentence is produced by `SYSTEM_PROMPT` Rule 7 (`streamlit_app.py:157-166`, `bowen_rag_gui.py:188`), which asks the model to judge verbatim presence in the provided excerpts. When retrieval hands it zero containing chunks the notice is true of the excerpts and false of the corpus. The model's own matching accuracy over 30 excerpts was not measured; the pipeline must not depend on it either way.

v1's three caveats stand (grep over PDFs is not ground truth; OCR under-reports; chunk seams), with one addition: PDF extraction sometimes drops a space between words, so a whole-word zero is not final — E5 reports the substring count alongside.

## Design principles

v1's four, plus:

- **Normalise once, identically, on both sides.** A scan copy of each chunk is built at load: NFKC, soft hyphens removed, `word-\nword` re-joined, curly quotes and en/em dashes mapped to ASCII, whitespace runs collapsed to one space; lower-cased only when `case=False`. The needle gets the same function. The copy lives in a parallel list (`self._scan_texts`); `self.chunks` is never mutated.
- **A zero is reported with its mode.** `exact_presence` in `word` mode also returns the substring count so a lost-space or plural variant shows as "0 whole-word, 1 loose" rather than a clean zero. OCR provenance is shown per document as in v1.
- **The user is told when no check was run.** If the phrase-extraction rule (E6a) yields nothing, the UI and the prompt say so and say how to get one (quote the term).
- **One implementation, two owners.** The scan, presence aggregate, union and prompt-block functions live in `exact_match.py` (stdlib only, no Streamlit/tkinter). Each `IndexManager` holds an `ExactIndex` and exposes `exact_search` / `exact_presence` as thin delegations, so both GUIs and `bowen_ask.py` get identical behaviour and `test_index_consistency.py` can pin that the two delegations exist.

## Requirements

Prefix `E` (no clash with `U`/`M`). Tests are in `test_exact_match.py` unless stated.

| ID | Requirement | Test |
|---|---|---|
| E0 | `exact_match.normalize(text, case=False)` as defined above; idempotent; `ExactIndex(chunks)` builds `_scan_texts` once (measured ≈1.0 s for 14,083 chunks; ≈17 MB). `IndexManager.load` constructs it after `self.chunks`; both `IndexManager` classes. | `test_e0_normalize_*`, `test_idx_both_apps_build_exact_index` (in `test_index_consistency.py`, by AST like the TF-IDF pin) |
| E1 | `IndexManager.exact_search(needle, mode="word", case=False)` → **every** matching chunk as `{**chunk, "hits": n, "mode": "exact"}` sorted by `hits` desc, then chunk `id`. No `top_k`, no per-doc collapse, no boost. By value; `self.chunks` untouched. | `test_e1_returns_all_no_truncation`, `test_e1_case_insensitive`, `test_e1_does_not_mutate_chunks` |
| E2 | `mode="word"`: `(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])` on the normalised text (`[A-Za-z0-9]` when `case=True`). Pins: "differentiation" matches "differentiation" and "self-differentiation", not "differentiated"; "self" does not match "yourself"/"selfish"; "emotional programming" matches across a newline and a soft-wrapped hyphen in the raw text. | `test_e2_word_boundary`, `test_e2_matches_across_newline` |
| E3 | `mode="substring"`: normalised `str.count`, no boundaries ("fu" matches "fusion"). Docstring says so. | `test_e3_substring` |
| E4 | `mode="regex"`: raw regex over the normalised text (`re.I` when `case=False`); `re.error` propagates; never a silent `[]`. | `test_e4_regex`, `test_e4_bad_pattern_raises` |
| E5 | `IndexManager.exact_presence(needle, mode="word", case=False)` → `{"found", "total_hits", "chunk_count", "doc_count", "docs": [{doc_name, hits, chunks, ocr, ocr_confidence, locators: [str]}], "loose_total_hits", "loose_doc_count"}` where the `loose_*` fields are the substring-mode counts (equal to the main counts when `mode != "word"`). `docs` sorted by `hits` desc. Locators via `citations.passage_locator` (omitted when empty). Same `_scan` core as E1. | `test_e5_presence_aggregate`, `test_e5_absent_is_found_false`, `test_e5_loose_counts_lost_space` |
| E6a | **Phrase extraction** `exact_match.phrases_for(query)` — deterministic: (1) every `"…"`/`“…”` quoted span; else (2) the whole query, stripped of surrounding punctuation, if it has ≤ 6 words; else (3) nothing. Returns `(phrases, reason)` where `reason ∈ {"quoted", "short-query", "none"}`. | `test_e6a_quoted`, `test_e6a_short_query`, `test_e6a_long_query_none` |
| E6 | **Auto-union.** `exact_match.union_exact(ranked, exact_hits, budget)`: prepend up to `budget` exact hits (by `hits` desc) not already in `ranked` (by `id`), then the ranked list minus duplicates, trimmed to `len(ranked)` so the user's k is unchanged. Exact hits already in `ranked` count toward the budget. `budget = min(EXACT_UNION_BUDGET, max(1, k // 2))`; `EXACT_UNION_BUDGET` default 5, overridable by env var and a UI control. Applied in every retrieval dispatch that feeds a model (Chat, Report, `bowen_ask.py`, GUI Report/Chat) for each phrase from E6a; **not** applied on the Search page (it shows what the ranker did). Acceptance on the real index: query "emotional programming" at Chat defaults (hybrid or bm25, k=12, boost on) → the returned set contains ≥1 chunk with `doc_name` containing `Bowen Theory Secrets_Chapter09_Emotional_Programming`. | `test_e6_auto_union` (fixture), `test_e6_budget_rule`, `test_e6_emotional_programming_chapter09` (corpus, `skipUnless` gated) |
| E7 | `bowen_ask.py --find "phrase" [--mode word|substring|regex] [--case] [--json]`: runs `exact_presence`, prints per-document hits with locators and OCR flags (or `0 occurrences in the indexed text (N loose)`), makes **no** LLM call, and never imports `anthropic` on that path (move the import into the report branch). Exit 0 found, 1 not found, 2 bad input/regex error (message on stderr). | `test_e7_find_cli_found_exit0`, `test_e7_find_cli_absent_exit1`, `test_e7_find_cli_bad_regex_exit2`, `test_e7_find_does_not_import_anthropic` |
| E8 | **Prompt guard.** (a) `exact_match.verification_block(presence_by_phrase, provided_chunks, reason)` renders an `EXACT-MATCH VERIFICATION` block: per phrase, corpus counts (hits/chunks/docs, loose counts) and how many of the provided excerpts contain it; or the one-line "no phrase was quoted and the question is longer than 6 words, so no exact-match check was run". (b) The block is placed at the top of the user message in Chat (`streamlit_app.py:1435`), the Report prompt (`:1605`), `bowen_ask.build_prompt`, and the GUI equivalents, and shown in the UI above the answer (`st.caption`; GUI label). (c) `SYSTEM_PROMPT` Rule 7 in both apps is replaced by: the model restates the verification block's finding, may say "does not appear in the excerpts provided" but never "in the corpus"/"in the sources"; the "emotional contact" vs "emotional connection" rationale stays. (d) Existing sessions carry `st.session_state.system_prompt`; bump it when the default changes (compare to the old constant on `_init_session`). | `test_e8_block_text`, `test_e8_prompt_contains_block` (assert on `build_prompt` output), `test_e8_rule7_wording` (assert the forbidden phrase "does not appear in the provided sources" is gone from both `SYSTEM_PROMPT` constants) |
| E9 | GUI: a "Find exact phrase" input + mode selector + case toggle in both apps rendering `exact_presence` (doc list, locators, hit counts, OCR flag, loose counts), expandable to chunk text with the phrase highlighted; and an "Exact phrase" entry in the Search-page mode list returning `exact_search` with the corpus counts in the caption even when the list is truncated. | hand-checked; `test_e9_search_mode_registered` (AST: the mode tuple exists in both apps) |
| E10 | **Keyword mode** (`streamlit_app.py:404-446`, `bowen_rag_gui.py:465-509`): tokenise with `re.findall(r"[\w'-]+", remainder)` so punctuation never sticks to a term; an unquoted multi-word query is first tried as a phrase (its hits ranked on top, labelled "phrase"), then word hits require **all** terms present (AND). Quoted-phrase matching moves onto the E0 normalised text. | `test_e10_punctuation_stripped`, `test_e10_all_terms_required`, `test_e10_unquoted_phrase_first` |
| E11 | **Authority boost.** Decision required (plan, Tier 3). Options measured: (a) cap Tier-1 at 1.5 (BM25 top-12: 9/12 here); (b) apply the boost after ranking as a rank bonus (RRF-style third list) so it reorders among comparable chunks; (c) apply it only within the top-4k raw candidates. Whatever is chosen is recorded in `authority_tiers.yml` and the badge help; `authority_boost()` keeps its signature. | `test_e11_boost_never_removes_all_literal_hits` (corpus-gated: for each term in `tests/known_terms.yml`, BM25 k=12 boost on returns ≥1 containing chunk) |
| E12 | **Known-terms regression** `tests/known_terms.yml`: ≈20 Bowen terms with minimum expected `doc_count` from the current index (e.g. `emotional programming` 22, `emotional contact` 72, `emotional cutoff` 74, `nodal events` 18, `emotional shock wave` 11). Corpus-gated test asserts `exact_presence(term).doc_count >= min` and that each page's default retrieval + E6 union hands the model ≥1 containing chunk. | `test_e12_known_terms_present`, `test_e12_known_terms_reach_the_model` |
| E13 | Docs: `USER_GUIDE.md` (search-mode table; "Terminology precision warning" rewritten around the verification block; Find feature), `CLAUDE.md` (search modes, tests line, `exact_match.py`), `LEARNINGS.md` fix-log entry (P7 boost, P2 punctuation, P6 model-judged presence). | review by hand |

## Interfaces (normative)

```python
# exact_match.py  (stdlib only)
def normalize(text: str, case: bool = False) -> str
class ExactIndex:
    def __init__(self, chunks: list[dict]): ...            # builds _scan_texts; never mutates chunks
    def scan(self, needle, mode="word", case=False) -> list[tuple[int, int]]   # (chunk_idx, hits)
    def search(self, needle, mode="word", case=False) -> list[dict]           # E1
    def presence(self, needle, mode="word", case=False, locator_fn=None) -> dict   # E5
def phrases_for(query: str) -> tuple[list[str], str]                          # E6a
def union_exact(ranked: list[dict], exact_hits: list[dict], budget: int) -> list[dict]   # E6
def union_budget(k: int, configured: int = EXACT_UNION_BUDGET) -> int
def verification_block(presence_by_phrase: dict, provided: list[dict], reason: str) -> str   # E8
def highlight(text: str, phrase: str) -> str                                  # E9
EXACT_UNION_BUDGET = int(os.environ.get("EXACT_UNION_BUDGET", 5))
```

`IndexManager` (both files): `self.exact = ExactIndex(self.chunks)` in `load()`; `exact_search(...)` and `exact_presence(...)` delegate, passing `citations.passage_locator` as `locator_fn`.

## Tests

`cd "/Volumes/CrucialX9/a Downloads/bowen_rag" && python3 -m unittest test_exact_match -v`, added to the unittest line in `CLAUDE.md` and to CI. Fixtures are built offline with `build_index.build(src, out, config_dir=src)` as in `test_index_consistency.py`, then an `IndexManager` is loaded against `out`; the fixture corpus must include: a phrase present in more chunks than `top_k`; "differentiation"/"self-differentiation"/"differentiated"; "self"/"yourself"; a chunk with the phrase split by `\n`; one by `-\n`; one lost-space variant ("thatemotional programming"); an `ocr: true` document (via a fixture `ocr_manifest.yml`). Corpus-gated tests (`test_e6_emotional_programming_chapter09`, E11, E12) use `skipUnless(REFS/chunk_metadata.json exists and names a chunk containing "Emotional_Programming")` and skip cleanly when the volume is unmounted. The `literal_search.py` prototype's 17 tests are folded into this file (same cases, new names) and the prototype files are deleted once `exact_match.py` lands.

## Not covered

Fuzzy matching; OCR re-matching; index rebuild or schema change; raw-source scanning for chunk seams (`source=chunks|source_files` later); checking the model's *quotations* against the chunks after the stream ends (natural follow-on: the same `ExactIndex`).

## Known limits

Exact match on OCR'd documents is a floor, shown as such. Word mode treats hyphen and apostrophe as boundaries; use `substring`/`regex` for internal punctuation. E6a's ≤6-word rule means a long prose question gets no union and an explicit "no check was run" line; quoting the term is the documented way to force one. The model's own behaviour under Rule 7 (whether it misjudges presence even when the phrase is in the excerpts) is still unobserved: `api.deepseek.com` is refused by the network policy in both sandboxes; the experiment script is in `review_2026-11-01/llm_presence_experiment.py`.

## Decisions needed before Tier 3

1. Boost option (E11 a/b/c) — my recommendation is (b), with (a) as the one-line interim.
2. Whether Kerr's *Bowen Theory Secrets* (26 documents, currently ×1.0, no pattern in `authority_tiers.yml`) should have a tier. Editorial; not assumed.
3. `EXACT_UNION_BUDGET` default: 5 (measured sufficient) versus v1's 10 (at Chat k=12 that leaves 2 ranked chunks).
