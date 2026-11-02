# Implementation plan — exact-match search and provable negatives

**Spec:** `docs/spec_exact_match_search_v2.md` (requirements E0–E13). Read it first; this plan only orders the work.
**Repo:** `/Volumes/CrucialX9/a Downloads/bowen_rag`. Test command for every tier: `cd "/Volumes/CrucialX9/a Downloads/bowen_rag" && python3 -m unittest test_exact_match -v`, followed by the full line from `CLAUDE.md` before each commit.
**Reporting rule for every tier:** paste the real `unittest -v` output (counts, skips, failures) and, for anything corpus-gated, say whether the index was present when it ran.

## Tier 0 — baseline (do first, ~15 minutes, no judgement needed)

1. Bring CrucialX9 to `c5ac598` (`git pull` from `main`, or `git fetch` from the Downloads clone and fast-forward). Confirm `git status` shows only the untracked notes and the index files are unchanged (`git diff --stat` empty for `rag-document-search/references/`).
2. Copy from the Downloads copy into CrucialX9: `REVIEW-bowen_rag-2026-11-01-literal-search.md`, `review_2026-11-01/`, `literal_search.py`, `test_literal_search.py` (the last two are inputs to Tier 1 and are deleted at its end). Commit the review and scripts.
3. Run the existing suite once and keep the output as the "before" record.

## Tier 1 — the deterministic core: E0–E5 (mechanical; cheap-model-safe)

Files: new `exact_match.py`, new `test_exact_match.py`; two-line additions to `streamlit_app.py:IndexManager.load` (after `self.chunks` is set, line 288) and `bowen_rag_gui.py:IndexManager.load` (the matching spot after its `json.load`), plus the two delegating methods in each class; one AST test in `test_index_consistency.py`.

Start from `literal_search.py` — it already passes the equivalent of E0–E3 and E5 (`normalize`, `LiteralIndex.search/presence`, `search_regex`). Map: `normalize` → keep, add the `case` parameter; `LiteralIndex` → `ExactIndex` with `scan()` factored out and the three modes from the spec (word via `[a-z0-9]` lookaround, substring via `str.count`, regex); `Hit` dataclass → the `{**chunk, "hits", "mode": "exact"}` dict the spec requires; `presence` → add `chunk_count`, per-doc `chunks`, `ocr`, `ocr_confidence`, `locators` (through the injected `locator_fn`), and the `loose_*` counts. Delete the prototype's whole-word default (the spec's default is word; the loose counts cover the lost-space case).

Acceptance: `test_e0_*`, `test_e1_*`, `test_e2_*`, `test_e3_*`, `test_e4_*`, `test_e5_*` pass on the offline fixture; `test_idx_both_apps_build_exact_index` passes; `python3 -c 'import streamlit_app as a; i=a.IndexManager(); i.load(); print(i.exact_presence("emotional programming")["doc_count"])'` prints `22` on the mounted index (and `93` for `total_hits` in substring mode). Load-time cost ≤ 1.5 s.

## Tier 2 — retrieval and prompt: E6a, E6, E7, E8 (careful; verify against the mounted corpus)

Files: `exact_match.py` (`phrases_for`, `union_exact`, `union_budget`, `verification_block`); `streamlit_app.py` `page_chat` retrieval (`:1395-1438`), `page_report` via `_gather_chunks` (`:816-841`) and the prompt (`:1605`), `SYSTEM_PROMPT` (`:144-167`), `_init_session` (`:752`); `bowen_ask.py` (`main` dispatch, `build_prompt`, new `--find`, `import anthropic` moved into the report branch); `bowen_rag_gui.py` Report/Chat retrieval and its `SYSTEM_PROMPT` (`:188`).

Order inside the tier: E6a and E6 as pure functions with fixture tests first; then wire E6 into the four dispatch points; then E8 (block + Rule 7 + session bump); then E7. Do not apply the union on the Search page.

Acceptance (all must be shown, not asserted): `test_e6_auto_union`, `test_e6_budget_rule`, `test_e8_*`, `test_e7_*` pass offline; with the index mounted, `test_e6_emotional_programming_chapter09` runs (not skipped) and passes; `echo "emotional programming" | python3 bowen_ask.py --find` exits 0 and lists `Bowen Theory Secrets_Chapter09_Emotional_Programming` first; `python3 bowen_ask.py --find "no such phrase xyz"` exits 1; `--find "(" --mode regex` exits 2 with the `re.error` text on stderr. One paste of the Chat user message for "emotional programming" at defaults showing the `EXACT-MATCH VERIFICATION` block with "22 documents" and "≥1 of the excerpts provided". Confirm by grep that the string "does not appear in the provided sources" no longer exists in either `SYSTEM_PROMPT`.

Risk to watch: `_gather_chunks` returns staged chunks unchanged when any are staged — apply the union there too, or the verification block will truthfully say "0 of the excerpts provided contain it" for a hand-staged report; either is defensible, but the block must describe what actually happened. The chat context window (`get_context_window(window=1)`) can add the phrase from neighbouring chunks; count presence over the text actually sent, not over the chunk list.

## Tier 3 — ranking defects: E10, E11 (careful; one decision from Dave)

E10 is mechanical once decided (both `keyword_search` copies, same diff twice; tests on the fixture: "programming?" matches, two-term query requires both, unquoted phrase hits rank first). E11 needs the decision in the spec's last section; implement one option in all four boost sites per file (`streamlit_app.py:357, 462, 517, 547`; `bowen_rag_gui.py:391, 414, 525` and its `top_docs_search`), update `authority_tiers.yml` comments and the four badge-help strings. Acceptance: `test_e10_*` offline; `test_e11_boost_never_removes_all_literal_hits` on the mounted index; re-run `review_2026-11-01/diagnose_boost_and_keyword.py` and paste the new BM25 top-12 table (the "before" is in the committed `.output.txt`).

## Tier 4 — regression harness and docs: E12, E13 (mechanical)

`tests/known_terms.yml` seeded from `review_2026-11-01/measure_literal.output.txt` (use the substring-mode `docs` column as the minimum, rounded down by 10 % so a corpus edit does not flap the test); `test_e12_*` corpus-gated; CI step. Docs per E13; `LEARNINGS.md` fix-log entry in the house format (issue → root cause Pn → what would have caught it → fix → rule).

## Tier 5 — GUI: E9 (separate, after everything above is merged)

Streamlit: "Exact phrase" mode on the Search page (`:1105-1114`, `:1168-1180`), caption with corpus counts, highlight in `_result_card`/`_show_section_dialog`; a "Find" expander on the Index page or a sixth page. Tkinter: a Find frame on the Search tab. Hand-checked; one AST test that the mode tuple exists.

## What to paste back after each tier

Tier number; files and functions changed (`file:function`); the full `python3 -m unittest test_exact_match -v` output; for Tier 2 the four acceptance commands with their stdout/stderr and exit codes; for Tier 3 the before/after BM25 table; and a one-line statement of whether the corpus-gated tests ran or skipped.

## Notes on the hand-off text

The draft prompt's constraints are kept except two: "IndexManager is the single implementation point" (the tkinter app has its own `IndexManager`, so the shared module is the single point and both classes delegate) and "no new dependencies (stdlib `re` only)" (unchanged in substance — `exact_match.py` uses `re`, `unicodedata`, `os`, `dataclasses`). The `test_e7_*` requirement that no Anthropic client is constructed can only be met if `bowen_ask.py` stops importing `anthropic` at module level; that move is part of E7. The question of running an independent-model review pass on the v1 spec is answered by this document: the corrections it found are listed under *What changed from v1* in the spec.
