# Review: bowen_rag (whole repo) — 2026-10-01

## Verdict
The citation post-processor is well designed, but the data under it is wrong in places. Twelve records in `sources.yml` credit Murray Bowen with other authors' work, and about 90% of records have the year "n.d.", so most in-text citations come out identical. The index build silently drops Section 1 of files that begin with a section heading. The web app and the desktop app are two diverging copies: the report prompt, the tier defaults and the YAML loading already differ. The packaged desktop app does not ship its YAML config. There is no CI test workflow, and nothing tests search ranking or the report flow where the project's worst past bug occurred. Fix first: the attribution data, the Section-1 drop, and the wrong page locators.

## Coverage
| Area | G-CORRECT | G-SEC | G-STRUCT | G-TEST |
|---|---|---|---|---|
| streamlit_app.py (deployed web app) | covered | covered | partial | partial |
| bowen_rag_gui.py (desktop) | covered | partial (key handling, grep sweeps) | partial (lines 1–830, 1965–2205) | partial (grep only) |
| citations.py, seed_sources.py, YAML config | partial (sources.yml checked by grep; 2851 lines not read in full) | partial (grep only) | not reviewed (citations internals, seed_sources) | partial |
| Index pipeline (build_index, process_transcripts, split_fsj_issues, semantic_search) | covered (FSJ TOC parsers not run against real PDFs) | partial (grep only) | partial | partial |
| CI / build.yml / requirements | n/a | covered | covered | covered |

Read: about 12 code files, roughly 7,500 LOC of Python, plus config. Nothing was executed. The test worker had no shell, so **the test suite was not run**. The challenge worker also ran nothing.
Not reviewed: GUI `App` class body (about lines 830–1965, 2205–2715) for duplication (STRUCT); `seed_sources.py` and `citations.py` internals for STRUCT; git history for committed secrets (`git log --all -- .env` not run); FSJ parsers against real PDFs; live LLM provider behaviour.
Excluded: `source_files/`, `references/*.json|npy|npz`, `outputs/`, `sources.seeded.yml`, docs, `__pycache__`.

## Findings

### Blockers
**B1. DATA · `sources.yml` (lines 966, 1128, 1308, 1578, 1695, 1704, 1785, 1911, 1920, 2145, 2405); cause `author_map.yml:103`**
- WHAT: Eleven records for other authors (Jones, Riordan, Harrison, Ferrera, McKnight, Millikin, Holt, Caskie ×2, Thompson, Lerner) list "Murray Bowen" as author. 51 records in total carry a Bowen author.
- WHY: The bare `pattern: "Bowen"` catch-all in `author_map.yml` matches any filename containing "Bowen" and the seeder copied it. Reports print "(Bowen, n.d.)" for a Jones article, and the author filter is wrong too. This is fabricated attribution in a source-fidelity tool.
- FIX: Correct the records to the author named in the filename, or set `authors: []`. Stop the seeder from using the bare "Bowen" catch-all.
- confidence high · **verified** (challenge pass rated it major; I keep it blocker because it is live in shipped data and violates "never fabricate"). Found by: citations-config.

**B2. CORRECT · `rag-document-search/scripts/build_index.py:117`**
- WHAT: The splitter regex needs a newline before `## Section`, so a file that starts with its first heading loses Section 1. Nothing is logged.
- WHY: Confirmed in the shipped index: "Bowen Basic Series Tape 7 … Emotional Cutoff" and "2011 04 Making a Difference … Kerr" start with `## Section 1` at byte 0, and their first sections are absent from `chunk_metadata.json`. Primary Bowen content is missing from search.
- FIX: `re.split(r'(?:^|\n)(## Section \d+ – [^\n]+)\n', content)`, index or log any preamble, then rebuild TF-IDF and the embeddings.
- confidence high · **verified** (challenge rated it major; same reasoning for blocker). Found by: build-pipeline.

### Major
**M1. CORRECT · `streamlit_app.py:1478-1483`, `bowen_rag_gui.py:2041-2064`** (corroborated by two lenses/workers)
- WHAT: The `(p. N)` offered to the model is computed from the retrieved chunks, but the text sent also includes ±window neighbour chunks that may be on other pages. A quote from a neighbour gets a confidently wrong page.
- FIX: Build the page set from every chunk id actually included in the prompt and offer a locator only when that set has one page.
- confidence high · **verified**.

**M2. CORRECT · `citations.py:384-429`**
- WHAT: Works by the same first author with the same year, including "n.d.", get identical in-text citations and no a/b suffix. 297 of 330 records are "n.d.", so about 90% of the corpus collides (22 FTCP chapters all become "(Bowen, n.d.)").
- FIX: Detect same-author/same-year groups among cited works and add suffixes in both the in-text citation and the reference entry.
- confidence high · **verified**.

**M3. ARCH · `.github/workflows/build.yml:39-41`, `bowen_rag_gui.py:94-116, 119, 142, 158`** (corroborated by structure, GUI correctness and challenge)
- WHAT: The PyInstaller bundle does not include `authority_tiers.yml`, `author_map.yml` or `sources.yml`, and the frozen app reads them from `~/Documents/BowenRAG` where nothing puts them. The in-code default tiers also lack `("FSJ ", 1.3)` (they still have `"Copy of "`).
- WHY: In the packaged app every author is "Unknown", there is no bibliography data, and FSJ articles get 1.0x instead of 1.3x. Silent, because a missing file returns `[]`.
- FIX: Add `--add-data` for the three YAML files and load from the bundle when absent in `BASE_DIR`. Better, move loading into one shared module with no code-side defaults.
- confidence high · **verified**.

**M4. ARCH · `streamlit_app.py:1502-1543` vs `bowen_rag_gui.py:2074-2121`**
- WHAT: The report prompt was copy-pasted and has drifted. The deployed web app lacks "Do not paraphrase without attribution", "if sources disagree, quote both", and "use the numbers in the source headers, not document names".
- WHY: The same topic gets differently constrained reports depending on the front-end. The Streamlit system prompt is also user-editable.
- FIX: One shared `build_report_prompt` module.
- confidence high · **verified**.

**M5. PERF · `streamlit_app.py:289`, `bowen_rag_gui.py:281`**
- WHAT: `load_npz(...).toarray()` makes a dense float64 TF-IDF matrix: 11,252 × 8,000 × 8 bytes, about 720 MB resident.
- WHY: On Railway this sits next to PyTorch, BM25 tokens and the embedding matrix, and grows with the corpus. The rows are already normalised, so a sparse matrix works directly.
- FIX: Keep CSR and use sparse `cosine_similarity` or `matrix @ qvec.T`.
- confidence high · **verified**.

**M6. ERROR · `streamlit_app.py:553-626`, 1557-1584; same gap in `bowen_rag_gui.py`**
- WHAT: `stop_reason` / `finish_reason` is never checked anywhere in the repo, so a report cut off at `max_tokens` gets a References section and is shown as complete.
- FIX: Read the final stop reason and show a visible warning on a length stop.
- confidence high · **verified**.

**M7. CORRECT · `build_index.py:267-290`; loaders `streamlit_app.py:458`, `bowen_rag_gui.py:363`** (corroborated by pipeline, tests, GUI)
- WHAT: `embed_matrix.npy` staleness is detected only by row count, at search time (CLAUDE.md says "startup error"). A rebuild with the same chunk count but different text silently pairs old embeddings with the wrong chunks. The build is also not atomic, and `vectorizer.json` records `max_features: 500` (real value 8000).
- FIX: Store a content-hash fingerprint with the embeddings and check it on load. Write index files via temp + rename.
- confidence high · **verified**.

**M8. CORRECT · `streamlit_app.py:1172-1179, 1306-1309`**
- WHAT: Interview, Coach and Quiz modes seed history with an assistant message, so the first API call starts with role `assistant`. Nothing strips it.
- WHY: Anthropic's Messages API is believed to require a leading `user` turn; this is **not verified** for Claude or DeepSeek. If so, these modes fail on every first turn.
- FIX: Drop the leading assistant turn when sending (still display it), and test live.
- confidence medium · verified that nothing strips it; provider behaviour untested.

**M9. CORRECT · `streamlit_app.py:1243-1248, 1343-1348`**
- WHAT: Chat "View ↗" opens the first chunk of the document, not the retrieved chunk. `sources` entries hold only `doc` and `excerpt`.
- FIX: Store the chunk id in each source and open that chunk.
- confidence high · **verified**.

**M10. ERROR · Chat history, `bowen_rag_gui.py:2683-2684`, `streamlit_app.py:1351-1353`**
- WHAT: An empty assistant reply (for example DeepSeek spending its whole budget on thinking) is appended to history. Later turns send an empty assistant message, which the APIs likely reject, until the user clears chat.
- FIX: Skip the append and show an error on empty output.
- confidence high that no guard exists · **verified**.

**M11. CORRECT · `bowen_rag_gui.py:791, 812, 1784-1786`**
- WHAT: Non-streaming DeepSeek calls use `max_tokens=32000`. Installed anthropic 0.78.0 raises "Streaming is required" above 600 s expected time (3600 × 32000 / 128000 = 900), so GUI "Test Connection" always fails for DeepSeek. Streamlit's test streams and is unaffected.
- FIX: Use a small `max_tokens` for the probe, or stream internally.
- confidence high · **verified** (challenge: minor to major).

**M12. CONCUR · `bowen_rag_gui.py:269-314`**
- WHAT: `IndexManager.load` replaces `chunks`, `matrix` and the rest in place on a background thread while `loaded` stays True, so a search during reload can pair new chunks with the old matrix.
- FIX: Build into locals and swap in one step.
- confidence high · **verified** (desktop only, a window of seconds).

**M13. CONCUR · `bowen_rag_gui.py:1486-1509, 1528-1607, 2564-2691`**
- WHAT: No in-flight guard on Generate Report, Chat Send, Rebuild, Import or Build Embeddings, and several patch global `builtins.print`. A double-click can leave `print` patched. This breaks the global rule that buttons launching background work are disabled immediately.
- FIX: Per-operation busy flag, set before the thread starts; use `contextlib.redirect_stdout` under a lock.
- confidence high · not independently verified.

**M14. SEC · `streamlit_app.py:684-701`**
- WHAT: The only access control is one optional shared password: no rate limit, non-constant-time compare, and when `APP_PASSWORD` is unset the app is fully open and server API keys load into every session. A report call can use 32,000 output tokens.
- FIX: Fail closed on Railway without `APP_PASSWORD`, use `hmac.compare_digest`, add attempt limiting and a call cap.
- confidence high · **verified**. Becomes a blocker if Railway runs without `APP_PASSWORD` (question below).

**M15. SEC · `streamlit_app.py:1757-1766, 1970-1971`**
- WHAT: Settings and the sources editor are open to every authenticated user, and Save writes `sources.yml` on the shared filesystem (non-atomic, last writer wins).
- FIX: Separate admin gate; on hosted deployments allow Download only; write via temp + `os.replace` and merge only the edited record.
- confidence medium. Found by: streamlit security, streamlit correctness.

**M16. SEC · `streamlit_app.py:1302-1323, 1900-1915`**
- WHAT: Personal family disclosures typed in Interview and Coach modes go to a third-party provider (DeepSeek by default) with no notice.
- FIX: Show a notice naming the provider, or require opt-in.
- confidence medium (design/privacy judgement).

**M17. DEPS · `requirements.txt:1-14`, `build.yml`** (corroborated by both security workers and the challenge pass)
- WHAT: All 14 dependencies and pyinstaller are unpinned, with no lockfile. Railway deploys on push to main and the release `.app` builds from whatever is newest.
- FIX: Pin exact versions (or a hashed lock). Move build-only PyPDF2 out of the server requirements (it is the deprecated package; `pypdf` replaces it).
- confidence high · **verified**.

**M18. TEST · `.github/workflows/` (only `build.yml`, tag/dispatch triggered)**
- WHAT: There is a test suite and no `tests.yml`, and pushes to main auto-deploy to Railway with no tests run. Global rule violated.
- FIX: Add `tests.yml`: install `requirements.txt` + pytest, run `pytest` on push and PR.
- confidence high · **verified**.

**M19. TEST · whole repo**
- WHAT: No tests for search ranking (`authority_boost`, RRF in `hybrid_search`, `use_boost` off), the stale-embedding check, `_gather_chunks` staged-replaces-retrieval, the DeepSeek text-block selection, the empty-output fail-loud branch, or the `max_tokens` behaviour. The report flow is where the past references-only bug lived, and the post-stream logic is inline in `page_report`, so it cannot be called from a test.
- FIX: Extract the post-stream logic and text-block selector into pure functions; add synthetic-index ranking tests, including an adversarial case (a high raw score from a low-authority document must not outrank unless the boost says so).
- confidence high. Found by: tests, corroborated by GUI/Streamlit correctness.

**M20. ERROR · `build_index.py:50-134` (pipeline-02, 04)**
- WHAT: A document that passes the `^## Section` detector but not the strict splitter (en dash, non-empty title) yields zero chunks silently. Image-only or failing PDFs are skipped with only console prints; a mis-encoded file is indexed as mojibake via the UTF-16 fallback. No "N of M files skipped" summary.
- WHY: No file currently triggers the zero-chunk case (challenge: partial/minor), but the encoding and PDF-skip paths are unverified against the corpus.
- FIX: Same regex for detection and split; summary of skipped, partial and zero-chunk documents; try UTF-16 only on a BOM.
- confidence medium.

**M21. CORRECT · `split_fsj_issues.py:745-757, 646-652, 777-781`**
- WHAT: Full-issue PDFs are moved out of the indexed folder even if no articles were extracted, the script cannot be re-run (the PDFs are already moved), and output filenames can collide so a later article overwrites an earlier one. Hard-coded `/Users/davemini2/...` path; imports `fitz`, which is not in `requirements.txt`.
- FIX: Move only successfully split issues; detect duplicate output names; take the source path from a CLI argument.
- confidence medium (read-only, not run).

### Minor
- **CORRECT · `streamlit_app.py:1105-1111`**: help text says Report combines staged chunks plus fresh retrieval; code and CLAUDE.md say staged chunks replace retrieval. Verified, wrong text only.
- **CORRECT · `streamlit_app.py:427-431`, `bowen_rag_gui.py:491-496`**: "Both" mode sorts keyword hit counts with cosine scores, so keyword results always win. Corroborated by two workers.
- **CORRECT · `streamlit_app.py:1054-1055, 1280-1284`**: author filter applied after top-k, so it can return fewer than k, or nothing.
- **CORRECT · `streamlit_app.py:1606-1613`**: appendix built from all retrieved documents, not only cited ones.
- **CORRECT · `streamlit_app.py:1500, 1563-1574`**: `last_rpt_context` is overwritten before generation succeeds, so the audit can show chunks that did not produce the displayed report.
- **CORRECT · `streamlit_app.py:675-681`**: `_init_session` resets provider and keys from env on every rerun, so Settings changes do not stick when env vars are set. Also makes the Ollama SSRF path less reachable than first reported.
- **SEC · `streamlit_app.py:603-626, 1891-1893`**: Ollama URL taken from the UI with no validation (SSRF). Partial: reachable when `LLM_PROVIDER` is unset or `APP_PASSWORD` is unset.
- **SEC · `streamlit_app.py:1851`**: Settings shows the last 6 characters of server keys; the standard allows 4.
- **SEC · `bowen_rag_gui.py:955`**: `.env` written with default umask (not 0600), non-atomic. Save also omits `DEEPSEEK_API_KEY`/`DEEPSEEK_MODEL`/Ollama settings and rewrites empty fields as `KEY=` (corroborated by security and GUI correctness).
- **DEPS · `build.yml`**: actions pinned to tags, no `permissions:` block, `macos-13` runner may be retired; `SentenceTransformer("all-MiniLM-L6-v2")` unpinned at runtime.
- **CORRECT · `citations.py:342-358`**: Harvard article entries with no container render a stray comma; MLA/Chicago entries with "n.d." end in a double period. Affects most of the corpus.
- **CORRECT · `citations.py:457-462`**: `[[3, page 45]]` or `[[3, pg. 45]]` is read as sources 3 and 45, which can cite an unrelated source.
- **ERROR · `citations.py:60-70`, `streamlit_app.py:79-115`**: all YAML loaders swallow every exception and fall back silently, so a typo in `sources.yml` degrades every reference with no signal. Unknown numbers inside a grouped marker are dropped silently.
- **DATA · `seed_sources.py`, `citations.py:119-124`**: year regex misses underscore-delimited years (Richardson_1996), takes the first year in "(1943-2017)", and the title cleaner strips real leading numbers.
- **DATA · `citations.py:169-216`**: author strings split on commas ("Kerr, Michael" becomes two authors).
- **DATA · `build_index.py:235-242`**: documents keyed by file stem, so `foo.txt` and `foo.pdf` merge.
- **CORRECT · `process_transcripts.py:26-98`**: the `(?i)test` skip pattern excludes any filename containing "test" (Contest, Latest); duplicate titles overwrite each other; CRLF breaks frontmatter parsing; outputs from renamed transcripts are never removed.
- **CORRECT · `build_index.py:157-189`**: no hard cap on chunk size; an oversized chunk's tail is beyond the 256-token limit of MiniLM and not searchable by embedding.
- **ARCH · `semantic_search.py`** (used only by `test_skill.py` and the CLI documented in SKILL.md): a third stale copy of retrieval (`.npy`, 500 features), incompatible with the current index. Verified. Dead for the apps.
- **PERF · `streamlit_app.py:299-314, 336-337, 441-442, 496-497`**: per-query Python work over all 11k chunks (authority_boost per chunk, `.lower().count()` for keyword); the vectorizer is refit and BM25 retokenised at every cold start while `vectorizer.json` goes unused. The embedding model is lazily constructed with no lock.
- **MAINT · both apps**: editorial content in code (system prompt, interview/coach/quiz prompts, report structure, BM25 stop words, boost help text "3x / 1.3x / 1.15x" hard-coded in four places). The comment "identical to bowen_rag_gui.py" at `streamlit_app.py:264` is false: `build_embeddings` is missing and error behaviour differs.
- **CONCUR · `bowen_rag_gui.py:2137-2175`**: worker threads read Tk variables and widgets directly.
- **CORRECT · `bowen_rag_gui.py:1043-1049, 1977-2024`**: switching to the Report tab overwrites the topic; staged chunks survive an index rebuild and their ids are looked up in the new index; blank spinboxes raise an uncaught `TclError`.
- **CORRECT · `streamlit_app.py:258, 577-581`**: `o1`/`o1-mini` offered but the call uses `max_tokens` and a system message. Unverified against the current OpenAI contract.
- **TEST**: `test_citations.py::test_m1c_match_source_first_wins` cannot tell first-wins from longest-wins (and is misnamed); `rag-document-search/test_skill.py` asserts nothing and its eval prompts refer to a different corpus; `test_report_export.py` imports the whole `streamlit_app` (heavy on a blank machine; PDF checks are header + size only); `docs/spec_coverage.md` marks M3.A, M4.A, M4.B and M4.D "done" with "UI/smoke" as evidence.

## Systemic patterns
1. **Two diverging front-ends with no shared core.** The report prompt, tier defaults, YAML loading, "Both" ranking, page-locator logic and chat-history handling are copied, and several copies already differ (M3, M4, M1, M10, the "Both" ranking, and the false "identical" comment). A shared module for index/search, config loading and report prompt/post-processing would fix M1, M3, M4 and the "Both" ranking together.
2. **Silent failure and silent fallback (P2/P9).** Dropped Section 1 (B2), swallowed YAML errors, missing config in the frozen app (M3), unknown citation numbers dropped, skipped PDFs, truncation shown as complete (M6), empty replies saved to history (M10). Nothing is counted or announced.
3. **Index/embedding consistency is checked weakly.** Row count only, at search time (M7); the loaders refit the vectorizer instead of using `vectorizer.json`; `semantic_search.py` has drifted. B2 means the index must now be rebuilt, which makes this live.
4. **Seeded data carries guesses forward as record.** B1, M2 and the year/title seeding issues share a cause: the seeder's heuristics are written to `sources.yml`, and `verified: false` is only displayed as a count.
5. **No automated gate before deploy.** No `tests.yml`, unpinned dependencies, and Railway deploying on push (M17, M18). Tests that exist cover citations well and almost nothing else.

## Refuted in challenge pass
No finding was fully refuted. Corrections made by the challenge worker:
- B1: the cause is `author_map.yml` alone; `seed_sources.py` has no Bowen pattern of its own.
- Pipeline-02 (zero-chunk sectioned docs): the code path is real, but no file in the current corpus triggers it. Downgraded to latent.
- Ollama SSRF: only partly reachable, because the `LLM_PROVIDER` override resets the provider choice on rerun. Downgraded to minor.
- M11 (DeepSeek "Streaming is required"): affects the GUI only. Streamlit's test streams.
- M8 and M10: the provider rejection is likely but was not tested.
- Not challenged (single-worker findings, not independently verified): M13, M15, M16, M20, M21, and most Minor items.

## Questions only the author can answer
1. Does Railway always run with `APP_PASSWORD` set? If not, M14 is a blocker.
2. Are the "Wisdom of the Ages 1998 / 2002 / 2009" records meant to show event years as authorship dates "(Bowen, 1998)"? Bowen died in 1990.
3. Is the `[assistant opening, user]` first-turn failure (M8) already observed, or have those modes only been tried on a tolerant provider?
4. Is `tfidf_matrix.npy` in `references/` still used by anything? If not it is a stale artifact.
5. Are FSJ 16.1 and 16.2 (`REDUNDANT_ISSUES`) really duplicated by article files that already exist?
6. In the packaged app, does a user who builds embeddings before ever rebuilding the index lose them on restart (`build_embeddings` writes to `USER_REFS_DIR`; `_bg_load_index` falls back to the bundled dir)? Not confirmed.
7. Did an earlier commit ever contain `.env`? Not checked (`git log --all -- .env`).
