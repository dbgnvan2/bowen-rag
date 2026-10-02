# Review (round 2): bowen_rag (whole repo) — 2026-10-01, at commit a88a36b

Round 1 is `REVIEW-bowen_rag-2026-10-01.md` (at commit 7736978). The round-2 workers were not shown it, so findings they re-found independently are marked **re-found**. The only code change between the rounds is the daily token limit feature (`usage_limit.py` and its hooks in `streamlit_app.py`). **None of round 1's findings has been fixed.**

## Verdict
The new token limiter has a hole: it admits a call whose own input is larger than the visitor's remaining budget, so one request can spend well past the cap. The limiter is also the only thing standing between an anonymous visitor and your API spend, and the Settings page is an unauthenticated admin surface that bypasses it in several ways (other providers, the pricier `deepseek-v4-pro` model, and a server-side URL fetch). The two data defects from round 1 are still live: a dozen records credit Bowen with other authors' work, and the first section of some transcripts is missing from the shipped index. Citation formatting for the roughly 190 author-less records is wrong in every style. There is still no CI test run and no test of the report or limiter wiring. Fix first: the attribution data, the Section-1 drop, the limiter's oversized-call hole, and gating Settings in deployed mode.

## Coverage
| Area | G-CORRECT | G-SEC | G-STRUCT | G-TEST |
|---|---|---|---|---|
| streamlit_app.py + usage_limit.py (deployed) | covered | covered | partial (lines 30–140, 262–785, 1365–1395, 2050–2090) | partial |
| bowen_rag_gui.py (desktop) | covered | partial (key handling, greps) | partial (lines 30–180, 256–660, 1971–1996) | partial (greps) |
| citations.py, seed_sources.py, YAML config | partial (sources.yml by grep; 330 records not all read) | partial (greps) | not reviewed (citations/seed internals) | partial |
| Index pipeline (build_index, process_transcripts, split_fsj_issues, semantic_search) | covered (FSJ parsers not run on real PDFs) | partial (greps) | partial | partial |
| CI / build.yml / requirements | n/a | covered | covered | covered |

Read: about 12 code files, roughly 7,900 LOC of Python (including 247 lines of `usage_limit.py` and 228 of its tests), plus config. **Nothing was executed**: no worker had a shell for the test run, so the suite was not run in this round (I ran 65 tests myself before the commit). The challenge pass also ran nothing.
Not reviewed: most of the GUI `App` class body for duplication; `seed_sources.py` and `citations.py` internals for STRUCT; git history for a committed `.env` (the security worker had no shell; `git log --all -- .env` not run); GitHub's current runner list; DeepSeek's actual usage reporting and pricing; live Railway behaviour (headers, cookie write from the iframe).
Excluded: `source_files/`, `references/*.json|npy|npz`, `outputs/`, `sources.seeded.yml`, `__pycache__`, other untracked notes.

## Findings
"verified" = confirmed against the code by a challenge pass (round 2 unless noted). "round 1 verified" = confirmed in round 1 and the code is unchanged since.

### Blockers
**B1. DATA · `sources.yml` (lines 966, 1128, 1308, 1578, 1695, 1704, 1785, 1911, 1920, 2145, 2405); cause `author_map.yml:103`**
- WHAT: Eleven records for other authors (Jones, Riordan, Harrison, Ferrera, McKnight, Millikin, Holt, Caskie ×2, Thompson, Lerner) list "Murray Bowen" as author. The bare `"Bowen"` pattern in `author_map.yml` matches any filename containing the word, and the seeder copied it. The author filter is wrong for these too.
- WHY: Reports print "(Bowen, n.d.)" for a Jones article. This is wrong attribution in a source-fidelity tool.
- FIX: Correct the records or set `authors: []`. Move the catch-all below the specific rules and re-seed.
- confidence high · round 1 verified · **re-found** by the citations and desktop-security workers.

**B2. CORRECT · `rag-document-search/scripts/build_index.py:117`**
- WHAT: The splitter regex requires a newline before `## Section`, so a transcript that starts with its first heading loses Section 1. Nothing is logged. Confirmed in the shipped index: "Bowen Basic Series Tape 7" and "2011 04 Making a Difference … Kerr" are both missing their opening sections.
- FIX: `re.split(r'(?:^|\n)(## Section \d+ – [^\n]+)\n', content)`, keep or log any preamble, then rebuild TF-IDF and embeddings together.
- confidence high · round 1 verified · **re-found** by the pipeline worker.

### Major
**M1. CORRECT (new, in my code) · `usage_limit.py:178-192` (`begin`)** — found independently by the security and correctness workers
- WHAT: The reservation is clamped to the remaining budget (`min(..., user_rem, global_rem)`) instead of refusing a call whose estimated input alone exceeds it. `finish()` then records the real usage with no ceiling. Nothing else bounds the prompt: the Report form allows top-N up to 150 and chunks per source up to 20 (19 chunks per document); chat allows up to 100 chunks.
- WHY: Report only needs 60K tokens remaining to start. A visitor with 61K left can run a report of roughly 285K tokens (for example top 30, 20 chunks per source). Chat has a floor of 1 token, so a visitor with almost nothing left can send about 150K tokens of input. The challenge pass notes the absolute maximum (about 1.4M tokens) would probably be rejected by DeepSeek's context limit, so the realistic overshoot is a few hundred thousand tokens per call, not unbounded. The spec's "known limits" line does not mention this.
- FIX: In `begin()`, raise `UsageLimitError` when `estimate_tokens(input_chars) + 16,000` exceeds `user_rem` or `global_rem`, and cap total prompt size in `_llm_stream`. Add a test for it (the current tests pass because none exercises this case).
- confidence high · **verified (scope narrowed)**.

**M2. SEC (new) · Settings is an unauthenticated admin surface; with no `APP_PASSWORD`, every visitor can use it.** Four instances, one root cause:
- **a. Other providers, unmetered.** `streamlit_app.py:1904-1907, 1965-1976`: with Anthropic or OpenAI keys in Railway env, a visitor can pick that provider in Settings and click "Test connection" repeatedly. Only the DeepSeek branch is metered. Challenge: Chat and Report are *not* affected while `LLM_PROVIDER` is set, because the page's radio state is dropped when the visitor leaves Settings and `_init_session` resets the provider. If `LLM_PROVIDER` is unset on Railway, every page is affected. Requires the operator to have set those keys; `CLAUDE.md` lists only the DeepSeek key.
- **b. Pricier model.** `streamlit_app.py:260, 1941-1945, 604`: the session's `deepseek_model` selects `deepseek-v4-pro`; the cap counts tokens, not money, and nothing pins the model server-side. Pricing not verified.
- **c. Server-side request to any URL.** `streamlit_app.py:617-636, 1952-1956`: the Ollama "Server URL" is free text and not reset by `_init_session`. "Test connection" POSTs to it and echoes connection errors, which acts as an open/closed oracle for Railway's internal network. Chat and Report cannot reach it while `LLM_PROVIDER` is set.
- **d. Shared file overwrite.** `streamlit_app.py:1818-1827`: "Save this source" writes the server's `sources.yml` for any visitor; other sessions pick it up on their next rerun, a stale snapshot overwrites concurrent edits, and the write is not atomic. A bad file silently falls back to `[]` for everyone. (Edits are lost on redeploy on Railway.)
- Also: the system prompt is visitor-editable, so the app is a general DeepSeek proxy within the cap (reported by one worker as a note).
- FIX: In deployed mode, hide Claude/OpenAI/Ollama and the model picker, pin provider and model from env, make `sources.yml` saving opt-in (`ALLOW_SOURCES_WRITE`), and require `LLM_PROVIDER` in deployment. Or add an `ADMIN_PASSWORD` gate on Settings only.
- confidence high · **verified** (a and c partial as stated). Corroborated by three workers.

**M3. SEC/CONCUR (new) · `streamlit_app.py:664-687`, `usage_limit.py:60-65`: the visitor id is unsigned.**
- WHAT: Any 32-hex cookie value is accepted and gets a fresh 300K, and a visitor who opens several tabs before the cookie is stored gets one id per tab. A single script making fresh sessions needs about 7 ids to reach the 2M global cap, after which every other visitor is told "total daily capacity has been reached" until 00:00 UTC.
- WHY: This is the accepted cookie-method design, but the denial-of-service side was not spelled out when you chose it: the global cap protects spend, not availability.
- FIX: Sign the cookie (HMAC with a server secret) and rate-limit new-id creation per IP (`X-Forwarded-For` via `st.context.headers`; not verified on Railway), or reserve part of the global cap per day.
- confidence high · **verified**.

**M4. TEST · limiter wiring and cookie path untested (new) · `streamlit_app.py:554-613, 664-687`**
- WHAT: `test_usage_limit.py` tests the module in isolation. Nothing tests that `_llm_stream` calls `begin`/`meter`/`finish` correctly, that a capped visitor is refused before any client is created, that non-DeepSeek providers skip the limiter (U7), or the cookie-missing path (U6). M1 above is a hole the isolated tests cannot see.
- FIX: Drive `_llm_stream` with a fake Anthropic client and a stub `st.session_state`; add boundary tests at exactly cap and cap-1 (per-user and global).
- confidence high. Spec IDs U1–U6 map to real tests; U7 and U8 are declared integration-only.

**M5. CORRECT · citation rendering for author-less and "n.d." records · `citations.py:331, 341, 361, 371-375, 384-398`** — found by the citations worker; consolidated from several findings
- WHAT: About 158–190 of 330 records have no author and about 297 have year "n.d.". Consequences: (i) the in-text label for author-less records is `title.split(".")[0][:40]`, so different FSJ articles all render as "(FSJ 10, n.d.)" (84 FSJ records are affected); (ii) APA/Harvard/Chicago put the date first for author-less works ("(n.d.). *Title*."), which is not their convention, and `test_citations.py:188` locks that in; (iii) article records with no container or volume (all ~100 `article-journal` records) render a stray comma in Harvard, "n.d.." in MLA and Chicago, and "n.d.;." in Vancouver; (iv) same-author, same-year works get no a/b suffix, so one citation maps to several references (22 FTCP chapters, 7 Bowen tapes, 6 Wisdom of the Ages recordings).
- FIX: Split titles on ". " (period plus space) at a word boundary; put the title in the author position per style; build container/volume pieces only when present; add a/b/c suffixes; add golden-string tests per style for an author-less record.
- confidence high · (i), (ii), (iii) verified in round 2; (iv) round 1 verified.

**M6. DATA · seeded years and authors that are not real · `sources.yml:1200-1207, 2749-2851`; cause `seed_sources.py:65-71`** (verified)
- WHAT: The seeder takes the first 4-digit number in a filename as the year. "FSJ 12.2 Noone Jaak Panksepp (1943-2017)" is cited as "(Noone, 1943)", Panksepp's birth year. The Wisdom of the Ages Houston 2009 / Pittsburgh 1998 / 2002 records are "(Bowen, 2009)" etc. with event years as publication years; Bowen died in 1990. All are `verified: false`, which reports show only as a count.
- FIX: Seed `n.d.` unless the year follows "Surname YYYY"; correct these records by hand; confirm who the author of the Wisdom of the Ages records is.
- confidence high.

**M7. CORRECT · wrong page locator in citations · `streamlit_app.py:1478-1495`, `bowen_rag_gui.py:2020-2046`**
- WHAT: The "(p. N)" offered to the model is built from the retrieved chunks, but the text sent includes ±window neighbour chunks that may be on other pages. A quote from a neighbour gets a confidently wrong page.
- FIX: Build the page set from every chunk id included in the prompt; offer a locator only if it has one page.
- confidence high · round 1 verified · **re-found** (desktop worker).

**M8. DATA · author filter applied after top-k · `streamlit_app.py:1108-1109, 1334-1335`, `bowen_rag_gui.py:1251-1260, 2600-2605`** (verified)
- WHAT: With the author filter on and k=15, a visitor may see 2 results or none, and Chat says "No chunks found for author". Streamlit shows no count and an empty list reads "Run a search to see results". Report has no author filter.
- FIX: Pass the filter into retrieval or over-fetch, and show "N of M".
- confidence high.

**M9. ARCH · packaged desktop app ships no config; defaults diverge · `build.yml:39-41`, `bowen_rag_gui.py:94-116, 119, 142, 158`**
- WHAT: The PyInstaller bundle does not include `authority_tiers.yml`, `author_map.yml` or `sources.yml`, and the frozen app reads them from `~/Documents/BowenRAG`. The GUI's built-in tiers lack the `("FSJ ", 1.3)` entry (still `"Copy of "`). In the frozen app FSJ articles rank at 1.0×, every author is "Unknown", and references fall back to filenames.
- FIX: Add `--add-data` for the three YAML files, or move loading into one shared module; add a test that the defaults equal the YAML.
- confidence high · round 1 verified · **re-found** (structure worker).

**M10. ARCH · report prompt drift between the apps · `streamlit_app.py:1502-1543` vs `bowen_rag_gui.py:2074-2121`** (round 1 verified; not re-found this round)
- WHAT: The deployed web app's report prompt lacks "Do not paraphrase without attribution", "if sources disagree, quote both", and "use the numbers shown in the source headers". The two IndexManager copies also differ (`embedding_search` behaviour when unloaded, stale-check order, silent YAML errors in the web app), and the comment "identical to bowen_rag_gui.py" is false.
- FIX: One shared prompt builder and a shared core module for index/search/config.
- confidence high.

**M11. PERF · dense TF-IDF matrix · `streamlit_app.py:289`, `bowen_rag_gui.py:281`**
- WHAT: `.toarray()` on an 11,252 × 8,000 matrix is about 720 MB resident (about 360 MB if float32; dtype not checked), and each query also re-normalises it and loops over all chunks calling `authority_boost`. Concurrent searches multiply temporary memory on the shared index.
- FIX: Keep CSR and use `matrix @ qvec.T`; precompute the boost array at load.
- confidence medium-high · round 1 verified · **re-found**.

**M12. ERROR · silent truncation not detected (round 1 verified; not re-found this round)**
- WHAT: `stop_reason`/`finish_reason` is never checked anywhere. A report cut off at `max_tokens` gets a References section and is shown as complete. The spec for the limiter notes this as out of scope.
- FIX: Read the final stop reason and show a warning on a length stop.
- confidence high.

**M13. ERROR · empty chat reply saved to history · `streamlit_app.py:1377-1410`, `bowen_rag_gui.py:2682-2684`** (verified; **re-found** by two workers)
- WHAT: An empty streamed response (for example DeepSeek spending its budget on thinking) is appended to history. Later turns send an empty assistant message, which the API likely rejects (not tested), so chat errors every turn until cleared. Ollama `{"error": ...}` lines are also ignored in the GUI.
- FIX: Check `response.strip()`; skip the append and show an error; raise on Ollama error lines.

**M14. CORRECT · chat "View ↗" opens the wrong chunk · `streamlit_app.py:1297-1302, 1400-1405`** (round 1 verified; **re-found**)
- WHAT: It opens the first chunk of the document, not the retrieved one; `sources` entries hold only `doc` and `excerpt`.
- FIX: Store the chunk id in each source.

**M15. CORRECT · interview/coach/quiz first turn (round 1, unverified at provider level; not re-found)**
- WHAT: `streamlit_app.py` seeds history with an assistant message, so the first API call starts with role `assistant` and nothing strips it. Whether Claude or DeepSeek reject this is untested.
- FIX: Drop the leading assistant turn when sending; test live.
- confidence medium.

**M16. ERROR · embeddings staleness detected by row count only · `build_index.py:267-290`, loaders `streamlit_app.py:458`, `bowen_rag_gui.py:363`** (round 1 verified; **re-found** by pipeline and GUI workers)
- WHAT: A rebuild that keeps the chunk count but changes the text silently pairs old embeddings with the wrong chunks. The TF-IDF vectorizer is refit at every load with parameters duplicated by hand, `vectorizer.json` records `max_features: 500` (real value 8000), and a stale `tfidf_matrix.npy` still sits next to the `.npz`.
- FIX: Write a `build_info.json` (chunk count, content hash, vectorizer params) and verify it at load; write index files atomically.

**M17. CORRECT · `bowen_rag_gui.py:791, 812, 1783-1786`** (verified in round 1; **re-found**)
- WHAT: Non-streaming DeepSeek calls use `max_tokens=32000`; with anthropic 0.78.0 the SDK raises "Streaming is required", so the GUI's Test Connection always fails for DeepSeek. Streamlit's test streams and is unaffected.
- FIX: Use a small `max_tokens` for the probe, or stream internally.

**M18. CONCUR · desktop GUI** (round 1 verified; **re-found**)
- `bowen_rag_gui.py:269-314`: `IndexManager.load` replaces `chunks`, `matrix` etc. in place on a background thread while the UI can search it.
- `bowen_rag_gui.py:1486-1607, 2135-2202, 2655-2691`: no in-flight guard on Rebuild, Import, Build Embeddings, Generate Report or Chat Send, and several jobs patch the global `builtins.print`, so a double-click can leave `print` redirected. Breaks the project rule on buttons that launch background work.
- FIX: Build state in locals and swap once; per-job busy flag set before the thread starts; `contextlib.redirect_stdout` under a lock.

**M19. DEPS · `requirements.txt`** (**re-found** by three workers)
- WHAT: Everything except `streamlit>=1.37` is unpinned, and there is no lockfile, so each Railway deploy and release build takes whatever is newest. `PyPDF2` (deprecated, build-only) ships to the server.
- FIX: Pin exact versions or a hashed lock; move build-only packages to a separate file.

**M20. DEPS · `.github/workflows/build.yml:14-17, 63-67`**
- WHAT: No `permissions:` block (the release step needs `contents: write`), actions pinned to tags not SHAs, and the Intel leg uses `macos-13`. Challenge: the file confirms all three; GitHub's runner availability could not be checked offline. One worker recalls the `macos-13` image being retired around December 2025, in which case with fail-fast one dead leg cancels the arm64 build too.
- FIX: Add `permissions: contents: read` at the top with write only on the release job, pin the release action to a SHA, replace the Intel runner, set `fail-fast: false`.
- confidence high for the file contents, unverified for the runner.

**M21. TEST · repo-wide**
- No `.github/workflows/tests.yml`; pushes to `main` redeploy Railway with no tests run (global rule). `build.yml` fires only on `v*` tags.
- No tests for search ranking (authority boost, RRF, `use_boost` off), staged-chunk replacement, the stale-embedding check, `build_index` chunking (heading at byte 0, hyphen heading, CRLF), or the report fail-loud and DeepSeek text-block selection (the post-stream logic is inline in `page_report` and cannot be called from a test). `rag-document-search/test_skill.py` asserts nothing and needs a built index. `test_report_export.py` imports the whole `streamlit_app`.
- FIX: tests.yml running `pytest`; extract the post-stream validation into a pure function; synthetic-index ranking tests including an adversarial case.

**M22. ERROR · index build drops or damages input without a summary · `build_index.py:50-134`** (round 1 verified in part; **re-found**)
- A document detected as sectioned (`^## Section \d+`) but with headings the splitter does not match (hyphen, CRLF, heading at EOF) yields zero chunks silently; no file in the current corpus triggers this (latent). Image-only or failing PDFs are skipped or truncated with only a console line and no "N of M" summary. A non-UTF-8 file would be decoded as BOM-less UTF-16 and indexed as mojibake (no file in the corpus is currently invalid UTF-8, so this is latent too).
- FIX: One shared heading regex; warn and fall back to word-count when a sectioned doc yields fewer chunks than headings; BOM check before UTF-16; print a skipped/partial summary.

**M23. CORRECT · `split_fsj_issues.py:745-757, 646-652, 30`** (**re-found**)
- Full-issue PDFs are moved even if no articles were extracted, the script cannot be re-run, output filenames can collide so a later article overwrites an earlier one, and the path is hard-coded to `/Users/davemini2/...`. It also imports `fitz` (PyMuPDF, AGPL), which is not in `requirements.txt`.

### Minor
- **SEC** `streamlit_app.py:1851`: Settings shows the last 6 characters of server API keys to every visitor (standard allows 4).
- **SEC** `bowen_rag_gui.py:921-957`: "Save to .env" writes with default umask (not 0600), is non-atomic, omits `DEEPSEEK_*`, `OLLAMA_*` and `CITATION_STYLE`, and rewrites an existing key to empty when the field is blank. The GUI also never loads `.env` into `os.environ`, so `CITATION_STYLE` in `.env` is ignored (verified).
- **SEC** privacy: Interview and Coach chat text goes to the third-party provider with no notice (round 1; not re-found).
- **SEC** `usage_limit.py:19, 65`: `re.match` with `$` accepts a trailing newline; use `fullmatch`. Challenge: no practical consequence.
- **ERROR** `usage_limit.py` (observation): reservations have no timeout, so a stream that never closes holds its reservation until restart; I could not show it without a runtime test. The corrupt-state recovery fails open (counters restart at zero) and an unset `USAGE_DIR` resets everything on redeploy. Both are logged but not user-visible.
- **ERROR** `usage_limit.py` (observation): only `input_tokens + output_tokens` are summed; cache-read tokens, if DeepSeek reports them separately, are not counted. Reasoning-token accounting is unverified (the spec says so).
- **CORRECT** `bowen_rag_gui.py:453, 471` and `streamlit_app.py:405-409`: keyword search counts substrings ("self" in "itself") and keeps punctuation on query terms, so "triangles?" matches nothing (verified).
- **CORRECT** `combined_search` ("Both" mode) in both apps sorts keyword hit counts with cosine scores, so keyword results always win.
- **CORRECT** `streamlit_app.py:675-681` (`_init_session`): env values reset provider and keys on every rerun, so Settings changes do not stick when env vars are set.
- **CORRECT** `streamlit_app.py:1608-1625`: the Audit expander can show chunks that did not produce the displayed report; the appendix includes uncited sources; the download filename uses the last search query.
- **CORRECT** `streamlit_app.py:1105-1111` / `1165-1169`: help text says Report combines staged chunks plus fresh retrieval; the code replaces (verified, text only).
- **CORRECT** `streamlit_app.py:258` / `bowen_rag_gui.py:68-74`: `o1`/`o1-mini` are offered but the calls use `max_tokens` and a system message (OpenAI's current contract not checked).
- **CORRECT** `bowen_rag_gui.py:1043-1049`: switching to the Report tab overwrites the topic with the Search query; staged chunks survive an index reload and are expanded against the new index; blank spinboxes raise an uncaught `TclError`; Save writes frontmatter into `.docx`/`.pdf` and can save the "Generating…" banner.
- **CORRECT** `citations.py:457-462`: `[[3, page 45]]` or `[[3, pg. 45]]` is read as sources 3 and 45; `[[1234]]` parses as 123 and 4.
- **ERROR** YAML loaders swallow every exception and fall back silently (`citations.py:60-70`, `streamlit_app.py:79-115`, GUI prints only).
- **DATA** `seed_sources.py`: `speech` type renders as "[Audio recording]" for anything with "lecture", "webinar" or "series" in the name; title cleaner strips real leading numbers; author strings split on commas.
- **DATA** `process_transcripts.py:26-98` (`(?i)test` skip pattern excludes "Contest", "Latest"; same-title transcripts overwrite each other; CRLF breaks frontmatter parsing; outputs of renamed transcripts are never removed), `build_index.py:235-242` (`foo.txt` and `foo.pdf` merge into one doc), `build_index.py:157-189` (no cap on chunk size; MiniLM truncates at 256 word pieces).
- **ARCH** `rag-document-search/scripts/semantic_search.py` (used only by `test_skill.py` and the CLI in SKILL.md): loads the stale `.npy` and refits with `max_features=500`, so it cannot work against the current index. The apps do not use it.
- **MAINT** editorial content in code (Interview/Coach/Quiz prompts, report structure, BM25 stop words, boost help text "3x / 1.3x / 1.15x" in four places); `max_tokens` literals repeated about 12 times; `begin()` raises a bare `ValueError` for an invalid visitor id (not `UsageLimitError`).
- **PERF** per-query Python loops over all chunks; the embedding model is created lazily with no lock on the shared index.
- **TEST** `test_m1c_match_source_first_wins` cannot tell first-wins from longest-wins (and is misnamed); `docs/spec_coverage.md` marks M3.A, M4.A, M4.B and M4.D done with "UI/smoke" evidence and does not list the usage-limit tests.

## Systemic patterns
1. **Unauthenticated admin surface plus a spend limiter that covers one path.** M1, M2 and M3 are all about what an anonymous visitor can do on the deployed app once `APP_PASSWORD` is dropped. The limiter is correct in isolation (all U1–U6 tests pass) and wrong at its edges: oversized calls, other providers, other models, unsigned ids. A server-side deployed-mode switch (pin provider and model from env, hide Settings admin controls, cap prompt size) would address most of it in one change.
2. **Seeded guesses written into `sources.yml` as record.** B1, M5 and M6 share a cause: the seeder's heuristics (year from the first digits, author from a catch-all, type from keywords) are persisted, and `verified: false` is shown only as a count. The citation formatters then print them with full authority.
3. **Two front-ends with no shared core.** M7, M8, M9, M10, M13, M14 and the Both/keyword ranking issues all have a copy in each app (some already differ). A shared module for index, search, config loading and the report prompt would remove most of them.
4. **Silent failure and silent fallback (P2/P9).** Dropped Section 1 (B2), zero-chunk documents, skipped PDFs, swallowed YAML errors, truncation shown as complete (M12), empty chat replies saved (M13), fail-open limiter recovery.
5. **No automated gate before deploy.** No `tests.yml`, unpinned dependencies, Railway deploys on every push (M19, M21), and the new limiter's integration is untested (M4).

## Refuted or corrected in the challenge pass
No finding was refuted outright. Corrections:
- **Oversized-call hole (M1):** the extreme case (top 150, 19 chunks per source, about 1.4M tokens) would probably be rejected by DeepSeek's context limit before spending; the realistic overshoot is a few hundred thousand tokens per call. Not a guarantee: the DeepSeek limit was not checked.
- **Provider switch (M2a):** Chat and Report are not reachable through the Settings radio while `LLM_PROVIDER` is set; only "Test connection" is.
- **Ollama URL (M2c):** same: reachable only through "Test connection" while `LLM_PROVIDER` is set.
- **Trailing-newline cookie value:** real in the regex, but no meaningful consequence; downgraded to a nit.
- **"Save to .env" empty overwrite:** only happens if the user clears a field or the key was unreadable by the loader; minor.
- **Invalid-UTF-8 fallback (M22):** latent; no file in `source_files/` is invalid UTF-8.
- **Round 1 corrections still stand:** the Bowen attribution comes from `author_map.yml` alone, and `semantic_search.py` is dead for the apps.
- **Not independently verified by the challenge pass** (single-worker findings): the GUI frontmatter and export issues, the `semantic_search`/`SKILL.md` claims beyond round 1, FSJ page-boundary fallbacks, the GUI Tk-thread reads, the PyMuPDF licence note, and the claims in M15, M20 (runner) and the limiter's cache-token and reservation-timeout observations.

## Questions only the author can answer
1. Is `LLM_PROVIDER` set on Railway, and are `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` set there? That decides how serious M2a is.
2. Do you want Settings (provider, model, Ollama URL, system prompt, Save source) hidden or locked in the deployed app, or is a visitor-editable system prompt acceptable?
3. Are the "Wisdom of the Ages 1998 / 2002 / 2009" files transcripts of events rather than Bowen's own text? If so, the author attribution is wrong as well as the year.
4. Is the desktop `.app` still distributed? If yes, M9 is user-visible today.
5. Is the `[assistant opening, user]` first-turn failure (M15) already observed on Claude or DeepSeek?
6. Did an earlier commit ever contain `.env`? Not checked in either round (`git log --all -- .env`).
7. Is `tfidf_matrix.npy` in `references/` still used by anything, or is it a stale artifact?
