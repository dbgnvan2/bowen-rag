# Review: false "not in the corpus" results, and a literal-search feature — 2026-11-01, at commit c5ac598

Scope: why the app reports a phrase as absent when it is in the corpus ("emotional programming" is the reported case), and whether a literal string search should sit beside the TF-IDF / BM25 / embedding modes. Everything below was run in a cloud sandbox against a copy of the shipped index (`chunk_metadata.json`, `tfidf_matrix.npz`, `vectorizer.json`, `embed_matrix.npy`, 14,083 chunks, 401 documents) and the current `streamlit_app.py` `IndexManager`, loaded exactly as the app loads it.

## Verdict

The index is not at fault: "emotional programming" occurs 93 times in 66 chunks across 22 documents, and it is even one of the 8,000 TF-IDF features. The false negative is produced downstream, by three things that compound. First, the ×3 authority multiplier is applied to raw similarity scores, and for a two-word term where one word ("emotional", 12,769 occurrences) is far commoner than the other ("programming", 171), it pushes every chunk that contains the phrase below unboosted-irrelevant chunks from Tier-1 sources: with boost on, BM25 at the Chat default (12 chunks) returns **0 of 12** chunks containing the phrase, and the best phrase-bearing chunk ranks 25th; Hybrid, the default mode, also returns **0 of 12** (addendum below), and the context the model receives contains the phrase zero times. Second, Keyword mode scores an unquoted query as a sum of raw word counts, so "emotional" alone decides the ranking (0 of 15 with boost), and a trailing question mark makes the last word unmatchable ("programming?" is never found). Third, system-prompt Rule 7 asks the model itself to judge whether the term appears verbatim in the *provided* excerpts and to print "does not appear in the provided sources"; a reader takes that as "not in the corpus", and when the retrieval modes above have handed the model zero containing chunks, the notice is literally true and substantively wrong. None of this is caught by a test, because no test asks the pipeline the question "is the phrase in what the model was given?".

Recommended fix order: (1) compute term presence in code and hand the model the answer (`literal_search.corpus_check_line`), retiring the model's own judgement; (2) guarantee that chunks containing the user's phrase reach the model when any exist (`ensure_literal_hits`); (3) add an Exact-phrase search mode and a CLI; (4) stop multiplying the authority boost into raw scores, or lower it (data below); (5) fix Keyword mode's punctuation and OR-scoring; (6) add a regression test over a list of Bowen terms known to be in the corpus. A working `literal_search.py` with 17 passing unit tests is delivered with this review; it is a proposal, not yet wired into the apps.

## What was executed, and what was not

Executed in the cloud: every retrieval mode that does not need the sentence-transformer model — Top Docs, Semantic (TF-IDF), Keyword, Both, BM25 — at the defaults each page uses (Search 15, Chat 12, Report 30), with the boost on and off, for three phrasings of the query (`review_2026-11-01/repro_emotional_programming.py`, `diagnose_boost_and_keyword.py`, `measure_literal.py`; their output is in `review_2026-11-01/*.txt`). The literal-search prototype and its tests.

**Executed later in the same session (addendum 2026-11-01, after the model cache was connected):** Embedding and Hybrid modes, with the cached `all-MiniLM-L6-v2` loaded offline (it re-encodes chunk 0 to cosine 1.0000 against the stored `embed_matrix.npy`, so the setup matches the index). Hybrid, the default mode, boost on: **0 of 12** containing chunks at the Chat default, 0/15 at the Search default, 2/30 at the Report default (bare query); boost off: 12/12, 15/15, 27/30. For "What is emotional programming?": 0/12, 0/15, 3/30 on; 12/12, 15/15, 28/30 off. For the long question form: 0/12, 0/15, 3/30 on; 9/12, 12/15, 20/30 off. Embedding alone, boost on: 0 at every k and phrasing; off: 9/12, 11/15, 14/30 (bare). The Chat context actually sent — the 12 chunks plus their ±1 neighbours — contains the phrase **0 times** for all three phrasings. Still not executed: any LLM call. The cloud sandbox and the VM on your Mac are both refused by the network policy for `api.deepseek.com` (403 from the egress gateway); the model's behaviour under Rule 7 therefore remains argued, not observed, and `review_2026-11-01/llm_presence_experiment.py` is ready to run wherever that host is reachable.

## Ground truth

`grep -ric "emotional programming" source_files/*.txt` on your Mac: 18 text files, 56 lines. In the shipped index (substring, case-insensitive): 93 occurrences, 66 chunks, 22 documents. The heaviest sources are Kerr's "Bowen Theory Secrets" (chapter 9 is titled *Emotional Programming*; 8 chunks in that chapter, 25 across the book) and FSJ 4.2 Kerr "Bowen Theory and Evolutionary Theory" (14–15 chunks in each of its two copies). `vectorizer.json` lists `emotional programming` as a bigram feature. So the phrase is indexed by every mode; it is the ranking that loses it.

## Findings

### Blockers

**B1. P7 (over-weighted scoring proxy) · multiplicative authority boost · `streamlit_app.py:357, 462, 517, 547`; `bowen_rag_gui.py:391, 414, 525`, and `top_docs_search` in both**
- WHAT: `boosted = raw × authority_boost(doc)` with Tier 1 = 3.0. For "emotional programming", BM25 boost-on at k=12 returns 0 chunks containing the phrase (boost-off: 12 of 12); the best phrase-bearing chunk has raw BM25 10.77 (from "Bowen Theory Secrets", ×1.0) and ranks 25th boosted, behind twelve ×3 chunks from "Bowen Basic Series Tape 5", "Family Evaluation ch. 8" and "BOWEN-KERR INTERVIEW SERIES #7" that contain "programmed"/"programming" but not the phrase (the 12th-place boosted score is 14.96). Keyword boost-on k=15: 0 of 15 (all winners are ×3 chunks with 8–12 occurrences of "emotional" and none of "programming"). TF-IDF k=15: 12 of 15 boost-on vs 15 of 15 off. Top Docs k=12: 8 of 12 vs 11 of 12. Report top-docs k=30 with the question phrasing: 5 of 30 vs 20 of 30.
- Sensitivity (Tier-1 multiplier scaled, lower tiers scaled proportionally): BM25 top-12 phrase chunks = 12 / 9 / 9 / 5 / 0 at ×1.0 / 1.3 / 1.5 / 2.0 / 3.0; BM25 top-30 = 27 / 26 / 23 / 18 / 18; TF-IDF top-15 = 15 / 15 / 15 / 14 / 12.
- WHY: The boost is applied to scores that already encode relevance, so a weak Tier-1 match outranks a strong match from an unboosted document by construction. The documents that contain this phrase are all ×1.0 or ×1.3 (see M2), so for this term the boost acts as a filter against the right answer. The badge help text says the boost "does not eliminate other sources"; at k=12 it did.
- FIX (choose one; the first is the least invasive): (a) apply the boost after rank fusion as a rank bonus (for example RRF with the tier as a third ranked list, or `score + λ·log(mult)` on normalised scores) so it reorders among comparably relevant chunks instead of overriding relevance; (b) keep the multiplier but cap it at about 1.5 (the table above shows ×1.3–1.5 keeps 9 of 12 here); (c) apply the boost only within the top-N raw candidates (N = 4k, as Hybrid already pools) rather than over all 14,083. Whichever you pick, record the choice in `authority_tiers.yml` and in the badge help. Confidence high; measured.

**B2. P7/P2 · Keyword mode: OR-scored raw counts, and punctuation is never stripped · `streamlit_app.py:404-446` (terms at 411, scoring at 429); identical copy `bowen_rag_gui.py:465-509`**
- WHAT: (i) `raw_terms = remainder.split()` keeps punctuation, so in "What is emotional programming?" the term is `programming?` and `_stems` yields `['programming?']`; it matches nothing, and the whole query reduces to counting "emotional". Result: 0 of 15 with or without boost for the question form. (ii) Unquoted terms are summed as raw occurrence counts with no document-frequency weighting and no requirement that all terms be present, so a word that is common across the corpus dominates: "emotional" occurs 12,769 times, "programming" 171. Even for the bare query "emotional programming" with boost off, Keyword returns 4 of 15.
- WHY: The user guide (lines 65, 159, 239, 247) tells users to use Keyword mode to "verify whether the term appears in the corpus at all", and this mode cannot do that unless the phrase is quoted. A user who types the phrase unquoted gets a confident wrong answer.
- FIX: tokenise with `re.findall(r"[\w'-]+", remainder)`; either require every term (AND) or weight hits by inverse document frequency (BM25 already does this — Keyword could simply become "BM25 with quoted-phrase filtering"); and for an unquoted multi-word query, also try it as a phrase first and surface those hits on top. Quoted-phrase search (`tl.count(p)`) is sound and agrees with the literal prototype document-for-document (22, 18, 11 documents for three test phrases; for "pseudo-self" it is 48 vs 48 in substring mode and 48 vs 45 in the prototype's optional whole-word mode, which excludes "pseudo-selfs"). Confidence high; measured.

**B3. P6 (trusting a judgement without verifying the artefact) · Rule 7 delegates verbatim-presence checking to the model · `streamlit_app.py:157-166`; `bowen_rag_gui.py:188`; `USER_GUIDE.md:152-159, 247`**
- WHAT: The system prompt asks the model to "check whether the exact term or phrase from the user's query appears verbatim in the provided excerpts" and, if not, to open with `⚠️ Note: The exact term/phrase "…" does not appear in the provided sources`. The model can only see the k excerpts; it cannot know about the corpus. After B1/B2 have handed it zero containing chunks, the notice is correct about the excerpts and is read as a statement about the corpus. Separately, a language model is not a reliable substring matcher across 30 excerpts of ~1,500 characters (this part is my expectation, not something I measured in this review — see "not executed").
- WHY: This is the exact sentence the user reported. Rule 7 is the right instinct (make the inference visible) implemented in the wrong place (a probabilistic reader instead of a string scan). The project's own prime directive in `LEARNINGS.md` is "make negatives provable"; this negative is not.
- FIX: Compute it. `literal_search.LiteralIndex.corpus_check_line(phrase, provided_chunks)` returns, from a scan of the whole index, "the exact phrase "X" occurs N times in P passages across D documents in the indexed corpus; K of the excerpts provided below contain it verbatim" (or "does not occur anywhere in the indexed corpus"). Prepend it to the user message in Chat (`streamlit_app.py:1435-1438`), the Report prompt (`:1605`), and `bowen_ask.build_prompt`, and show the same line in the UI above the answer. Rewrite Rule 7 to: "A CORPUS CHECK line in the user message states, computed by the system, whether the exact phrase occurs in the corpus and in the excerpts. Begin your answer by restating it. Do not make your own claim about whether a term appears in the corpus." Confidence high for the mechanism; the model's own matching accuracy is unmeasured.

### Major

**M1. P3 (narrow scope) · nothing guarantees that containing chunks reach the model**
- WHAT: Every page takes the top-k of one ranking and sends it. When the phrase exists in the corpus but ranks below k, the model answers from non-containing text. At the defaults, that was the case for Chat (BM25 half: 0/12; Keyword 0/12) and in part for Report (top-docs, question phrasing: 5/30 boost-on).
- FIX: `LiteralIndex.ensure_literal_hits(results, phrase, reserve=5)` — if the phrase occurs anywhere, make sure at least `reserve` containing chunks (ranked by occurrence count, ties by authority) are in the set handed to the model, inserted at the front and trimmed from the tail so the count the user chose is unchanged. Measured: Chat BM25 0/12 → 5/12; Chat Keyword 0/12 → 5/12; Report top-docs 5/30 → 5/30 (already met); Search semantic 12/15 → 12/15. Phrase extraction: quoted phrases in the query if any, else the whole query (`extract_phrases`); for a long natural-language question the whole query will not match and the guarantee is a no-op, which is correct — it never inserts non-matching chunks. A later refinement would extract candidate noun phrases, but that reintroduces inference. Confidence high; measured.

**M2. DATA (editorial, please verify) · Kerr's "Bowen Theory Secrets" has no authority tier · `authority_tiers.yml`**
- WHAT: All 26 "Bowen Theory Secrets_*" documents in the index resolve to ×1.0 — no pattern in `authority_tiers.yml` matches them — while FSJ articles get ×1.3. If this book is Michael Kerr's *Bowen Theory's Secrets* (2019), it is a primary Kerr source under your own tiering, and for this phrase it holds 25 of the 66 containing chunks, including the chapter named for the term. I have not confirmed the attribution from the files; the decision is yours. Note that giving it ×3 would not by itself fix B1 for terms that live in other unboosted documents.

**M3. TEST · no test asks whether retrieval returns the phrase it was asked for · `test_semantic_search.py`, `test_index_consistency.py`**
- WHAT: The retrieval tests check that the index loads, that the files agree, and that a six-document toy corpus ranks by topic. Nothing tests, against the shipped index, "for each of these known Bowen terms, does the default pipeline hand the model at least one chunk containing it?" That test fails today for "emotional programming" at the Chat defaults.
- FIX: a regression test (skip when the index files are absent, as `test_index_consistency` already does) over a YAML list of terms with their expected minimum document counts — the 19 phrases in `measure_literal.py` are a start, with counts from the current index (substring counts: `emotional contact` 181 chunks / 72 docs, `emotional cutoff` 196 / 74, `nodal events` 36 / 18, `emotional shock wave` 20 / 11, and so on). Assert (a) `LiteralIndex.presence(term).documents >= expected` (catches index regressions such as the ligature case already in the fix log), and (b) for each page's default mode and k, at least one returned chunk contains the term (catches B1/B2-type ranking regressions). Run it in CI.

### Minor

**m1.** `hybrid_search` pools `4k` candidates from each side (`:482`); with k=12 the pool is 48 and the best phrase chunk sits at BM25 rank 25 boosted. Measured after the addendum: it does not survive fusion — Hybrid boost-on returns 0 of 12. With the boost off the same fusion returns 12 of 12, so the loss is the boost (B1), not RRF.

**m2.** `keyword_search` returns one chunk per document (`doc_best`), so "Chunks = 12" on the Chat page means 12 documents in Keyword mode, each represented by its highest-count chunk. The guide does not say so.

**m3.** `USER_GUIDE.md:154` describes the ⚠️ notice as evidence that the term "does not appear verbatim in the retrieved sources" and `:247` as "The exact phrase is not in the retrieved sources"; both are accurate about the excerpts, but the surrounding text sends the user to Keyword mode to check the corpus, which (B2) only works quoted. Rewrite once the corpus-check line exists.

## The literal-search feature

### Why a separate mode rather than fixing the rankers

The three existing families answer "what is most similar to this query?" (TF-IDF and BM25 by weighted term overlap, embeddings by meaning). None of them can answer "does this string occur, and where?", and none can prove a negative, because each returns a top-k of a ranking rather than a membership set. Quoted Keyword mode is already a literal filter, but it is hidden behind a syntax, one chunk per document, and it is combined with raw-count scoring. A dedicated exact-match mode that returns every containing chunk, with counts, is the only way to make "the corpus does/does not contain X" a statement that is true by construction. It also gives the LLM pipeline a fact to repeat instead of a judgement to make.

### What is delivered

`literal_search.py` (standard library only; no Streamlit or tkinter import) and `test_literal_search.py` (17 tests, all passing). Measured on the shipped index: building the normalised copy of all 14,083 chunks takes about 1.0 s at load; a phrase query takes about 17–45 ms in the default substring mode and 70–160 ms with `whole_words=True` (a regex scan of ~17 million characters). API:

- `normalize(text)` — lower-case, NFKC (ﬁ → fi), soft hyphens removed, "word-\nword" re-joined, curly quotes and dashes mapped to ASCII, whitespace runs collapsed. Applied identically to chunks and query, so a phrase matches across a line break or a PDF soft-wrap. Note: the index already has ligatures cleaned at build time (fix log 2026-10-02), so NFKC is belt-and-braces there; the whitespace collapse matters (84,388 newlines inside chunk texts).
- `LiteralIndex(chunks).search(phrase, top_k=None, whole_words=False)` — every containing chunk as `Hit(chunk, occurrences, first_pos, snippet)`, sorted by occurrences then (optionally) authority then document order. Substring is the default on purpose: the real corpus has "thatemotional programming" (a lost space) and "pseudo-selfs"; a presence check must not call those absent. `whole_words=True` is available for the strict reading.
- `.search_regex(pattern)` — for variants (`emotional(ly)? programm\w*`).
- `.presence(phrase)` → occurrences, chunks, documents, document names.
- `.corpus_check_line(phrase, provided)` — the sentence for the prompt and the UI (B3).
- `.ensure_literal_hits(results, phrase, reserve)` — the retrieval guarantee (M1).
- `extract_phrases(query)` — quoted phrases, else the whole query.
- `highlight(text, phrase)` — for result cards and the View dialog.
- CLI: `python3 literal_search.py "emotional programming"` prints the counts and the top passages with locators; exit status 1 when absent, so it can be used in scripts and in the regression test.

### Integration points (not applied; line numbers at c5ac598)

1. `IndexManager.load` (`streamlit_app.py:280`; GUI `bowen_rag_gui.py:270`): `self.literal = literal_search.LiteralIndex(self.chunks)` after `self.chunks` is loaded. ~1 s, once, under `@st.cache_resource`.
2. Search page (`:1105-1114`): add `("Exact phrase", "literal")`; in the dispatcher (`:1168-1180`) `results = [{**h.chunk, "score": h.occurrences, "score_label": f"{h.occurrences} exact", "mode": "literal"} for h in idx.literal.search(query, top_k=top_k)]`; show the `presence()` counts as the caption ("93 occurrences in 66 passages across 22 documents — showing 15") so the user sees the whole-corpus answer even when k truncates the list; highlight the phrase in `_result_card` and the View dialog. Same three edits in Chat (`:1319`, `:1396-1408`) and Report (`:1517`, `_gather_chunks :830-841`), and in `bowen_ask.py`'s mode switch.
3. Chat, Report and `bowen_ask`: after retrieval and before building context, `for p in literal_search.extract_phrases(query): chunks = idx.literal.ensure_literal_hits(chunks, p, reserve=5, boost_fn=authority_boost)`; then `check = "\n".join(idx.literal.corpus_check_line(p, chunks) for p in phrases)`; put `check` at the top of `current_content` (`:1435`) and of the Report prompt (`:1605`), and `st.caption(check)` above the streamed answer. Add a "Reserve exact-match passages" number input (default 5, 0 to disable) next to "Chunks".
4. `SYSTEM_PROMPT` Rule 7 (`:157-166`, GUI `:188`): replace the model-side check with the restate-the-CORPUS-CHECK rule quoted in B3. Keep the sentence about "emotional contact" vs "emotional connection" as the rationale.
5. Keyword mode (`:404-446`, GUI `:465-509`): the tokeniser and AND/IDF fix from B2, or fold Keyword into BM25-plus-quoted-filter.
6. Boost (`:357, 462, 517, 547` and GUI): one of the three options in B1; update `authority_tiers.yml` comments and the badge help in four places (the r2 review already lists them).
7. Tests: `test_literal_search.py` into the unittest line in `CLAUDE.md`; the regression test from M3; `test_index_consistency.py` can assert that `LiteralIndex` is built from the same chunk list the fingerprint covers.
8. Docs: `USER_GUIDE.md` search-mode table and the "Terminology precision warning" section; `CLAUDE.md` search-modes table.

### What this does not solve

A literal search is exact by definition: it does not find "emotionally programmed" when asked for "emotional programming" (5 chunks here; `search_regex` or a second quoted search does), nor British/American spellings, nor the concept under another name. That is the job of the semantic modes, and the right UI is to show both answers side by side, not to pick one. The retrieval guarantee only fires for phrases the user quoted or typed as the whole query; a question in prose gets the corpus-check line (whole-query, which will say "does not occur") but no reserved chunks, and that line should be worded so a user does not read a non-match of a whole sentence as a non-match of its key term — the prototype wording already names the string it checked. Finally, nothing here checks the model's *quotations* against the chunks (open item in the fix log, 2026-10-02); the same `LiteralIndex` is the natural place to do that after the stream ends.

## What would have caught it

A test that asks the shipped pipeline, for a list of terms that are certainly in the corpus, whether at least one returned chunk contains the term (M3); and a rule that any "absent" statement shown to the user is produced by code that looked at the whole corpus, never by the model (B3). Both are now cheap: the first is a loop over `presence()` and the page defaults; the second is one prepended line.

## Addendum 2026-11-01 (b) — LLM presence experiment, run locally against DeepSeek

Command: `python3 review_2026-11-01/llm_presence_experiment.py --provider deepseek --reps 2 --words 500`, run on davemini2 from `/Volumes/CrucialX9/a Downloads/bowen_rag` at `50da727` (app code identical to `c5ac598`; unchanged `SYSTEM_PROMPT`, Rule 7 as shipped). Model `deepseek-v4-flash`, 9 calls, exit 0. Query: "What is emotional programming?". Per-call figures are in `review_2026-11-01/llm_presence_results.json`; the nine replies are in `review_2026-11-01/llm_presence_responses/` (not committed: they quote the corpus at length, and the corpus is kept out of git).

Printed table:

```
chat/hybrid/k12/boost-on  (app default)    rep1  sent=12 containing= 0 occ=  0  claims_absent=True  ⚠=True   -> correct absent (nothing containing it was sent)
chat/hybrid/k12/boost-on  (app default)    rep2  sent=12 containing= 0 occ=  0  claims_absent=True  ⚠=True   -> correct absent (nothing containing it was sent)
chat/hybrid/k12/boost-off                  rep1  sent=12 containing=12 occ= 34  claims_absent=False ⚠=False  -> no absence claim; phrase was sent
chat/hybrid/k12/boost-off                  rep2  sent=12 containing=12 occ= 34  claims_absent=False ⚠=False  -> no absence claim; phrase was sent
chat/semantic/k12/boost-on                 rep1  sent=12 containing= 9 occ= 28  claims_absent=False ⚠=False  -> no absence claim; phrase was sent
chat/semantic/k12/boost-on                 rep2  sent=12 containing= 9 occ= 28  claims_absent=False ⚠=False  -> no absence claim; phrase was sent
report/top-docs/k30/boost-on (default)     rep1  sent=30 containing=17 occ= 28  claims_absent=False ⚠=False  -> no absence claim; phrase was sent
report/hybrid/k30/boost-on                 rep1  sent=30 containing= 3 occ= 11  claims_absent=False ⚠=True   -> no absence claim; phrase was sent
report/hybrid/k30/boost-off                rep1  sent=30 containing=28 occ= 52  claims_absent=False ⚠=False  -> no absence claim; phrase was sent
```

The containing-chunk counts (0, 12, 9, 17, 3, 28) equal the expected ground truth for the six conditions.

Per condition, read from the reply files (a: ⚠ Rule-7 notice printed; b: claims the term is absent; c: claim true of the excerpts sent; d: absence worded about "the corpus"/"the sources" rather than "the excerpts provided"):

| # | Condition | containing | (a) ⚠ | (b) absent claim | (c) true of excerpts | (d) worded as corpus/sources |
|---|---|---|---|---|---|---|
| 1 | chat/hybrid/k12/boost-on (default), rep 1 | 0 | yes | yes | yes | yes — "does not appear in the provided **sources**" (the Rule-7 template), then "The sources use the word 'programming' … without defining it as a formal concept called 'emotional programming'" |
| 1 | same, rep 2 | 0 | yes | yes | yes | yes — the same template sentence; the body then says "The provided excerpts use … 'programming' without the qualifier 'emotional'" |
| 2 | chat/hybrid/k12/boost-off, reps 1–2 | 12 | no | no | — | — |
| 3 | chat/semantic/k12/boost-on, reps 1–2 | 9 | no | no | — | — |
| 4 | report/top-docs/k30/boost-on (default) | 17 | no | no | — | — |
| 5 | report/hybrid/k30/boost-on | 3 | yes, but inverted: "The exact phrase 'emotional programming' **does** appear verbatim in the provided sources [[27, 28, 30]]" | no | — (the positive claim is true: the three cited numbers are where the phrase was) | — |
| 6 | report/hybrid/k30/boost-off | 28 | no | no | — | — |

The two questions:

1. **Condition 1 (default Chat, 0 containing chunks) reproduces the reported failure in both repetitions.** The model prints the Rule-7 notice "The exact term/phrase "emotional programming" does not appear in the provided sources", which is true of the 12 excerpts and false of the corpus (22 documents). The wording is the system prompt's own: "provided sources", not "corpus". The word "corpus" did not occur in any reply (`absence_worded_as_corpus` false for all 9). A reader cannot tell "the 12 excerpts I was given" from "the sources" — this is the defect, and it comes from the template text, not from model error.
2. **The model never claimed absence when the phrase was in its excerpts** (0 of 7 replies in conditions 2–6). In condition 5, with only 3 of 30 excerpts containing it, it located those three and cited them by number. In the 7 replies that had the phrase, Rule-7 matching was correct every time.

What this changes: B3's mechanism is confirmed (a true statement about the excerpts is read as a statement about the corpus); B3's secondary concern, that the model is an unreliable substring matcher, was not observed at this sample size (7 replies, one model). The fix order stands — retrieval (B1, M1) decides what the model sees, and the absence wording must come from code that scanned the whole index (E8). Seven replies do not establish reliability; the pipeline still must not depend on it.

## Addendum 2026-11-01 (c) — experiment re-run after Tier 2 (exact-match guard, new Rule 7)

`llm_presence_experiment.py --provider deepseek --reps 1 --words 500`, now building messages with the app's pipeline (`exact_guard`, `build_chat_message`, `build_report_prompt`). Results in `review_2026-11-01/llm_presence_results_tier2.json` and `…_tier2_quoted.json`; replies in `llm_presence_responses_tier2*/` (not committed: they quote the corpus). "sent containing" counts chunks, "occ" counts occurrences in the excerpt text sent (Chat includes the ±1 neighbours). The script's verdict column is a regex; every match was read by hand (last column).

Default question `What is emotional programming?`:

| Condition | sent containing | occ | regex verdict | hand reading |
|---|---|---|---|---|
| chat hybrid k12 boost on (app default) | 0 | 0 | absent (correct) | "does not appear in the excerpts provided" — true; no corpus claim |
| chat hybrid k12 boost off | 12 | 34 | no claim | — |
| chat semantic k12 boost on | 9 | 28 | no claim | — |
| report top-docs k30 boost on (default) | 17 | 27 | no claim | — |
| report hybrid k30 boost on | 3 | 10 | false negative | "do not contain a single, formal, standalone definition" — not an absence claim |
| report hybrid k30 boost off | 28 | 51 | false negative | says the whole question "What is emotional programming" is not in the excerpts — true |

Condition 1 still sends 0 containing chunks: E6a checks a short unquoted question as one phrase, "What is emotional programming", which occurs nowhere, so the union does not fire. This is a spec gap (see the Tier 2 report), not a code fault.

Quoted question `What is "emotional programming"?`:

| Condition | sent containing | occ | regex verdict | hand reading |
|---|---|---|---|---|
| chat hybrid k12 boost on (app default) | 5 | 20 | false negative | "do not contain a single formal definition sentence" — not an absence claim; reply opens with the block's count (91 / 65 / 22, 6 of 32 excerpts) |
| chat hybrid k12 boost off | 12 | 34 | no claim | — |
| chat semantic k12 boost on | 9 | 28 | no claim | — |
| report top-docs k30 boost on (default) | 17 | 27 | false negative | "do not contain information on these points" (named gaps) — not an absence claim |
| report hybrid k30 boost on | 5 | 18 | no claim | — |
| report hybrid k30 boost off | 27 | 49 | false negative | "do not contain a single agreed formal definition" — not an absence claim |

Result: with the term quoted, the app default Chat condition now sends 5 containing chunks (was 0) and the model makes no absence claim. In 12 replies, no reply claims the term is absent from the corpus or from excerpts that contain it. One rep per condition; this is a spot check, not a rate.
