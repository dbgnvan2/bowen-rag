# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Global standards

Read the relevant file from `~/.claude/standards/` before starting work:

| Standard | When |
|---|---|
| `learnings.md` | Any RAG pipeline, search, scoring, or report code — P1–P10 checklist |
| `llm-integration.md` | Any LLM call, prompt building, model selection, or output parsing |
| `external-api.md` | Any HTTP call to LLM providers or external APIs |
| `security.md` | API keys in `.env`, Railway env vars, input validation — deployed app, higher stakes |
| `file-maintainability.md` | Any new module or significant refactor |



**User-facing documentation:** [`USER_GUIDE.md`](USER_GUIDE.md) — keep it current whenever features, search modes, score badges, or admin workflows change.

## Setup

```bash
pip install -r requirements.txt
```

`requirements-build.txt` (adds PyMuPDF, AGPL-licensed) is needed only to *build* the index from PDFs and to run the book/issue splitters; the deployed apps do not need it. `sentence-transformers` is included in requirements and enables Embedding and Hybrid search modes. It pulls in PyTorch (~500 MB one-time download). The app runs without it — those search options are hidden if the import fails.

`python-dotenv` is included and is used by both apps to load `.env` at startup.

`python-docx` and `reportlab` are included for Word (.docx) and PDF (.pdf) report export. If either import fails, the corresponding download/save format is unavailable but the rest of the app still works.

## Environment / API keys

Copy `.env.example` to `.env` and fill in your keys. Both apps read this file at startup:

```bash
cp .env.example .env
```

Supported variables:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `LLM_PROVIDER` | Default provider: `claude`, `openai`, `deepseek`, or `ollama` |
| `ANTHROPIC_MODEL` | Default Claude model (e.g. `claude-sonnet-4-6`) |
| `OPENAI_MODEL` | Default OpenAI model |
| `DEEPSEEK_MODEL` | Default DeepSeek model (default: `deepseek-v4-flash`) |
| `OLLAMA_MODEL` | Default Ollama model |
| `OLLAMA_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `CLAUDE_EXTRA_MODELS` | Comma-separated extra Claude model IDs added to the dropdown |
| `OPENAI_EXTRA_MODELS` | Comma-separated extra OpenAI model IDs added to the dropdown |

The recommended default is `LLM_PROVIDER=deepseek` (cheapest, good quality).

## Running the desktop GUI

```bash
./bowen_rag.sh          # launches tkinter GUI (starts Ollama if not running)
python3 bowen_rag_gui.py  # launch directly without the shell wrapper
```

## Running the Streamlit web app (local)

```bash
./bowen_rag_web.sh      # starts Ollama if needed, then runs Streamlit on port 8501
streamlit run streamlit_app.py  # launch directly
```

The web app is available at `http://localhost:8501`.

## Railway deployment

The Streamlit app is deployed to Railway. Push to `main` on GitHub triggers an automatic redeploy (usually 2–3 minutes).

**Required Railway environment variables:**
- `LLM_PROVIDER` — e.g. `deepseek`
- `DEEPSEEK_API_KEY` — your DeepSeek key
- `APP_PASSWORD` — optional; if set, users must enter this password to access the app
- `DAILY_TOKEN_CAP_PER_USER` (default 300000), `DAILY_TOKEN_CAP_GLOBAL` (default 2000000), `REPORT_MIN_BUDGET` (default 60000), `USAGE_DIR` — DeepSeek daily token limits, see below

**Daily token limits** (`usage_limit.py`, spec `docs/spec_usage_limits.md`, tests `test_usage_limit.py` and `test_llm_stream_limits.py`): the web app has no logins, so a visitor is a `bowen_uid` browser cookie. Each visitor gets 300K tokens per UTC day and all visitors share a 2M global cap. The budget is in *weighted* tokens so it covers every provider that uses a server-side key (Claude, OpenAI, DeepSeek; Ollama is not metered): a model's real tokens are multiplied by its weight in `model_weights.yml` (deepseek-v4-flash = 1, unlisted models use `default_weight`, 10 — placeholders to set from the providers' prices, not prices; until you do, any model other than deepseek-v4-flash needs about 160K weighted tokens in reserve for even a short chat and cannot run a full report). A call is refused *before* any API request if its estimated cost ((input estimate + 16K output floor) × weight) does not fit the visitor's or the global remaining budget, or `REPORT_MIN_BUDGET` for a Report; the reservation is released and real usage recorded when the stream ends (estimated, with a logged warning, if the API gives none). Clearing cookies resets a visitor's own budget and the cookie is not signed — the global cap is the spend backstop, and a script minting fresh ids can exhaust it. Counters persist best-effort to `USAGE_DIR`; Railway's filesystem is ephemeral, so mount a volume and point `USAGE_DIR` at it, otherwise counts reset on every redeploy.

The `Procfile` tells Railway how to start the app:
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --logger.level=warning
```

## Rebuilding the search index

```bash
python3 rag-document-search/scripts/build_index.py source_files/ rag-document-search/references/
```

Run this after adding or changing documents in `source_files/` (needs `pip install -r requirements-build.txt`). The builder is paragraph-aligned and metadata-enriched; see *Chunking strategy*. It reports instead of skipping: files with no extractable text (for example scanned PDFs with no text layer — see *OCR of scanned PDFs* below), duplicate document names, sectioned transcripts with empty sections, and documents with no metadata are all listed, and a malformed `chapter_map.yml`/`headers_candidates.yml` or a missing PDF library is an error. The TF-IDF settings are defined in `TFIDF_PARAMS` in `build_index.py`; the CLI search imports them, and the two apps repeat the values when they re-fit the vectorizer on load — `test_index_consistency.py` fails if the copies drift. Both apps build their query vectorizer on the saved feature list in `vectorizer.json` (`vocabulary=`) instead of re-choosing 8,000 features: 295 terms tie at the cutoff and only 5 are kept, so a fresh `max_features` fit can pick different words on another CPU/numpy build and broke the Railway deploy. `vectorizer.json` also stores a hash of the chunk text (`chunks_sha256`). Both apps also refuse to load an index whose matrix, chunk metadata and `vectorizer.json` disagree (no silent fallback to an old dense `tfidf_matrix.npy`). The script writes three files to `references/`: `chunk_metadata.json`, `tfidf_matrix.npz`, and `vectorizer.json` (all three are written under temporary names and renamed together).

**After rebuilding, always rebuild the embedding index too** (`python3 rag-document-search/scripts/build_embeddings.py`, or the GUI) — the chunk count changes and a stale `embed_matrix.npy` makes Embedding/Hybrid search raise on every query.

## Building the embedding index

After rebuilding the TF-IDF index, rebuild the embedding index. You can do this via the GUI (**Index tab → Build Embeddings**) or directly:

```python
python3 - <<'EOF'
import json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
REFS = Path("rag-document-search/references")
chunks = json.load(open(REFS / "chunk_metadata.json"))
model = SentenceTransformer("all-MiniLM-L6-v2")
vecs = model.encode([c["text"] for c in chunks], show_progress_bar=True, batch_size=64, convert_to_numpy=True)
np.save(str(REFS / "embed_matrix.npy"), vecs)
print(f"Saved {len(vecs):,} embeddings")
EOF
```

This encodes all chunks with `all-MiniLM-L6-v2` and saves `embed_matrix.npy` alongside the TF-IDF files. First run downloads the model (~90 MB to `~/.cache/huggingface/`). Expect a few minutes on CPU for a large corpus.

`build_embeddings.py` (and the GUI's Build Embeddings) also write `embed_meta.json` — a hash of the chunk text the embeddings were built from. Both apps check it on load and refuse an `embed_matrix.npy` that was built from different text, even when the chunk count happens to match (`index_fingerprint.py`; an index with no `embed_meta.json` is still accepted).

The embedding index is required for Embedding and Hybrid search modes. It is loaded automatically on startup if the file exists. `embed_matrix.npy` (~16 MB) is committed to the repo so Railway gets it on deploy.

## OCR of scanned PDFs

Scans with no text layer cannot be searched or cited. `rag-document-search/scripts/ocr_scans.py` reads them with tesseract (install: `brew install tesseract`; also `requirements-build.txt`) and writes one `.txt` per PDF plus `ocr_manifest.yml`:

```bash
python3 rag-document-search/scripts/ocr_scans.py source_files source_files_ocr --manifest ocr_manifest.yml
cp source_files_ocr/*.txt source_files/        # the PDFs stay; a .txt wins over a PDF with the same name
```

Why not plain `ocrmypdf`: the scans are often a *spread* — two facing book pages on one sheet, rotated 90°, each page skewed differently — and whole-sheet OCR interleaves and garbles them. The tool picks each sheet's orientation (tesseract also reads a sheet turned one way, so the rule is: when two adjacent rotations both read, the upright one is the later of the pair — `choose_orientation`, tested on real score tables), splits a spread at the gutter, deskews each page separately, rebuilds paragraphs from tesseract's layout data and removes line-end hyphens. `ocr_manifest.yml` records each document's mean word confidence, the pages below 70, the engine and the date. `build_index.py` marks those documents' chunks `ocr: true` (with the confidence) and the web app shows an **OCR** badge on search result cards, a warning in the View dialog and "(scan, OCR)" in Chat sources; the web Report page and `bowen_ask.py` end with "N of M cited passages come from scanned documents read by OCR…". **The desktop GUI does not show the flag yet** (its Report tab is the old pipeline). OCR chunks have no PDF page number (`page` is null; the locator is the paragraph number in the OCR text), so a quotation cannot be mapped to a page automatically. `source_files/` and `source_files_ocr/` are not in git: keep the OCR `.txt` files with the corpus (the drive copy); on a checkout without them `build_index.py` warns that the manifest's documents have no `.txt` and leaves them out. Two scans (Freeman "Developmental Challenges", Nel "Theology Triangels") are blank page images and are not indexed. OCR is a machine reading: **spot-check a document against its page images before relying on a quotation from it**, and re-run `extract_headers.py` (below) after adding documents.

`build_index.py` also cleans extraction artefacts when it reads a document (`clean_text`: ﬁ/ﬂ ligatures become letters, soft hyphens inside words are removed and the word re-joined) — a ligature made "deﬁne" unfindable and printed as a wrong glyph in the PDF report. `extract_headers.py` takes a periodical's year from its dateline ("SPRING 1989") before any year mentioned in the text; a mis-printed or mis-read dateline still gives a wrong year (Family Center Reports Vol 6 No 3 reads "1981", probably 1984), so years stay unverified. `extract_headers.py` regenerates `headers_candidates.yml` from scratch, so never hand-edit that file: fix the cause instead (the year rule in the script, `author_map.yml`), and `test_extract_headers.py` / `test_sources_data.py` fail if a known correction is lost.

## Processing transcripts

`process_transcripts.py` reads `*yaml.md` files from `~/transcripts/projects/` (recursively), strips YAML frontmatter, and writes clean `.txt` files to `source_files/`. Files without `## Section N –` headings are silently skipped.

```bash
python3 process_transcripts.py                        # default source and output dirs
python3 process_transcripts.py --dry-run              # preview what would be written
python3 process_transcripts.py --transcripts-dir DIR  # override transcript source
python3 process_transcripts.py --source-dir DIR       # override output destination
```

After importing, rebuild the index (the script prints the command as a reminder).

## Running evals / tests

```bash
cd rag-document-search
python3 test_skill.py   # runs 3 sample queries; writes results to test_results.json
```

`rag-document-search/scripts/semantic_search.py` is a plain TF-IDF command-line search (`python3 scripts/semantic_search.py references/ "query" 5`). It stops with a "rebuild the index" error if the index files disagree. It has no authority boost and no embedding/hybrid mode, so its ranking differs from the apps.

The unit tests run with `python3 -m unittest test_citations test_seed_sources test_sources_data test_report_export test_usage_limit test_llm_stream_limits test_build_index test_semantic_search test_index_consistency test_ocr_scans test_extract_headers` (the original citations tests also run with `python3 test_citations.py`).

## Building the macOS app

CI (GitHub Actions) builds `.app` bundles via PyInstaller on push to a `v*` tag — one for `arm64` (macos-14) and one for `x86_64` (macos-13). To build locally:

```bash
pip install -r requirements.txt pyinstaller
pyinstaller --name "Bowen RAG" --windowed --onedir \
  --add-data "rag-document-search/scripts:rag-document-search/scripts" \
  --add-data "rag-document-search/references:rag-document-search/references" \
  --add-data "process_transcripts.py:." \
  --hidden-import scipy.sparse \
  --hidden-import sklearn.feature_extraction.text \
  bowen_rag_gui.py
```

Note: including `sentence-transformers` in a PyInstaller bundle significantly increases bundle size (~500 MB+). Consider excluding it for distribution builds.

## Architecture

There are two front-ends that share the same `IndexManager` backend logic:

### Desktop GUI (`bowen_rag_gui.py`, ~2300 lines)

Built with tkinter. Four classes:

- **`IndexManager`** — loads the prebuilt TF-IDF index from `rag-document-search/references/` at startup. Exposes `semantic_search`, `keyword_search`, `combined_search`, `top_docs_search`, `embedding_search`, `bm25_search`, `hybrid_search`, and `build_embeddings`. Also handles rebuilding the index in a background thread.

- **`LLMClient`** — static methods for calling Claude (`anthropic` SDK), OpenAI, DeepSeek (Anthropic SDK with custom `base_url`), and Ollama. Streaming is supported for all four. Single-turn methods are used by the Report tab; multi-turn chat methods accept a full `messages` list and are used by the Chat tab.

- **`App`** — main window. Five notebook tabs:
  - **Search** — query + ranked results with checkboxes to stage excerpts for the report
  - **Index** — rebuild TF-IDF index, import transcripts, build embedding index
  - **LLM Settings** — provider config (Claude / OpenAI / DeepSeek / Ollama), model selection, system prompt editor, connection test, Save to .env
  - **Report Generator** — one-shot report from staged or freshly retrieved chunks; cites sources by reference number; authority boost toggle; chunk audit expander
  - **Chat** — multi-turn conversational interface; each turn retrieves fresh chunks, history carries only bare Q&A (not chunks); sources shown in expander

- **`Tooltip`** — click-to-show helper widget, used on `?` buttons.

### Streamlit web app (`streamlit_app.py`, ~1400 lines)

Designed for Railway deployment and browser access. Uses `@st.cache_resource` for the shared index and `st.session_state` for per-user state. Contains a copy of `IndexManager` (no tkinter dependency) and a `_llm_stream` generator for streaming LLM responses.

Five pages (sidebar navigation, each with a `?` help button):
- **Search** — query + ranked results with checkboxes; stage selected chunks for Report; each result has a **View ↗** button that opens a formatted modal dialog with the full section text
- **Chat** — conversational Q&A; compact single-line control bar; sources expander includes **View ↗** per source
- **Report** — generate a cited report; audit chunks used; optional **Include sources as Appendix** checkbox appends full source texts to the report and download. Three download formats: Markdown (.md), Word (.docx), PDF (.pdf).
- **Index** — admin: index statistics and document list
- **Settings** — LLM provider/key/model config; default search mode; system prompt; connection test

API keys in Settings are masked — only the last 6 characters are shown.

Key helper functions in `streamlit_app.py`:

| Function | Purpose |
|---|---|
| `_format_chunk_text(text)` | Strips `[Section Title]` prefix, collapses PDF soft-wrap newlines, normalises whitespace — used by the View dialog and appendix |
| `_show_section_dialog()` | `@st.dialog` modal — shows doc name, section title, page/position caption and formatted full text |
| `_result_card(result, key)` | Renders a single search result card with score/author/page badges, excerpt, and View ↗ button |
| `md_to_docx_bytes(md, title)` | Converts markdown report to a .docx file (returns bytes). Handles `#`/`##`/`###` headings, `**bold**`, `*italic*`, bullet/numbered lists, horizontal rules. Used by the Word download button. |
| `md_to_pdf_bytes(md, title)` | Converts markdown report to a PDF (returns bytes) using reportlab. Same markdown subset as the docx converter. Used by the PDF download button. |

### Search modes

| Mode | Method | Description |
|---|---|---|
| Top Docs | `top_docs_search` | Aggregates chunk scores per document; best single chunk per doc returned. Best for most queries. |
| Semantic (TF-IDF) | `semantic_search` | Cosine similarity on TF-IDF vectors. Fast, exact-vocabulary matching. |
| Keyword | `keyword_search` | Counts exact word matches with simple stemming. Good for names and specific terms. |
| Both | `combined_search` | Merges semantic and keyword results. |
| Embedding | `embedding_search` | Cosine similarity on sentence-transformer vectors. Finds conceptual matches regardless of exact wording. Requires `embed_matrix.npy`. |
| Hybrid | `hybrid_search` | Reciprocal Rank Fusion (RRF, K=60) of BM25 and Embedding results. Best overall quality. Requires `embed_matrix.npy`. |

### Search ranking

Raw similarity scores are boosted by `authority_boost()` before ranking. The multipliers are defined in `AUTHORITY_TIERS` (top of `bowen_rag_gui.py` and `streamlit_app.py`, also overridable via `authority_tiers.yml`) — primary Bowen/Kerr sources get 3.0×, Family Systems Journal articles 1.3×, other named theorists 1.15×. This is editorial content; keep it in the config files rather than hardcoding elsewhere.

### DeepSeek integration

DeepSeek uses the Anthropic SDK with a custom `base_url`:

```python
client = anthropic.Anthropic(api_key=key, base_url="https://api.deepseek.com/anthropic")
```

Available models: `deepseek-v4-flash` (default, fast/cheap), `deepseek-v4-pro`.

**Reasoning-model gotcha:** `deepseek-v4-flash` emits a large `thinking` block that counts against `max_tokens`. With a low ceiling (was 8000), a full-size report's thinking could consume the entire budget, leaving no visible text — the report came back as references only. All DeepSeek calls use `max_tokens=32000` to leave room for thinking + a complete report. Also, the model's first content block is the thinking block, so non-streaming reads must select the `text` block (`b.type == "text"`), not `content[0]`. Because the thinking pass runs before any text streams, the report UI shows a spinner/"analyzing sources" status during the initial silent phase, and fails loud (not references-only) if the text ever comes back empty.

### Chat context strategy

The Chat tab keeps conversation history as bare Q&A pairs — the user's question and the assistant's answer only. Retrieved source chunks are included only for the current turn and are not stored in history. This keeps context size flat regardless of conversation length while preserving conversational continuity. Each source in the sources expander has a **View ↗** button to open the full section text in a modal.

### Report citation format & citation styles

**The LLM never writes a citation.** It emits **double-bracket** numbered placeholders `[[1]]`, `[[2]]` (grouped `[[1, 3]]`; the web Report page and `bowen_ask.py` tell the model not to put page numbers inside the brackets — each numbered source is one chunk with its own locator, which goes in the reference list). Double brackets are deliberate: a single-bracket `[1]` can appear inside quoted source text (the corpus has footnote numbers in passages the report may quote verbatim), and a single-bracket scheme would rewrite those into wrong citations — so citations use `[[…]]`, which can't collide. After the stream completes, a deterministic post-processor in `citations.py` (`apply_intext_citations`) rewrites those markers into the user's chosen style and builds the reference list; grouped markers are expanded and joined. This preserves the no-fabrication guarantee — the model cannot invent an author, year, or publisher because it never writes one — and keeps the reference list the single source of truth (the LLM is still told NOT to output a References section).

Post-processing is guarded in both apps: a malformed hand-edited `sources.yml` record degrades to plain numbered references instead of crashing the report. (The desktop GUI's Report tab still uses the older per-document pipeline — one source per document, a page locator offered only when its retrieved chunks resolve to a single page — so its reports are structured differently from the web app's and `bowen_ask.py`'s.)

**Selectable styles** (Settings → Citations in Streamlit; Report tab in the desktop GUI; default from `CITATION_STYLE` env var, else Vancouver):

| Style | In-text | Reference list |
|---|---|---|
| APA, Harvard | `(Bowen, 1978, p. 45)` | alphabetical by author |
| MLA | `(Bowen 45)` | alphabetical by author |
| Chicago (author–date) | `(Bowen 1978, 45)` | alphabetical by author |
| Vancouver | `[1]` | numbered, citation order |

In the desktop GUI, for author–date/author–page styles the reference list contains **only the works actually cited** (parsed from the report body via `citations.cited_numbers`); Vancouver keeps the numbered markers. **In the web Report page and `bowen_ask.py` the style applies to the in-text markers only**; the reference list is always the traceability format `Author (Year) *Title* [Ch. N ¶ M]. In *Container*.`, one entry per cited chunk in number order, followed by a footer saying how many of those passages rest on unverified bibliographic data (`citations.unverified_footer`).

**`citations.py`** (shared by both apps, no tkinter/streamlit deps): `load_sources` / `match_source` (longest-substring-match wins), `record_for_doc` (sources.yml record or a filename fallback), `format_reference`, `format_intext`, `order_references`, `build_reference_list_md`, `apply_intext_citations`, `cited_numbers`. Missing fields render with each style's real convention (APA `n.d.`); nothing is invented. Known simplifications (Chicago = author–date not footnotes; Harvard = "Cite Them Right"; et-al. thresholds) are documented in the module docstring. Tests: `test_citations.py`.

**`sources.yml`** — bibliographic records (author/year/title/publisher/journal/volume/pages/type), pattern-matched to documents like `author_map.yml`. Editable in the app via **Settings → Citations → "Edit bibliographic records"** (`_sources_editor` in `streamlit_app.py`): a structured per-source form with a live reference preview that writes via `citations.dump_sources` and updates `st.session_state.sources` so reports reflect edits immediately (the report reads `st.session_state.get("sources") or SOURCES`). Save persists locally; on Railway (ephemeral FS) use Download + commit. Created by **`seed_sources.py`** (seeds author from `author_map.yml`, year from filename digits, title from cleaned filename; writes `sources.seeded.yml`, refuses to overwrite `sources.yml`; every field `verified: false` until a human confirms it). Editorial content — keep it in the YAML file, not code. Tests: `test_seed_sources.py`.

**Chunk-level references and who wins.** Reports (Streamlit Report page and the headless `bowen_ask.py`) cite each retrieved *chunk*: `citations.record_from_chunk(chunk, sources)` builds the record, `passage_locator` the locator (`Ch. 4 ¶ 12–14`, `¶ 12`, `p. 63`), and `format_passage_reference` prints it in the reference list. Two unverified sources describe each document — the chunk's own metadata and the matching `sources.yml` record — and the precedence is: a `sources.yml` record with `verified: true` wins outright (that is how you correct a wrong extraction in Settings → Citations); otherwise the chunk metadata is used; an unverified `sources.yml` value only fills a field the chunk lacks (an "n.d." placeholder never does). `test_sources_data.py` fails if the two disagree on an author for any document.

**Headless report (`bowen_ask.py`).** One command that runs the Report pipeline for the email bot: `echo "question" | python3 bowen_ask.py` (loads `streamlit_app.py` by path; `BOWEN_REPO` overrides the repo location). Exit codes: 0 ok, 1 no answer, 2 no question, 3 the model produced no text, 4 configuration fault. It has no daily token limit (that lives in the web app's `_llm_stream`) and does not check for a truncated reply.

**Report structure (generated by the LLM):**
1. **Executive Summary** (300–500 words) — concise overview
2. **Full Report** — 8 sections (Introduction & Definition, Theoretical Foundations, Key Dimensions, Relationships, Clinical Presentation, Clinical Implications, Direct Quotations, Gaps & Limitations)
3. **References** — appended programmatically in the chosen style (not generated by LLM)

**Include sources as Appendix** — optional checkbox on the Report page. When enabled, a formatted appendix containing the full text of every cited source is added after the report body. The appendix is included in all download formats.

**Authority boost sync** — when the user toggles Authority boost on the Search page, the same setting carries over to the Report page automatically (via `_rpt_use_boost` in the GUI and `st.session_state.rpt_boost` in Streamlit). Both Search and Report use the synced value when retrieving chunks.

**Staging behavior** — if staged chunks are present, `_gather_chunks` returns ONLY the staged chunks; no fresh retrieval is performed. This is the intentional behavior so the user can hand-pick the exact sources for a report. If no chunks are staged, fresh retrieval runs based on the topic query.

### Report download formats

The Report page in both apps exports to three formats. The Streamlit app exposes three download buttons (Markdown / Word / PDF); the desktop GUI uses a single Save dialog where the file extension chosen drives the export format.

| Format | Library | Notes |
|---|---|---|
| `.md` | (none) | Original markdown with YAML frontmatter (topic + timestamp) |
| `.docx` | `python-docx` | Headings, bold, italic, bullet/numbered lists preserved with Word styles |
| `.pdf` | `reportlab` | Letter size, 0.75" margins, headings sized 18/15/13pt, body 11pt |

The markdown→docx and markdown→pdf converters handle a deliberately small subset of markdown: `#`/`##`/`###` headings, `**bold**`, `*italic*`, `-`/`*` bullet lists, `1.` numbered lists, `---` horizontal rules, paragraphs. Tables and code blocks are not handled (reports do not use them).

### Chunking strategy

`build_index.py` builds chunks from whole **paragraphs** (about 1,500 characters; a paragraph is never split, so an overlong paragraph is one oversized chunk). Paragraphs are numbered per document, so each chunk records `paragraph_start`/`paragraph_end` for a stable "Ch. N, ¶ M" locator. For `.txt` transcripts with `## Section N –` headings (output of `process_transcripts.py`; en dash, em dash or hyphen, LF or CRLF, heading allowed on the very first line) a chunk **never spans two sections**, so `section_title` is always where the text came from; a long section becomes several chunks that all carry its title, and a section with no text is reported. A one-line `# Transcript Formatting …` banner before the first heading is dropped. PDFs are read with PyMuPDF text blocks and keep their page number. There is no chunk overlap.

### Chunk metadata fields

Each entry in `chunk_metadata.json` has:

| Field | Type | Description |
|---|---|---|
| `id` | int | Global chunk index |
| `doc_name` | str | Source document filename (without extension) |
| `section_title` | str | Section heading for transcript chunks; empty otherwise |
| `text` | str | Full chunk text (transcript chunks are prefixed with `[Section Title]\n\n`) |
| `char_count` | int | Character length of `text` |
| `page` | int or null | PDF page number of the chunk's first paragraph; `null` for `.txt` files |
| `paragraph_start`, `paragraph_end` | int | 1-based paragraph numbers (within the document) the chunk covers |
| `author` | list of str | Full-name strings from `chapter_map.yml` (book chapters) or `headers_candidates.yml` (everything else); empty if unknown |
| `date`, `title`, `container` | str or null | Year, title and containing work (book / journal) from the same two files |
| `chapter`, `chapter_label`, `chapter_title` | int / str / str, or null | Book chapter number, front-matter label (Introduction, Epilogue) and chapter title |
| `chunk_pos` | int | 1-based position of this chunk within its document |
| `doc_chunk_count` | int | Total number of chunks in this document |
| `preview` | str | First 150 characters of `text` |

`chapter_map.yml` and `headers_candidates.yml` are **extracted data, all unverified** (`headers_candidates.yml` records are `verified: false`). The `page` and `chunk_pos`/`doc_chunk_count` fields are used by the Streamlit UI to display location badges (`p.5` for PDFs, `~33%` for text files) on search result cards.

### Paths at runtime

| Context | `BASE_DIR` | index refs |
|---|---|---|
| Script / dev | directory of `bowen_rag_gui.py` | same |
| PyInstaller bundle | `~/Documents/BowenRAG/` | `sys._MEIPASS/.../references/` (read-only bundle) |
| Streamlit / Railway | directory of `streamlit_app.py` | `rag-document-search/references/` |

### Claude models (current as of May 2026)

| Model | ID |
|---|---|
| Opus 4.7 (latest, most capable) | `claude-opus-4-7` |
| Sonnet 4.6 | `claude-sonnet-4-6` |
| Haiku 4.5 (fastest) | `claude-haiku-4-5` |
| Opus 4.6 (legacy) | `claude-opus-4-6` |
| Opus 4.5 (legacy) | `claude-opus-4-5` |
| Sonnet 4.5 (legacy) | `claude-sonnet-4-5` |

### LLM system prompt

Defined as `SYSTEM_PROMPT` constant near the top of both `bowen_rag_gui.py` and `streamlit_app.py`. It instructs the model to cite only the provided source excerpts and not draw on outside knowledge. Editable at runtime via the LLM Settings tab / Settings page. Do not soften these constraints — the app is used for Bowen Family Systems Theory research where source fidelity matters.
