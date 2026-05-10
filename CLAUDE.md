# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

`sentence-transformers` is included in requirements and enables Embedding and Hybrid search modes. It pulls in PyTorch (~500 MB one-time download). The app runs without it — those search options are hidden if the import fails.

`python-dotenv` is included and is used by both apps to load `.env` at startup.

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

The `Procfile` tells Railway how to start the app:
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --logger.level=warning
```

## Rebuilding the search index

```bash
python3 rag-document-search/scripts/build_index.py source_files/ rag-document-search/references/
```

Run this after adding or changing documents in `source_files/`. The script writes three files to `references/`: `chunk_metadata.json`, `tfidf_matrix.npz`, and `vectorizer.json`.

**After rebuilding, always rebuild the embedding index too** — the chunk count changes and a stale `embed_matrix.npy` will cause a startup error.

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

The embedding index is required for Embedding and Hybrid search modes. It is loaded automatically on startup if the file exists. `embed_matrix.npy` (~16 MB) is committed to the repo so Railway gets it on deploy.

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
- **Report** — generate a cited report; audit chunks used; optional **Include sources as Appendix** checkbox appends full source texts to the report and download
- **Index** — admin: index statistics and document list
- **Settings** — LLM provider/key/model config; default search mode; system prompt; connection test

API keys in Settings are masked — only the last 6 characters are shown.

Key helper functions in `streamlit_app.py`:

| Function | Purpose |
|---|---|
| `_format_chunk_text(text)` | Strips `[Section Title]` prefix, collapses PDF soft-wrap newlines, normalises whitespace — used by the View dialog and appendix |
| `_show_section_dialog()` | `@st.dialog` modal — shows doc name, section title, page/position caption and formatted full text |
| `_result_card(result, key)` | Renders a single search result card with score/author/page badges, excerpt, and View ↗ button |

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

### Chat context strategy

The Chat tab keeps conversation history as bare Q&A pairs — the user's question and the assistant's answer only. Retrieved source chunks are included only for the current turn and are not stored in history. This keeps context size flat regardless of conversation length while preserving conversational continuity. Each source in the sources expander has a **View ↗** button to open the full section text in a modal.

### Report citation format

Reports cite sources by reference number (`[1]`, `[2]`) rather than document name. The reference list appears at the end of the generated report. An "Audit: show chunks sent to LLM" expander lets you inspect exactly what was passed to the model.

**Include sources as Appendix** — optional checkbox on the Report page. When enabled, a formatted appendix containing the full text of every cited source is added after the report body. The appendix is also included in the downloaded `.md` file.

### Chunking strategy

`build_index.py` uses two chunking modes: if a document contains `## Section N –` headings (formatted transcript output from `process_transcripts.py`), each section becomes one chunk. Otherwise it falls back to overlapping word-count chunks (~1500 chars, 200-char overlap at sentence boundaries).

### Chunk metadata fields

Each entry in `chunk_metadata.json` has:

| Field | Type | Description |
|---|---|---|
| `id` | int | Global chunk index |
| `doc_name` | str | Source document filename (without extension) |
| `section_title` | str | Section heading for transcript chunks; empty for word-count chunks |
| `text` | str | Full chunk text (transcript chunks are prefixed with `[Section Title]\n\n`) |
| `char_count` | int | Character length of `text` |
| `page` | int or null | PDF page number of the first sentence in this chunk; `null` for `.txt` files |
| `chunk_pos` | int | 1-based position of this chunk within its document |
| `doc_chunk_count` | int | Total number of chunks in this document |
| `preview` | str | First 150 characters of `text` |

The `page` and `chunk_pos`/`doc_chunk_count` fields are used by the Streamlit UI to display location badges (`p.5` for PDFs, `~33%` for text files) on search result cards.

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
