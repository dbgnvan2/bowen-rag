# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

`sentence-transformers` is included in requirements and enables the Embedding search mode. It pulls in PyTorch (~500 MB one-time download). The app runs without it — embedding options are simply hidden if the import fails.

## Environment / API keys

Copy `.env.example` to `.env` and fill in your keys. The app reads this file at startup:

```bash
cp .env.example .env
```

Supported variables:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `LLM_PROVIDER` | Default provider: `claude`, `openai`, or `ollama` |
| `ANTHROPIC_MODEL` | Default Claude model (e.g. `claude-opus-4-7`) |
| `OPENAI_MODEL` | Default OpenAI model |
| `OLLAMA_MODEL` | Default Ollama model |
| `OLLAMA_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `CLAUDE_EXTRA_MODELS` | Comma-separated extra Claude model IDs added to the dropdown |
| `OPENAI_EXTRA_MODELS` | Comma-separated extra OpenAI model IDs added to the dropdown |

Keys can also be saved from inside the app: LLM tab → Claude Settings → **Save to .env**.

## Running the app

```bash
./bowen_rag.sh          # launches GUI (starts Ollama if not running)
python3 bowen_rag_gui.py  # launch directly without the shell wrapper
```

## Rebuilding the search index

```bash
python3 rag-document-search/scripts/build_index.py source_files/ rag-document-search/references/
```

Run this after adding or changing documents in `source_files/`. The script writes three files to `references/`: `chunk_metadata.json`, `tfidf_matrix.npz`, and `vectorizer.json`.

## Building the embedding index

After rebuilding the TF-IDF index, optionally build the semantic embedding index via the GUI:

**Index tab → Build Embeddings**

This encodes all chunks with `all-MiniLM-L6-v2` (sentence-transformers) and saves `embed_matrix.npy` alongside the TF-IDF files. First run downloads the model (~90 MB to `~/.cache/huggingface/`). Subsequent runs load from cache. Expect a few minutes on CPU for a large corpus.

The embedding index is loaded automatically on startup if the file exists.

## Manual search (no GUI)

```bash
python3 rag-document-search/scripts/semantic_search.py rag-document-search/references/ "your query" 5
```

**Note:** `semantic_search.py` attempts to load `tfidf_matrix.npy` (dense NumPy format), but `build_index.py` now saves `tfidf_matrix.npz` (sparse SciPy format). Running it directly from the CLI will fail until it is updated to use `scipy.sparse.load_npz`. The GUI (`IndexManager`) uses its own inline loading logic and is unaffected.

## Processing transcripts

`process_transcripts.py` reads `*yaml.md` files from `~/transcripts/projects/` (recursively), strips YAML frontmatter, and writes clean `.txt` files to `source_files/`. Files without `## Section N –` headings are silently skipped — they haven't been section-formatted yet and aren't ready to index.

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

Note: including `sentence-transformers` in a PyInstaller bundle significantly increases bundle size (~500 MB+). Consider excluding it for distribution builds and documenting it as an optional install.

## Architecture

Everything runs in a single GUI file (`bowen_rag_gui.py`, ~2100 lines) built with tkinter. Four classes:

- **`IndexManager`** — loads the prebuilt TF-IDF index from `rag-document-search/references/` at startup. Exposes `semantic_search`, `keyword_search`, `combined_search`, `top_docs_search`, `embedding_search`, and `build_embeddings`. Also handles rebuilding the index in a background thread. The frozen-app path (`sys._MEIPASS`) and dev path diverge here: writable output goes to `~/Documents/BowenRAG/` in a bundle, next to the script otherwise.

- **`LLMClient`** — static methods for calling Claude (`anthropic` SDK), OpenAI, and Ollama. Streaming is supported for all three. Single-turn methods (`call_claude`, `call_openai`, `call_ollama`) are used by the Report tab. Multi-turn chat methods (`call_claude_chat`, `call_openai_chat`, `call_ollama_chat`) accept a full `messages` list and are used by the Chat tab. Provider selection and API keys are stored as tkinter `StringVar`s in `App`.

- **`App`** — main window. Five notebook tabs:
  - **Search** — query + ranked results with checkboxes to stage excerpts for the report
  - **Index** — rebuild TF-IDF index, import transcripts, build embedding index
  - **LLM Settings** — provider config (Claude / OpenAI / Ollama), model selection, system prompt editor, connection test, Save to .env
  - **Report Generator** — one-shot report from staged or freshly retrieved chunks; cites sources by reference number
  - **Chat** — multi-turn conversational interface; each turn retrieves fresh chunks, history carries only bare Q&A (not chunks)

- **`Tooltip`** — click-to-show helper widget, used on `?` buttons.

### Search modes

| Mode | Method | Description |
|---|---|---|
| Top Docs | `top_docs_search` | Aggregates chunk scores per document; best single chunk per doc returned. Best for most queries. |
| Semantic (TF-IDF) | `semantic_search` | Cosine similarity on TF-IDF vectors. Fast, exact-vocabulary matching. |
| Keyword | `keyword_search` | Counts exact word matches with simple stemming. Good for names and specific terms. |
| Both | `combined_search` | Merges semantic and keyword results. |
| Embedding | `embedding_search` | Cosine similarity on sentence-transformer vectors. Finds conceptual matches regardless of exact wording. Requires `embed_matrix.npy` to be built first. |

### Search ranking

Raw similarity scores are boosted by `authority_boost()` before ranking. The multipliers are defined in `AUTHORITY_TIERS` (top of `bowen_rag_gui.py`) — primary Bowen/Kerr sources get 3.0×, Family Systems Journal articles 1.3×, other named theorists 1.15×. This is editorial content and should stay in that list rather than being hardcoded elsewhere.

### Chat context strategy

The Chat tab keeps conversation history as bare Q&A pairs — the user's question and the assistant's answer only. Retrieved source chunks are included only for the current turn and are not stored in history. This keeps context size flat regardless of conversation length while preserving conversational continuity.

### Report citation format

Reports cite sources by reference number (`[1]`, `[2]`) rather than document name. The reference list mapping numbers to document names appears in the "Reference List" panel in the UI and at the end of every generated report.

### Chunking strategy

`build_index.py` uses two chunking modes: if a document contains `## Section N –` headings (formatted transcript output from `process_transcripts.py`), each section becomes one chunk. Otherwise it falls back to overlapping word-count chunks (~1500 chars, 200-char overlap at sentence boundaries).

### Paths at runtime

| Context | `BASE_DIR` | index refs |
|---|---|---|
| Script / dev | directory of `bowen_rag_gui.py` | same |
| PyInstaller bundle | `~/Documents/BowenRAG/` | `sys._MEIPASS/.../references/` (read-only bundle) |

When the index is rebuilt inside a frozen app, output goes to `~/Documents/BowenRAG/references/` (writable), and `IndexManager.load()` prefers that over the bundle's read-only copy.

### Claude models (current as of May 2026)

| Model | ID |
|---|---|
| Opus 4.7 (latest, most capable) | `claude-opus-4-7` |
| Sonnet 4.6 | `claude-sonnet-4-6` |
| Haiku 4.5 (fastest) | `claude-haiku-4-5` |
| Opus 4.6 (legacy) | `claude-opus-4-6` |
| Opus 4.5 (legacy) | `claude-opus-4-5` |
| Sonnet 4.5 (legacy) | `claude-sonnet-4-5` |
| Opus 4.1 (legacy) | `claude-opus-4-1` |

### LLM system prompt

Defined as `SYSTEM_PROMPT` constant near the top of `bowen_rag_gui.py`. It instructs the model to cite only the provided source excerpts and not draw on outside knowledge. Editable at runtime via LLM Settings tab. Do not soften these constraints without understanding the intent — the app is used for Bowen Family Systems Theory research where source fidelity matters.
