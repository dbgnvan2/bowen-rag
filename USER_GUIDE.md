# Bowen RAG — User Guide

**Bowen RAG** is a research tool for Bowen Family Systems Theory literature. It searches a private corpus of indexed source documents and uses an AI language model to synthesise answers grounded strictly in those sources — every claim is cited, and the model is instructed not to draw on outside knowledge.

There are two front-ends that share the same search engine:

| Front-end | When to use it |
|---|---|
| **Streamlit web app** (`streamlit_app.py`) | Browser access, Railway deployment, collaborative use |
| **Desktop GUI** (`bowen_rag_gui.py`) | Local Mac/Windows use, offline work |

This guide covers the Streamlit web app. The desktop GUI has the same features with a slightly different layout.

---

## Table of Contents

- [Part 1 — User Guide](#part-1--user-guide)
  - [Navigation](#navigation)
  - [Search](#search)
  - [Chat](#chat)
  - [Report](#report)
  - [Search Tips](#search-tips)
  - [Understanding Score Badges](#understanding-score-badges)
- [Part 2 — Admin Guide](#part-2--admin-guide)
  - [Environment Setup](#environment-setup)
  - [Adding Documents](#adding-documents)
  - [Rebuilding the Search Index](#rebuilding-the-search-index)
  - [Building the Embedding Index](#building-the-embedding-index)
  - [Processing Transcripts](#processing-transcripts)
  - [LLM Provider Configuration](#llm-provider-configuration)
  - [Authority Tiers](#authority-tiers)
  - [Railway Deployment](#railway-deployment)
  - [Desktop App Build](#desktop-app-build)

---

# Part 1 — User Guide

## Navigation

The sidebar has five pages. Use the nav buttons to switch pages; click the **?** beside each button for a one-line description of that page. The sidebar also shows the corpus size (documents, chunks, embeddings) and the active LLM provider and model.

---

## Search

**When to use Search:** You want to find specific passages in the corpus, read them in context, or select a set of excerpts to feed into a Report. Search is the best starting point for any research question.

### Running a search

1. Type your query in the search box and press **Enter** or click **Search**.
2. Results appear ranked by relevance. Each card shows the document name, a short excerpt, location badges, and a score badge.
3. Click **↗** on any card to open the full section text in a formatted popup.
4. Check the boxes on results you want to keep, then click **Stage selected for Report** to carry them into the Report page.

### Search modes

Choose the mode that matches how you are thinking about your question.

| Mode | When to use it | Why |
|---|---|---|
| **Top Docs** | Most queries — broad topics, concepts, questions about the theory | Aggregates chunk scores per document and returns the best match per source. Avoids returning five chunks from the same document and gives a wider view across the corpus. Best default. |
| **Semantic (TF-IDF)** | When you want Bowen's precise vocabulary to drive matching | Matches your query terms against all chunks using TF-IDF cosine similarity. Fast and exact-vocabulary. Use when you know Bowen uses a specific word and want passages that contain it. |
| **Keyword** | Proper names, exact phrases, specific technical terms | Counts exact word occurrences with simple stemming. Reliable for names ("Papero", "Kerr"), diagnostic terms, or short phrases. Use quoted phrases (`"multigenerational transmission"`) to require an exact string. |
| **Both** | When you are unsure whether vocabulary will match | Merges Semantic and Keyword results. Broader than either alone, but the combined score scale can be uneven — try Top Docs first. |
| **Embedding** | Conceptual questions where your phrasing may differ from the source | Uses a sentence-transformer neural model (`all-MiniLM-L6-v2`) to find passages that are *semantically similar* even if the exact words don't match. A query about "emotional autonomy" will find passages about "differentiation of self". Requires the embedding index to be built. |
| **Hybrid** | Highest-quality results for complex questions | Combines BM25 (vocabulary-based) and Embedding (semantic) using Reciprocal Rank Fusion. Best overall quality for nuanced research questions. Requires the embedding index. |

**Recommended workflow:**
- Start with **Top Docs** or **Hybrid** for most questions.
- Switch to **Keyword** when searching for a specific name, phrase, or technical term.
- Use **Embedding** when your query is conceptual and you are not sure how Bowen would have phrased it.

### Authority boost

The **Authority boost** checkbox multiplies scores for primary sources before ranking. This means a passage from a Bowen lecture tape or the Bowen-Kerr interview series ranks above an equivalent passage from a secondary paper, all else being equal.

**When to leave it on (default):** Most research — you want primary Bowen and Kerr material to surface first.

**When to turn it off:** You want to compare all sources on equal footing, or you are specifically looking for secondary commentary and primary sources are crowding out the results.

See [Authority Tiers](#authority-tiers) for the exact multipliers.

### Filtering by author

The author filter dropdown narrows results to a single author's documents. Use it when you want to see how a specific writer (e.g., Kerr, Papero, Friedman) addresses a topic, rather than the full corpus view.

### Quoted phrase search (Keyword mode)

In Keyword mode, wrapping a phrase in double quotes requires that exact phrase to be present in the chunk:

- `emotional cutoff` — finds chunks containing both words, anywhere
- `"emotional cutoff"` — finds chunks containing the exact phrase; phrase hits are weighted 3× over individual word hits

Mix quoted and unquoted terms: `"emotional cutoff" differentiation` finds chunks containing the exact phrase plus the word "differentiation".

### Viewing a full section

Click **↗** on any result card to open the full section text in a popup. The popup shows the document name, section title, and page or position information, with the text formatted for reading (not raw source markup). Use this to read context around an excerpt before deciding whether to stage it.

### Staging results for a Report

Check the boxes on results you want to use, then click **Stage selected for Report**. Staged chunks appear on the Report page and are merged with any fresh retrieval you do there. This lets you hand-pick the most relevant passages and supplement them with additional retrieval.

---

## Chat

**When to use Chat:** You want to explore a topic conversationally, ask follow-up questions, or think through a concept interactively. Chat is better than repeated Report generation for open-ended inquiry.

**When to use Report instead:** You want a structured, citable document suitable for sharing or saving.

### Chat interaction modes

The **Mode** selector at the top of the Chat page changes how the AI engages with you. Switching modes clears the current conversation and starts a fresh session in the new mode.

| Mode | When to use it | What the AI does |
|---|---|---|
| **Standard** | Research questions, looking something up, exploring a topic | Answers your questions from the retrieved source corpus. Default mode. |
| **Interview** | Exploring your own family system, reflective self-study | Takes the role of interviewer — asks one focused question at a time, reflects your answers back through Bowen theory concepts, then asks the next question. You respond; it asks. |
| **Coach** | Applying Bowen theory to your own functioning or clinical work | Acts as a Bowen-informed coach — offers one observation grounded in the sources, then asks one question to help you go deeper. Non-directive: it helps you think, not tells you what to do. |
| **Quiz** | Testing your knowledge of Bowen theory, studying for clinical work | Asks one question at a time drawn from the source material, evaluates your answer against the sources, gives feedback with citations, and tracks your running score. |

**Interview and Coach modes** are grounded in the same source corpus as Standard mode — the AI's questions and reflections draw from Bowen and Kerr's actual writing, not generic therapy frameworks.

**Quiz mode** generates questions directly from retrieved source excerpts, so questions vary based on what the corpus contains on the topic you seed the conversation with.

### How Chat works

Each message you send triggers a fresh search of the corpus. The retrieved chunks are included as context for the AI's answer for that turn only — they are not stored in conversation history. Prior turns are kept as bare question-and-answer pairs, so the conversation context grows slowly regardless of how long the conversation goes.

This means:
- The AI always answers from fresh retrieval, not from memory of prior sources.
- You can explore many different facets of a topic without the context window filling up.
- Each answer is as well-sourced as the first.

### Sources expander

Every AI response shows a **Sources** expander beneath it. Expand it to see which document chunks were retrieved for that answer. Each source has a **↗** button to open the full section text. Use this to verify that the AI's answer is grounded in the source, and to read the passage in full context.

### Terminology precision warning

If the exact term or phrase in your question does not appear verbatim in the retrieved sources, the AI will open its response with a **⚠️ notice** identifying the term and explaining what related concept it is drawing on instead. This is important for Bowen theory, where precise terminology matters — "emotional contact" and "emotional connection" are not interchangeable in Bowen's framework.

If you see this warning, consider:
- Rephrasing your query using Bowen's exact terminology.
- Checking whether the concept you are looking for is indexed under a different term.
- Using Keyword search with the exact phrase in quotes to verify whether the term appears in the corpus at all.

---

## Report

**When to use Report:** You want a structured written synthesis — suitable for notes, summaries, or sharing — with numbered citations and a reference list. Reports are generated in Markdown and can be downloaded.

### Controls

| Control | What it does | Guidance |
|---|---|---|
| **Retrieve top N** | How many source documents to pull in for the report | 15–20 for focused questions; 30–40 for broad topics or when the corpus is large. More sources = more complete report but longer generation time. |
| **Mode** | Which search method to use for retrieval | Hybrid or Embedding for nuanced topics; Keyword for reports built around specific terms or names. |
| **Target words** | Minimum word count for the generated report | 400–600 for a summary; 800–1200 for a detailed synthesis. The AI will expand to this length using only the provided sources. |
| **Chunks per source** | How many text chunks to include per source document | 1–2 for broad coverage; 3–4 when you want depth on each source. Higher values use more of the context window. |
| **Authority boost** | Apply source authority multipliers before retrieval | Leave on for most reports; turn off when you specifically want secondary sources included on equal footing. |
| **Include sources as Appendix** | Append full section texts after the report body | Use when you want a self-contained document with all sources readable in one place. Adds significant length. Included in the downloaded `.md` file. |

### Staged chunks

If you staged chunks from the Search page, they appear listed above the Generate button. They will be merged with any fresh retrieval when the report is generated. Remove staged chunks by clearing them in the Search page before navigating here.

### Audit: show chunks sent to LLM

After generating, expand **Audit: show chunks sent to LLM** to see exactly what source text was passed to the model. Use this to verify the report's claims, understand why certain sources were cited, or debug cases where the report missed something you expected.

### Downloading a report

Click **Download report** after generating to save the report as a `.md` file. If **Include sources as Appendix** was checked, the appendix is included in the download.

---

## Search Tips

**For best results on conceptual questions:**
Use Hybrid or Embedding mode. These find passages that are semantically related to your question even when the exact words differ.

**For best results on specific terms or names:**
Use Keyword mode with quoted phrases for exact matching. Example: `"nodal events"` rather than `nodal events`.

**If results seem off-topic:**
- Check that you are using the right search mode.
- Try turning off Authority boost to see if a secondary source has better coverage.
- Try rephrasing using Bowen's own terminology — the corpus is indexed on his language, not paraphrases.

**If the AI warns about a missing term (⚠️):**
The exact phrase is not in the retrieved sources. Run a Keyword search with the term in quotes to confirm whether it appears in the corpus at all before drawing conclusions.

**For thorough reports:**
Increase Retrieve top N to 30–40 and use Hybrid mode. Stage the best results from a Search first, then generate to supplement with additional retrieval.

---

## Understanding Score Badges

Each search result shows a score badge indicating how well the result matched your query.

| Badge | Mode | What it means |
|---|---|---|
| `45%` | Top Docs | Aggregate similarity score across the best chunks for that document |
| `45% ★` | Semantic (TF-IDF) | Cosine similarity; ★ means an authority boost was applied |
| `45% ⬡` | Hybrid | Reciprocal Rank Fusion (RRF) score; ⬡ identifies Hybrid mode |
| `45% ⬡ ★` | Hybrid + boosted | RRF score with authority boost applied |
| `45% ✦` | Embedding | Cosine similarity on sentence-transformer vectors |
| `6 hits ×3 ★` | Keyword | Raw hit count × authority multiplier = effective rank score |
| `3.42 ★` | BM25 | BM25 relevance score |

**The ★ and multiplier:** When authority boost is on, a result from a primary Bowen/Kerr source is multiplied by up to 3× before ranking. A result showing `3 hits ×3 ★` has an effective score of 9 and will rank above `6 hits` (score 6) from a non-primary source. This is intentional — primary sources should surface first. Turn off Authority boost in the search controls if you want raw scores only.

**The position badge:** Every result also shows a grey location badge — either a PDF page number (`p.5`) or a percentage through the document (`~33%`). This is not a score; it tells you where in the source document the chunk comes from.

---

# Part 2 — Admin Guide

## Environment Setup

Copy `.env.example` to `.env` and fill in your API keys before starting either app.

```bash
cp .env.example .env
```

| Variable | Purpose | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude API key | Required for Claude provider |
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI provider |
| `DEEPSEEK_API_KEY` | DeepSeek API key | Required for DeepSeek provider |
| `LLM_PROVIDER` | Default provider at startup | `claude`, `openai`, `deepseek`, or `ollama` |
| `ANTHROPIC_MODEL` | Default Claude model | e.g. `claude-sonnet-4-6` |
| `OPENAI_MODEL` | Default OpenAI model | e.g. `gpt-4o` |
| `DEEPSEEK_MODEL` | Default DeepSeek model | Default: `deepseek-v4-flash` |
| `OLLAMA_MODEL` | Default Ollama model | e.g. `llama3` |
| `OLLAMA_URL` | Ollama server URL | Default: `http://localhost:11434` |
| `APP_PASSWORD` | Optional access password | Web app only; leave blank for open access |
| `CLAUDE_EXTRA_MODELS` | Extra Claude model IDs | Comma-separated; added to the model dropdown |
| `OPENAI_EXTRA_MODELS` | Extra OpenAI model IDs | Comma-separated; added to the model dropdown |

**Recommended default:** `LLM_PROVIDER=deepseek` — cheapest, good quality. Switch to `claude` or `openai` for highest-quality report generation.

### Current Claude models (as of May 2026)

| Model | ID | Notes |
|---|---|---|
| Opus 4.7 | `claude-opus-4-7` | Most capable |
| Sonnet 4.6 | `claude-sonnet-4-6` | Fast, high quality — good default |
| Haiku 4.5 | `claude-haiku-4-5` | Fastest, cheapest |

---

## Adding Documents

Place `.txt` or `.pdf` files in `source_files/`. After adding files, [rebuild the search index](#rebuilding-the-search-index).

**File format notes:**
- `.txt` files are read as UTF-8 (falls back to UTF-16 and latin-1 if needed).
- `.pdf` files are read with PyPDF2; text extraction quality depends on how the PDF was produced. Scanned PDFs without an OCR text layer will not index usefully.
- Formatted transcripts (files with `## Section N –` headings from `process_transcripts.py`) are chunked one section per chunk, which preserves semantic boundaries.
- All other documents are split into overlapping ~1500-character chunks with 200-character overlap at sentence boundaries.

---

## Rebuilding the Search Index

Run after adding, removing, or changing documents in `source_files/`:

```bash
python3 rag-document-search/scripts/build_index.py source_files/ rag-document-search/references/
```

This writes three files to `references/`:
- `chunk_metadata.json` — chunk text, section titles, page numbers, positions
- `tfidf_matrix.npz` — sparse TF-IDF matrix
- `vectorizer.json` — TF-IDF vocabulary

**After rebuilding the TF-IDF index, always rebuild the embedding index.** The chunk count changes and a stale `embed_matrix.npy` will cause a startup error.

You can also trigger a rebuild from within the app:
- **Web app:** Index page → Rebuild Index
- **Desktop GUI:** Index tab → Rebuild Index

---

## Building the Embedding Index

The embedding index enables Embedding and Hybrid search modes. It is required for best-quality results. Build it after every TF-IDF index rebuild.

**From the app:**
- Web app: Index page → Build Embeddings
- Desktop GUI: Index tab → Build Embeddings

**From the command line:**

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

The first run downloads the model (~90 MB to `~/.cache/huggingface/`). On CPU, encoding a large corpus takes a few minutes. The resulting `embed_matrix.npy` (~16 MB for ~11,000 chunks) is committed to the repo so Railway gets it on deploy.

---

## Processing Transcripts

`process_transcripts.py` imports formatted transcript files and converts them to indexed source files.

**What it does:** Reads `*yaml.md` files from `~/transcripts/projects/` (recursively), strips YAML frontmatter, and writes clean `.txt` files to `source_files/`. Files without `## Section N –` headings are skipped silently — they are not transcripts in the expected format.

```bash
python3 process_transcripts.py                        # default paths
python3 process_transcripts.py --dry-run              # preview without writing
python3 process_transcripts.py --transcripts-dir DIR  # override source directory
python3 process_transcripts.py --source-dir DIR       # override output directory
```

After importing, rebuild the index:

```bash
python3 rag-document-search/scripts/build_index.py source_files/ rag-document-search/references/
```

The script prints this command as a reminder after it runs.

---

## LLM Provider Configuration

Configure the LLM provider in **Settings** (web app) or **LLM Settings** (desktop GUI). Changes made here persist to `.env` if you click **Save to .env**.

### Provider comparison

| Provider | Model | Cost | Quality | Notes |
|---|---|---|---|---|
| **DeepSeek** | `deepseek-v4-flash` | Lowest | Good | Recommended default; uses Anthropic SDK with custom base_url |
| **DeepSeek** | `deepseek-v4-pro` | Low | High | Better for complex synthesis |
| **Claude** | `claude-sonnet-4-6` | Medium | Very high | Best for nuanced Bowen theory questions |
| **Claude** | `claude-opus-4-7` | High | Highest | Use for the most demanding reports |
| **OpenAI** | `gpt-4o` | Medium | Very high | Good alternative to Claude |
| **Ollama** | any local model | Free | Varies | Fully offline; quality depends on model |

### System prompt

The system prompt instructs the model to cite only the provided source excerpts and not draw on outside knowledge. It includes a terminology precision rule (Rule 7) that requires the model to warn the user when it is inferring from semantic similarity rather than citing verbatim source text.

**Do not soften these constraints.** The app is used for Bowen Family Systems Theory research where source fidelity is clinically important — "emotional contact" and "emotional connection" are not interchangeable.

Edit the system prompt in Settings if you need to adjust emphasis, but preserve the core citation and terminology-precision rules.

### API key security

In the Settings page, API keys are masked — only the last 6 characters are shown. Keys are stored in `.env` and are never logged or exposed in results.

---

## Authority Tiers

Search results are multiplied by a source-authority factor before ranking. This ensures that primary Bowen and Kerr material surfaces above secondary commentary for equivalent query matches.

| Multiplier | Sources |
|---|---|
| **3.0×** | Bowen lecture tapes, Bowen-Kerr interview series, *Family Therapy in Clinical Practice* chapters, Kerr primary texts |
| **1.3×** | Family Systems Journal articles, Family Center Reports |
| **1.15×** | Papero, Friedman, Fogarty, Guerin, Toman papers |
| **1.0×** | Everything else |

**To customise:** Create `authority_tiers.yml` in the project root. The app loads this file at startup if it exists, overriding the defaults in the code. Format:

```yaml
tiers:
  - multiplier: 3.0
    patterns: ["bowen_lecture", "kerr_interview", "ftcp_"]
  - multiplier: 1.3
    patterns: ["fsj_", "fcr_"]
  - multiplier: 1.15
    patterns: ["papero_", "friedman_", "fogarty_"]
```

The `patterns` values are matched against the document filename (without extension).

---

## Railway Deployment

The Streamlit web app is deployed to Railway. Pushing to `main` on GitHub triggers an automatic redeploy (typically 2–3 minutes).

**Required Railway environment variables** (set in Railway dashboard → Variables):

| Variable | Value |
|---|---|
| `LLM_PROVIDER` | `deepseek` (or your preferred provider) |
| `DEEPSEEK_API_KEY` | Your DeepSeek key |
| `APP_PASSWORD` | Optional; if set, users must enter this to access the app |

**Do not commit `.env` to git.** API keys for Railway are set in the Railway dashboard only.

The `Procfile` controls how Railway starts the app:

```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --logger.level=warning
```

**Deployment checklist after adding documents:**
1. Rebuild TF-IDF index locally
2. Rebuild embedding index locally
3. Commit `chunk_metadata.json`, `tfidf_matrix.npz`, `vectorizer.json`, `embed_matrix.npy`
4. Push to `main`
5. Confirm Railway build completes (check Railway dashboard)

**Note:** `embed_matrix.npy` is ~16 MB — large for git but necessary since Railway has no persistent disk and must have the index at deploy time.

---

## Desktop App Build

GitHub Actions builds `.app` bundles via PyInstaller on push to a `v*` tag — one for `arm64` (Apple Silicon, macOS-14) and one for `x86_64` (Intel, macOS-13).

**To build locally:**

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

**Note:** Including `sentence-transformers` in the bundle adds ~500 MB. Consider excluding it for distribution builds — Embedding and Hybrid search will be unavailable, but all other modes work.

**To trigger a release build:**

```bash
git tag v1.2.0
git push origin v1.2.0
```

GitHub Actions will build both architectures and attach the `.app` bundles to the release.

---

## Running the Apps Locally

**Desktop GUI:**
```bash
./bowen_rag.sh              # starts Ollama if not running, then launches GUI
python3 bowen_rag_gui.py    # launch directly
```

**Streamlit web app:**
```bash
./bowen_rag_web.sh          # starts Ollama if needed, then runs on port 8501
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

**Note:** The local Streamlit server does not auto-reload when code changes (file watching is disabled to suppress PyTorch log noise). Restart the server manually after editing `streamlit_app.py`.

---

## Evals / Tests

```bash
cd rag-document-search
python3 test_skill.py   # runs 3 sample queries; writes results to test_results.json
```

Use after rebuilding the index to confirm the search engine is returning sensible results.
