# Bowen RAG — Search & Analysis Guide

## What This Is

A **Retrieval-Augmented Generation (RAG)** research tool for the Bowen Family Systems Theory literature. It searches a corpus of indexed source documents and uses an LLM to synthesise answers grounded strictly in those sources — no outside knowledge, all claims cited.

---

## How Search Works

### TF-IDF (default)

Documents are split into chunks and converted to word-frequency vectors. Your query is converted to the same format and matched against all chunks using cosine similarity. Fast and exact — if your query words appear in a passage, it scores well. Misses conceptual matches where the phrasing differs.

### Semantic Embeddings (optional, recommended)

When the embedding index is built, each chunk is encoded by `all-MiniLM-L6-v2` (a sentence-transformer neural model) into a 384-dimensional vector that captures *meaning*, not just vocabulary. A query about "emotional autonomy" will find passages about "differentiation of self" even if those exact words don't overlap.

To enable: **Index tab → Build Embeddings** (one-time, takes a few minutes on CPU).

---

## Search Modes

| Mode | Best for |
|---|---|
| **Top Docs** | Most queries — aggregates scores across chunks per document, returns the best source per doc |
| **Semantic (TF-IDF)** | Conceptual queries using the theory's precise vocabulary |
| **Keyword** | Proper names, specific terms, phrases not captured by TF-IDF |
| **Both** | Merges semantic and keyword results |
| **Embedding** | Conceptual queries where the phrasing may vary from the source text |

---

## Tabs

### Search
Enter a query (natural language or keyword phrase). Results are ranked with authority boosts applied — primary Bowen/Kerr sources score 3× higher than general papers. Check results you want, then click **→ Send to Report**.

### Index
- **Reload Index** — reload from disk without rebuilding
- **Import Transcripts** — pull formatted transcripts from `~/transcripts/projects/`
- **Import + Rebuild** — import then immediately rebuild the TF-IDF index
- **Rebuild Index** — rebuild TF-IDF index from all `.txt` and `.pdf` files in the source directory
- **Build Embeddings** — encode all chunks with sentence-transformers; enables Embedding search mode

### LLM Settings
Configure provider (Claude / OpenAI / Ollama), model, and API keys. Use **Save to .env** to persist settings across restarts. Edit the system prompt here — it controls how strictly the LLM cites sources.

### Report Generator
Generates a structured Markdown report from retrieved or staged source excerpts. Sources are cited by reference number (`[1]`, `[2]`) for readability. The numbered reference list appears in the panel above and at the end of every report.

**Retrieval options:**
- *Retrieve top N* — how many source documents to pull in (20–40 recommended)
- *Mode* — which search method to use
- *Target length* — minimum word count for the report
- *Chunks per source* — context window expansion around each matched chunk

### Chat
Multi-turn conversational interface. Type a question and press **Enter** to send (Shift+Enter for newlines). Each message searches the corpus fresh and includes the retrieved chunks as context. Prior turns are kept as bare Q&A in memory — not chunks — so the conversation grows slowly regardless of length.

The **Sources** panel on the right shows which documents were retrieved for the last question.

---

## File Structure

```
bowen_rag/
├── bowen_rag_gui.py              # entire GUI application (~2100 lines)
├── bowen_rag.sh                  # launcher script (starts Ollama if needed)
├── process_transcripts.py        # transcript importer
├── requirements.txt
├── .env                          # API keys (gitignored)
├── .env.example                  # template
├── source_files/                 # indexed documents (gitignored)
├── outputs/                      # saved reports (gitignored)
└── rag-document-search/
    ├── scripts/
    │   ├── build_index.py        # builds TF-IDF index
    │   └── semantic_search.py    # CLI search (note: needs update for .npz format)
    ├── references/
    │   ├── chunk_metadata.json   # chunk text + metadata
    │   ├── tfidf_matrix.npz      # sparse TF-IDF matrix
    │   ├── vectorizer.json       # TF-IDF vocabulary
    │   └── embed_matrix.npy      # sentence-transformer embeddings (optional)
    └── test_skill.py             # runs 3 sample queries
```

---

## Authority Tiers

Search results are boosted by source authority before ranking:

| Multiplier | Sources |
|---|---|
| 3.0× | Bowen lecture tapes, Bowen-Kerr interview series, *Family Therapy in Clinical Practice* chapters, Kerr primary texts |
| 1.3× | Family Systems Journal articles, Family Center Reports |
| 1.15× | Papero, Friedman, Fogarty, Guerin, Toman papers |
| 1.0× | Everything else |

Multipliers are defined in `AUTHORITY_TIERS` at the top of `bowen_rag_gui.py`.

---

## Tips

- **Embedding mode + Top Docs** is the strongest combination for broad conceptual questions
- For proper names or exact terms (e.g. "Papero", "multigenerational transmission"), use **Keyword**
- In Chat, follow-up questions re-search the corpus independently — you can drill into a topic across many turns without losing thread
- If a report feels thin, increase *Retrieve top* and use Embedding mode
- The system prompt in LLM Settings enforces strict source citation — don't soften it without good reason

---

## Known Issues

- `semantic_search.py` (CLI) loads `tfidf_matrix.npy` (old dense format); `build_index.py` now writes `tfidf_matrix.npz` (sparse). CLI search will fail until `semantic_search.py` is updated to use `scipy.sparse.load_npz`. The GUI is unaffected.
