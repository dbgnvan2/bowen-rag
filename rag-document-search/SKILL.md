---
name: rag-document-search
description: |
  Search a collection of Bowen Family Systems Theory documents using natural language queries. Finds relevant passages across the corpus using TF-IDF or sentence-transformer embeddings, with authority boosting for primary Bowen/Kerr sources. Use this whenever you need to find information from the document library — e.g. "What does Bowen say about triangles?", "Find passages on differentiation of self", or "How is emotional cutoff described clinically?". Returns ranked chunks with source attribution.
compatibility: "Python 3.8+, numpy, scipy, scikit-learn. Optional: sentence-transformers (enables embedding search mode)"
---

# RAG Document Search

Searches the Bowen Family Systems Theory document corpus using natural language. Two retrieval modes are available:

- **TF-IDF** — fast word-frequency matching; good for exact terminology
- **Embedding** — sentence-transformer semantic search; finds conceptual matches regardless of exact wording (requires `embed_matrix.npy` to be built via the GUI)

## Scripts

### `build_index.py`

Processes source documents and builds the TF-IDF search index.

```bash
python3 scripts/build_index.py /path/to/source_files/ references/
```

Writes to `references/`:
- `chunk_metadata.json` — chunk text and document metadata
- `tfidf_matrix.npz` — sparse TF-IDF matrix
- `vectorizer.json` — feature names and the TF-IDF settings used (no IDF weights)

Chunking strategy: documents with `## Section N –` headings (transcript format) are split at headings. All others use overlapping word-count chunks (~1500 chars, 200-char overlap).

### `semantic_search.py`

CLI search against the TF-IDF index.

```bash
python3 scripts/semantic_search.py references/ "your query" 5
```

Reads the sparse `tfidf_matrix.npz` written by `build_index.py` and refits the query vectorizer with the same settings (`TFIDF_PARAMS`, imported from `build_index.py`). If the saved files disagree with each other it stops with a "rebuild the index" error instead of returning wrong chunks. An old dense `tfidf_matrix.npy` is ignored and can be deleted.

## References directory

| File | Description |
|---|---|
| `chunk_metadata.json` | Required — chunk text + doc metadata |
| `tfidf_matrix.npz` | Required — TF-IDF vectors |
| `vectorizer.json` | Required — vocabulary |
| `embed_matrix.npy` | Optional — sentence-transformer embeddings; enables Embedding search mode in GUI |

## Test runner

```bash
cd rag-document-search
python3 test_skill.py
```

Runs 3 sample queries and writes results to `test_results.json`.
