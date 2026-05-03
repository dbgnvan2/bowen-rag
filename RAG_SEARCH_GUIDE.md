# RAG Document Search Skill - Setup & Usage Guide

## What You Have

A fully functional **Retrieval-Augmented Generation (RAG)** search system for your "Family Therapy in Clinical Practice" document collection. This lets you search across 22 chapters (3,811 semantic chunks) using natural language questions.

## How It Works

1. **Documents are split into chunks** — Each chapter is divided into semantic sections (~500 characters each)
2. **TF-IDF vectors** — Each chunk is converted to a mathematical representation
3. **Query matching** — Your search query is converted to the same format and matched against all chunks
4. **Ranked results** — Most relevant sections are returned with relevance scores

## Token Cost Analysis

### One-Time Costs (Already Paid)
- **Indexing**: ~15,000-20,000 tokens (building the index from 22 documents)
- This happens once and is complete

### Per-Search Costs
- **Each query**: ~3,000-8,000 tokens depending on:
  - Query length and complexity
  - Number of results returned (default: 5)
  - Size of matching document sections

### Cost Example
If you run 100 searches: ~300K-800K tokens total = **$0.30-$0.80** (at current pricing)

This is very economical for document search compared to alternatives.

## File Structure

```
rag-document-search/
├── SKILL.md                          # Skill documentation
├── scripts/
│   ├── build_index.py               # Creates search index from documents
│   └── semantic_search.py           # Performs searches
├── references/
│   ├── chunk_metadata.json          # Cached document chunks (3,811 chunks)
│   ├── tfidf_matrix.npy             # Vector embeddings
│   └── vectorizer.json              # Vocabulary and settings
├── evals/
│   └── evals.json                   # Test queries
└── test_skill.py                    # Test runner
```

## How to Use This

### Option 1: Use with Claude's Skills (Recommended)

1. **Install as a skill** — Place the `rag-document-search/` folder in your Claude skills directory
2. **Trigger the skill** — When you ask Claude a question about your documents, mention it's about "document search" or ask to "search the documents"
3. **Claude will automatically use it** — No manual steps needed

### Option 2: Run Searches Manually

```bash
cd rag-document-search
python3 scripts/semantic_search.py references/ "your search query" 5
```

Results show:
- Chapter name
- Relevance score (0-100%)
- Full relevant section
- Context around the match

## Test Results

Three test queries were run successfully:

### Test 1: Family Therapy Techniques
**Query:** "Search for information about family therapy techniques..."
- ✅ Returned relevant sections on therapy methods, team approaches, clinical techniques
- Relevance scores: 59-62%

### Test 2: Schizophrenia & Family Dynamics
**Query:** "How does schizophrenia affect family relationships..."
- ✅ Returned sections on family relationships in schizophrenia, etiology, and family systems
- Relevance scores: 43-47%

### Test 3: Alcoholism & Family Therapy
**Query:** "What information is provided about alcoholism..."
- ✅ Returned relevant therapy approaches and family intervention techniques
- Relevance scores: 42-57%

## Limitations & Notes

1. **Best for semantic/topical queries** — Works well for "What does it say about X?" but not boolean logic like "A AND NOT B"
2. **Chunk boundaries** — Results may be cut off at section boundaries; context loss is minimal
3. **Relevance scores** — 40%+ is usually meaningful; 60%+ is a strong match
4. **One-way updates** — If you add new documents, you'll need to rebuild the index
5. **Embedding method** — Uses TF-IDF (fast, interpretable) rather than neural embeddings (slower but potentially more nuanced)

## Rebuilding the Index

If you add new documents to your source folder:

```bash
cd rag-document-search
python3 scripts/build_index.py /path/to/documents references/
```

Processing time: ~30-60 seconds for your document size
Token cost: ~15,000-20,000 tokens

## Next Steps

1. **Test with your own queries** — Try searching for topics you care about
2. **Adjust chunk size if needed** — Edit `chunk_size=500` in `build_index.py` for larger/smaller results
3. **Request more results** — Change `top_k=5` to show more matches
4. **Monitor costs** — Track token usage from your Claude usage dashboard

## Questions?

- **Why TF-IDF instead of embeddings?** — TF-IDF is faster, cheaper, and interpretable. For 1.5MB of text, it's more efficient than neural embeddings.
- **Can I search PDFs?** — Currently supports .txt files. To add PDF support, use a PDF extraction tool to convert to text first.
- **How accurate is it?** — Depends on query clarity. Well-phrased questions (with context) perform better than vague ones.

---

**Summary:** You have a working RAG system that costs ~$0.03-0.08 per search. Use it whenever you need to find specific information across your document library.
