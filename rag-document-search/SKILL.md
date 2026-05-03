---
name: rag-document-search
description: |
  Search a collection of text documents using natural language queries. This skill finds relevant sections across multiple documents by matching your question against the document content. Use this whenever you need to find information from a document library—for example, "What does this say about schizophrenia?", "Find sections on family therapy techniques", or "How are alcoholism and family dynamics discussed?". The skill works best with topical queries and returns full relevant sections ranked by relevance.
compatibility: Requires Python 3.8+, numpy, sklearn
---

# RAG Document Search Skill

This skill lets you search a collection of documents using natural language questions. Instead of keyword matching, it understands the semantic meaning of your query and finds the most relevant sections across all documents.

## How it works

1. Documents are preprocessed into semantic chunks
2. Each chunk is converted to a vector embedding
3. Your query is converted to the same embedding space
4. Chunks are ranked by similarity to your query
5. Top matching sections are returned with full context

## Using the skill

When you need to search documents, simply ask your question in natural language:

**Example queries:**
- "What approaches do therapists use for family therapy?"
- "How does schizophrenia affect family relationships?"
- "What does the book say about treating alcoholism?"
- "Find information about the role of fathers in family therapy"

## What you'll get back

Results include:
- **Chapter name** — where the content is from
- **Relevance score** — how well it matches your query (0-100%)
- **Full section** — the complete paragraph or section containing the answer
- **Context** — surrounding sentences to help understand the excerpt

## Token cost

**One-time setup (first time skill is used):**
- Loading and indexing all documents: ~15,000-25,000 tokens
- This happens automatically on first use

**Per search:**
- Query processing and ranking: ~3,000-8,000 tokens
- Depends on number of documents and result count

After the first search, subsequent searches are much more efficient because the document chunks are cached locally.

## Limitations

- Works best with topical or semantic queries (not boolean logic like "A AND NOT B")
- Returns sections as they appear in the original documents; some context may be cut off at chunk boundaries
- Search quality depends on how clearly your question is phrased

---

## Implementation Notes

The skill uses these files:
- `scripts/build_index.py` — Processes documents and builds the search index
- `scripts/semantic_search.py` — Handles search queries and ranking
- `references/chunk_metadata.json` — Cached chunk data (auto-generated on first use)

You don't need to run these manually — they execute automatically when needed.
