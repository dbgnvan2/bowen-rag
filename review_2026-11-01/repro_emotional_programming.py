#!/usr/bin/env python3
"""Reproduce the "emotional programming" false negative against the real index.

Runs every retrieval mode the apps expose, at the defaults each page uses, and reports
for each: how many of the returned chunks contain the literal phrase, and which
documents they come from. Ground truth is a plain substring scan of chunk_metadata.json.
"""
import re, sys, time, json
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))   # repo root (streamlit_app, literal_search)
import streamlit_app as app

PHRASE = sys.argv[1] if len(sys.argv) > 1 else "emotional programming"
QUERIES = [PHRASE,
           f"What is {PHRASE}?",
           f"What does Bowen theory say about {PHRASE}?"]

idx = app.IndexManager()
t = time.time(); info = idx.load(); print("loaded", info, f"{time.time()-t:.1f}s")

rx = re.compile(re.escape(PHRASE), re.I)
truth = [c for c in idx.chunks if rx.search(c["text"])]
truth_docs = sorted(set(c["doc_name"] for c in truth))
print(f"\nGROUND TRUTH: {len(truth)} chunks in {len(truth_docs)} docs contain '{PHRASE}' literally")
for d in truth_docs:
    print("   ", d, "x", sum(1 for c in truth if c["doc_name"] == d))

def run(mode, q, k, boost):
    if mode == "top-docs":
        return idx.top_docs_search(q, top_chunks=300, top_docs=k, use_boost=boost)
    if mode == "semantic":  return idx.semantic_search(q, k, use_boost=boost)
    if mode == "keyword":   return idx.keyword_search(q, k, use_boost=boost)
    if mode == "both":      return idx.combined_search(q, k, use_boost=boost)
    if mode == "bm25":      return idx.bm25_search(q, k, use_boost=boost)
    if mode == "embedding": return idx.embedding_search(q, k, use_boost=boost)
    if mode == "hybrid":    return idx.hybrid_search(q, k, use_boost=boost)

# (page, mode, k) — the defaults each page actually uses
SETTINGS = [("search", 15), ("chat", 12), ("report", 30)]
MODES = ["hybrid", "top-docs", "semantic", "keyword", "both", "bm25", "embedding"]

print("\nmode        | k  | boost | query                                              | hits/k | docs with phrase among results")
for q in QUERIES:
    for mode in MODES:
        for page, k in SETTINGS:
            for boost in (True, False):
                try:
                    res = run(mode, q, k, boost)
                except Exception as e:
                    print(f"{mode:11s} | {k:2d} | {str(boost):5s} | {q[:50]:50s} | ERROR {e}")
                    continue
                hits = [r for r in res if rx.search(r["text"])]
                docs = sorted(set(r["doc_name"][:28] for r in hits))
                print(f"{mode:11s} | {k:2d} | {str(boost):5s} | {q[:50]:50s} | {len(hits):2d}/{len(res):<3d} | {docs}")

# What the Chat page would actually send: context window ±1 around each hit (so the phrase
# can appear in context even when the retrieved chunk itself lacks it)
print("\nCHAT CONTEXT (hybrid, k=12, boost on) — does the phrase appear anywhere in the assembled context?")
for q in QUERIES:
    res = idx.hybrid_search(q, 12, use_boost=True)
    texts = []
    for r in res:
        texts += idx.get_context_window(r["id"], window=1)
    ctx = "\n".join(texts)
    print(f"  {q[:55]:55s} -> phrase occurrences in context: {len(rx.findall(ctx))}, chunks: {len(res)}")

# Keyword mode with quotes (the documented way to force an exact phrase)
print("\nKEYWORD with quotes:")
for k in (15, 30):
    res = idx.keyword_search(f'"{PHRASE}"', k, use_boost=True)
    print(f"  k={k}: {len(res)} results, {sum(1 for r in res if rx.search(r['text']))} contain phrase; "
          f"docs={len(set(r['doc_name'] for r in res))}")
