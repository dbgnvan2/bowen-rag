#!/usr/bin/env python3
"""Why do Keyword / BM25 / Top-Docs miss chunks that literally contain the phrase?"""
import re, numpy as np
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))   # repo root (streamlit_app, literal_search)
import streamlit_app as app

PHRASE = "emotional programming"
rx = re.compile(re.escape(PHRASE), re.I)
idx = app.IndexManager(); idx.load()
truth_ids = {c["id"] for c in idx.chunks if rx.search(c["text"])}

# 1. Authority tier of the documents that contain the phrase
print("Authority multiplier of each document containing the phrase:")
docs = sorted(set(idx.chunks[i]["doc_name"] for i in truth_ids))
for d in docs:
    print(f"  x{app.authority_boost(d):<4} {d}")

# 2. BM25, boost on, k=12: what wins instead, and why
print("\nBM25 boost=True k=12 — the 12 winners:")
res = idx.bm25_search(PHRASE, 12, use_boost=True)
for r in res:
    tl = r["text"].lower()
    print(f"  {r['score']:7.2f} x{app.authority_boost(r['doc_name']):<4} emotional={tl.count('emotional'):3d} "
          f"programm={tl.count('programm'):2d} phrase={'YES' if rx.search(r['text']) else 'no '} {r['doc_name'][:60]}")

raw = idx.bm25.get_scores(idx._tokenize(PHRASE))
best_true = max(truth_ids, key=lambda i: raw[i])
print(f"\n  best raw BM25 among phrase chunks: {raw[best_true]:.2f} (doc {idx.chunks[best_true]['doc_name'][:50]}, "
      f"x{app.authority_boost(idx.chunks[best_true]['doc_name'])})")
print(f"  12th boosted score: {res[-1]['score']:.2f}")
print("  rank of the best phrase chunk in the BOOSTED BM25 list:",
      int((np.array([raw[i]*app.authority_boost(idx.chunks[i]['doc_name']) for i in range(len(idx.chunks))]) > raw[best_true]*app.authority_boost(idx.chunks[best_true]['doc_name'])).sum()) + 1)

# 3. Keyword mode, boost on: why 0/15
print("\nKEYWORD boost=True k=15 — winners:")
res = idx.keyword_search(PHRASE, 15, use_boost=True)
for r in res:
    tl = r["text"].lower()
    print(f"  {r['score_label']:18s} emotional={tl.count('emotional'):3d} programm={tl.count('programm'):2d} "
          f"phrase={'YES' if rx.search(r['text']) else 'no '} {r['doc_name'][:55]}")

# 4. Keyword mode punctuation: does "programming?" ever match?
q = "What is emotional programming?"
remainder = re.sub(r'"[^"]+"', '', q)
raw_terms = [t.lower() for t in remainder.split() if len(t) > 2 and t.lower() not in idx._STOP]
print("\nkeyword_search terms for", repr(q), "->", raw_terms,
      "| stems of last:", idx._stems(raw_terms[-1]))

# 5. Semantic (TF-IDF) boost on: which non-phrase chunks beat phrase chunks?
print("\nSEMANTIC boost=True k=15 — non-phrase winners:")
for r in idx.semantic_search(PHRASE, 15, use_boost=True):
    if not rx.search(r["text"]):
        tl = r["text"].lower()
        print(f"  {r['score_label']:10s} emotional={tl.count('emotional'):3d} programm={tl.count('programm'):2d} {r['doc_name'][:60]}")

# 6. Top-docs: the ratio of phrase to non-phrase
print("\nTOP-DOCS boost=True k=15 — non-phrase winners:")
for r in idx.top_docs_search(PHRASE, top_chunks=300, top_docs=15, use_boost=True):
    if not rx.search(r["text"]):
        print(f"  {r['score_label']:10s} {r['doc_name'][:60]}")

# 7. How often does 'emotional' appear in the corpus vs 'programming'
tot_e = sum(c["text"].lower().count("emotional") for c in idx.chunks)
tot_p = sum(c["text"].lower().count("programming") for c in idx.chunks)
print(f"\ncorpus counts: 'emotional'={tot_e:,}  'programming'={tot_p:,}  chunks with phrase={len(truth_ids)}")
