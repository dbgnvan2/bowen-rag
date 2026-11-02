#!/usr/bin/env python3
"""Measure literal_search against the real index and show the pipeline effect."""
import re, time, json
import literal_search as ls
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))   # repo root (streamlit_app, literal_search)
import streamlit_app as app

idx = app.IndexManager(); idx.load()

t = time.time(); lix = ls.LiteralIndex(idx.chunks); print(f"LiteralIndex build: {time.time()-t:.2f}s for {len(idx.chunks):,} chunks")

PHRASES = ["emotional programming", "emotional contact", "emotional connection", "nodal events",
           "nodal event", "multigenerational transmission process", "societal regression",
           "emotional cutoff", "I-position", "differentiation of self scale", "basic self",
           "pseudo-self", "pseudo self", "the family projection process", "undifferentiated family ego mass",
           "overfunctioning", "emotional regression", "emotional shock wave", "sibling position",
           "this phrase is not in the corpus"]
print(f"\n{'phrase':42s} {'ms':>6s} {'occ':>5s} {'chunks':>6s} {'docs':>5s}")
for p in PHRASES:
    t = time.time(); pr = lix.presence(p); ms = (time.time()-t)*1000
    print(f"{p:42s} {ms:6.1f} {pr.occurrences:5d} {pr.chunks:6d} {pr.documents:5d}")

# Cross-check against the apps' own quoted-keyword mode, which is the documented way to
# test presence today. keyword_search returns ONE chunk per document (doc_best), so its
# count is documents, not chunks.
print("\nquoted keyword_search vs literal presence (documents):")
for p in ["emotional programming", "nodal events", "pseudo-self", "emotional shock wave"]:
    kw = idx.keyword_search(f'"{p}"', 10_000, use_boost=False)
    print(f"  {p:28s} keyword docs={len(kw):3d}  literal docs={lix.presence(p).documents:3d}")

# Pipeline effect at the defaults: how many phrase chunks reach the LLM with and without the guarantee
rx = re.compile(r"(?<!\w)emotional programming(?!\w)", re.I)
print("\nphrase chunks handed to the LLM, before -> after ensure_literal_hits(reserve=5):")
for label, fn in [("chat bm25 k=12 boost", lambda: idx.bm25_search("emotional programming", 12, True)),
                  ("chat keyword k=12 boost", lambda: idx.keyword_search("What is emotional programming?", 12, True)),
                  ("report top-docs k=30 boost", lambda: idx.top_docs_search("What does Bowen theory say about emotional programming?", 300, 30, True)),
                  ("search semantic k=15 boost", lambda: idx.semantic_search("emotional programming", 15, True))]:
    res = fn()
    before = sum(1 for r in res if rx.search(r["text"]))
    after_res = lix.ensure_literal_hits(res, "emotional programming", reserve=5, boost_fn=app.authority_boost)
    after = sum(1 for r in after_res if rx.search(r["text"]))
    print(f"  {label:30s} {before:2d}/{len(res)} -> {after:2d}/{len(after_res)}")

print("\n" + lix.corpus_check_line("emotional programming", idx.bm25_search("emotional programming", 12, True)))
print(lix.corpus_check_line("emotional connection", []))

# Top literal hits, as the UI would list them
print("\nTop 8 literal hits for 'emotional programming':")
for h in lix.search("emotional programming", top_k=8):
    print(f"  {h.occurrences}x  {h.doc_name[:55]:55s} ¶{h.chunk['paragraph_start']}  …{h.snippet[40:120]}…")
