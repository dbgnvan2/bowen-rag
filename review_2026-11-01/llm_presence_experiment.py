#!/usr/bin/env python3
"""Does the model's Rule-7 "verbatim presence" judgement match what it was actually given?

Builds the SAME user messages the app builds (Chat: streamlit_app.page_chat, context with
±1 neighbour chunks; Report: bowen_ask.build_prompt, identical to page_report) for a set
of retrieval conditions, sends each to the configured provider, and compares the model's
presence/absence claim with the ground truth (how many of the excerpts it was sent contain
the phrase). No app code is modified; the app's SYSTEM_PROMPT is used unchanged.

Run from the repo root (needs the index files, and the embedding model for hybrid):
    python3 review_2026-11-01/llm_presence_experiment.py [--provider deepseek|claude] [--reps 2]
Keys come from .env (DEEPSEEK_API_KEY / ANTHROPIC_API_KEY); they are never printed.
Output: review_2026-11-01/llm_presence_results.json and one .md per response.
"""
import argparse, json, os, re, sys, time, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = _pl.Path(__file__).resolve().parent / "llm_presence_responses"

from dotenv import dotenv_values
import anthropic
import streamlit_app as app
import bowen_ask                      # for build_prompt (same text as page_report)
import citations

PHRASE = "emotional programming"
RX = re.compile(re.escape(PHRASE), re.I)
CLAIM_ABSENT = re.compile(r"(does not appear|do not appear|not appear verbatim|no source excerpt uses|"
                          r"does not contain|do not contain|is not (used|found|present)|not (found|present) in)", re.I)
CLAIM_CORPUS = re.compile(r"\b(corpus|the sources|the literature|Bowen'?s? writings?)\b", re.I)

CONDITIONS = [  # (label, page, mode, k, boost)
    ("chat/hybrid/k12/boost-on  (app default)", "chat",   "hybrid",   12, True),
    ("chat/hybrid/k12/boost-off",               "chat",   "hybrid",   12, False),
    ("chat/semantic/k12/boost-on",              "chat",   "semantic", 12, True),
    ("report/top-docs/k30/boost-on (default)",  "report", "top-docs", 30, True),
    ("report/hybrid/k30/boost-on",              "report", "hybrid",   30, True),
    ("report/hybrid/k30/boost-off",             "report", "hybrid",   30, False),
]

def retrieve(idx, mode, q, k, boost):
    if mode == "top-docs": return idx.top_docs_search(q, top_chunks=300, top_docs=k, use_boost=boost)
    if mode == "semantic": return idx.semantic_search(q, k, use_boost=boost)
    if mode == "hybrid":   return idx.hybrid_search(q, k, use_boost=boost)
    raise ValueError(mode)

def chat_message(idx, chunks, prompt):
    """Verbatim copy of page_chat's context assembly (streamlit_app.py:1419-1438)."""
    docs = {}
    for c in chunks:
        cid = c.get("id")
        txts = idx.get_context_window(cid, window=1) if cid is not None else [c["text"]]
        seen = set(docs.get(c["doc_name"], []))
        for t in txts:
            if t not in seen:
                docs.setdefault(c["doc_name"], []).append(t); seen.add(t)
    context = "\n\n---\n\n".join(f"### [{dn}]\n" + "\n…\n".join(txts) for dn, txts in docs.items())
    return f"[Retrieved {len(chunks)} chunks from {len(docs)} documents]\n\n{context}\n\n---\nQuestion: {prompt}"

def report_message(chunks, query, words):
    """Same as page_report / bowen_ask.main (chunk-level sources with locators)."""
    parts, refs = [], []
    for n, c in enumerate(chunks, 1):
        loc = citations.passage_locator(c); tag = f" — {loc}" if loc else ""
        parts.append(f"### [[{n}]] {c['doc_name']}{tag}\n{c['text']}"); refs.append(f"[[{n}]] {c['doc_name']}{tag}")
    return bowen_ask.build_prompt(query, "\n\n---\n\n".join(parts), "\n".join(refs), len(chunks), words)

def call(client, model, system, content, max_tokens):
    r = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                               messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, {"input": r.usage.input_tokens, "output": r.usage.output_tokens}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek", choices=["deepseek", "claude"])
    ap.add_argument("--reps", type=int, default=2, help="repeats per CHAT condition (reports run once)")
    ap.add_argument("--words", type=int, default=500, help="report target words (cost control)")
    ap.add_argument("--query", default="What is emotional programming?")
    a = ap.parse_args()
    env = {**dotenv_values(ROOT / ".env"), **os.environ}
    if a.provider == "deepseek":
        client = anthropic.Anthropic(api_key=env["DEEPSEEK_API_KEY"], base_url=app.DEEPSEEK_BASE_URL, timeout=900)
        model, max_tokens = env.get("DEEPSEEK_MODEL") or "deepseek-v4-flash", 32000
    else:
        client = anthropic.Anthropic(api_key=env["ANTHROPIC_API_KEY"], timeout=900)
        model, max_tokens = env.get("ANTHROPIC_MODEL") or "claude-sonnet-4-6", 16000

    idx = app.IndexManager(); idx.load()
    OUT.mkdir(exist_ok=True)
    results = []
    for label, page, mode, k, boost in CONDITIONS:
        reps = a.reps if page == "chat" else 1
        chunks = retrieve(idx, mode, a.query, k, boost)
        content = chat_message(idx, chunks, a.query) if page == "chat" else report_message(chunks, a.query, a.words)
        n_chunks_with = sum(1 for c in chunks if RX.search(c["text"]))
        n_in_sent = len(RX.findall(content.split("---\nQuestion:")[0] if page == "chat" else content.split("## STRICT INSTRUCTIONS")[0]))
        for r in range(reps):
            t = time.time()
            try:
                text, usage = call(client, model, app.SYSTEM_PROMPT, content, max_tokens)
            except Exception as e:
                text, usage = f"ERROR {type(e).__name__}: {e}", {}
            rec = {"condition": label, "page": page, "mode": mode, "k": k, "boost": boost, "rep": r + 1,
                   "model": model, "chunks_sent": len(chunks), "chunks_containing_phrase": n_chunks_with,
                   "phrase_occurrences_in_sent_text": n_in_sent,
                   "model_claims_absent": bool(CLAIM_ABSENT.search(text)),
                   "warning_marker": "⚠️" in text,
                   "absence_worded_as_corpus": bool(CLAIM_ABSENT.search(text) and CLAIM_CORPUS.search(
                       text[max(0, CLAIM_ABSENT.search(text).start() - 200):CLAIM_ABSENT.search(text).end() + 200])),
                   "seconds": round(time.time() - t, 1), "usage": usage}
            rec["verdict"] = ("FALSE NEGATIVE: claims absent, phrase was in the excerpts" if rec["model_claims_absent"] and n_in_sent > 0
                              else "correct absent (nothing containing it was sent)" if rec["model_claims_absent"]
                              else "no absence claim; phrase was sent" if n_in_sent > 0
                              else "FALSE POSITIVE?: no absence claim although nothing containing it was sent")
            results.append(rec)
            fn = OUT / f"{page}_{mode}_k{k}_{'on' if boost else 'off'}_r{r+1}.md"
            fn.write_text(f"# {label} rep {r+1}\n\nsent: {len(chunks)} chunks, {n_chunks_with} contain the phrase, "
                          f"{n_in_sent} occurrences in the sent text\n\n---\n\n{text}", encoding="utf-8")
            print(f"{label:42s} rep{r+1}  sent={len(chunks):2d} containing={n_chunks_with:2d} occ={n_in_sent:3d}  "
                  f"claims_absent={rec['model_claims_absent']!s:5s} ⚠={rec['warning_marker']!s:5s}  -> {rec['verdict']}")
    (OUT.parent / "llm_presence_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{len(results)} responses written to {OUT}/ ; summary in llm_presence_results.json")

if __name__ == "__main__":
    main()
