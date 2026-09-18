#!/usr/bin/env python3
"""bowen_ask.py — headless port of the Bowen RAG app's **Report** page.

The email bot runs exactly one command; this is it. The output is the same
artefact the app produces on its Report page: numbered sources, `[[N]]` inline
citations with page locators where a page is unambiguous, the 8-section report
structure, then those markers rewritten into the chosen citation style with a
reference list built from the cited sources only.

  echo "What does Bowen theory say about triangles?" | python3 bowen_ask.py

Options (defaults match the Report page's defaults):
  --mode NAME  retrieval mode                (default hybrid; also top-docs, semantic,
                                              keyword, both, embedding)
  --top N      retrieved documents           (default 30; page: "Retrieve top")
  --words N    target minimum word count     (default 2000; page: "Target words")
  --cpd N      chunks per source             (default 5; page: "Chunks per source")
  --style NAME citations.DEFAULT_STYLE default (vancouver)

Exit codes: 0 ok · 1 no answer produced (index unreadable, retrieval failed, or no
relevant sources found) · 2 bad input (no question) · 3 the model produced no text ·
4 configuration fault (API key missing).
"""
import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

# Default to this file's own directory — the repo it lives in — so the CLI runs on any
# machine without an env var. BOWEN_REPO overrides for a checkout kept elsewhere.
REPO = Path(os.environ.get("BOWEN_REPO") or Path(__file__).resolve().parent)
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO / ".env")

spec = importlib.util.spec_from_file_location("bowen_app", REPO / "streamlit_app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
import citations  # noqa: E402
import anthropic  # noqa: E402

DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"


def build_prompt(query, docs, refs_md, doc_page, target_words):
    """The Report page's prompt, verbatim (streamlit_app.py page_report)."""
    context_parts = [
        f"### [[{i + 1}]] {dn}" + (f" (p. {doc_page[dn]})" if dn in doc_page else "") + "\n"
        + "\n…\n".join(txts)
        for i, (dn, txts) in enumerate(sorted(docs.items()))
    ]
    context = "\n\n---\n\n".join(context_parts)
    return f"""Write a comprehensive report on the following topic using ONLY the source excerpts provided below.

**Topic / Question:** {query}

---

## SOURCE EXCERPTS ({len(docs)} documents)

{context}

---

## STRICT INSTRUCTIONS

- **Use only the excerpts above.** Do not add any information from outside these sources.
- **Do not infer, assume, or extrapolate.** If the sources do not explicitly address a point, write: "The provided sources do not address this point."
- **Every factual claim must be cited** immediately after the claim using the reference number in DOUBLE brackets, e.g. [[1]] or [[3]]. To cite several sources at once, group them: [[1, 3]]. Always use double brackets so your citations are never confused with bracketed numbers that appear inside quoted source text.
- **When you quote a specific passage** and that source's header shows a page (e.g. "(p. 45)"), cite it as [[N, p. 45]]. If no page is shown, just use [[N]]. Do not invent page numbers.
- Write at least {target_words} words total. Develop each section fully using evidence from the excerpts.

## REPORT STRUCTURE

### 1. Executive Summary (300–500 words)
A concise overview of the topic drawing from the sources. Mention key themes, major concepts, and main findings. This should give the reader a complete but brief understanding of the topic.

### 2. Full Report
Develop the topic in depth with these sections:

1. **Introduction & Definition** — what do the sources say this concept is?
2. **Theoretical Foundations** — how do the sources describe its origins and place in Bowen theory?
3. **Key Dimensions** — what distinct aspects or components do the sources identify?
4. **Relationship to Other Bowen Concepts** — what connections do the sources explicitly draw?
5. **Clinical Presentation** — how do the sources describe this appearing in families or individuals?
6. **Clinical Implications & Therapeutic Approach** — what do the sources say about working with this clinically?
7. **Direct Quotations & Illustrations** — include key verbatim or near-verbatim passages from the sources
8. **Gaps & Limitations** — what does this topic lack coverage on in the provided sources?

**IMPORTANT: Do NOT include a References section in your output.** The reference list will be appended automatically. Just use [[1]], [[2]], etc. (double brackets) for inline citations throughout your report, using the numbers from the source list below.

## Source numbers (for inline citations only — do NOT reproduce this list in your output)
{refs_md}
"""


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--words", type=int, default=2000)
    ap.add_argument("--cpd", type=int, default=5)
    ap.add_argument("--mode", default="hybrid",
                    choices=["hybrid", "top-docs", "semantic", "keyword", "both",
                             "embedding"],
                    help="retrieval mode; hybrid matches the app's Report default")
    ap.add_argument("--style", default=None)
    ap.add_argument("--file", default=None, help="read the question from this file")
    args = ap.parse_args()

    if args.file:
        query = Path(args.file).read_text(encoding="utf-8").strip()
    else:
        query = sys.stdin.read().strip()
    if not query:
        print("no question supplied (stdin or --file)", file=sys.stderr)
        sys.exit(2)

    idx = app.IndexManager()
    idx.load(app.REFS_DIR)
    if not idx.chunks:
        print("could not read the index — is the corpus volume mounted?", file=sys.stderr)
        sys.exit(1)

    # Retrieval mode — hybrid by default, matching the app's Report page. Downgrade to
    # top-docs only when the requested mode's own pieces are missing; embedding needs the
    # embedding index alone, hybrid needs BM25 as well (the app offers them on exactly
    # those terms). Neither raises once its pieces are present.
    mode = args.mode
    mode_substituted_from = None
    have_emb = bool(app.EMBEDDING_AVAILABLE) and idx.embed_matrix is not None
    mode_available = {"hybrid": have_emb and bool(app.BM25_AVAILABLE),
                      "embedding": have_emb}
    if mode in mode_available and not mode_available[mode]:
        print(f"({mode} unavailable — falling back to top-docs)", file=sys.stderr)
        mode_substituted_from = mode
        mode = "top-docs"

    try:
        if mode == "top-docs":
            chunks = idx.top_docs_search(query, top_chunks=300, top_docs=args.top,
                                         use_boost=True)
        elif mode == "semantic":
            chunks = idx.semantic_search(query, args.top, use_boost=True)
        elif mode == "keyword":
            chunks = idx.keyword_search(query, args.top, use_boost=True)
        elif mode == "both":
            chunks = idx.combined_search(query, args.top, use_boost=True)
        elif mode == "embedding":
            chunks = idx.embedding_search(query, args.top, use_boost=True)
        else:
            chunks = idx.hybrid_search(query, args.top, use_boost=True)
    except RuntimeError as e:
        print(f"retrieval failed: {e}", file=sys.stderr)
        sys.exit(1)
    if not chunks:
        print("no relevant sources found for that question", file=sys.stderr)
        sys.exit(1)

    # Group chunks by document, expanding a context window around each hit —
    # mirroring the Report page's "Chunks per source" setting.
    docs: dict = {}
    window = max(0, (args.cpd - 1) // 2)
    for c in chunks:
        cid = c.get("id")
        expanded = (idx.get_context_window(cid, window=window)
                    if cid is not None and hasattr(idx, "_doc_chunk_ids")
                    else [c["text"]])
        existing = set(docs.get(c["doc_name"], []))
        for t in expanded:
            if t not in existing:
                docs.setdefault(c["doc_name"], []).append(t)
                existing.add(t)

    ref_map = {name: i + 1 for i, name in enumerate(sorted(docs))}

    # A page locator only when the retrieved chunks for that document all sit on
    # ONE page — otherwise it would be confidently wrong.
    doc_pages: dict = {}
    for c in chunks:
        p = c.get("page")
        if p is not None:
            doc_pages.setdefault(c["doc_name"], set()).add(p)
    doc_page = {dn: next(iter(ps)) for dn, ps in doc_pages.items() if len(ps) == 1}

    refs_md = "\n".join(f"[[{num}]] {name}"
                        for name, num in sorted(ref_map.items(), key=lambda x: x[1]))
    srcs = citations.load_sources(REPO)
    num_to_record = {ref_map[dn]: citations.record_for_doc(dn, srcs, app.doc_author)
                     for dn in docs}

    prompt = build_prompt(query, docs, refs_md, doc_page, args.words)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY is not set (missing from the environment or the "
              "corpus repo's .env)", file=sys.stderr)
        # Config fault, not caller error: keep it distinct from exit 2 (empty question) so
        # an operator monitoring exit codes sees a misconfigured bot, not bad input.
        sys.exit(4)

    client = anthropic.Anthropic(api_key=api_key, base_url=DEEPSEEK_BASE_URL,
                                 timeout=900.0, max_retries=2)
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    try:
        with client.messages.stream(model=model, max_tokens=32000,
                                    system=app.SYSTEM_PROMPT,
                                    messages=[{"role": "user", "content": prompt}]) as s:
            result = "".join(s.text_stream)
    except Exception as e:
        print(f"the model call failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    if not result.strip():
        print("the model returned no report text (token budget spent on reasoning)",
              file=sys.stderr)
        sys.exit(3)

    style = citations.normalize_style(args.style or citations.DEFAULT_STYLE)
    try:
        styled_body = citations.apply_intext_citations(result, num_to_record, style)
        raw_cited = citations.cited_numbers(result, set(num_to_record))
        if not raw_cited and len(result.strip()) > 200:
            # Same loud-fail as the app (streamlit_app.py page_report): zero markers means
            # the model ignored the required format, so the list below names every
            # retrieved source with nothing in the body pointing at them. Warn, and still
            # exit 0 — an unmarked real report beats no report.
            print("warning: the model emitted no [[N]] citation markers — the reference "
                  "list below covers all retrieved sources, not the cited ones",
                  file=sys.stderr)
        cited = raw_cited or set(num_to_record)
        refs_body = citations.build_reference_list_md(num_to_record, style, cited)
        final_report = styled_body + f"\n\n## References\n\n{refs_body}\n"
    except Exception as e:  # a bad sources.yml record must not lose the report
        plain = re.sub(r"\[\[\s*(\d[^\]]*?)\s*\]\]", r"[\1]", result)
        plain_refs = "\n".join(f"{n}. {nm}"
                               for nm, n in sorted(ref_map.items(), key=lambda x: x[1]))
        final_report = plain + f"\n\n## References\n\n{plain_refs}\n"
        print(f"(citation styling failed: {e})", file=sys.stderr)

    if mode_substituted_from:
        # The delivered artifact has to show that a substitution happened: the email
        # consumer sees stdout only, so a stderr-only notice never reaches the report.
        final_report = (f"[retrieval: {mode} — {mode_substituted_from} unavailable]\n\n"
                        + final_report)

    sys.stdout.write(final_report)


if __name__ == "__main__":
    main()
