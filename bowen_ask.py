#!/usr/bin/env python3
"""bowen_ask.py — headless port of the Bowen RAG app's **Report** page.

The email bot runs exactly one command; this is it. The output is the same
artefact the app produces on its Report page: each retrieved chunk becomes a
numbered source with its own precise locator (chapter + paragraph, or page),
the 8-section report structure, then the `[[N]]` markers rewritten into the
chosen citation style with a reference list that carries the locators.

  echo "What does Bowen theory say about triangles?" | python3 bowen_ask.py

Options:
  --mode NAME  retrieval mode                (default hybrid; also top-docs, semantic,
                                              keyword, both, embedding)
  --top N      retrieved chunks              (default 30)
  --words N    target minimum word count     (default 2000)
  --style NAME citation style                (default citations.DEFAULT_STYLE = vancouver)

Exit codes: 0 ok · 1 no answer produced (index unreadable, retrieval failed, or no
relevant sources found) · 2 bad input (no question) · 3 the model produced no text ·
4 configuration fault (API key missing).

Exact-phrase lookup (spec E7; no model call, `anthropic` is never imported):

  python3 bowen_ask.py --find "emotional programming" [--mode word|substring|regex]
                       [--case] [--json]
  echo "emotional programming" | python3 bowen_ask.py --find

  Prints every document containing the phrase with hit counts, locators and OCR flags.
  With --find, --mode is the match mode (default word), not the retrieval mode.
  Exit codes: 0 found · 1 not found · 2 bad input (empty phrase, unknown mode, bad
  regex; message on stderr) · 4 the index could not be read.
"""
import argparse
import importlib.util
import json
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
import exact_match  # noqa: E402

DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"


def build_prompt(query, context, refs_md, n_sources, target_words, verification=""):
    """The Report prompt — the same function the web Report page uses. Each source header
    carries its locator (Ch. N ¶ M / p. X); `verification` is the EXACT-MATCH
    VERIFICATION block (spec E8), placed at the top."""
    return app.build_report_prompt(query, context, refs_md, n_sources, target_words,
                                   verification=verification)


def find(phrase: str, mode: str, case: bool, as_json: bool) -> int:
    """Exact-phrase lookup over the whole index; returns the exit code.

    Purpose: a scriptable, model-free answer to "is this phrase in the corpus, and where?"
    Spec:    docs/spec_exact_match_search_v2.md#E7
    Tests:   test_exact_match.py::TestE7FindCli
    """
    phrase = phrase.strip()
    if not phrase:
        print("no phrase supplied (--find PHRASE, or on stdin)", file=sys.stderr)
        return 2
    if mode not in exact_match.MODES:
        print(f"unknown match mode {mode!r}; use one of {', '.join(exact_match.MODES)}",
              file=sys.stderr)
        return 2
    try:
        idx = app.IndexManager()
        idx.load(app.REFS_DIR)
    except Exception as e:          # an unreadable index must not read as "not found"
        print(f"could not read the index: {type(e).__name__}: {e}", file=sys.stderr)
        return 4
    try:
        p = idx.exact_presence(phrase, mode=mode, case=case)
    except re.error as e:
        print(f"bad regular expression: re.error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"bad input: {e}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(p, ensure_ascii=False, indent=1))
        return 0 if p["found"] else 1

    label = f'"{phrase}" ({mode} match{", case-sensitive" if case else ""})'
    if not p["found"]:
        loose = (f" ({p['loose_total_hits']} loose)" if mode == "word" else "")
        print(f"{label}: 0 occurrences in the indexed text{loose} "
              f"— {p['corpus_chunks']:,} passages scanned")
        return 1
    loose = ""
    if mode == "word" and p["loose_total_hits"] != p["total_hits"]:
        loose = f"; {p['loose_total_hits']} counting partial-word (loose) matches"
    print(f"{label}: {p['total_hits']} occurrence{'s' if p['total_hits'] != 1 else ''} in "
          f"{p['chunk_count']} passage{'s' if p['chunk_count'] != 1 else ''} across "
          f"{p['doc_count']} document{'s' if p['doc_count'] != 1 else ''} "
          f"(of {p['corpus_chunks']:,} passages){loose}")
    for d in p["docs"]:
        ocr = ""
        if d["ocr"]:
            conf = d.get("ocr_confidence")
            ocr = f"  [scan, OCR{f' {conf}' if conf is not None else ''}]"
        noun = "passage " if d["chunks"] == 1 else "passages"
        print(f"  {d['hits']:4d}x  {d['chunks']:3d} {noun}  {d['doc_name']}{ocr}")
        if d["locators"]:
            print(f"                       {'; '.join(d['locators'])}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--words", type=int, default=2000)
    ap.add_argument("--mode", default=None,
                    help="retrieval mode (default hybrid, matching the app's Report page; "
                         "also top-docs, semantic, keyword, both, embedding). With --find: "
                         "the match mode (word, substring, regex; default word)")
    ap.add_argument("--style", default=None)
    ap.add_argument("--file", default=None, help="read the question from this file")
    ap.add_argument("--find", nargs="?", const="", default=None, metavar="PHRASE",
                    help="exact-phrase lookup, no model call (phrase from stdin if omitted)")
    ap.add_argument("--case", action="store_true", help="--find: case-sensitive")
    ap.add_argument("--json", action="store_true", help="--find: print the result as JSON")
    args = ap.parse_args()

    if args.find is not None:
        phrase = args.find
        if not phrase:
            phrase = (Path(args.file).read_text(encoding="utf-8") if args.file
                      else sys.stdin.read())
        sys.exit(find(phrase, args.mode or "word", args.case, args.json))

    retrieval_modes = ["hybrid", "top-docs", "semantic", "keyword", "both", "embedding"]
    if args.mode is None:
        args.mode = "hybrid"
    elif args.mode not in retrieval_modes:
        ap.error(f"argument --mode: invalid choice: {args.mode!r} "
                 f"(choose from {', '.join(retrieval_modes)})")

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
    # top-docs only when the requested mode's own pieces are missing.
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
    # Exact-match guard (spec E6/E8): passages containing the user's phrase reach the
    # model when any exist; the verification block is computed, never the model's guess.
    try:
        chunks, _, ex_reason, ex_presence = exact_match.exact_guard(
            idx, query, chunks, k=args.top)
    except (ValueError, RuntimeError) as e:
        print(f"exact-match check failed: {e}", file=sys.stderr)
        sys.exit(1)
    if not chunks:
        print("no relevant sources found for that question", file=sys.stderr)
        sys.exit(1)
    verification = exact_match.verification_block(ex_presence, chunks, ex_reason)

    # Each retrieved chunk is its own citable unit with a precise locator
    # (chapter + paragraph, or page). No context-window expansion — that would
    # smear one chunk across multiple paragraphs and make its locator wrong.
    num_to_chunk = dict(enumerate(chunks, start=1))
    num_to_record = {n: citations.record_from_chunk(c, app.SOURCES)
                     for n, c in num_to_chunk.items()}

    context_parts, refs_md_lines = [], []
    for n, c in num_to_chunk.items():
        loc = citations.passage_locator(c)
        tag = f" — {loc}" if loc else ""
        context_parts.append(f"### [[{n}]] {c['doc_name']}{tag}\n{c['text']}")
        refs_md_lines.append(f"[[{n}]] {c['doc_name']}{tag}")
    context = "\n\n---\n\n".join(context_parts)
    refs_md = "\n".join(refs_md_lines)

    prompt = build_prompt(query, context, refs_md, len(num_to_chunk), args.words,
                          verification=verification)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY is not set (missing from the environment or the "
              "corpus repo's .env)", file=sys.stderr)
        sys.exit(4)

    import anthropic        # only the report path needs it (spec E7: --find never imports it)
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
    final_report, note, warn = citations.assemble_report(
        result, num_to_record, num_to_chunk, style)
    if warn:
        # An unmarked or plain-reference report beats no report: warn, still exit 0.
        print(f"warning: {note}", file=sys.stderr)

    # The email consumer sees stdout only: show the computed exact-match statement there,
    # so any "is X in the corpus" claim the reader sees comes from the scan.
    final_report = "\n".join("> " + ln for ln in verification.splitlines()) + "\n\n" + final_report

    if mode_substituted_from:
        # The delivered artifact has to show that a substitution happened: the email
        # consumer sees stdout only, so a stderr-only notice never reaches the report.
        final_report = (f"[retrieval: {mode} — {mode_substituted_from} unavailable]\n\n"
                        + final_report)

    sys.stdout.write(final_report)


if __name__ == "__main__":
    main()
