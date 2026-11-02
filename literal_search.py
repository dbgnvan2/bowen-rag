"""literal_search.py — exact-phrase search over the chunk index, independent of TF-IDF,
BM25 and embeddings.

Purpose: make "is this phrase in the corpus?" a provable, deterministic answer. The
ranking modes are similarity proxies and can legitimately rank a chunk that contains the
exact phrase below chunks that do not (see REVIEW-bowen_rag-2026-11-01-literal-search.md).
This module scans the chunk text itself.

Design:
  * One normalised copy of every chunk text is built once at load (`LiteralIndex`).
    Normalisation: lower-case, Unicode NFKC (ﬁ → fi, curly quotes → straight), soft
    hyphens removed, "word-\nword" re-joined, all whitespace runs collapsed to one space.
    The same function normalises the query, so a phrase matches across a line break or a
    PDF soft-wrap.
  * `search(phrase)` returns EVERY chunk containing the phrase, with the occurrence count,
    ranked by count (ties by document order). No authority boost is applied to membership;
    boost is only a tie-break if requested. There is no top-k cut-off by default: a
    literal search that drops hits cannot be used to prove a negative.
  * `presence(phrase)` returns the corpus-level facts (occurrences, chunks, documents).
  * `corpus_check_line(phrase, provided)` renders a one-line, machine-computed statement
    for the LLM prompt and the UI, replacing the model's own guess at verbatim presence.
  * `ensure_literal_hits(results, phrase, reserve)` guarantees that, when the phrase exists
    in the corpus, at least `reserve` chunks containing it are among the chunks handed to
    the LLM.

No dependencies beyond the standard library. Pure functions; nothing here touches
Streamlit or tkinter.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional

_SOFT_HYPHEN = "­"
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+")


def normalize(text: str) -> str:
    """Canonical form used for both chunk text and query. Idempotent."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace(_SOFT_HYPHEN, "")
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-").replace("‑", "-")
    t = _HYPHEN_BREAK.sub(r"\1\2", t)     # "program-\nming" -> "programming"
    t = _WS.sub(" ", t)
    return t.lower().strip()


def extract_phrases(query: str) -> list[str]:
    """Quoted phrases in a query, or the whole query if nothing is quoted.

    'What is "emotional programming"?' -> ['emotional programming']
    'emotional programming'            -> ['emotional programming']
    """
    quoted = re.findall(r'["“]([^"”]+)["”]', query)
    if quoted:
        return [normalize(q) for q in quoted if normalize(q)]
    q = normalize(query)
    return [q] if q else []


@dataclass
class Hit:
    chunk: dict
    occurrences: int
    first_pos: int
    snippet: str

    @property
    def doc_name(self) -> str:
        return self.chunk["doc_name"]


@dataclass
class Presence:
    phrase: str
    occurrences: int
    chunks: int
    documents: int
    doc_names: list = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.chunks > 0


class LiteralIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self._norm = [normalize(c["text"]) for c in chunks]

    # ── core ────────────────────────────────────────────────────────────────
    def search(self, phrase: str, top_k: Optional[int] = None,
               whole_words: bool = False, snippet_chars: int = 160,
               boost_fn=None) -> list[Hit]:
        """All chunks containing `phrase` (normalised). `top_k=None` returns every hit.

        Substring matching is the default on purpose: PDF extraction sometimes drops a
        space ("thatemotional programming" occurs in the real corpus) and plurals such as
        "pseudo-selfs" exist; a presence check must not call those absent. Pass
        `whole_words=True` for the stricter match.
        """
        p = normalize(phrase)
        if not p:
            return []
        rx = re.compile((r"(?<!\w)" + re.escape(p) + r"(?!\w)") if whole_words else re.escape(p))
        hits: list[Hit] = []
        for c, nt in zip(self.chunks, self._norm):
            n = 0
            first = -1
            for m in rx.finditer(nt):
                if first < 0:
                    first = m.start()
                n += 1
            if n:
                s0 = max(0, first - snippet_chars // 2)
                snippet = nt[s0:first + len(p) + snippet_chars // 2]
                hits.append(Hit(c, n, first, snippet))
        boost = boost_fn or (lambda _dn: 1.0)
        hits.sort(key=lambda h: (-h.occurrences, -boost(h.doc_name), h.chunk["id"]))
        return hits if top_k is None else hits[:top_k]

    def search_regex(self, pattern: str, flags: int = re.I, top_k: Optional[int] = None) -> list[Hit]:
        """Regex over the normalised text, for variants such as r'emotional(ly)? programm\\w+'."""
        rx = re.compile(pattern, flags)
        hits = []
        for c, nt in zip(self.chunks, self._norm):
            ms = list(rx.finditer(nt))
            if ms:
                first = ms[0].start()
                hits.append(Hit(c, len(ms), first, nt[max(0, first - 80):ms[0].end() + 80]))
        hits.sort(key=lambda h: (-h.occurrences, h.chunk["id"]))
        return hits if top_k is None else hits[:top_k]

    def presence(self, phrase: str, whole_words: bool = False) -> Presence:
        hits = self.search(phrase, whole_words=whole_words)
        docs = sorted(set(h.doc_name for h in hits))
        return Presence(normalize(phrase), sum(h.occurrences for h in hits), len(hits), len(docs), docs)

    # ── helpers for the LLM pipeline ────────────────────────────────────────
    def corpus_check_line(self, phrase: str, provided: Iterable[dict]) -> str:
        """A factual sentence computed by code, to prepend to the prompt and show in the UI.

        Replaces the system-prompt rule that asks the model to judge verbatim presence
        itself: the model cannot see the corpus, and its judgement over 30 excerpts is not
        reliable. Both statements below are true by construction.
        """
        pr = self.presence(phrase)
        p = normalize(phrase)
        in_provided = sum(1 for c in provided if p in normalize(c["text"]))
        if not pr.found:
            return (f'CORPUS CHECK: the exact phrase "{phrase}" does not occur anywhere in the '
                    f"indexed corpus ({len(self.chunks):,} passages). Any discussion of it is an "
                    f"inference from related wording and must be labelled as such.")
        return (f'CORPUS CHECK: the exact phrase "{phrase}" occurs {pr.occurrences} times in '
                f"{pr.chunks} passages across {pr.documents} documents in the indexed corpus; "
                f"{in_provided} of the excerpts provided below contain it verbatim.")

    def ensure_literal_hits(self, results: list[dict], phrase: str, reserve: int = 5,
                            boost_fn=None) -> list[dict]:
        """Return `results` with up to `reserve` literal-hit chunks guaranteed present.

        Literal hits already in `results` count toward the reserve. Missing ones are
        inserted at the front (they are, by definition, the most relevant passages for a
        terminology question) and the list is trimmed from the tail to its original length
        when it would otherwise grow. Returns a new list; input order is otherwise kept.
        """
        hits = self.search(phrase, boost_fn=boost_fn)
        if not hits:
            return list(results)
        present_ids = {r.get("id") for r in results}
        have = sum(1 for h in hits if h.chunk["id"] in present_ids)
        need = max(0, reserve - have)
        to_add = []
        for h in hits:
            if need == 0:
                break
            if h.chunk["id"] not in present_ids:
                to_add.append({**h.chunk, "score": float(h.occurrences),
                               "score_label": f"{h.occurrences} exact", "mode": "literal"})
                need -= 1
        if not to_add:
            return list(results)
        merged = to_add + list(results)
        return merged[:max(len(results), len(to_add))]


def highlight(text: str, phrase: str, mark: str = "**") -> str:
    """Wrap each occurrence of `phrase` in `text` (case-insensitive, raw text) with `mark`."""
    p = re.escape(normalize(phrase)).replace(r"\ ", r"\s+")
    return re.sub(f"({p})", lambda m: f"{mark}{m.group(1)}{mark}", text, flags=re.I)


if __name__ == "__main__":      # python3 literal_search.py "emotional programming" [--whole-words] [--regex]
    import argparse, json, sys
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Exact-phrase search over chunk_metadata.json")
    ap.add_argument("phrase")
    ap.add_argument("--refs", default=str(Path(__file__).parent / "rag-document-search" / "references"))
    ap.add_argument("--whole-words", action="store_true")
    ap.add_argument("--regex", action="store_true", help="treat PHRASE as a regular expression")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    chunks = json.load(open(Path(a.refs) / "chunk_metadata.json", encoding="utf-8"))
    ix = LiteralIndex(chunks)
    hits = ix.search_regex(a.phrase) if a.regex else ix.search(a.phrase, whole_words=a.whole_words)
    docs = sorted(set(h.doc_name for h in hits))
    print(f'"{a.phrase}": {sum(h.occurrences for h in hits)} occurrences in {len(hits)} passages '
          f"across {len(docs)} documents (of {len(chunks):,} passages)")
    for h in hits[:a.top]:
        c = h.chunk
        loc = f"p.{c['page']}" if c.get("page") else f"¶{c.get('paragraph_start')}"
        print(f"  {h.occurrences:2d}x  {c['doc_name'][:60]:60s} {loc:7s} …{h.snippet}…")
    if len(hits) > a.top:
        print(f"  … {len(hits) - a.top} more passages")
    sys.exit(0 if hits else 1)
