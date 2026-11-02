"""exact_match.py — exact term / phrase presence over the chunk index.

Purpose: answer "does this exact term occur in the corpus, and where?" by scanning the
chunk text itself, so a negative is provable. The ranking modes (TF-IDF, BM25,
embeddings) return a top-k of a similarity ranking and cannot prove absence.
Spec:    docs/spec_exact_match_search_v2.md#E0-E5

One implementation, two owners: `streamlit_app.IndexManager` and
`bowen_rag_gui.IndexManager` each hold an `ExactIndex` and delegate `exact_search` /
`exact_presence` to it (test_index_consistency.py pins that both do).

Standard library only; nothing here imports Streamlit or tkinter. `chunks` is never
mutated: the normalised copies live in a parallel list.
"""
from __future__ import annotations

import re
import unicodedata

MODES = ("word", "substring", "regex")

_SOFT_HYPHEN = "­"
_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'",       # curly single quotes
    "“": '"', "”": '"',       # curly double quotes
    "–": "-", "—": "-",       # en / em dash
    "‑": "-",                      # non-breaking hyphen
})
_HYPHEN_BREAK = re.compile(r"(\w)-[ \t]*\n\s*(\w)")
_WS = re.compile(r"\s+")


def normalize(text: str, case: bool = False) -> str:
    """Canonical form used for both chunk text and needle. Idempotent.

    Purpose: make a phrase match regardless of PDF soft-wraps, ligatures and quote styles.
    Spec:    docs/spec_exact_match_search_v2.md#E0
    Tests:   test_exact_match.py::TestE0Normalize

    NFKC (ﬁ -> fi), soft hyphens removed, "word-\\nword" re-joined, curly quotes and
    en/em dashes mapped to ASCII, whitespace runs collapsed to one space, stripped.
    Lower-cased unless `case=True`.
    """
    t = unicodedata.normalize("NFKC", text)
    t = t.replace(_SOFT_HYPHEN, "").translate(_CHAR_MAP)
    t = _HYPHEN_BREAK.sub(r"\1\2", t)
    t = _WS.sub(" ", t).strip()
    return t if case else t.lower()


def _word_pattern(needle: str, case: bool) -> re.Pattern:
    cls = "[A-Za-z0-9]" if case else "[a-z0-9]"
    return re.compile(f"(?<!{cls}){re.escape(needle)}(?!{cls})")


class ExactIndex:
    """Normalised copy of every chunk text, built once at load.

    Purpose: one scan core for exact search and presence (E1-E5).
    Spec:    docs/spec_exact_match_search_v2.md#E0
    Tests:   test_exact_match.py::TestE0Normalize::test_e0_index_does_not_mutate_chunks
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self._scan_texts = [normalize(c["text"]) for c in chunks]
        self._scan_texts_cased = None        # built on first case=True query

    def _texts(self, case: bool) -> list[str]:
        if not case:
            return self._scan_texts
        if self._scan_texts_cased is None:
            self._scan_texts_cased = [normalize(c["text"], case=True) for c in self.chunks]
        return self._scan_texts_cased

    def scan(self, needle: str, mode: str = "word", case: bool = False) -> list[tuple[int, int]]:
        """(chunk_index, hits) for every chunk containing `needle`, in chunk order.

        Purpose: the single matching core behind search() and presence().
        Spec:    docs/spec_exact_match_search_v2.md#E2 (word), #E3 (substring), #E4 (regex)
        Tests:   test_exact_match.py::TestE2Word, TestE3Substring, TestE4Regex

        mode="word": the normalised needle bounded by non-alphanumerics on both sides, so
            "self" does not match "yourself" but "differentiation" matches
            "self-differentiation" (hyphen and apostrophe are boundaries).
        mode="substring": plain occurrence count, no boundaries ("fu" matches "fusion").
        mode="regex": `needle` is a regular expression applied to the normalised text
            (re.I unless case=True). A bad pattern raises re.error; it never returns [].
        An empty needle raises ValueError: "found nothing" must not be a typo's result.
        """
        if mode not in MODES:
            raise ValueError(f"unknown exact-match mode {mode!r}; use one of {MODES}")
        if mode == "regex":
            if not needle:
                raise ValueError("empty regular expression")
            rx = re.compile(needle, 0 if case else re.I)
            return [(i, n) for i, t in enumerate(self._texts(case))
                    if (n := sum(1 for _ in rx.finditer(t)))]
        n_needle = normalize(needle, case=case)
        if not n_needle:
            raise ValueError("empty search phrase")
        texts = self._texts(case)
        if mode == "substring":
            return [(i, t.count(n_needle)) for i, t in enumerate(texts) if n_needle in t]
        rx = _word_pattern(n_needle, case)
        return [(i, n) for i, t in enumerate(texts)
                if n_needle in t and (n := sum(1 for _ in rx.finditer(t)))]

    def search(self, needle: str, mode: str = "word", case: bool = False) -> list[dict]:
        """Every matching chunk as {**chunk, "hits": n, "mode": "exact"}.

        Purpose: return ALL containing chunks — no top_k, no per-document collapse, no
            boost — sorted by hits (desc), then chunk id.
        Spec:    docs/spec_exact_match_search_v2.md#E1
        Tests:   test_exact_match.py::TestE1Search
        """
        out = [{**self.chunks[i], "hits": n, "mode": "exact"}
               for i, n in self.scan(needle, mode=mode, case=case)]
        out.sort(key=lambda c: (-c["hits"], c.get("id", 0)))
        return out

    def presence(self, needle: str, mode: str = "word", case: bool = False,
                 locator_fn=None) -> dict:
        """Corpus-level facts for `needle`, from one scan of the whole index.

        Purpose: the provable answer to "is X in the corpus?" (counts per document,
            locators, OCR provenance) plus loose (substring) counts, so a lost space or
            a plural shows as "0 whole-word, 1 loose" rather than a clean zero.
        Spec:    docs/spec_exact_match_search_v2.md#E5
        Tests:   test_exact_match.py::TestE5Presence

        `loose_*` are the substring-mode counts in mode "word"; in the other modes they
        equal the main counts. `docs` is sorted by hits (desc), then document name.
        """
        hits = self.search(needle, mode=mode, case=case)
        docs: dict = {}
        for c in hits:
            d = docs.setdefault(c["doc_name"], {
                "doc_name": c["doc_name"], "hits": 0, "chunks": 0,
                "ocr": bool(c.get("ocr")), "ocr_confidence": c.get("ocr_confidence"),
                "locators": []})
            d["hits"] += c["hits"]
            d["chunks"] += 1
            if locator_fn is not None:
                loc = locator_fn(c)
                if loc:
                    d["locators"].append(loc)
        total = sum(c["hits"] for c in hits)
        if mode == "word":
            loose = self.scan(needle, mode="substring", case=case)
            loose_total = sum(n for _, n in loose)
            loose_docs = len({self.chunks[i]["doc_name"] for i, _ in loose})
        else:
            loose_total, loose_docs = total, len(docs)
        return {
            "needle": needle, "mode": mode, "case": case,
            "found": bool(hits),
            "total_hits": total,
            "chunk_count": len(hits),
            "doc_count": len(docs),
            "docs": sorted(docs.values(), key=lambda d: (-d["hits"], d["doc_name"])),
            "loose_total_hits": loose_total,
            "loose_doc_count": loose_docs,
            "corpus_chunks": len(self.chunks),
        }
