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

import os
import re
import string
import sys
import unicodedata
from pathlib import Path

MODES = ("word", "substring", "regex")

# How many exact-match passages a model-feeding retrieval guarantees (spec E6). The
# budget for a request is min(this, max(1, k // 2)); 0 disables the union.
EXACT_UNION_BUDGET = int(os.environ.get("EXACT_UNION_BUDGET", 5))
# A whole unquoted query of at most this many words is checked as one phrase (E6a).
MAX_SHORT_QUERY_WORDS = 6

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


def count_matches(text: str, needle: str, mode: str = "word", case: bool = False) -> int:
    """Occurrences of `needle` in one raw `text`, by the same rules as ExactIndex.scan.

    Purpose: count presence over the text actually sent to a model (which may include
        neighbouring chunks), not over the chunk list.
    Spec:    docs/spec_exact_match_search_v2.md#E8
    Tests:   test_exact_match.py::TestE8Verification::test_e8_counts_text_actually_sent
    """
    found = ExactIndex([{"text": text}]).scan(needle, mode=mode, case=case)
    return found[0][1] if found else 0


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
        equal the main counts. `docs` is sorted by hits (desc), then document name; each document's
        locators are in reading (chunk) order.
        """
        hits = self.search(needle, mode=mode, case=case)
        docs: dict = {}
        for c in sorted(hits, key=lambda c: c.get("id", 0)):     # locators in reading order
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

    def keyword_hits(self, query: str, stop, stems) -> list[tuple[int, int, bool]]:
        """(chunk_index, hits, phrase) for Keyword mode: every chunk that has all the terms.

        Purpose: Keyword mode's matching, on the normalised text, shared by both apps.
        Spec:    docs/spec_exact_match_search_v2.md#E10
        Tests:   test_exact_match.py::TestE10Keyword

        Every "quoted" phrase must occur (whole-word, as in ExactIndex.scan) and counts 3x.
        Every unquoted term must occur (AND), counted by its `stems` variants as substrings
        (the old Keyword counting). `phrase` is True when the chunk contains a quoted phrase
        or the unquoted terms as one phrase (keyword_terms); callers rank those first.
        """
        quoted, phrase, terms = keyword_terms(query, stop)
        if not quoted and not terms:
            return []
        candidates = None
        quoted_hits: dict = {}
        for q in quoted:
            found = dict(self.scan(q))
            candidates = set(found) if candidates is None else candidates & set(found)
            for i, n in found.items():
                quoted_hits[i] = quoted_hits.get(i, 0) + 3 * n
        in_phrase = {i for i, _ in self.scan(phrase)} if phrase else set()
        term_sets = [set(stems(t)) for t in terms]
        out = []
        for i in (sorted(candidates) if candidates is not None else range(len(self.chunks))):
            t = self._scan_texts[i]
            counts = [max(t.count(v) for v in variants) for variants in term_sets]
            if not all(counts):
                continue
            hits = quoted_hits.get(i, 0) + sum(counts)
            if hits:
                out.append((i, hits, bool(quoted) or i in in_phrase))
        return out


_TOKEN = re.compile(r"[\w'-]+")


def keyword_terms(query: str, stop) -> tuple[list[str], str | None, list[str]]:
    """(quoted phrases, unquoted phrase or None, unquoted terms) for Keyword mode.

    Purpose: tokenise so punctuation never sticks to a term ("programming?" -> "programming").
    Spec:    docs/spec_exact_match_search_v2.md#E10
    Tests:   test_exact_match.py::TestE10Keyword::test_e10_punctuation_stripped

    Tokens are re.findall(r"[\\w'-]+") with leading/trailing apostrophes and hyphens
    stripped (so 'programmed' in single quotes is the word). Terms are the tokens longer
    than 2 characters that are not in `stop`. The unquoted phrase is the run of tokens from
    the first term to the last, when it spans two or more tokens: "What is emotional
    programming?" -> "emotional programming"; "differentiation of self" keeps "of".
    """
    quoted = [q for q in (normalize(m) for m in _QUOTED.findall(query)) if q]
    remainder = _QUOTED.sub(" ", query)
    tokens = [t.strip("'-") for t in _TOKEN.findall(normalize(remainder))]
    tokens = [t for t in tokens if t]
    keep = [i for i, t in enumerate(tokens) if len(t) > 2 and t not in stop]
    terms = [tokens[i] for i in keep]
    phrase = " ".join(tokens[keep[0]:keep[-1] + 1]) if keep and keep[-1] > keep[0] else None
    return quoted, phrase, terms


# ── Retrieval guarantee and prompt guard (spec E6a, E6, E8) ───────────────────

_QUOTED = re.compile(r'["\u201c]([^"\u201c\u201d]+)["\u201d]')
_EDGE = string.whitespace + string.punctuation + "\u2018\u2019\u201c\u201d\u2026\u2013\u2014\u00bf\u00a1"


QUESTION_FRAMES_FILE = Path(__file__).resolve().parent / "question_frames.yml"
_FRAMES_CACHE: dict = {}


def load_question_frames(path: Path = QUESTION_FRAMES_FILE) -> tuple[list, list]:
    """(leading, trailing) question-frame word tuples from question_frames.yml, longest
    first. A missing or unreadable file is reported on stderr and strips nothing.

    Spec:  docs/spec_exact_match_search_v2.md#E6a (frame stripping, Dave 2026-11-01)
    Tests: test_exact_match.py::TestE6aPhrases::test_e6a_frames_file_missing_is_reported
    """
    key = str(path)
    if key in _FRAMES_CACHE:
        return _FRAMES_CACHE[key]
    try:
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        frames = tuple(
            sorted({tuple(normalize(str(f)).split()) for f in data.get(side) or []} - {()},
                   key=len, reverse=True)
            for side in ("leading", "trailing"))
    except Exception as e:
        print(f"WARNING: could not load {path} ({e}); short questions are checked "
              "whole, with no question words removed.", file=sys.stderr)
        frames = ((), ())
    _FRAMES_CACHE[key] = frames
    return frames


def strip_question_frame(text: str, frames=None) -> str:
    """`text` with leading/trailing question-frame words removed (whole words, repeatedly).

    Purpose: "What is emotional programming?" -> "emotional programming".
    Spec:    docs/spec_exact_match_search_v2.md#E6a
    Tests:   test_exact_match.py::TestE6aPhrases::test_e6a_question_frame_stripped
    """
    leading, trailing = frames if frames is not None else load_question_frames()
    words = text.split()
    norm = [normalize(w.strip(_EDGE)) for w in words]
    a, b = 0, len(words)
    changed = True
    while changed and a < b:
        changed = False
        for f in leading:
            if tuple(norm[a:a + len(f)]) == f and a + len(f) < b:
                a += len(f); changed = True; break
        for f in trailing:
            if tuple(norm[b - len(f):b]) == f and b - len(f) > a:
                b -= len(f); changed = True; break
    if (a, b) == (0, len(words)):
        return text.strip(_EDGE)            # nothing removed: keep the text as typed
    return " ".join(words[a:b]).strip(_EDGE)


def phrases_for(query: str, frames=None) -> tuple[list[str], str]:
    """The phrases to check exactly for a query, and why.

    Purpose: a deterministic rule for which strings get an exact-match check.
    Spec:    docs/spec_exact_match_search_v2.md#E6a
    Tests:   test_exact_match.py::TestE6aPhrases

    (1) every "…" / “…” quoted span -> reason "quoted"; else (2) the query stripped of
    surrounding punctuation and of question-frame words (question_frames.yml: "What is",
    "define", a trailing "mean", ...), if at most MAX_SHORT_QUERY_WORDS words remain ->
    "question-term" when frame words were removed, "short-query" when none were;
    else (3) nothing -> "none". Duplicates (after normalising) and spans that normalise
    to nothing are dropped.
    """
    def dedupe(items):
        seen, out = set(), []
        for it in items:
            key = normalize(it)
            if key and key not in seen:
                seen.add(key)
                out.append(it.strip())
        return out

    quoted = dedupe(_QUOTED.findall(query))
    if quoted:
        return quoted, "quoted"
    whole = query.strip(_EDGE)
    term = strip_question_frame(whole, frames)
    if term and len(term.split()) <= MAX_SHORT_QUERY_WORDS and normalize(term):
        return [term], ("short-query" if normalize(term) == normalize(whole) else "question-term")
    return [], "none"


def union_budget(k: int, configured: int | None = None) -> int:
    """min(configured, max(1, k // 2)); configured defaults to EXACT_UNION_BUDGET.

    Purpose: never let guaranteed exact-match passages crowd out more than half of k.
    Spec:    docs/spec_exact_match_search_v2.md#E6
    Tests:   test_exact_match.py::TestE6Union::test_e6_budget_rule
    """
    configured = EXACT_UNION_BUDGET if configured is None else configured
    if configured <= 0:
        return 0
    return min(configured, max(1, k // 2))


def union_exact(ranked: list[dict], exact_hits: list[dict], budget: int,
                k: int | None = None) -> list[dict]:
    """`ranked` with up to `budget` exact-match chunks guaranteed present.

    Purpose: make sure passages containing the user's phrase reach the model when any
        exist, without changing how many passages the user asked for.
    Spec:    docs/spec_exact_match_search_v2.md#E6
    Tests:   test_exact_match.py::TestE6Union

    Exact hits already in `ranked` (by id) count toward the budget. Missing ones are
    prepended (best first: `exact_hits` is sorted by hits). The result is trimmed to
    `k` (default len(ranked)), dropping non-containing passages from the tail first so
    the guarantee survives the trim. Returns a new list; inputs are not modified.
    Deviation from the spec text: `k` is optional; the spec trims to len(ranked), which
    is the default. Callers pass the user's k so a ranker that returned fewer than k
    passages (top-docs, author filter) does not also cap the guarantee at zero.
    """
    limit = len(ranked) if k is None else max(k, 0)
    hit_ids = {h["id"] for h in exact_hits}
    present = {r.get("id") for r in ranked}
    need = max(0, budget - len(hit_ids & present))
    added = []
    for h in exact_hits:
        if need == 0:
            break
        if h["id"] not in present:
            added.append({**h, "score": float(h["hits"]),
                          "score_label": f"{h['hits']} exact", "exact_union": True})
            need -= 1
    out, seen = [], set()
    for c in added + list(ranked):
        if c.get("id") not in seen:
            seen.add(c.get("id"))
            out.append(c)
    while len(out) > limit:
        drop = next((j for j in range(len(out) - 1, -1, -1)
                     if out[j].get("id") not in hit_ids), len(out) - 1)
        out.pop(drop)
    return out


def exact_guard(idx, query: str, ranked: list[dict], k: int | None = None,
                union: bool = True, budget: int | None = None, keep=None):
    """Run E6a, the presence check and (optionally) the E6 union for one request.

    Purpose: the one entry point every model-feeding dispatch calls (Chat, Report,
        bowen_ask.py, the desktop GUI), so they behave identically.
    Spec:    docs/spec_exact_match_search_v2.md#E6
    Tests:   test_exact_match.py::TestE6Union::test_e6_guard_*

    `idx` is either IndexManager (anything with exact_search / exact_presence). `keep`
    filters candidate exact hits (the Chat author filter). Returns
    (chunks, phrases, reason, presence_by_phrase).
    """
    phrases, reason = phrases_for(query)
    k = len(ranked) if k is None else k
    b = union_budget(k) if budget is None else budget
    out = list(ranked)
    presence = {}
    for p in phrases:
        presence[p] = idx.exact_presence(p)
        if union and b > 0 and presence[p]["found"]:
            hits = idx.exact_search(p)
            if keep is not None:
                hits = [h for h in hits if keep(h)]
            out = union_exact(out, hits, b, k=k)
    return out, phrases, reason, presence


VERIFICATION_HEADER = "EXACT-MATCH VERIFICATION"


def _n(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def verification_block(presence_by_phrase: dict, provided: list, reason: str,
                       note: str = "") -> str:
    """The computed statement placed at the top of the model's user message and in the UI.

    Purpose: every "is X in the corpus" statement the user sees comes from a scan of
        the whole index, never from the model's reading of k excerpts.
    Spec:    docs/spec_exact_match_search_v2.md#E8
    Tests:   test_exact_match.py::TestE8Verification

    `provided` is the list of texts actually sent to the model (dicts with "text", or
    strings); "K of N" is counted over them with the same matching rules. `note` is
    appended verbatim (e.g. that the excerpts were hand-staged).
    """
    texts = [t["text"] if isinstance(t, dict) else t for t in provided]
    if reason == "none" or not presence_by_phrase:
        lines = [f"{VERIFICATION_HEADER}: no exact-match check was run (no phrase was "
                 f"quoted and the question is longer than {MAX_SHORT_QUERY_WORDS} words). "
                 "Nothing here says whether any term occurs in the corpus. To have a term "
                 'checked, put it in double quotes, e.g. "emotional cutoff".']
    else:
        corpus = next(iter(presence_by_phrase.values())).get("corpus_chunks")
        scope = f"all {corpus:,} indexed passages" if corpus else "the whole indexed corpus"
        lines = [f"{VERIFICATION_HEADER} (computed by the system by scanning {scope}; "
                 "a count, not a judgement):"]
        for phrase, p in presence_by_phrase.items():
            sent = sum(1 for t in texts if count_matches(t, phrase, p["mode"], p["case"]))
            in_sent = f"{sent} of the {len(texts)} excerpts provided contain it."
            loose = ""
            if p["mode"] == "word" and p["loose_total_hits"] > p["total_hits"]:
                extra = p["loose_total_hits"] - p["total_hits"]
                loose = (f" A further {extra} occurrence{'s run' if extra != 1 else ' runs'} "
                         "into a neighbouring word (for example a space lost in the scan) "
                         f"and {'are' if extra != 1 else 'is'} not counted as whole-word matches.")
            if p["found"]:
                lines.append(f'- "{phrase}": occurs {_n(p["total_hits"], "time")} in '
                             f'{_n(p["chunk_count"], "passage")} across '
                             f'{_n(p["doc_count"], "document")} of the indexed corpus. '
                             f"{in_sent}{loose}")
            elif reason == "short-query":
                lines.append(f'- "{phrase}" (the whole question, checked as one phrase): '
                             "does not occur verbatim in the indexed corpus. This says "
                             "nothing about the individual words or any shorter term in it; "
                             "no shorter term was checked. To have a term checked, put it in "
                             f"double quotes. {in_sent}{loose}")
            elif reason == "question-term":
                lines.append(f'- "{phrase}" (the question with its question words removed): '
                             "does not occur as a whole word or phrase anywhere in the "
                             "indexed corpus. To have a different term checked, put it in "
                             f"double quotes. {in_sent}{loose}")
            else:
                lines.append(f'- "{phrase}": does not occur as a whole word or phrase '
                             f"anywhere in the indexed corpus. {in_sent}{loose}")
    if note:
        lines.append(note)
    return "\n".join(lines)


def highlight(text: str, phrase: str, mark: str = "**") -> str:
    """Wrap each occurrence of `phrase` in raw `text` with `mark` (case-insensitive;
    whitespace in the phrase matches any whitespace run, so soft-wraps still match).

    Spec:  docs/spec_exact_match_search_v2.md#E9
    Tests: test_exact_match.py::TestCarriedOver::test_highlight
    """
    p = re.escape(normalize(phrase)).replace(r"\ ", r"\s+")
    return re.sub(f"({p})", lambda m: f"{mark}{m.group(1)}{mark}", text, flags=re.I)
