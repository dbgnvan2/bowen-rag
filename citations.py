"""
Citation formatting for Bowen RAG reports.

Purpose: Render in-text citations and reference-list entries in a user-selected
         style (APA, MLA, Chicago author-date, Harvard, Vancouver), reading only
         from verifiable bibliographic data in sources.yml — never fabricating.
Spec:    docs/spec_citation_styles.md  (M1, M2)
Tests:   test_citations.py

Design contract (why the LLM can never fabricate a citation):
  The report LLM only ever emits numbered placeholders [N] (optionally [N, p. X]
  when quoting). This module deterministically rewrites those placeholders into
  the chosen style and builds the reference list, using ONLY the fields present
  in sources.yml. Unknown fields render with each style's real "unknown"
  convention (e.g. APA "n.d."). Nothing is invented.

Known simplifications (documented, not bugs):
  - Chicago is the author-date variant (parenthetical), not notes-bibliography.
  - Harvard follows the "Cite Them Right" variant.
  - Multi-author et-al. handling is pragmatic: in-text uses "et al." for 3+
    authors; reference lists name every author we hold (this corpus rarely
    exceeds three).
  - Vancouver numbering follows the source-list order handed to the LLM (a fixed
    numbered bibliography), not strict order-of-first-appearance.
"""

from __future__ import annotations

import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Style registry ───────────────────────────────────────────────────────────
STYLES = ["APA", "MLA", "Chicago", "Harvard", "Vancouver"]
NUMBERED_STYLES = {"Vancouver"}
DEFAULT_STYLE = "APA"


def normalize_style(style: str) -> str:
    """Return a canonical style name, defaulting to APA for anything unknown."""
    if not style:
        return DEFAULT_STYLE
    for s in STYLES:
        if s.lower() == str(style).strip().lower():
            return s
    return DEFAULT_STYLE


# ── Bibliography data: load + match ──────────────────────────────────────────
def load_sources(base_dir: Path | None = None) -> list:
    """
    Load sources.yml into a list of record dicts (each carries its own `pattern`).
    Missing/invalid file → [] (callers then fall back to filename-derived records).
    """
    base = Path(base_dir) if base_dir else BASE_DIR
    config = base / "sources.yml"
    if not config.exists():
        return []
    try:
        import yaml
        with open(config, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        records = []
        for r in data.get("sources", []):
            if isinstance(r, dict) and r.get("pattern"):
                records.append(r)
        return records
    except Exception:
        return []


def dump_sources(records: list) -> str:
    """
    Serialize records back to sources.yml text (for the in-app editor). Drops
    internal keys (those starting with '_', e.g. the synth fallback marker) and
    empty values, so a saved file stays clean and re-loadable by load_sources.
    """
    import yaml
    FIELD_ORDER = ["pattern", "type", "authors", "year", "title", "container_title",
                   "publisher", "publisher_place", "volume", "issue", "page",
                   "editor", "url", "doi", "verified"]
    clean = []
    for r in records:
        rec = {}
        for k in FIELD_ORDER:
            if k not in r:
                continue
            v = r[k]
            if v in (None, "", []) and k != "verified":
                continue
            rec[k] = v
        # keep any non-internal keys not in FIELD_ORDER
        for k, v in r.items():
            if k not in rec and not k.startswith("_") and v not in (None, "", []):
                rec[k] = v
        clean.append(rec)
    return yaml.safe_dump({"sources": clean}, sort_keys=False, allow_unicode=True, width=100)


def match_source(doc_name: str, sources: list) -> dict | None:
    """
    Record whose `pattern` is a substring of doc_name (case-insensitive). When
    several patterns match, the LONGEST (most specific) wins — so a full-filename
    seed pattern always beats a short hand-authored catch-all, and matching does
    not depend on record order. Ties break on first occurrence.
    """
    dn = (doc_name or "").lower()
    best = None
    best_len = -1
    for r in sources:
        pat = str(r.get("pattern", "")).lower()
        if pat and pat in dn and len(pat) > best_len:
            best, best_len = r, len(pat)
    return best


# ── Filename → fallback record (no fabrication: only what the name/author map gives) ─
def clean_title_from_filename(doc_name: str) -> str:
    """Strip a leading 'NNN_' index and underscores to make a readable title guess."""
    name = re.sub(r'^\d+[_\s\-]+', '', doc_name or '')
    name = name.replace('_', ' ')
    name = re.sub(r'\s{2,}', ' ', name).strip()
    return name or (doc_name or 'Untitled')


def synth_record(doc_name: str, author_lookup=None) -> dict:
    """
    Build a fallback record for a doc with no sources.yml entry.
    `author_lookup` is an optional callable(doc_name) -> display string (the app's
    doc_author). Nothing here is invented: unknown author → "Unknown", no year → n.d.
    """
    author_str = ""
    if author_lookup is not None:
        try:
            author_str = author_lookup(doc_name) or ""
        except Exception:
            author_str = ""
    if author_str and author_str.lower() != "unknown":
        authors = parse_author_string(author_str)
    else:
        authors = []
    return {
        "pattern": doc_name,
        "type": "generic",
        "authors": authors,
        "year": "n.d.",
        "title": clean_title_from_filename(doc_name),
        "container_title": "",
        "publisher": "",
        "publisher_place": "",
        "volume": "",
        "issue": "",
        "page": "",
        "editor": "",
        "url": "",
        "doi": "",
        "verified": False,
        "_fallback": True,
    }


def record_for_doc(doc_name: str, sources: list, author_lookup=None) -> dict:
    """Matched sources.yml record, else a filename/author-map fallback record."""
    return match_source(doc_name, sources) or synth_record(doc_name, author_lookup)


# ── Author parsing + name formatting ─────────────────────────────────────────
def parse_author_string(s: str) -> list:
    """
    Parse a display string like "Murray Bowen", "Bowen & Kerr", "Kerr & Bowen"
    into a list of {"family", "given"} dicts. Best-effort — seeded records are
    marked unverified so a human can correct given names/initials.
    """
    if not s:
        return []
    parts = re.split(r'\s*(?:&|,|;|/|\band\b)\s*', s.strip())
    authors = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        tokens = p.split()
        if len(tokens) == 1:
            authors.append({"family": tokens[0], "given": ""})
        else:
            authors.append({"family": tokens[-1], "given": " ".join(tokens[:-1])})
    return authors


def _authors_list(record: dict) -> list:
    """Normalize a record's authors into a list of {family, given} dicts."""
    a = record.get("authors")
    if not a:
        return []
    if isinstance(a, str):
        return parse_author_string(a)
    out = []
    for item in a:
        if isinstance(item, dict):
            out.append({"family": (item.get("family") or "").strip(),
                        "given": (item.get("given") or "").strip()})
        elif isinstance(item, str):
            out.extend(parse_author_string(item))
    return out


def _initials(given: str, period=True, spaced=True) -> str:
    """'Murray' -> 'M.'  ;  'Michael E' -> 'M. E.'  ;  plain -> 'ME'."""
    if not given:
        return ""
    letters = [t[0].upper() for t in re.split(r'[\s\-]+', given) if t]
    if not period and not spaced:
        return "".join(letters)
    sep = " " if spaced else ""
    return sep.join(f"{ch}." for ch in letters)


def _year_str(record: dict) -> str:
    y = record.get("year")
    if y is None or y == "":
        return "n.d."
    return str(y)


def _page_locator(page, style: str) -> str:
    """Format a page locator for an in-text quote, e.g. 'p. 45' or 'pp. 45-47'."""
    if page in (None, ""):
        return ""
    p = str(page).strip()
    plural = bool(re.search(r'[–\-]', p))
    if style == "MLA":            # MLA in-text uses a bare number
        return p
    if style == "Chicago":        # Chicago author-date uses a bare number after the year
        return p
    return f"{'pp.' if plural else 'p.'} {p}"


# ── Author formatting per style (reference-list form) ────────────────────────
def _fmt_authors_reference(authors: list, style: str) -> str:
    if not authors:
        return ""

    def apa_name(a):
        ini = _initials(a["given"])
        return f"{a['family']}, {ini}" if ini else a["family"]

    def mla_first(a):
        return f"{a['family']}, {a['given']}" if a["given"] else a["family"]

    def natural(a):  # "Given Family" for subsequent Chicago/MLA authors
        return f"{a['given']} {a['family']}".strip() if a["given"] else a["family"]

    def vanc_name(a):
        ini = _initials(a["given"], period=False, spaced=False)
        return f"{a['family']} {ini}".strip() if ini else a["family"]

    if style in ("APA", "Harvard"):
        names = [apa_name(a) for a in authors]
        if len(names) == 1:
            return names[0]
        # APA puts a comma before '&' even for two authors; Harvard uses 'and', no comma.
        if style == "APA":
            return ", ".join(names[:-1]) + ", & " + names[-1]
        return ", ".join(names[:-1]) + " and " + names[-1]

    if style == "MLA":
        if len(authors) == 1:
            return mla_first(authors[0])
        if len(authors) == 2:
            return f"{mla_first(authors[0])}, and {natural(authors[1])}"
        return f"{mla_first(authors[0])}, et al."

    if style == "Chicago":
        if len(authors) == 1:
            return mla_first(authors[0])
        if len(authors) == 2:
            return f"{mla_first(authors[0])}, and {natural(authors[1])}"
        middle = ", ".join(natural(a) for a in authors[1:-1])
        return f"{mla_first(authors[0])}, {middle}, and {natural(authors[-1])}"

    if style == "Vancouver":
        return ", ".join(vanc_name(a) for a in authors)

    return ", ".join(a["family"] for a in authors)


def _fmt_authors_intext(authors: list, style: str) -> str:
    """Family-name form used inside a parenthetical citation."""
    if not authors:
        return ""
    fams = [a["family"] for a in authors]
    if len(fams) == 1:
        return fams[0]
    if len(fams) == 2:
        joiner = " & " if style in ("APA", "Harvard") else " and "
        return f"{fams[0]}{joiner}{fams[1]}"
    return f"{fams[0]} et al."


# ── Reference-list entry per style ───────────────────────────────────────────
def _italic(s: str) -> str:
    return f"*{s}*" if s else ""


def _clean_join(parts, sep=" ") -> str:
    return sep.join(p for p in parts if p)


def format_reference(record: dict, style: str) -> str:
    """One reference-list entry string in the given style."""
    style = normalize_style(style)
    rtype = (record.get("type") or "generic").lower()
    authors = _authors_list(record)
    year = _year_str(record)
    title = (record.get("title") or "").strip()
    container = (record.get("container_title") or "").strip()
    publisher = (record.get("publisher") or "").strip()
    place = (record.get("publisher_place") or "").strip()
    vol = str(record.get("volume") or "").strip()
    issue = str(record.get("issue") or "").strip()
    pages = str(record.get("page") or "").strip()
    medium = {"speech": " [Audio recording]", "interview": " [Interview]",
              "webpage": "", "report": " [Report]",
              "manuscript": " [Unpublished manuscript]"}.get(rtype, "")

    A = _fmt_authors_reference(authors, style)
    is_article = rtype in ("article-journal", "chapter")

    if style == "APA":
        head = f"{A} ({year})." if A else f"({year})."
        if is_article:
            vi = f"{_italic(vol)}({issue})" if issue else _italic(vol)  # APA italicizes volume
            tail = _clean_join([f"{_italic(container)}," if container else "",
                                f"{vi}," if vi else "", pages], " ").rstrip(",")
            return _clean_join([head, f"{title}.", tail + ("." if tail else "")], " ")
        return _clean_join([head, f"{_italic(title)}{medium}.",
                            f"{publisher}." if publisher else ""], " ")

    if style == "Harvard":
        head = f"{A} ({year})" if A else f"({year})"
        if is_article:
            vi = f"{vol}({issue})" if issue else vol
            loc = _clean_join([f"{_italic(container)},", vi and f"{vi},", pages and f"pp. {pages}"], " ")
            return _clean_join([f"{head}", f"'{title}',", loc]).rstrip(",") + "."
        pub = _clean_join([place and f"{place}:", publisher], " ")
        return _clean_join([f"{head}", f"{_italic(title)}{medium}.", pub and pub + "."], " ")

    if style == "MLA":
        if is_article:
            loc = _clean_join([_italic(container) + "," if container else "",
                               f"vol. {vol}," if vol else "",
                               f"no. {issue}," if issue else "",
                               f"{year}," if year else "",
                               f"pp. {pages}" if pages else ""], " ").rstrip(",")
            return _clean_join([f"{A}." if A else "", f'"{title}."', loc]).rstrip(",") + "."
        return _clean_join([f"{A}." if A else "", f"{_italic(title)}{medium}.",
                            _clean_join([publisher + "," if publisher else "", year]).rstrip(",") + "."], " ")

    if style == "Chicago":
        head = f"{A}. {year}." if A else f"{year}."
        if is_article:
            vi = f"{vol} ({issue})" if issue else vol
            loc = _clean_join([_italic(container), vi and f"{vi}:", pages], " ")
            return _clean_join([head, f'"{title}."', loc]).rstrip(":") + "."
        pub = _clean_join([place and f"{place}:", publisher], " ")
        return _clean_join([head, f"{_italic(title)}{medium}.", pub and pub + "."], " ")

    if style == "Vancouver":
        if is_article:
            vip = _clean_join([f"{year};" if year else "", vol,
                               f"({issue})" if issue else "",
                               f":{pages}" if pages else ""], "")
            return _clean_join([f"{A}." if A else "", f"{title}.",
                                container and f"{container}.", vip]).rstrip(".") + "."
        pub = _clean_join([place and f"{place}:", publisher], " ")
        tail = _clean_join([pub, year and f"; {year}" if pub else (year and f"{year}")], "")
        return _clean_join([f"{A}." if A else "", f"{title}{medium}.", tail]).rstrip(";. ") + "."

    return f"{A}. {title}."


# ── In-text citation per style ───────────────────────────────────────────────
def _intext_inner(record: dict, style: str, page=None) -> str:
    """The in-text citation WITHOUT surrounding parentheses, so grouped citations
    can be joined inside one set of parentheses (e.g. 'Bowen, 1978; Kerr, 1988')."""
    authors = _authors_list(record)
    fam = _fmt_authors_intext(authors, style)
    if not fam:
        # No author at all: fall back to a short title so the citation still points somewhere.
        fam = (record.get("title") or "Source").strip().split(".")[0][:40]
    year = _year_str(record)
    loc = _page_locator(page, style)
    if style == "MLA":                       # Bowen 45 — author + page, no year
        return f"{fam} {loc}".rstrip() if loc else fam
    if style == "Chicago":                   # Bowen 1978, 45
        return f"{fam} {year}, {loc}" if loc else f"{fam} {year}"
    return f"{fam}, {year}, {loc}" if loc else f"{fam}, {year}"   # APA / Harvard


def format_intext(record: dict, style: str, page=None, number: int | None = None) -> str:
    """
    In-text citation for a SINGLE source. Numbered styles return '[n]';
    author-date/author-page styles return a parenthetical, adding a page locator
    when `page` is supplied (quotes).
    """
    style = normalize_style(style)
    if style in NUMBERED_STYLES:
        return f"[{number}]" if number is not None else ""
    return "(" + _intext_inner(record, style, page=page) + ")"


# ── Ordering + list building ─────────────────────────────────────────────────
def order_references(num_records: list, style: str) -> list:
    """
    num_records: list of (number, record). Returns it ordered per style:
    numbered styles keep number order; author styles sort alphabetically by first
    author family then year.
    """
    style = normalize_style(style)
    if style in NUMBERED_STYLES:
        return sorted(num_records, key=lambda nr: nr[0])

    def key(nr):
        rec = nr[1]
        authors = _authors_list(rec)
        fam = authors[0]["family"].lower() if authors else (rec.get("title") or "").lower()
        return (fam, str(rec.get("year") or ""))
    return sorted(num_records, key=key)


def build_reference_list_md(num_to_record: dict, style: str, cited_numbers=None) -> str:
    """
    Build the markdown reference list. `num_to_record` maps the LLM's [N] to a
    record. `cited_numbers` optionally restricts to numbers actually cited.
    Numbered styles → 'N. entry' in number order; author styles → alphabetical,
    unnumbered, one entry per line.
    """
    style = normalize_style(style)
    items = [(n, r) for n, r in num_to_record.items()
             if cited_numbers is None or n in cited_numbers]
    ordered = order_references(items, style)
    if style in NUMBERED_STYLES:
        return "\n".join(f"{n}. {format_reference(r, style)}" for n, r in ordered)
    return "\n\n".join(format_reference(r, style) for _, r in ordered)


# ── Post-processor: rewrite [[N]] / [[N, p. X]] / [[1, 3]] markers ────────────
# Citations use DOUBLE brackets so they can never collide with SINGLE brackets
# that appear inside quoted source text — a source's own footnote "[1]", "[sic]",
# "[...]" — which must pass through untouched. This matters in a fidelity tool:
# the corpus contains bracketed footnote numbers inside passages the report may
# quote verbatim, and rewriting one into "(Author, year)" would corrupt the quote.
_CITE_RE = re.compile(r'\[\[\s*([0-9][^\[\]]*?)\s*\]\]')
# Inside a group, pull each number with an optional page. The page REQUIRES a 'p'
# so that "1, 3" parses as two numbers (not number 1 on page 3).
_NUM_PAGE_RE = re.compile(r'(\d{1,3})(?:\s*[,;]?\s*p+\.?\s*(\d+(?:\s*[–\-]\s*\d+)?))?')


def _parse_group(inner: str) -> list:
    """Parse the inside of a [[...]] marker into [(number, page_or_None), ...]."""
    return [(int(n), (pg or None)) for n, pg in _NUM_PAGE_RE.findall(inner)]


def apply_intext_citations(text: str, num_to_record: dict, style: str) -> str:
    """
    Replace [[N]] / [[N, p. X]] / [[1, 3]] markers with styled in-text citations.
    Grouped markers are expanded and joined (author styles: '(A, 1978; B, 1988)';
    numbered: '[1, 3]'). A marker with no KNOWN reference number is left untouched.
    Single brackets in the text (quoted source footnotes) are never matched.
    """
    style = normalize_style(style)

    def repl(m):
        pairs = _parse_group(m.group(1))
        known = [(n, pg) for n, pg in pairs if n in num_to_record]
        if not known:
            return m.group(0)             # nothing citable inside — leave as-is
        if style in NUMBERED_STYLES:
            return "[" + ", ".join(str(n) for n, _ in known) + "]"
        parts = [_intext_inner(num_to_record[n], style, page=pg) for n, pg in known]
        return "(" + "; ".join(parts) + ")"

    return _CITE_RE.sub(repl, text or "")


def cited_numbers(text: str, valid: set | None = None) -> set:
    """
    Reference numbers actually cited in `text` ([[N]] / grouped [[1, 3]] markers).
    If `valid` is given, only those numbers count.
    """
    nums = set()
    for m in _CITE_RE.finditer(text or ""):
        for n, _ in _parse_group(m.group(1)):
            if valid is None or n in valid:
                nums.add(n)
    return nums
