#!/usr/bin/env python3
"""Extract bibliographic headers from non-chapter source files (stream B).

The non-chapter material carries its citation info two ways:
  1. A structured metadata block at the top of the file (FSJ articles):
        Family Systems Journal Vol. 13.1
        Title: Triangles, Fusion, and the Challenge to be a Self
        Author: Brooks
        Pages: 63-80
  2. Encoded in the filename (transcripts, lectures, reviews):
        "Beautiful Boy Review - Kathleen Kerr - 2023-02-16.txt"
        "2011 04 Making a Difference ... - Michael Kerr - 2011-04-23.txt"

Chapter docs (matched by chapter_map.yml doc_pattern) are skipped — their metadata
comes from the chapter map, not the file header.

Output: headers_candidates.yml — one candidate record per non-chapter file, every
field marked `verified: false` for human review (matches seed_sources.py convention).
Nothing here overwrites sources.yml; it is a review artifact.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
SRC = REPO / "source_files"
OUT = REPO / "headers_candidates.yml"

YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
# "(1943-2017)" are life dates, "1976-1978" a span: neither is a publication year.
RANGE_RE = re.compile(r"(?<!\d)\d{4}\s*[-\u2013]\s*\d{4}(?!\d)")
DATE_RE = re.compile(r"\b(19|20)\d{2}[-/]\d{2}[-/]\d{2}\b")


def load_yaml(p: Path):
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


def chapter_patterns() -> list[str]:
    cm = load_yaml(REPO / "chapter_map.yml")
    return [c["doc_pattern"].lower() for c in cm.get("chapters", []) if c.get("doc_pattern")]


def author_pairs() -> list[tuple[str, str]]:
    am = load_yaml(REPO / "author_map.yml")
    return [(a["pattern"].lower(), a["author"]) for a in am.get("authors", []) if a.get("pattern")]


def read_head(path: Path, n_chars: int = 2500) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            import fitz
            doc = fitz.open(str(path))
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
                if len(text) >= n_chars:
                    break
            return text[: n_chars * 3]
        except Exception as e:
            return f"(pdf read error: {e})"
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:n_chars]
    except Exception as e:
        return f"(read error: {e})"


def extract_fsj_header(text: str) -> dict | None:
    """Match the FSJ metadata block. Gated on the 'Family Systems Journal Vol. X.Y'
    line — a bare 'Title:'/'Author:' line is NOT evidence of FSJ (transcripts also
    carry a 'Title:' line, and mis-labelling them as FSJ was a real false positive)."""
    vol = re.search(r"Family Systems (?:Journal|Forum)\s*,?\s*Vol\.?\s*([\d.]+)", text, re.I)
    if not vol:
        return None
    title = re.search(r"^\s*Title\s*:\s*(.+)$", text, re.M)
    author = re.search(r"^\s*Author\s*:\s*(.+)$", text, re.M)
    pages = re.search(r"^\s*Pages?\s*:\s*(\d+\s*[-–]\s*\d+)", text, re.M)
    return {
        "container": "Family Systems Journal",
        "volume_issue": vol.group(1),
        "title": title.group(1).strip() if title else None,
        "author": author.group(1).strip() if author else None,
        "pages": pages.group(1).strip() if pages else None,
    }


def extract_transcript_header(text: str) -> dict | None:
    """Transcript metadata block: 'Title: "..." Presenter: "..." Lecture date: "YYYY"'.
    Gated on 'Presenter:' (distinctive to the lecture-transcript header format)."""
    presenter = re.search(r'Presenter\s*:\s*"?([^"]+)"?', text)
    if not presenter:
        return None
    title = re.search(r'Title\s*:\s*"?([^"]+)"?', text)
    date = re.search(r'Lecture\s+date\s*:\s*"?(\d{4})', text)
    return {
        "title": re.sub(r"\s+", " ", title.group(1)).strip() if title else None,
        "author": re.sub(r"\s+", " ", presenter.group(1)).strip(),
        "year": date.group(1) if date else None,
    }


def author_from_filename(doc_name: str, pairs: list[tuple[str, str]]) -> str | None:
    dn = doc_name.lower()
    for pat, author in pairs:
        if pat in dn:
            return author
    return None


def year_from(text: str, doc_name: str) -> tuple[str | None, str]:
    # Prefer an explicit date (YYYY-MM-DD) or year in the FILENAME, then the header text.
    for src, s in (("filename", doc_name), ("header", text)):
        m = YEAR_RE.search(RANGE_RE.sub(" ", s))
        if m:
            return m.group(1), src
    return None, "none"


COPYRIGHT_RE = re.compile(r"\u00a9\s*Georgetown Family Center,?\s*((?:19|20)\d{2})")


def fsj_year(f: Path) -> tuple[str | None, str]:
    """An FSJ article's year is its copyright line. The first year anywhere in the text is
    often a reference-list entry (the Panksepp obituary got 2013 from a Bowen citation)."""
    if f.suffix.lower() == ".txt":
        m = COPYRIGHT_RE.search(f.read_text(encoding="utf-8", errors="ignore"))
        if m:
            return m.group(1), "copyright"
    return None, "none"


def title_from_filename(doc_name: str) -> str:
    # Strip a leading "NNN_" index and trailing "- Author - date" bits.
    name = re.sub(r"^\d+[\s_\-]+", "", doc_name)
    name = re.sub(r"\s*-\s*[A-Z][A-Za-z .]+-\s*\d{4}[-\d]*\s*$", "", name)
    name = re.sub(r"\s*-\s*\d{4}[-\d]*\s*$", "", name)
    name = name.replace("_", " ").replace(".txt", "").replace(".pdf", "")
    return re.sub(r"\s{2,}", " ", name).strip()


def is_chapter(doc_name: str, pats: list[str]) -> bool:
    dn = doc_name.lower()
    return any(p in dn for p in pats)


def main(src: Path = SRC, out: Path = OUT):
    pats = chapter_patterns()
    pairs = author_pairs()
    files = sorted(Path(src).iterdir())
    # One record per document: when both foo.txt and foo.pdf exist (an OCR'd scan keeps its
    # PDF) the .txt is the one the index is built from, so it is the one described.
    txt_stems = {f.stem for f in files if f.suffix.lower() == ".txt"}
    files = [f for f in files if not (f.suffix.lower() == ".pdf" and f.stem in txt_stems)]
    candidates: list[dict] = []
    skipped_chapter = 0
    n = 0

    for f in files:
        if f.suffix.lower() not in (".txt", ".pdf"):
            continue
        n += 1
        if is_chapter(f.name, pats):
            skipped_chapter += 1
            continue

        head = read_head(f, 2500)
        fsj = extract_fsj_header(head)
        transcript = extract_transcript_header(head) if not fsj else None

        rec = {
            "doc_name": f.stem,
            "verified": False,
            "header_excerpt": re.sub(r"\s+", " ", head[:180]).strip(),
        }

        if fsj:
            rec["source"] = "header"
            rec["container"] = fsj["container"]
            if fsj["volume_issue"]:
                m = re.match(r"(\d+)\.(\d+)", fsj["volume_issue"])
                rec["volume"], rec["issue"] = (int(m.group(1)), int(m.group(2))) if m else (None, None)
            rec["title"] = fsj["title"]
            rec["author"] = fsj["author"]
            rec["pages"] = fsj["pages"]
        elif transcript:
            rec["source"] = "header"
            rec["title"] = transcript["title"]
            rec["author"] = transcript["author"]
            rec["pages"] = None
        else:
            rec["source"] = "filename"
            rec["title"] = title_from_filename(f.stem)
            rec["author"] = author_from_filename(f.stem, pairs)
            rec["pages"] = None

        year, ysrc = fsj_year(f) if fsj else year_from(head, f.name)
        rec["year"] = year
        rec["year_source"] = ysrc

        # Clean out empty values to keep the file readable.
        candidates.append({k: v for k, v in rec.items() if v not in (None, "")})

    yaml.safe_dump(
        {"candidates": candidates},
        open(out, "w", encoding="utf-8"),
        sort_keys=False, allow_unicode=True, width=100,
    )
    with_author = sum(1 for c in candidates if c.get("author"))
    with_year = sum(1 for c in candidates if c.get("year"))
    no_year = [c["doc_name"] for c in candidates if c.get("container") and not c.get("year")]
    if no_year:
        print(f"{len(no_year)} journal articles have no year (no copyright line in a .txt): "
              f"{no_year[:6]}")
    print(f"files scanned: {n}  (skipped {skipped_chapter} chapter files)")
    print(f"candidates: {len(candidates)}  | with author: {with_author}  | with year: {with_year}")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
