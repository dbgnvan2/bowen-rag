#!/usr/bin/env python3
"""
Seed sources.yml with bibliographic records — a one-time admin helper.

Purpose: Pre-fill one record per source document (author from author_map.yml,
         year scraped from the filename, a title guess from the cleaned filename)
         so a human only has to VERIFY/correct the works they actually cite,
         rather than type every record from scratch.
Spec:    docs/spec_citation_styles.md  (M1.B)
Tests:   test_seed_sources.py

Safety / no-fabrication:
  - Writes to `sources.seeded.yml`. REFUSES to overwrite an existing `sources.yml`.
  - Every seeded field is a GUESS and is marked `verified: false`.
  - Author comes only from author_map.yml; year only from digits in the filename;
    title only from the cleaned filename. Nothing is invented from outside the data.
  - The authority_tiers note (when present) is emitted as a COMMENT hint, never as
    the title value, so a descriptive note ("FTCP book chapters") can't masquerade
    as bibliographic content.

Usage:
    python3 seed_sources.py
    python3 seed_sources.py --out sources.seeded.yml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import citations as C

BASE_DIR = Path(__file__).resolve().parent
CHUNKS = BASE_DIR / "rag-document-search" / "references" / "chunk_metadata.json"

_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')


def _load_yaml(path: Path, key: str) -> list:
    if not path.exists():
        return []
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get(key, []) or []
    except Exception as e:
        print(f"Warning: could not load {path.name} ({e})", file=sys.stderr)
        return []


def _first_match(doc_name: str, entries: list, value_key: str):
    """First entry whose 'pattern' is a case-insensitive substring of doc_name."""
    dn = doc_name.lower()
    for e in entries:
        pat = str(e.get("pattern", "")).lower()
        if pat and pat in dn:
            return e.get(value_key)
    return None


def guess_year(doc_name: str):
    """Return the first plausible 4-digit year (1900–2030) in the filename, else None."""
    for m in _YEAR_RE.finditer(doc_name):
        y = int(m.group(0))
        if 1900 <= y <= 2030:
            return y
    return None


def guess_type(doc_name: str) -> str:
    """Conservative type guess from filename keywords; defaults to 'generic'."""
    dn = doc_name.lower()
    if "interview" in dn:
        return "interview"
    if any(k in dn for k in ("tape", "lecture", "webinar", "webcast", "series",
                             "conference", "presentation", "talk")):
        return "speech"
    if dn.startswith("fsj ") or dn.startswith("copy of ") or "journal" in dn:
        return "article-journal"
    return "generic"


def _q(s) -> str:
    """Single-quote a YAML scalar, escaping embedded single quotes."""
    return "'" + str(s).replace("'", "''") + "'"


def build_records(doc_names, author_map, authority_tiers) -> list:
    records = []
    for dn in doc_names:
        author_str = _first_match(dn, author_map, "author")
        authors = C.parse_author_string(author_str) if author_str else []
        note = _first_match(dn, authority_tiers, "note")
        boost = _first_match(dn, authority_tiers, "multiplier") or 1.0
        records.append({
            "doc_name": dn,
            "type": guess_type(dn),
            "authors": authors,
            "year": guess_year(dn),
            "title": C.clean_title_from_filename(dn),
            "note_hint": note,
            "_boost": boost,
        })
    # Primary Bowen/Kerr sources first so the worthwhile records are easy to verify.
    records.sort(key=lambda r: (-float(r["_boost"]), r["doc_name"].lower()))
    return records


def render_yaml(records: list) -> str:
    out = [
        "# sources.yml — bibliographic records for citation formatting.",
        "#",
        "# AUTO-SEEDED by seed_sources.py. EVERY field below is a GUESS until you set",
        "# `verified: true`. For each work you actually cite, confirm the author(s),",
        "# year, and title, and add publisher / journal / volume / issue / pages.",
        "#",
        "# Matching: a record whose `pattern` is a case-insensitive substring of the",
        "# document filename applies; when several match, the LONGEST (most specific)",
        "# pattern wins, regardless of order. Records are listed primary-sources-first",
        "# only for ease of review.",
        "#",
        "# type: book | chapter | article-journal | speech | interview | report | webpage | generic",
        "",
        "sources:",
        "",
    ]
    for r in records:
        out.append(f"  - pattern: {_q(r['doc_name'])}")
        out.append(f"    type: {r['type']}")
        if r["authors"]:
            inner = ", ".join(
                "{family: %s, given: %s}" % (_q(a["family"]), _q(a["given"]))
                for a in r["authors"])
            out.append(f"    authors: [{inner}]")
        else:
            out.append(f"    authors: []            # UNKNOWN — add author(s)")
        out.append(f"    year: {r['year'] if r['year'] is not None else _q('n.d.')}"
                   f"{'' if r['year'] is not None else '        # no year in filename'}")
        out.append(f"    title: {_q(r['title'])}")
        if r["note_hint"]:
            out.append(f"    # title hint (from authority_tiers): {r['note_hint']}")
        out.append(f"    publisher: ''")
        out.append(f"    verified: false")
        out.append("")
    return "\n".join(out)


def coverage(records: list) -> str:
    n = len(records)
    with_author = sum(1 for r in records if r["authors"])
    with_year = sum(1 for r in records if r["year"] is not None)
    return (f"Seeded {n} records — author guessed for {with_author}/{n}, "
            f"year for {with_year}/{n}. All verified: false (need human review).")


def main():
    ap = argparse.ArgumentParser(description="Seed sources.yml from existing metadata.")
    ap.add_argument("--out", default="sources.seeded.yml",
                    help="output file (default: sources.seeded.yml)")
    ap.add_argument("--chunks", default=str(CHUNKS), help="chunk_metadata.json path")
    args = ap.parse_args()

    live = BASE_DIR / "sources.yml"
    out_path = BASE_DIR / args.out
    if out_path.name == "sources.yml" and live.exists():
        print("Refusing to overwrite existing sources.yml. "
              "Write to sources.seeded.yml and merge by hand.", file=sys.stderr)
        return 2

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"chunk_metadata.json not found at {chunks_path}", file=sys.stderr)
        return 1
    chunks = json.load(open(chunks_path, encoding="utf-8"))
    doc_names = sorted({c["doc_name"] for c in chunks})

    author_map = _load_yaml(BASE_DIR / "author_map.yml", "authors")
    authority = _load_yaml(BASE_DIR / "authority_tiers.yml", "tiers")

    records = build_records(doc_names, author_map, authority)
    out_path.write_text(render_yaml(records), encoding="utf-8")
    print(coverage(records))
    print(f"Wrote {out_path}")
    print("Review it, then: mv sources.seeded.yml sources.yml (or merge into an existing one).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
