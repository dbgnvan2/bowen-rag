#!/usr/bin/env python3
"""Enriched, paragraph-aligned Bowen RAG indexer.

Rebuilds chunk_metadata.json + tfidf_matrix.npz + vectorizer.json with every chunk
carrying author / date / chapter / paragraph metadata, resolved from:
  - chapter_map.yml        (chapter docs -> book citation + chapter number/title)
  - headers_candidates.yml (articles/transcripts -> header/author/year/title)

Documents listed in ocr_manifest.yml (written by ocr_scans.py) were read from scans by OCR;
their chunks carry `ocr: true` and the document's mean word confidence, so reports can say
so. Chunking is paragraph-aligned: chunks are built from WHOLE paragraphs (a chunk never
splits a paragraph), so a stable "Ch. N, ¶ M" locator is possible, and a chunk NEVER
spans two `## Section N –` transcript sections, so its section_title is always the
section its text came from.

Nothing is skipped silently: documents with no extractable text, duplicate stems,
unreadable PDFs and sectioned transcripts with empty sections are all reported, and a
malformed config file or a missing PDF library is an error, not an empty result.

Run:
    python3 build_index.py [source_dir] [output_dir]     # defaults: source_files -> references
Then rebuild the embedding index (build_embeddings.py) — the chunk count changes.
"""
import json
import os
import re
import sys
from pathlib import Path

import yaml
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import fitz  # PyMuPDF (requirements-build.txt; AGPL — only needed to BUILD the index)
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "source_files"
REFS = REPO / "rag-document-search" / "references"
CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 200   # a chunk is not closed below this size (no page-header stubs)

# The TF-IDF settings live here ONCE. semantic_search.py imports them, and the two apps
# repeat the values when they re-fit the vectorizer on load (test_index_consistency.py
# fails if a copy drifts), so the query vectorizer can never differ from the one the
# saved matrix was built with.
TFIDF_PARAMS = dict(
    max_features=8000,
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

# "## Section 3 – Title". Accepts an en dash, em dash or hyphen. Used for both the
# "is this a sectioned document" test and the split, so the two cannot disagree.
_SECTION_HEADING = r"## Section \d+ [–—-] [^\r\n]+"
_SECTION_TITLE = re.compile(r"## Section \d+ [–—-] (.+?)(?:\s*\(\[[\d:]+\]\))?\.?\s*$")


class IndexBuildError(RuntimeError):
    """A condition that would otherwise produce a silently wrong or incomplete index."""


# ── config / text reading ────────────────────────────────────────────────────

def load_yaml(p: Path) -> dict:
    """A missing file is an empty config (the caller reports it); a malformed file is an
    error — it used to return {} and silently strip all citation metadata."""
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise IndexBuildError(f"{p.name} is not valid YAML: {e}") from e


def read_text(path: Path) -> str:
    """Decode a text source. UTF-16 is used only when the file has a BOM, because almost
    any even-length byte string decodes as UTF-16 and would be indexed as mojibake."""
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        print(f"  Note: {path.name} is not UTF-8; read as cp1252")
        try:
            return raw.decode("cp1252")
        except UnicodeDecodeError:
            return raw.decode("latin-1")


def load_metadata(config_dir: Path):
    cm = load_yaml(config_dir / "chapter_map.yml")
    chapters = [(c.get("doc_pattern", "").lower(), c)
                for c in cm.get("chapters", []) if c.get("doc_pattern")]
    books = [(b.get("doc_pattern", "").lower(), b)
             for b in cm.get("books", {}).values() if b.get("doc_pattern")]
    hd = load_yaml(config_dir / "headers_candidates.yml")
    headers = {c.get("doc_name"): c for c in hd.get("candidates", []) if c.get("doc_name")}
    return chapters, books, headers


def longest_match(doc_name: str, pairs):
    dn = doc_name.lower()
    best, bl = None, -1
    for pat, val in pairs:
        if pat and pat in dn and len(pat) > bl:
            best, bl = val, len(pat)
    return best


def resolve_metadata(doc_name, chapters, books, headers):
    ch = longest_match(doc_name, chapters)
    if ch is not None:
        bk = longest_match(doc_name, books) or {}
        authors = bk.get("authors") or []
        return {
            "kind": "chapter",
            "author": [(f"{a.get('given','')} {a.get('family','')}".strip() or a.get("family", ""))
                       for a in authors],
            "date": str(bk["year"]) if bk.get("year") else None,
            "chapter": ch.get("chapter"),
            "chapter_label": ch.get("chapter_label"),
            "chapter_title": ch.get("title"),
            "container": bk.get("title"),
        }
    h = headers.get(doc_name)
    if h:
        au = h.get("author")
        return {
            "kind": "article",
            "author": [au] if au else [],
            "date": h.get("year"),
            "title": h.get("title"),
            "chapter": None,
            "chapter_label": None,
            "chapter_title": None,
            "container": h.get("container"),
        }
    return {"kind": "unknown", "author": [], "date": None, "chapter": None,
            "chapter_label": None, "chapter_title": None, "container": None}


# ── paragraphs ───────────────────────────────────────────────────────────────

def split_long(text: str, limit: int = CHUNK_CHARS) -> list:
    """Split an overlong paragraph into pieces of at most `limit` characters, at sentence
    boundaries (a sentence longer than `limit` is split at a word boundary). No text is
    dropped. Paragraphs with no blank lines (some transcripts) used to become one chunk of
    up to ~47,000 characters, of which an embedding model sees only the first ~1,000."""
    if len(text) <= limit:
        return [text]
    pieces, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        while len(sent) > limit:                     # no sentence break: cut at a space
            cut = sent.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if cur and len(cur) + 1 + len(sent) > limit:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip() if cur else sent
    if cur:
        pieces.append(cur)
    return [p for p in pieces if p]


def txt_paragraphs(text: str, stats: dict = None) -> list:
    """Split a text source into paragraphs, each tagged with its transcript section.

    `stats` (optional dict) receives `headings` and `empty` (sections with no text), so
    the caller can report sections that produced nothing.
    """
    text = text.lstrip("﻿").replace("\r\n", "\n")
    paras = []

    def add_body(body, section):
        for p in re.split(r"\n\s*\n", body):
            p = re.sub(r"\s+", " ", p).strip()
            if not p:
                continue
            pieces = split_long(p)
            if stats is not None and len(pieces) > 1:
                stats["split"] = stats.get("split", 0) + 1
            for piece in pieces:
                paras.append({"text": piece, "page": None, "section": section})

    if re.search(r"^" + _SECTION_HEADING, text, re.M):
        # The newline after a heading is NOT consumed (lookahead), so a heading that
        # directly follows another heading is still a heading, and a heading on the very
        # first line (how process_transcripts.py writes files) is recognised.
        parts = re.split(r"(?:^|\n)(" + _SECTION_HEADING + r")(?=\n|\Z)", text)
        preamble = parts[0].strip()
        # A one-line "# Transcript Formatting ..." banner is not source content.
        if preamble and not (preamble.startswith("#") and "\n" not in preamble):
            add_body(preamble, None)
        headings = empty = 0
        it = iter(parts[1:])
        for heading in it:
            body = next(it, "")
            m = _SECTION_TITLE.match(heading)
            before = len(paras)
            add_body(body, m.group(1).strip() if m else heading.strip())
            headings += 1
            empty += len(paras) == before
        if stats is not None:
            stats.update(headings=headings, empty=empty)
    else:
        add_body(text, None)
    return paras


def pdf_paragraphs(path: Path) -> list:
    if not PDF_SUPPORT:
        raise IndexBuildError(
            f"PyMuPDF is required to index {path.name} but is not installed "
            "(pip install -r requirements-build.txt). Refusing to index PDF bytes as text.")
    doc = fitz.open(str(path))
    paras = []
    for pi in range(doc.page_count):
        for b in sorted(doc[pi].get_text("blocks"), key=lambda b: (b[1], b[0])):
            if b[6] != 0:          # skip image blocks
                continue
            t = re.sub(r"\s+", " ", b[4]).strip()
            for piece in (split_long(t) if t else []):
                paras.append({"text": piece, "page": pi + 1, "section": None})
    return paras


def build_chunks(doc_name: str, paras: list) -> list:
    """Whole-paragraph chunks of about CHUNK_CHARS; never spanning two sections.

    No chunk is left smaller than MIN_CHUNK_CHARS unless it is a whole section on its own:
    a small group is not closed early (so a page number followed by a big paragraph is not
    a stub), and a small final group is absorbed into the previous chunk of the same
    section. Otherwise a 28-character tail would be embedded and cited as a passage.
    """
    for i, p in enumerate(paras):
        p["num"] = i + 1

    def size(group):
        return sum(len(p["text"]) + 2 for p in group)

    groups, buf = [], []
    for p in paras:
        if buf and (p["section"] != buf[0]["section"]
                    or (size(buf) + len(p["text"]) > CHUNK_CHARS and size(buf) >= MIN_CHUNK_CHARS)):
            groups.append(buf)
            buf = []
        buf.append(p)
    if buf:
        groups.append(buf)

    merged = []
    for g in groups:
        if (merged and size(g) < MIN_CHUNK_CHARS
                and merged[-1][0]["section"] == g[0]["section"]):
            merged[-1] = merged[-1] + g
        else:
            merged.append(g)

    chunks = []
    for g in merged:
        sec = g[0]["section"]
        body = "\n\n".join(p["text"] for p in g)
        text = (f"[{sec}]\n\n" + body) if sec else body
        chunks.append({
            "text": text,
            "char_count": len(text),
            "page": g[0]["page"],
            "section_title": sec or "",
            "paragraph_start": g[0]["num"],
            "paragraph_end": g[-1]["num"],
        })
    return chunks


# ── build ────────────────────────────────────────────────────────────────────

def _source_files(doc_dir: Path, log) -> list:
    """Indexable files, one per document stem. When both foo.txt and foo.pdf exist the
    .txt is used; any other duplicate stem is an error (the two would merge into one
    doc_name and corrupt chunk positions and citations)."""
    by_stem = {}
    for f in sorted(doc_dir.iterdir()):
        if f.suffix.lower() in (".txt", ".pdf") and not f.name.startswith("."):
            by_stem.setdefault(f.stem, []).append(f)
    files = []
    for stem, group in by_stem.items():
        if len(group) > 1:
            kinds = sorted(g.suffix.lower() for g in group)
            if kinds == [".pdf", ".txt"]:
                log(f"  Note: {stem}: both .txt and .pdf present; using the .txt")
                group = [g for g in group if g.suffix.lower() == ".txt"]
            else:
                raise IndexBuildError(f"duplicate document name {stem!r}: "
                                      f"{[g.name for g in group]}")
        files.append(group[0])
    return files


def build(doc_dir, out_dir, config_dir=REPO, log=print) -> dict:
    """Build the index from `doc_dir` into `out_dir`; returns a stats dict.

    `config_dir` holds chapter_map.yml / headers_candidates.yml. A config file that is
    absent is reported (every document then has unknown metadata); one that is present
    but malformed raises IndexBuildError.
    """
    doc_dir, out_dir, config_dir = Path(doc_dir), Path(out_dir), Path(config_dir)
    for name in ("chapter_map.yml", "headers_candidates.yml"):
        if not (config_dir / name).exists():
            log(f"  WARNING: {name} not found in {config_dir}; chunks will carry no "
                f"author/date/chapter metadata from it")
    chapters, books, headers = load_metadata(config_dir)
    ocr_docs = {e["doc_name"]: e for e in load_yaml(config_dir / "ocr_manifest.yml").get("ocr", [])
                if e.get("doc_name")}

    files = _source_files(doc_dir, log)
    if not files:
        raise IndexBuildError(f"no .txt or .pdf files in {doc_dir}")
    log(f"Found {len(files)} documents")

    all_chunks, skipped, unknown, kinds, split_docs = [], [], [], {}, {}
    for f in files:
        try:
            stats = {}
            paras = pdf_paragraphs(f) if f.suffix.lower() == ".pdf" else \
                txt_paragraphs(read_text(f), stats)
        except IndexBuildError:
            raise
        except Exception as e:
            skipped.append((f.name, f"unreadable: {type(e).__name__}: {e}"))
            continue
        if not paras:
            skipped.append((f.name, "no extractable text"))
            continue
        if stats.get("split"):
            split_docs[f.stem] = stats["split"]
        if stats.get("empty"):
            log(f"  WARNING: {f.stem}: {stats['empty']} of {stats['headings']} sections "
                f"have no text")
        meta = resolve_metadata(f.stem, chapters, books, headers)
        kinds[meta["kind"]] = kinds.get(meta["kind"], 0) + 1
        if meta["kind"] == "unknown":
            unknown.append(f.stem)
        for c in build_chunks(f.stem, paras):
            c.update({
                "ocr": f.stem in ocr_docs,
                "ocr_confidence": (ocr_docs[f.stem].get("mean_word_confidence")
                                   if f.stem in ocr_docs else None),
                "doc_name": f.stem,
                "author": meta["author"],
                "date": meta["date"],
                "title": meta.get("title"),
                "chapter": meta["chapter"],
                "chapter_label": meta["chapter_label"],
                "chapter_title": meta["chapter_title"],
                "container": meta["container"],
            })
            all_chunks.append(c)

    if ocr_docs:
        present = {f.stem for f in files}
        missing = sorted(set(ocr_docs) - present)
        indexed_ocr = sorted({c["doc_name"] for c in all_chunks if c.get("ocr")})
        log(f"  Note: {len(indexed_ocr)} documents were read from scans by OCR "
            f"(ocr_manifest.yml); their chunks are marked ocr: true")
        weak = [n for n in indexed_ocr
                if (ocr_docs[n].get("mean_word_confidence") or 100) < 85
                or ocr_docs[n].get("low_confidence_pages")]
        if weak:
            log(f"  Note: {len(weak)} of them have mean confidence under 85 or pages under "
                f"{70}: {weak[:4]}{' ...' if len(weak) > 4 else ''}")
        # a manifest entry whose file is still the bare PDF: the OCR .txt is not in this
        # checkout (source_files/ is not in git), so the document drops out of the index
        bare = sorted(f.stem for f in files if f.suffix.lower() == ".pdf" and f.stem in ocr_docs
                      and (ocr_docs[f.stem].get("words") or 0) > 0)
        if bare:
            log(f"  WARNING: {len(bare)} documents in ocr_manifest.yml have no OCR .txt beside "
                f"the PDF, so they are not searchable here (copy source_files_ocr/*.txt into "
                f"{doc_dir}): {bare[:3]}")
        if missing:
            log(f"  WARNING: ocr_manifest.yml lists {len(missing)} documents with no source "
                f"file in {doc_dir} (first few: {missing[:3]})")
    if split_docs:
        log(f"  Note: {sum(split_docs.values())} overlong paragraphs in {len(split_docs)} "
            f"documents were split at sentence boundaries (no blank lines in the source)")
    if skipped:
        log(f"  WARNING: {len(skipped)} of {len(files)} files were NOT indexed:")
        for name, why in skipped:
            log(f"    - {name}: {why}")
    if unknown:
        log(f"  WARNING: {len(unknown)} documents have no author/date metadata "
            f"(first few: {unknown[:5]})")
    if not all_chunks:
        raise IndexBuildError("no chunks were produced")

    doc_seq = {}
    for i, c in enumerate(all_chunks):
        doc_seq.setdefault(c["doc_name"], []).append(i)
    pos_map = {}
    for ids in doc_seq.values():
        for pos, idx in enumerate(ids):
            pos_map[idx] = (pos + 1, len(ids))

    metadata = [{
        "id": i,
        "doc_name": c["doc_name"],
        "author": c["author"],
        "date": c["date"],
        "title": c.get("title"),
        "chapter": c["chapter"],
        "chapter_label": c["chapter_label"],
        "chapter_title": c["chapter_title"],
        "container": c["container"],
        "paragraph_start": c["paragraph_start"],
        "paragraph_end": c["paragraph_end"],
        "page": c["page"],
        "ocr": c["ocr"],
        "ocr_confidence": c["ocr_confidence"],
        "section_title": c["section_title"],
        "text": c["text"],
        "char_count": c["char_count"],
        "chunk_pos": pos_map[i][0],
        "doc_chunk_count": pos_map[i][1],
        "preview": c["text"][:150],
    } for i, c in enumerate(all_chunks)]

    vec = TfidfVectorizer(**TFIDF_PARAMS)
    matrix = vec.fit_transform([c["text"] for c in metadata])

    # Write all three files under temporary names, then rename, so an interruption
    # cannot leave a new chunk_metadata.json next to an old matrix.
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = {n: out_dir / (n + ".tmp") for n in
           ("chunk_metadata.json", "vectorizer.json", "tfidf_matrix.npz")}
    with open(tmp["chunk_metadata.json"], "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False)
    with open(tmp["vectorizer.json"], "w", encoding="utf-8") as fh:
        json.dump({"feature_names": vec.get_feature_names_out().tolist(),
                   "max_features": TFIDF_PARAMS["max_features"],
                   "ngram_range": list(TFIDF_PARAMS["ngram_range"]),
                   "min_df": TFIDF_PARAMS["min_df"],
                   "sublinear_tf": TFIDF_PARAMS["sublinear_tf"]}, fh)
    with open(tmp["tfidf_matrix.npz"], "wb") as fh:
        sp_sparse.save_npz(fh, matrix)
    for name, path in tmp.items():
        os.replace(path, out_dir / name)

    stats = {
        "documents": len(files) - len(skipped),
        "files_skipped": [n for n, _ in skipped],
        "chunks": len(metadata),
        "kinds": kinds,
        "unknown_metadata": unknown,
        "with_author": sum(1 for c in metadata if c["author"]),
        "with_date": sum(1 for c in metadata if c["date"]),
        "with_chapter": sum(1 for c in metadata if c["chapter"]),
        "with_paragraph": sum(1 for c in metadata if c["paragraph_start"]),
        "ocr_chunks": sum(1 for c in metadata if c["ocr"]),
        "vectorizer_features": len(vec.get_feature_names_out()),
    }
    log(f"docs: {stats['documents']} (chapter {kinds.get('chapter', 0)}, "
        f"article {kinds.get('article', 0)}, unknown {kinds.get('unknown', 0)}); "
        f"chunks: {stats['chunks']}")
    log(f"  author: {stats['with_author']}  date: {stats['with_date']}  "
        f"chapter: {stats['with_chapter']}  paragraph: {stats['with_paragraph']}")
    log(f"saved -> {out_dir}")
    return stats


def main():
    doc_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else REFS
    try:
        build(doc_dir, out_dir)
    except IndexBuildError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
