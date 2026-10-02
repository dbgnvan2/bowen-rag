#!/usr/bin/env python3
"""Enriched, paragraph-aligned Bowen RAG indexer.

Rebuilds chunk_metadata.json + tfidf_matrix.npz with every chunk carrying
author / date / chapter / paragraph metadata, resolved from:
  - chapter_map.yml        (chapter docs -> book citation + chapter number/title)
  - headers_candidates.yml (articles/transcripts -> header/author/year/title)

Chunking is paragraph-aligned: chunks are built from WHOLE paragraphs (a chunk
never splits a paragraph), so a stable "Ch. N, ¶ M" locator is possible.

Run:
    python3 build_index.py            # repo defaults (source_files -> references)
Then rebuild the embedding index (build_embeddings.py) — the chunk count changes.
"""
import json
import re
import sys
from pathlib import Path

import yaml
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "source_files"
REFS = REPO / "rag-document-search" / "references"
CHUNK_CHARS = 1500


def load_yaml(p: Path):
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def load_metadata():
    cm = load_yaml(REPO / "chapter_map.yml")
    chapters = [(c.get("doc_pattern", "").lower(), c)
                for c in cm.get("chapters", []) if c.get("doc_pattern")]
    books = [(b.get("doc_pattern", "").lower(), b)
             for b in cm.get("books", {}).values() if b.get("doc_pattern")]
    hd = load_yaml(REPO / "headers_candidates.yml")
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


def txt_paragraphs(text: str) -> list:
    paras = []

    def add_body(body, section):
        for p in re.split(r"\n\s*\n", body):
            p = re.sub(r"\s+", " ", p).strip()
            if p:
                paras.append({"text": p, "page": None, "section": section})

    if re.search(r"^## Section \d+", text, re.M):
        parts = re.split(r"\n(## Section \d+ – [^\n]+)\n", text)
        if parts and parts[0].strip():
            add_body(parts[0], None)
        it = iter(parts[1:])
        for heading in it:
            body = next(it)
            m = re.match(r"## Section \d+ – (.+?)(?:\s*\(\[[\d:]+\]\))?\.?\s*$", heading)
            add_body(body, m.group(1).strip() if m else heading.strip())
    else:
        add_body(text, None)
    return paras


def pdf_paragraphs(path: Path) -> list:
    doc = fitz.open(str(path))
    paras = []
    for pi in range(doc.page_count):
        for b in sorted(doc[pi].get_text("blocks"), key=lambda b: (b[1], b[0])):
            if b[6] != 0:          # skip image blocks
                continue
            t = re.sub(r"\s+", " ", b[4]).strip()
            if t:
                paras.append({"text": t, "page": pi + 1, "section": None})
    return paras


def paragraphs_for(path: Path) -> list:
    if path.suffix.lower() == ".pdf" and PDF_SUPPORT:
        return pdf_paragraphs(path)
    return txt_paragraphs(read_text(path))


def build_chunks(doc_name: str, paras: list) -> list:
    for i, p in enumerate(paras):
        p["num"] = i + 1
    chunks, buf, chars = [], [], 0

    def flush():
        sec = buf[0]["section"]
        body = "\n\n".join(p["text"] for p in buf)
        text = (f"[{sec}]\n\n" + body) if sec else body
        return {
            "text": text,
            "char_count": len(text),
            "page": buf[0]["page"],
            "section_title": sec or "",
            "paragraph_start": buf[0]["num"],
            "paragraph_end": buf[-1]["num"],
        }

    for p in paras:
        if buf and chars + len(p["text"]) > CHUNK_CHARS:
            chunks.append(flush())
            buf, chars = [], 0
        buf.append(p)
        chars += len(p["text"]) + 2
    if buf:
        chunks.append(flush())
    return chunks


def main():
    doc_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else REFS
    chapters, books, headers = load_metadata()

    all_chunks = []
    stats = {"docs": 0}
    for f in sorted(doc_dir.iterdir()):
        if f.suffix.lower() not in (".txt", ".pdf"):
            continue
        paras = paragraphs_for(f)
        if not paras:
            continue
        meta = resolve_metadata(f.stem, chapters, books, headers)
        stats["docs"] += 1
        stats[meta["kind"]] = stats.get(meta["kind"], 0) + 1
        for c in build_chunks(f.stem, paras):
            c.update({
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

    doc_seq = {}
    for i, c in enumerate(all_chunks):
        doc_seq.setdefault(c["doc_name"], []).append(i)
    pos_map = {}
    for ids in doc_seq.values():
        for pos, idx in enumerate(ids):
            pos_map[idx] = (pos + 1, len(ids))

    metadata = []
    for i, c in enumerate(all_chunks):
        metadata.append({
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
            "section_title": c["section_title"],
            "text": c["text"],
            "char_count": c["char_count"],
            "chunk_pos": pos_map[i][0],
            "doc_chunk_count": pos_map[i][1],
            "preview": c["text"][:150],
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunk_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    vec = TfidfVectorizer(max_features=8000, stop_words="english", lowercase=True,
                          ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    matrix = vec.fit_transform([c["text"] for c in metadata])
    sp_sparse.save_npz(out_dir / "tfidf_matrix.npz", matrix)
    with open(out_dir / "vectorizer.json", "w", encoding="utf-8") as f:
        json.dump({"feature_names": vec.get_feature_names_out().tolist(),
                   "max_features": 8000, "ngram_range": [1, 2]}, f)

    with_auth = sum(1 for c in metadata if c["author"])
    with_date = sum(1 for c in metadata if c["date"])
    with_chap = sum(1 for c in metadata if c["chapter"])
    with_para = sum(1 for c in metadata if c["paragraph_start"])
    print(f"docs: {stats['docs']} (chapter {stats.get('chapter', 0)}, "
          f"article {stats.get('article', 0)}, unknown {stats.get('unknown', 0)})")
    print(f"chunks: {len(metadata)}")
    print(f"  author: {with_auth}  date: {with_date}  chapter: {with_chap}  paragraph: {with_para}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
