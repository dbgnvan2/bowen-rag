#!/usr/bin/env python3
"""Split Kerr's "Bowen Theory's Secrets" full-book PDF into per-chapter .txt files.

Uses PyMuPDF block extraction so PARAGRAPH structure is preserved (PyPDF2 loses
paragraph breaks — it flattens the text into soft-wrapped lines). Each layout
text block becomes one paragraph, paragraphs separated by a blank line.

Output: source_files/Bowen Theory Secrets_<ChapterNN|Introduction|Epilogue>_<Title>.txt
"""
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/Users/davemini2/ObsidianSyncVault/Bowen Theory_s Secrets_ Revealing the Hidden Life of Families.pdf"
)
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "/Volumes/CrucialX9/a Downloads/bowen_rag/source_files"
)

# (label, chapter_number, title, start_page, end_page) — 1-based PDF page numbers inclusive.
CHAPTERS = [
    ("Introduction", None, "Introduction", 8, 22),
    ("Chapter01", 1, "Systems Thinking", 24, 28),
    ("Chapter02", 2, "Evolution and the Emotional System", 29, 35),
    ("Chapter03", 3, "The Molecule of an Emotional System", 36, 45),
    ("Chapter04", 4, "Patterns of Emotional Functioning", 46, 71),
    ("Chapter05", 5, "Differentiation of Self", 72, 92),
    ("Chapter06", 6, "Emotional Regression", 93, 106),
    ("Chapter07", 7, "Emotional Regression and the Individuality-Togetherness Balance", 107, 116),
    ("Chapter08", 8, "Emotional Objectivity", 117, 122),
    ("Chapter09", 9, "Emotional Programming", 123, 137),
    ("Chapter10", 10, "Chronic Anxiety", 138, 149),
    ("Chapter11", 11, "The Multigenerational Family Organism", 150, 164),
    ("Chapter12", 12, "Sibling Position", 165, 174),
    ("Chapter13", 13, "Emotional Cutoff", 175, 180),
    ("Chapter14", 14, "Societal Emotional Process", 181, 199),
    ("Chapter15", 15, "Key Ingredients in the Process of Differentiation", 201, 215),
    ("Chapter16", 16, "Personal Vignettes of the Process of Differentiation", 216, 238),
    ("Chapter17", 17, "Clinical Example of the Process of Differentiation", 239, 251),
    ("Chapter18", 18, "The Process of Differentiation: Theory, Method, and Technique", 252, 263),
    ("Chapter19", 19, "The Unabomber and His Family", 266, 281),
    ("Chapter20", 20, "Gary Gilmore and His Family", 282, 303),
    ("Chapter21", 21, "Adam Lanza and His Family", 304, 316),
    ("Chapter22", 22, "John Nash: A Beautiful Mind", 317, 341),
    ("Chapter23", 23, "Unidisease: A Proposed New Concept in Bowen Theory", 343, 370),
    ("Chapter24", 24, "Toward a Systems Concept of Supernatural Phenomena", 371, 379),
    ("Epilogue", None, "Applying Bowen Theory to My Own Family", 380, 405),
]


def slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")


def chapter_text(doc: fitz.Document, start: int, end: int) -> str:
    """Extract a chapter as blank-line-separated paragraphs (one per layout block)."""
    paragraphs = []
    for i in range(start - 1, end):
        page = doc[i]
        blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))
        for b in blocks:
            if b[6] != 0:          # skip image blocks
                continue
            t = b[4].strip()
            if not t:
                continue
            t = re.sub(r"\s+", " ", t)   # collapse soft-wraps within the block
            paragraphs.append(t)
    return "\n\n".join(paragraphs) + "\n"


def main():
    doc = fitz.open(str(PDF))
    n = doc.page_count
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_written = 0
    for label, num, title, start, end in CHAPTERS:
        body = chapter_text(doc, start, min(end, n))
        fname = f"Bowen Theory Secrets_{label}_{slug(title)}.txt"
        (OUT_DIR / fname).write_text(body, encoding="utf-8")
        total_written += len(body)
        print(f"{fname:70s}  {len(body):7d} chars  {body.count(chr(10)+chr(10)):4d} paragraphs")
    print(f"\nTotal: {len(CHAPTERS)} files, {total_written} chars -> {OUT_DIR}")


if __name__ == "__main__":
    main()
