#!/usr/bin/env python3
"""OCR scanned PDFs (no text layer) into per-document .txt files plus a manifest.

Why not just `ocrmypdf`: the scans in source_files/ are often a *spread* — two facing book
pages on one sheet, rotated 90 degrees, each page skewed differently. OCR of the whole sheet
interleaves and garbles the two pages. This tool, for every page of every scan:
  1. renders it and picks the orientation (0/90/180/270) that yields the most confident words;
  2. if the upright sheet is landscape (a spread), splits it at the gutter, left page first;
  3. deskews each page on its own;
  4. runs tesseract (word confidences) and rebuilds paragraphs from its layout data;
  5. de-hyphenates words broken across line ends.
Nothing is overwritten: the PDF stays where it is. The .txt is written to a separate folder
(copy it next to the PDF; build_index.py prefers a .txt over a PDF with the same stem) and
every document is listed in ocr_manifest.yml with its mean word confidence, so reports can
say a passage came from OCR and the quality of each document can be audited.

Run:
    python3 ocr_scans.py <scan_dir> <out_dir> [--manifest ocr_manifest.yml] [--only NAME ...]
                         [--jobs N] [--dpi 300]
OCR output is a MACHINE reading of a scan: it can contain recognition errors. Spot-check it
against the page images before relying on a quotation.
"""
import argparse
import csv
import datetime
import io
import re
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

LANG = "eng"
MIN_WORD_CONF = 60          # a word counts as "confident" for orientation scoring
MIN_ORIENT_WORDS = 12       # confident words needed to trust an orientation guess
SPREAD_RATIO = 1.1          # upright width/height above this = two facing pages
LOW_CONF_PAGE = 70.0        # mean word confidence below this flags a page for review


# ── pure helpers (unit-tested without tesseract) ─────────────────────────────

def dehyphenate(lines: list) -> str:
    """Join OCR lines into a paragraph, removing a hyphen that only marks a line break
    ("schizo-" + "phrenia" -> "schizophrenia") when the next line continues in lower case.
    A genuine hyphenated compound broken at a line end keeps its hyphen only if the next
    word starts with a capital or digit; otherwise it is ambiguous and joined."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if out.endswith("-") and line[:1].islower() and len(out) > 1 and out[-2].isalpha():
            out = out[:-1] + line
        else:
            out = f"{out} {line}".strip()
    return out


def tsv_to_paragraphs(rows: list) -> tuple:
    """Rebuild paragraphs from tesseract TSV rows (dicts).

    Returns (paragraphs, confidences) where `confidences` are the per-word confidences
    (words tesseract itself marked -1 are not words and are skipped).
    """
    paras, confs = [], []
    cur_key, lines, cur_line_key, words = None, [], None, []

    def end_line():
        nonlocal words
        if words:
            lines.append(" ".join(words))
        words = []

    def end_para():
        nonlocal lines
        end_line()
        text = dehyphenate(lines)
        if text:
            paras.append(text)
        lines = []

    for r in rows:
        if r.get("level") != "5":                 # 5 = word
            continue
        try:
            conf = float(r["conf"])
        except (KeyError, ValueError):
            continue
        text = (r.get("text") or "").strip()
        if conf < 0 or not text:
            continue
        key = (r["block_num"], r["par_num"])
        if key != cur_key:
            end_para()
            cur_key, cur_line_key = key, None
        line_key = (r["block_num"], r["par_num"], r["line_num"])
        if line_key != cur_line_key:
            end_line()
            cur_line_key = line_key
        words.append(text)
        confs.append(conf)
    end_para()
    return paras, confs


def find_gutter(gray: np.ndarray) -> int:
    """x position to split a two-page spread: the quietest column of ink in the middle
    40% of the sheet (smoothed so a single speck does not decide)."""
    h, w = gray.shape
    ink = (gray < 128).sum(axis=0).astype(float)
    k = max(5, w // 100)
    smooth = np.convolve(ink, np.ones(k) / k, mode="same")
    lo, hi = int(w * 0.4), int(w * 0.6)
    return lo + int(np.argmin(smooth[lo:hi]))


def estimate_skew(gray: np.ndarray, max_deg: float = 5.0) -> float:
    """Skew angle in degrees (positive = rotate the image counter-clockwise to fix it),
    by maximising the variance of the horizontal ink profile (text lines line up)."""
    small = Image.fromarray(gray)
    scale = 700 / max(small.size)
    if scale < 1:
        small = small.resize((int(small.width * scale), int(small.height * scale)))
    base = small.point(lambda v: 255 if v > 140 else 0)

    def score(angle):
        rot = base.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
        prof = (np.asarray(rot) < 128).sum(axis=1).astype(float)
        return prof.var()

    coarse = max(np.arange(-max_deg, max_deg + 0.01, 0.5), key=score)
    return float(max(np.arange(coarse - 0.5, coarse + 0.51, 0.1), key=score))


# ── tesseract-backed steps ───────────────────────────────────────────────────

def run_tesseract(img: Image.Image, psm: int = 3) -> list:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.png"
        img.save(path)
        res = subprocess.run(["tesseract", str(path), "stdout", "-l", LANG, "--psm", str(psm),
                              "tsv"], capture_output=True, text=True, check=True)
    return list(csv.DictReader(io.StringIO(res.stdout), delimiter="\t", quoting=csv.QUOTE_NONE))


_DICT = None


def _dictionary():
    """English word list for judging whether OCR output is language, or None if the
    machine has none (then confident-word counts are used instead)."""
    global _DICT
    if _DICT is None:
        _DICT = set()
        for path in ("/usr/share/dict/words", "/usr/share/dict/web2"):
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    _DICT |= {w.strip().lower() for w in fh if len(w.strip()) >= 3}
            except OSError:
                pass
    return _DICT or None


def confident_words(rows: list) -> int:
    """Score for how readable a page is in the orientation tested: the number of words that
    are real English words (or, with no word list, words tesseract is confident about).
    Real words matter because tesseract reads text turned 90 degrees one way, and gibberish
    read in the wrong orientation can still look confident."""
    dictionary = _dictionary()
    n = 0
    for r in rows:
        try:
            if r["level"] != "5" or float(r["conf"]) < 0:
                continue
            word = re.sub(r"[^A-Za-z]", "", r.get("text") or "").lower()
            if len(word) < 3:
                continue
            if dictionary is not None:
                n += word in dictionary
            elif float(r["conf"]) >= MIN_WORD_CONF:
                n += 1
        except (KeyError, ValueError):
            pass
    return n


def choose_orientation(scores: dict) -> int:
    """Pick the rotation (degrees counter-clockwise: 0, 90, 180, 270) that makes the sheet
    upright, from {rotation: readability score}.

    Tesseract also reads a sheet turned 90 degrees one way, so the upright rotation u scores
    high AND so does u+270 (mod 360). Two adjacent rotations (r, r+90) that both read
    therefore mean the upright one is r+90; otherwise the best score wins. Checked against
    real scans and a synthetic spread (test_ocr_scans.py::TestChooseOrientation).
    """
    best = max(scores.values())
    if best <= 0:
        return 0
    readable = [a for a, s in scores.items() if s >= 0.6 * best]
    if len(readable) == 2:
        for r in readable:
            if (r + 90) % 360 in readable:
                return (r + 90) % 360
    return max(scores, key=scores.get)


def best_orientation(img: Image.Image, sizes=(1100, 1800, 2800)) -> int:
    """Rotation that makes the sheet upright, judged on a reduced copy for speed and
    repeated at the next larger size when too little was readable to trust the answer.
    Returns 0 when nothing was readable at any size (the page is blank or unreadable and
    is then reported as such)."""
    best_angle, best_total = 0, 0
    for size in sizes:
        small = img.copy()
        small.thumbnail((size, size))
        scores = {a: confident_words(run_tesseract(small.rotate(a, expand=True, fillcolor=255)))
                  for a in (0, 90, 180, 270)}
        angle = choose_orientation(scores)
        if max(scores.values()) > best_total:
            best_angle, best_total = angle, max(scores.values())
        if max(scores.values()) >= MIN_ORIENT_WORDS:
            return angle
        if max(small.size) >= max(img.size):          # already at full size
            break
    return best_angle


def ocr_page(page_img: Image.Image) -> list:
    """OCR one upright single page: deskew, then (paragraphs, word confidences)."""
    gray = np.asarray(page_img.convert("L"))
    angle = estimate_skew(gray)
    if abs(angle) >= 0.2:
        page_img = page_img.convert("L").rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    rows = run_tesseract(page_img.convert("L"))
    return tsv_to_paragraphs(rows)


def ocr_sheet(img: Image.Image) -> list:
    """One scanned sheet -> list of (label, paragraphs, confidences), in reading order."""
    img = img.convert("L")
    angle = best_orientation(img)
    if angle:
        img = img.rotate(angle, expand=True, fillcolor=255)
    parts = [("page", img)]
    if img.width / img.height > SPREAD_RATIO:
        x = find_gutter(np.asarray(img))
        parts = [("left", img.crop((0, 0, x, img.height))),
                 ("right", img.crop((x, 0, img.width, img.height)))]
    out = []
    for label, part in parts:
        paras, confs = ocr_page(part)
        out.append((label, paras, confs))
    return out


def ocr_pdf(args) -> dict:
    """OCR every page of one PDF; write <out_dir>/<stem>.txt; return its manifest entry."""
    pdf, out_dir, dpi = args
    doc = fitz.open(str(pdf))
    all_paras, all_confs, low_pages, parts_n, blank = [], [], [], 0, 0
    for pno in range(doc.page_count):
        pix = doc[pno].get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        for label, paras, confs in ocr_sheet(img):
            parts_n += 1
            if not confs:
                blank += 1
                continue
            all_paras += paras
            all_confs += confs
            if sum(confs) / len(confs) < LOW_CONF_PAGE:
                low_pages.append(f"{pno + 1}{'' if label == 'page' else label[0]}")
    text = "\n\n".join(all_paras)
    Path(out_dir, pdf.stem + ".txt").write_text(text + "\n", encoding="utf-8")
    ver = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
    return {
        "doc_name": pdf.stem,
        "source_pdf": pdf.name,
        "pages": doc.page_count,
        "page_images": parts_n,
        "blank_page_images": blank,
        "words": len(all_confs),
        "mean_word_confidence": round(sum(all_confs) / len(all_confs), 1) if all_confs else 0.0,
        "low_confidence_pages": low_pages,
        "engine": (ver.stdout or ver.stderr).splitlines()[0].strip(),
        "dpi": dpi,
        "ocr_date": datetime.date.today().isoformat(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scan_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--manifest", default="ocr_manifest.yml")
    ap.add_argument("--only", nargs="*", help="PDF file names to process (default: every "
                    "PDF with no text layer and no .txt sibling)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    if fitz is None:
        sys.exit("PyMuPDF is required (pip install -r requirements-build.txt)")
    scan_dir, out_dir = Path(a.scan_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = []
    for f in sorted(scan_dir.glob("*.pdf")):
        if a.only is not None and f.name not in a.only:
            continue
        if a.only is None:
            if (scan_dir / (f.stem + ".txt")).exists():
                continue
            if any(p.get_text("text").strip() for p in fitz.open(str(f))):
                continue                       # already has a text layer
        pdfs.append(f)
    print(f"OCR of {len(pdfs)} PDFs, {a.jobs} jobs", flush=True)
    entries = []
    with Pool(a.jobs) as pool:
        for e in pool.imap_unordered(ocr_pdf, [(f, out_dir, a.dpi) for f in pdfs]):
            entries.append(e)
            print(f"  done {e['source_pdf'][:60]:60s} words={e['words']:>6} "
                  f"conf={e['mean_word_confidence']:>5} low-pages={len(e['low_confidence_pages'])}",
                  flush=True)
    entries.sort(key=lambda e: e["doc_name"].lower())
    manifest = Path(a.manifest)
    existing = {}
    if manifest.exists():
        existing = {e["doc_name"]: e for e in (yaml.safe_load(manifest.read_text()) or {}).get("ocr", [])}
    existing.update({e["doc_name"]: e for e in entries})
    manifest.write_text(
        "# OCR provenance. Every document listed here was read from a scan by OCR (a machine\n"
        "# reading, not a transcription): build_index.py marks its chunks `ocr: true` and\n"
        "# reports end with a note. Quality figures are tesseract word confidences (0-100).\n"
        + yaml.safe_dump({"ocr": sorted(existing.values(), key=lambda e: e["doc_name"].lower())},
                         sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
