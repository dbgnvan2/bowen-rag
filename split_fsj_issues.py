#!/usr/bin/env python3
"""
Split FSJ full-issue PDFs into individual article .txt files,
then move all full-issue PDFs to source_files/fsj_full_issues/.

TOC token patterns observed (TAB format):
  - Lines ending with \t become TAB tokens (content without trailing \t)
  - Lines that are purely digits become PAGE tokens
  - Section labels become HEADER tokens
  - Everything else is TEXT

Article entry patterns:
  A) Standard: TAB(title) -> PAGE(n) -> [TAB(cont)]* -> TEXT*(author)
  B) Multi-line title with split: TAB(title1) -> PAGE(n) -> TAB(cont)+ -> TEXT*(author)
  C) Author-before-page (FROM THE EDITOR style):
       TEXT("FROM THE EDITOR") -> TAB(author,credentials) -> PAGE(n)
  D) TAB(FROM THE EDITOR) -> PAGE(n) -> TEXT(author,credentials)  [9.1, 9.2 style]
  E) Author line ends with TAB (12.2 style):
       TAB(title) -> PAGE -> TAB(author,credentials) -> next TAB(title)/HEADER

For book reviews: scan forward after PAGE to find "Reviewed by: Name, Cred" line.
"""

import fitz
import os
import re
import shutil
import unicodedata

SOURCE = "/Users/davemini2/Downloads/bowen_rag/source_files"
DEST_SUBDIR = os.path.join(SOURCE, "fsj_full_issues")

ISSUES = [
    {"filename": "FSJ 4.2 Family Systems Journal Full Issue .pdf", "vol_issue": "4.2", "skip": True},
    {"filename": "FSJ 5.1 Family Systems Journal Full Issue OCR.pdf", "vol_issue": "5.1", "toc_pages": [0], "toc_format": "OCR"},
    {"filename": "FSJ 9.1 (1) Full Issue pdf.pdf", "vol_issue": "9.1", "toc_pages": [0], "toc_format": "TAB"},
    {"filename": "FSJ 9.2 (1) complete issue (1).pdf", "vol_issue": "9.2", "toc_pages": [0, 1], "toc_format": "TAB"},
    {"filename": "FSJ 10.2 complete issue fnl (1).pdf", "vol_issue": "10.2", "toc_pages": [0], "toc_format": "TAB"},
    {"filename": "FSJ 11.2 full issue.pdf", "vol_issue": "11.2", "toc_pages": [0], "toc_format": "TAB"},
    {"filename": "FSJ 12.1 (1) full issue (1).pdf", "vol_issue": "12.1", "toc_pages": [0], "toc_format": "TAB"},
    {"filename": "FSJ 12.2(1) Full Issue.pdf", "vol_issue": "12.2", "toc_pages": [0], "toc_format": "TAB"},
    {"filename": "FSJ 13.1 FSJ Digital Edition Full Issue.pdf", "vol_issue": "13.1", "toc_pages": [0], "toc_format": "TAB"},
    {"filename": "FSJ 13.2 Full Issue FSJ.pdf", "vol_issue": "13.2", "toc_pages": [0], "toc_format": "TAB"},
    {"filename": "FSJ 15.2 Full Digital Issue.pdf", "vol_issue": "15.2", "toc_pages": [2], "toc_format": "TAB"},
    {"filename": "FSJ 17.1 (1) Full Issue FS.pdf", "vol_issue": "17.1", "toc_pages": [2], "toc_format": "TAB"},
    {"filename": "FSJ 17.2 (1) full issue copy.pdf", "vol_issue": "17.2", "toc_pages": [2], "toc_format": "TAB"},
    {"filename": "FSJ 18.2 (1) full issue 2.pdf", "vol_issue": "18.2", "toc_pages": [2], "toc_format": "TAB"},
]

REDUNDANT_ISSUES = [
    "FSJ 16.1 (1) Family Systems full issue.pdf",
    "FSJ 16.2 (1) Family Systems full issue.pdf",
]

SECTION_LABELS = {
    "ARTICLES", "ARTICLE", "COMMENTARY", "BRIEF REPORTS", "BRIEF REPORT",
    "BOOK REVIEWS", "BOOK REVIEW", "FROM THE ARCHIVES", "FACULTY CASE CONFERENCE",
    "REFERENCES", "INDEX", "IN MEMORIAM", "CONTENTS",
    "THE IMPACT OF RELATIONSHIPS ON INDIVIDUAL VARIATION: THE FIFTH INTERDISCIPLINARY CONFERENCE",
    "INTRODUCTION TO CONFERENCE",
    "FIRST PANEL OF PRESENTATIONS", "SECOND PANEL OF PRESENTATIONS",
    "THIRD PANEL OF PRESENTATIONS", "FOURTH PANEL OF PRESENTATIONS",
    "FIFTH PANEL OF PRESENTATIONS", "SIXTH PANEL OF PRESENTATIONS",
    "SEVENTH PANEL OF PRESENTATIONS", "EIGHTH PANEL OF PRESENTATIONS",
}

STOP_SNIPPETS = [
    "Published by", "The Georgetown Family Center", "Washington, DC",
    "Bowen Center for the Study",
]

CRED_RE = re.compile(
    r'\b(PhD|MD|MSW|MA|RN|LCSW|LCPC|LCSW|EdD|MDiv|PsyD|JD|MEd|LMFT|LMFT|MSN|MSSW|MSAT|MS|MFT)\b'
)
# Credential patterns that appear at end of reviewer lines (non-name abbreviations)
TRAILING_CRED_RE = re.compile(
    r',?\s*\b(LCPC|LMFT|LCSW|CPC|MFT|RN|JD)\s*$'
)


# ─────────────────── utilities ───────────────────

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    return text.replace('­', '')


def is_section_label(s):
    return s.strip().rstrip('\t').upper() in SECTION_LABELS


def is_stop(s):
    return any(kw.lower() in s.lower() for kw in STOP_SNIPPETS)


def get_page_num(s):
    m = re.match(r'^\s*(\d{1,4})\s*$', s.strip())
    return int(m.group(1)) if m else None


def has_creds(s):
    return bool(CRED_RE.search(s))


def has_reviewer_prefix(s):
    return bool(re.match(r'(Reviewed by|Review:|Reviewer:)', s.strip()))


def extract_last_name(s):
    """Extract reviewer last name from an author credits line."""
    s = s.strip()
    for pfx in ("Reviewed by: ", "Reviewed by ", "Review: ", "Reviewer: ",
                "Presenter: ", "Introduction by: ", "Introduction by ",
                "Introduction: "):
        if s.startswith(pfx):
            s = s[len(pfx):]
    if " and " in s:
        s = s.split(" and ")[0].strip()
    # Remove trailing non-name credential abbreviations (LCPC, LMFT, etc.)
    s = TRAILING_CRED_RE.sub("", s).strip()
    # Remove all known credentials
    s = CRED_RE.sub("", s)
    s = re.sub(r',\s*$', '', s.strip()).strip()
    # Clean up leftover punctuation
    s = re.sub(r'\s+', ' ', s).strip()
    parts = s.split()
    if not parts:
        return "Unknown"
    return parts[-1].strip('.,;')


# ─────────────────── tokenizer ───────────────────

def tokenize_toc(lines):
    """
    Token types: TAB, PAGE, TEXT, HEADER, STOP.
    TAB: line ending with \t (stripped content, no trailing \t)
    PAGE: standalone integer
    HEADER: known section label
    STOP: publisher/end sentinel
    TEXT: everything else
    """
    tokens = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if is_stop(stripped):
            tokens.append(('STOP', stripped))
            break
        content = stripped.rstrip('\t').strip()
        ends_tab = raw.rstrip('\n\r').endswith('\t')
        pg = get_page_num(stripped)

        if pg is not None:
            tokens.append(('PAGE', pg))
        elif is_section_label(content):
            tokens.append(('HEADER', content.upper().rstrip()))
        elif ends_tab:
            tokens.append(('TAB', content))
        else:
            tokens.append(('TEXT', content))
    return tokens


# ─────────────────── TAB-format article parser ───────────────────

def is_new_entry_start(tokens, idx):
    """True if tokens[idx] is a TAB followed (at some point) by PAGE."""
    if idx >= len(tokens) or tokens[idx][0] != 'TAB':
        return False
    for j in range(idx + 1, min(idx + 4, len(tokens))):
        if tokens[j][0] == 'PAGE':
            return True
        if tokens[j][0] in ('TAB', 'HEADER', 'STOP'):
            break
    return False


def parse_tab_tokens(tokens):
    """
    Parse tokens into list of (title, author_last, start_page).

    Walk the token list. When we see TAB->PAGE, we have an entry.
    Collect title (and continuation TAB lines), then scan for author.

    Special cases:
    - TAB(author,creds)->PAGE: FROM THE EDITOR pattern (author before page)
    - After PAGE, TAB lines that are title continuations
    - After PAGE+title_conts, TEXT lines build up until we find one with creds
      OR a "Reviewed by:" pattern (for book reviews)
    - TAB(author,creds) immediately before next entry start or HEADER: it's the author
    """
    articles = []
    i = 0
    n = len(tokens)

    def next_page_distance(start):
        """How many tokens until next PAGE from start."""
        for k in range(start, min(start + 4, n)):
            if tokens[k][0] == 'PAGE':
                return k - start
        return 999

    while i < n:
        typ, val = tokens[i]

        if typ in ('TEXT', 'PAGE', 'STOP'):
            i += 1
            continue

        if typ == 'HEADER':
            i += 1
            continue

        # TAB token
        assert typ == 'TAB', f"unexpected {typ}"

        # Peek ahead to find PAGE
        j = i + 1
        while j < n and tokens[j][0] not in ('PAGE', 'HEADER', 'STOP'):
            if tokens[j][0] == 'TAB':
                # This intermediate TAB breaks the simple pattern - stop looking
                # unless we're within 1 step (i.e., j == i+1)
                if j > i + 1:
                    break
            j += 1

        if j >= n or tokens[j][0] != 'PAGE':
            # No PAGE found after this TAB within expected range
            # Check if this TAB is an author-before-HEADER pattern
            # (e.g. TAB(author) followed directly by HEADER or STOP)
            i += 1
            continue

        page_num = tokens[j][1]

        # Pattern check: is this TAB an AUTHOR before PAGE?
        # This happens in "FROM THE EDITOR" sections where editor name comes first:
        #   TEXT("FROM THE EDITOR") -> TAB(editor_name, creds) -> PAGE
        # A name-before-page line looks like "Firstname Lastname, Credentials"
        # NOT like "An Interview with..." or other sentence-starting phrases.
        looks_like_name = bool(re.match(
            r'^[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+\s*,\s*\w',
            val
        ))
        if has_creds(val) and j == i + 1 and looks_like_name:
            # Author-before-page pattern
            # Determine title from context (previous HEADER or TEXT token)
            title = "FROM THE EDITOR"
            look_back = i - 1
            while look_back >= 0:
                lb_typ, lb_val = tokens[look_back]
                if lb_typ == 'TEXT' and 'EDITOR' in lb_val.upper():
                    title = "FROM THE EDITOR"
                    break
                if lb_typ in ('TAB', 'HEADER'):
                    break
                look_back -= 1
            author_last = extract_last_name(val)
            articles.append((title, author_last, page_num))
            i = j + 1
            continue

        # Standard pattern: TAB(title) -> PAGE(n)
        title_parts = [val]
        k = j + 1  # start after PAGE

        # Collect title continuations and find author
        author_last = "Unknown"
        reviewer_found = None

        while k < n:
            ktyp, kval = tokens[k]

            if ktyp == 'STOP':
                break
            if ktyp == 'HEADER':
                break

            if ktyp == 'PAGE':
                # Another page? Shouldn't happen mid-article collection, stop
                break

            if ktyp == 'TAB':
                # TAB after PAGE could be:
                # (a) title continuation if next token is NOT PAGE and no creds
                # (b) author with credentials (followed by HEADER, STOP, or another TAB->PAGE)
                # (c) next article's title (followed by PAGE)
                # (d) reviewer line starting with "Reviewed by"
                # (e) book author (no creds) followed by TEXT(Reviewed by...)

                # Check: reviewer prefix in TAB -> it's the author
                if has_reviewer_prefix(kval):
                    author_last = extract_last_name(kval)
                    k += 1
                    break

                # Check if next meaningful token is PAGE -> this TAB starts a new entry
                m = k + 1
                while m < n and tokens[m][0] not in ('PAGE', 'TAB', 'HEADER', 'STOP', 'TEXT'):
                    m += 1
                next_meaningful_is_page = (m < n and tokens[m][0] == 'PAGE')

                if next_meaningful_is_page:
                    # This TAB is the start of a new entry - stop
                    break

                # Check if next meaningful token is HEADER or STOP
                next_meaningful_is_end = (m < n and tokens[m][0] in ('HEADER', 'STOP'))
                # Also: no next non-end token
                at_end = (m >= n)

                # Check if this TAB has credentials -> it's the author
                if has_creds(kval):
                    author_last = extract_last_name(kval)
                    k += 1
                    break

                # Check: TAB (no creds) followed by TEXT(Reviewed by...) -> book author, skip
                if m < n and tokens[m][0] == 'TEXT' and has_reviewer_prefix(tokens[m][1]):
                    # Skip this TAB (book author), let TEXT(Reviewed by) be handled next
                    k += 1
                    continue

                # Otherwise: title continuation
                title_parts.append(kval)
                k += 1
                continue

            if ktyp == 'TEXT':
                # TEXT after PAGE (and optional title continuations)
                # Could be: title continuation (no creds), book author, or article author

                # Check for reviewer prefix first
                if has_reviewer_prefix(kval):
                    author_last = extract_last_name(kval)
                    reviewer_found = True
                    k += 1
                    # Skip remaining TEXT lines (additional credits, intro lines)
                    while k < n and tokens[k][0] == 'TEXT':
                        k += 1
                    break

                # If this TEXT has credentials, it's the author
                if has_creds(kval):
                    author_last = extract_last_name(kval)
                    k += 1
                    # Continue scanning for "Reviewed by:" which overrides
                    while k < n:
                        ntyp, nval = tokens[k]
                        if ntyp == 'TEXT' and has_reviewer_prefix(nval):
                            author_last = extract_last_name(nval)
                            k += 1
                            break
                        if ntyp in ('TAB', 'HEADER', 'STOP'):
                            break
                        if ntyp == 'TEXT':
                            k += 1
                            continue
                        k += 1
                    break

                # TEXT without credentials - could be title continuation or book author
                # Look ahead to determine: if next TEXT has creds or is reviewer -> current is book info
                # If next is TAB or HEADER -> current is probably author (no creds, like "Murray Bowen, MD"
                # but without credential match... actually Murray Bowen, MD WOULD match)
                # For safety: look ahead
                m = k + 1
                while m < n and tokens[m][0] not in ('TEXT', 'TAB', 'HEADER', 'STOP'):
                    m += 1

                if m < n:
                    mtyp, mval = tokens[m]
                    if mtyp in ('TEXT', 'TAB'):
                        if has_reviewer_prefix(mval) or has_creds(mval):
                            # Current line is book subtitle/author info, not our author
                            k += 1
                            continue
                        # Another plain TEXT follows - current might be author
                        # Look one more ahead
                        mm = m + 1
                        while mm < n and tokens[mm][0] not in ('TEXT', 'TAB', 'HEADER', 'STOP'):
                            mm += 1
                        if mm < n:
                            mm_typ, mm_val = tokens[mm]
                            is_reviewer_ahead = (
                                (mm_typ == 'TEXT' and (has_reviewer_prefix(mm_val) or has_creds(mm_val))) or
                                (mm_typ == 'TAB' and (has_reviewer_prefix(mm_val) or has_creds(mm_val)))
                            )
                            if is_reviewer_ahead:
                                # Skip current (book subtitle) and next (book author)
                                k += 1
                                continue
                        # Default: this TEXT is probably the author
                        author_last = extract_last_name(kval)
                        k += 1
                        break
                    elif mtyp == 'TAB':
                        # Next is a TAB - check if that TAB has creds (it's the author)
                        if has_creds(mval):
                            # Current TEXT is a title continuation; TAB is the author
                            title_parts.append(kval)
                            k += 1
                            continue
                        # Next TAB has no creds - check if it's followed by PAGE
                        mm = m + 1
                        while mm < n and tokens[mm][0] not in ('PAGE', 'TAB', 'HEADER', 'STOP', 'TEXT'):
                            mm += 1
                        if mm < n and tokens[mm][0] == 'PAGE':
                            # Next TAB is start of new entry - current TEXT is the author
                            author_last = extract_last_name(kval)
                            k += 1
                            break
                        # Otherwise current TEXT is title continuation
                        title_parts.append(kval)
                        k += 1
                        continue
                    elif mtyp in ('HEADER', 'STOP'):
                        # Nothing useful follows -> it's the author
                        author_last = extract_last_name(kval)
                        k += 1
                        break
                else:
                    # End of tokens
                    author_last = extract_last_name(kval)
                    k += 1
                    break

        full_title = ' '.join(title_parts)
        full_title = re.sub(r'\s+', ' ', full_title).strip()

        if full_title and not is_section_label(full_title):
            articles.append((full_title, author_last, page_num))

        i = k

    return articles


# ─────────────────── OCR-format parser (FSJ 5.1) ───────────────────

def parse_toc_ocr(lines, vol_issue):
    """
    FSJ 5.1 OCR format (heavily garbled).
    Strategy: scan for lines that are purely numeric (page numbers),
    and treat the previous non-header, non-short line as the title.
    Author is next line with credentials after the page number.
    """
    OCR_HEADERS = {
        "CONTENTS", "ARTICLES", "COMMENTARY", "BRIEF REPORTS",
        "BOOK REVIEWS", "FROM THE ARCHIVES",
    }

    clean = []
    for ln in lines:
        s = ln.strip()
        if s and not is_stop(s):
            clean.append(s)
        elif is_stop(s):
            break

    # First pass: find all (title_idx, page_num, page_idx) triples
    entries = []  # (title_candidate, page_num, page_line_idx)

    # Special handling for year vs page
    year_like = {1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005}

    for j, line in enumerate(clean):
        pg = get_page_num(line)
        if pg is None or pg in year_like:
            continue
        # Bare "Si" OCR error - skip
        if line.strip() == 'Si':
            continue
        # Find title: look backward for a non-numeric, non-header line
        title = "Unknown"
        author_last = "Unknown"
        for back in range(j - 1, max(j - 5, -1), -1):
            candidate = clean[back]
            u_cand = candidate.upper().rstrip('_')
            pg_cand = get_page_num(candidate)
            if pg_cand is not None and pg_cand not in year_like:
                break  # hit another page number
            if u_cand in OCR_HEADERS or is_section_label(candidate):
                # The previous section label is the entry type
                # (e.g., "FROM THE EDITOR_" -> title is "FROM THE EDITOR")
                if 'EDITOR' in u_cand:
                    title = "FROM THE EDITOR"
                break
            if len(candidate) < 3:
                continue  # skip OCR junk like "F", "A"
            # Check it doesn't look like OCR garbage (starts with -)
            if candidate.startswith('-') or candidate.startswith('F') and len(candidate) <= 2:
                continue
            title = candidate.lstrip("'\"").strip()
            break

        # Look forward for author (scan multiple lines, skip title continuations)
        for fwd in range(j + 1, min(j + 6, len(clean))):
            al = clean[fwd]
            pg2 = get_page_num(al)
            if pg2 is not None and pg2 not in year_like:
                break
            if al.upper().rstrip('_') in OCR_HEADERS or is_section_label(al):
                break
            if is_stop(al):
                break
            if has_creds(al) or has_reviewer_prefix(al):
                author_last = extract_last_name(al)
                break
            # Skip OCR garbage / very short lines
            if len(al) <= 2:
                continue

        # Clean up title
        title = title.rstrip('_').strip()
        # Clean up title
        title = title.rstrip('_').strip()
        if title and title != "Unknown" and not is_section_label(title.rstrip('_').upper()):
            entries.append((title, author_last, pg))

    # FSJ 5.1 specific: add missing first article (Caskie) that OCR lost page num for
    # It starts at journal page 7 (PDF page 6) - Bowen Theory and Health Care Costs
    if vol_issue == "5.1":
        # Insert Caskie article at page 7 if not already present
        pages_found = {e[2] for e in entries}
        if 7 not in pages_found and 8 not in pages_found:
            caskie = ("Bowen Theory and Health Care Costs in the United States",
                      "Caskie", 7)
            # Insert after FROM THE GUEST EDITOR (page 5)
            insert_pos = 0
            for idx, (t, a, p) in enumerate(entries):
                if p == 5:
                    insert_pos = idx + 1
                    break
            entries.insert(insert_pos, caskie)

        # Also fix the HMO article (Quinn) - appears to start at PDF page 18 (journal ~19-20)
        # but may be captured under ARTICLES or missed
        if 18 not in pages_found and 19 not in pages_found and 20 not in pages_found:
            quinn = ("The Health Maintenance Organization as an Emotional System",
                     "Quinn", 19)
            # Insert before Anxiety and Differentiation (page 31)
            insert_pos2 = len(entries)
            for idx, (t, a, p) in enumerate(entries):
                if p == 31:
                    insert_pos2 = idx
                    break
            entries.insert(insert_pos2, quinn)

    return entries


# ─────────────────── page mapping ───────────────────

def build_page_map(doc):
    """
    Build {journal_page: pdf_page_index}.
    Handles multiple header formats:
      - "NNN"  (standalone integer)
      - "NNN\t..."  (tab-separated header)
      - "NNN | Family Systems..." (pipe-separated newer format)
      - "Section Title | NNN"  (page number at end after pipe)
    """
    page_map = {}
    for pdf_idx in range(len(doc)):
        text = doc[pdf_idx].get_text().strip()
        if not text:
            continue
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        found = False
        for line in lines[:6]:
            # Pattern 1: standalone integer
            m = re.match(r'^(\d{1,4})\s*$', line)
            if m:
                jp = int(m.group(1))
                if jp not in page_map:
                    page_map[jp] = pdf_idx
                found = True
                break
            # Pattern 2: "NNN\t..." (tab-separated)
            m2 = re.match(r'^(\d{1,4})\t', line)
            if m2:
                jp = int(m2.group(1))
                if jp not in page_map:
                    page_map[jp] = pdf_idx
                found = True
                break
            # Pattern 3: "NNN | Family Systems..." (newer digital format)
            m3 = re.match(r'^(\d{1,4})\s*\|\s*Family Systems', line)
            if m3:
                jp = int(m3.group(1))
                if jp not in page_map:
                    page_map[jp] = pdf_idx
                found = True
                break
            # Pattern 4: "Text | NNN" (page num at end after pipe)
            m4 = re.search(r'\|\s*(\d{1,4})\s*$', line)
            if m4:
                jp = int(m4.group(1))
                if jp not in page_map:
                    page_map[jp] = pdf_idx
                found = True
                break
        if not found:
            # Try all lines in the page for a standalone number
            for line in lines:
                m = re.match(r'^(\d{1,4})\s*$', line)
                if m:
                    jp = int(m.group(1))
                    if jp not in page_map:
                        page_map[jp] = pdf_idx
                    break
    return page_map


def find_pdf_page(page_map, jp, fallback=None):
    if jp in page_map:
        return page_map[jp]
    for d in range(1, 8):
        if jp - d in page_map:
            return page_map[jp - d]
        if jp + d in page_map:
            return page_map[jp + d]
    return fallback


def extract_article_text(doc, start_pdf, end_pdf):
    parts = []
    for idx in range(start_pdf, min(end_pdf + 1, len(doc))):
        text = doc[idx].get_text()
        if text.strip():
            parts.append(clean_text(text))
    return '\n\n'.join(parts)


# ─────────────────── filename generation ───────────────────

def sanitize_fname(s):
    s = re.sub(r"[<>:\"/\\|?*\x00-\x1f'''‘’]", '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def make_filename(vol_issue, author_last, title):
    words = title.split()
    if words and words[0].lower() in ('a', 'an', 'the', 'on'):
        words = words[1:]
    short = ' '.join(words[:5])
    raw = f"FSJ {vol_issue} {author_last} {short}.txt"
    return sanitize_fname(raw)


# ─────────────────── main ───────────────────

def process_issue(cfg):
    filename = cfg["filename"]
    vol_issue = cfg["vol_issue"]
    pdf_path = os.path.join(SOURCE, filename)

    print(f"\n{'='*60}")
    print(f"Processing: {filename}")

    if cfg.get("skip"):
        print("  SKIPPING: image-only PDF")
        return []

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"  PDF pages: {total_pages}")

    page_map = build_page_map(doc)
    if page_map:
        print(f"  Page map: {len(page_map)} entries "
              f"({min(page_map.keys())}-{max(page_map.keys())})")
    else:
        print("  WARNING: empty page map")

    toc_lines = []
    for pg in cfg["toc_pages"]:
        toc_lines.extend(doc[pg].get_text().split('\n'))

    fmt = cfg.get("toc_format", "TAB")
    if fmt == "OCR":
        articles = parse_toc_ocr(toc_lines, vol_issue)
    else:
        tokens = tokenize_toc(toc_lines)
        articles = parse_tab_tokens(tokens)

    print(f"  TOC articles: {len(articles)}")
    for title, author, pg in articles:
        print(f"    [{pg:4}] {author:15}: {title[:55]}")

    if not articles:
        print("  WARNING: no articles parsed!")
        doc.close()
        return []

    # Exclude year-like values (1990-2050) from max page calculation
    content_pages = {jp: pdfp for jp, pdfp in page_map.items() if jp < 1000}
    max_jp = max(content_pages.keys()) if content_pages else total_pages * 2
    results = []

    for idx, (title, author, start_jp) in enumerate(articles):
        if idx + 1 < len(articles):
            end_jp = articles[idx + 1][2] - 1
        else:
            end_jp = max_jp

        start_pdf = find_pdf_page(content_pages, start_jp)
        end_pdf = find_pdf_page(content_pages, end_jp,
                                fallback=(start_pdf or 0) + 30)

        if start_pdf is None:
            print(f"  WARN: cannot map page {start_jp} for '{title[:40]}'")
            continue

        # Ensure end_pdf >= start_pdf
        if end_pdf is None or end_pdf < start_pdf:
            end_pdf = min(start_pdf + 30, total_pages - 1)

        end_pdf = min(end_pdf, total_pages - 1)

        text = extract_article_text(doc, start_pdf, end_pdf)
        if not text.strip():
            print(f"  WARN: empty text for '{title[:40]}' (pdf {start_pdf}-{end_pdf})")

        header = (f"Family Systems Journal Vol. {vol_issue}\n"
                  f"Title: {title}\n"
                  f"Author: {author}\n"
                  f"Pages: {start_jp}-{end_jp}\n\n")
        full_text = header + text

        out_fname = make_filename(vol_issue, author, title)
        out_path = os.path.join(SOURCE, out_fname)
        results.append((out_fname, out_path, full_text))
        print(f"  -> {out_fname}")
        print(f"     (journal {start_jp}-{end_jp}, pdf {start_pdf}-{end_pdf})")

    doc.close()
    return results


def move_full_issues():
    os.makedirs(DEST_SUBDIR, exist_ok=True)
    all_fnames = [cfg["filename"] for cfg in ISSUES] + REDUNDANT_ISSUES
    moved, missing = [], []
    for fname in all_fnames:
        src = os.path.join(SOURCE, fname)
        dst = os.path.join(DEST_SUBDIR, fname)
        if os.path.exists(src):
            shutil.move(src, dst)
            moved.append(fname)
        else:
            missing.append(fname)
    return moved, missing


def main():
    print("FSJ Full-Issue PDF Splitter")
    print("="*60)

    all_results = []
    skipped = []

    for cfg in ISSUES:
        results = process_issue(cfg)
        if cfg.get("skip"):
            skipped.append(cfg["filename"])
        else:
            all_results.extend(results)

    print(f"\n{'='*60}")
    print("WRITING ARTICLE FILES...")
    total_written = 0
    for out_fname, out_path, text in all_results:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        total_written += 1
        print(f"  Written: {out_fname}")

    print(f"\n{'='*60}")
    print("MOVING FULL-ISSUE PDFs...")
    moved, missing = move_full_issues()
    for fname in moved:
        print(f"  Moved: {fname}")
    for fname in missing:
        print(f"  NOT FOUND: {fname}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Articles written:            {total_written}")
    print(f"  Issues skipped (image-only): {len(skipped)}")
    print(f"  Full-issue PDFs moved:       {len(moved)}")
    if missing:
        print(f"  PDFs NOT FOUND:              {len(missing)}")
        for m in missing:
            print(f"    - {m}")
    print("Done.")


if __name__ == "__main__":
    main()
