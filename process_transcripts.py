#!/usr/bin/env python3
"""
Import formatted transcript yaml.md files into the RAG source_files directory.

Reads transcript files from ~/transcripts/projects/ that have been processed
with section headings (## Section N – Title), strips the YAML frontmatter,
and writes clean .txt files to source_files/ ready for indexing.

Usage:
    python3 process_transcripts.py [--transcripts-dir DIR] [--source-dir DIR] [--dry-run]

Run from the bowen_rag directory:
    python3 process_transcripts.py
"""

import re
import sys
import shutil
import argparse
from pathlib import Path

TRANSCRIPTS_DIR = Path.home() / "transcripts" / "projects"
SOURCE_DIR      = Path(__file__).parent / "source_files"

# Files to skip (test files, duplicates, etc.)
SKIP_PATTERNS = [
    r"(?i)test",
    r"HIPAA-Safe-Harbor",
]


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata_dict, body_text)."""
    m = re.match(r'^---\n(.*?)\n---\n', text, re.DOTALL)
    if not m:
        return {}, text
    fm = {}
    for line in m.group(1).splitlines():
        if ':' in line and not line.startswith(' '):
            k, _, v = line.partition(':')
            fm[k.strip()] = v.strip().strip('"')
    body = text[m.end():]
    return fm, body


def has_sections(text: str) -> bool:
    return bool(re.search(r'^## Section \d+', text, re.MULTILINE))


def safe_filename(name: str) -> str:
    """Strip characters unsafe for filenames and trim to 200 chars."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name[:200]


def process_file(md_path: Path, source_dir: Path, dry_run: bool) -> str | None:
    """
    Parse one yaml.md transcript and write it to source_dir as a .txt file.
    Returns the output filename, or None if skipped.
    """
    text = md_path.read_text(encoding='utf-8')
    fm, body = parse_frontmatter(text)

    if not has_sections(body):
        return None  # Not yet section-formatted — skip

    title = fm.get('Title', md_path.stem)
    presenter = fm.get('Presenter', '')
    date = fm.get('Lecture date', '')

    # Build a clean doc name for the filename
    parts = [title]
    if presenter:
        parts.append(presenter)
    if date and date != '1980-01-01':   # suppress placeholder dates
        parts.append(date)
    doc_name = " - ".join(parts)
    out_name  = safe_filename(doc_name) + ".txt"
    out_path  = source_dir / out_name

    # Clean up body: normalize separator lines, preserve section headings
    body_clean = body.strip()
    # Remove bare --- separator lines (keep content)
    body_clean = re.sub(r'\n---\n', '\n\n', body_clean)
    # Remove timestamp placeholders like ([00:00:00])
    body_clean = re.sub(r'\s*\(\[[\d:]+\]\)', '', body_clean)
    # Normalise multiple blank lines
    body_clean = re.sub(r'\n{3,}', '\n\n', body_clean).strip()

    if dry_run:
        sections = len(re.findall(r'^## Section \d+', body_clean, re.MULTILINE))
        words    = len(body_clean.split())
        print(f"  [DRY RUN] Would write: {out_name}  ({sections} sections, {words:,} words)")
        return out_name

    source_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body_clean, encoding='utf-8')
    sections = len(re.findall(r'^## Section \d+', body_clean, re.MULTILINE))
    words    = len(body_clean.split())
    print(f"  Written: {out_name}  ({sections} sections, {words:,} words)")
    return out_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcripts-dir', type=Path, default=TRANSCRIPTS_DIR,
                        help=f'Transcript projects directory (default: {TRANSCRIPTS_DIR})')
    parser.add_argument('--source-dir', type=Path, default=SOURCE_DIR,
                        help=f'RAG source_files directory (default: {SOURCE_DIR})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be written without writing anything')
    args = parser.parse_args()

    print(f"Scanning: {args.transcripts_dir}")
    print(f"Output:   {args.source_dir}")
    if args.dry_run:
        print("(DRY RUN — no files will be written)\n")
    else:
        print()

    md_files = sorted(args.transcripts_dir.rglob('*yaml.md'))
    if not md_files:
        print("No yaml.md files found.")
        return

    written = []
    skipped_skip = []
    skipped_no_sections = []

    for md_path in md_files:
        name = md_path.name

        # Apply skip patterns
        if any(re.search(p, name) for p in SKIP_PATTERNS):
            skipped_skip.append(name)
            continue

        result = process_file(md_path, args.source_dir, args.dry_run)
        if result is None:
            skipped_no_sections.append(name)
        else:
            written.append(result)

    print(f"\nDone.")
    print(f"  Imported:  {len(written)}")
    if skipped_no_sections:
        print(f"  Skipped (no sections yet): {len(skipped_no_sections)}")
        for n in skipped_no_sections:
            print(f"    {n}")
    if skipped_skip:
        print(f"  Skipped (excluded):        {len(skipped_skip)}")
        for n in skipped_skip:
            print(f"    {n}")

    if written and not args.dry_run:
        print(f"\nNext step: rebuild the index.")
        print(f"  python3 rag-document-search/scripts/build_index.py source_files/ rag-document-search/references/")


if __name__ == '__main__':
    main()
