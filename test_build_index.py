"""
Tests for rag-document-search/scripts/build_index.py chunking and index files.

Key regression: a transcript whose FIRST line is "## Section 1 – ..." lost Section 1,
because the split required a newline before each heading and then discarded the text
before the first match. process_transcripts.py writes exactly that shape.

Run: python3 -m unittest test_build_index
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse as sp_sparse

sys.path.insert(0, str(Path(__file__).resolve().parent / "rag-document-search" / "scripts"))
import build_index as B  # noqa: E402

SECTIONED = (
    "## Section 1 – Overview of the Family\n\nThe family is an emotional unit.\n\n"
    "## Section 2 – Triangles ([00:05:10])\n\nTriangles stabilise two-person systems.\n\n"
    "## Section 3 – Cutoff\n\nCutoff is a way of managing anxiety.\n"
)


def chunk(content, doc="doc"):
    return B.DocumentIndexer(".").chunk_document(doc, content)


def chunk_quiet(content, doc="doc"):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = chunk(content, doc)
    return out, buf.getvalue()


class TestSectionSplit(unittest.TestCase):
    def test_idx_first_section_at_byte_zero_is_indexed(self):
        chunks, _ = chunk_quiet(SECTIONED)
        self.assertEqual([c["section_title"] for c in chunks],
                         ["Overview of the Family", "Triangles", "Cutoff"])

    def test_idx_chunk_count_equals_heading_count(self):
        # Adversarial: "looks right" (chunks exist) but one section is missing.
        chunks, _ = chunk_quiet(SECTIONED)
        self.assertEqual(len(chunks), SECTIONED.count("## Section"))

    def test_idx_text_before_first_heading_is_reported_not_silently_dropped(self):
        chunks, out = chunk_quiet("# Transcript Formatting\nnotes\n\n" + SECTIONED)
        self.assertEqual(len(chunks), 3)
        self.assertIn("before the first section heading are not indexed", out)

    def test_idx_hyphen_and_em_dash_headings_are_sections(self):
        content = ("## Section 1 - One\n\nalpha text\n\n"
                   "## Section 2 — Two\n\nbeta text\n")
        chunks, _ = chunk_quiet(content)
        self.assertEqual([c["section_title"] for c in chunks], ["One", "Two"])

    def test_idx_utf8_bom_does_not_hide_a_heading_at_byte_zero(self):
        # Reproduced by learning-qa: "\ufeff## Section 1 ..." lost Section 1 again.
        chunks, _ = chunk_quiet("\ufeff" + SECTIONED)
        self.assertEqual(len(chunks), 3)

    def test_idx_bom_file_is_read_without_the_bom(self):
        with tempfile.TemporaryDirectory() as src:
            (Path(src) / "a.txt").write_bytes(b"\xef\xbb\xbf" + SECTIONED.encode("utf-8"))
            docs = B.DocumentIndexer(src).load_documents()
        self.assertFalse(docs[0][1].startswith("\ufeff"))
        chunks, _ = chunk_quiet(docs[0][1])
        self.assertEqual(len(chunks), 3)

    def test_idx_headings_with_no_blank_line_between_them_keep_their_own_text(self):
        # Reproduced by learning-qa: the second heading was swallowed into the first
        # section's text and the warning wrongly blamed empty sections.
        content = ("## Section 1 – A\n## Section 2 – B\nbody of b\n"
                   "## Section 3 – C\nbody of c\n")
        chunks, out = chunk_quiet(content)
        self.assertEqual([c["section_title"] for c in chunks], ["B", "C"])
        self.assertIn("body of b", chunks[0]["text"])
        self.assertNotIn("## Section", chunks[0]["text"])
        self.assertIn("only 2 of 3 sections indexed", out)    # A really is empty

    def test_idx_crlf_line_endings(self):
        chunks, _ = chunk_quiet(SECTIONED.replace("\n", "\r\n"))
        self.assertEqual(len(chunks), 3)

    def test_idx_empty_sections_are_skipped_with_a_warning(self):
        content = "## Section 1 – One\n\n\n## Section 2 – Two\n\nreal text\n"
        chunks, out = chunk_quiet(content)
        self.assertEqual(len(chunks), 1)
        self.assertIn("only 1 of 2 sections indexed", out)

    def test_idx_sectioned_doc_with_no_usable_sections_falls_back_loudly(self):
        content = "## Section 1 – One\n\n## Section 2 – Two\n\n"
        chunks, out = chunk_quiet(content)
        self.assertIn("falling back to word-count", out)
        self.assertEqual(len(chunks), 1)      # fallback chunker kept the text

    def test_idx_plain_document_uses_wordcount_chunks(self):
        chunks, _ = chunk_quiet("One sentence here. Another sentence there. " * 100)
        self.assertGreater(len(chunks), 1)
        self.assertNotIn("section_title", chunks[0])


class TestBuildAndSave(unittest.TestCase):
    def test_idx_saved_files_agree_with_each_other(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            for i in range(3):
                (Path(src) / f"d{i}.txt").write_text(SECTIONED.replace("Cutoff", f"Cutoff{i}"))
            ix = B.DocumentIndexer(src)
            with contextlib.redirect_stdout(io.StringIO()):
                stats = ix.build_index()
                ix.save_index(out)
            meta = json.loads((Path(out, "chunk_metadata.json")).read_text())
            vec = json.loads((Path(out, "vectorizer.json")).read_text())
            matrix = sp_sparse.load_npz(str(Path(out, "tfidf_matrix.npz")))
            self.assertEqual(len(meta), 9)                      # 3 docs x 3 sections
            self.assertEqual(matrix.shape, (len(meta), len(vec["feature_names"])))
            self.assertEqual(stats["num_chunks"], len(meta))
            # vectorizer.json used to claim max_features=500 while 8000 was used
            self.assertEqual(vec["max_features"], B.TFIDF_PARAMS["max_features"])
            self.assertEqual(vec["min_df"], B.TFIDF_PARAMS["min_df"])


if __name__ == "__main__":
    unittest.main()
