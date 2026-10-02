"""
Tests for rag-document-search/scripts/build_index.py (the merged paragraph-aligned,
metadata-enriched builder).

Regressions covered, from both lines of work:
  * a transcript whose FIRST line is "## Section 1 – ..." lost Section 1 (old builder);
  * chunks could span two sections and carry only the first one's title (enriched builder);
  * a PDF with PyMuPDF missing was decoded as text and indexed as garbage;
  * a malformed config YAML silently stripped all citation metadata;
  * a cp1252 file decoded as BOM-less UTF-16 and was indexed as mojibake;
  * empty / unreadable documents were skipped without a word.

Run: python3 -m unittest test_build_index
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
from scipy import sparse as sp_sparse

sys.path.insert(0, str(Path(__file__).resolve().parent / "rag-document-search" / "scripts"))
import build_index as B  # noqa: E402

SECTIONED = (
    "## Section 1 – Overview of the Family\n\nThe family is an emotional unit.\n\n"
    "## Section 2 – Triangles ([00:05:10])\n\nTriangles stabilise two-person systems.\n\n"
    "## Section 3 – Cutoff\n\nCutoff is a way of managing anxiety.\n"
)


def chunks_of(text, doc="doc"):
    stats = {}
    paras = B.txt_paragraphs(text, stats)
    return B.build_chunks(doc, paras), stats


class TestSections(unittest.TestCase):
    def test_idx_first_section_at_byte_zero_keeps_its_title(self):
        chunks, _ = chunks_of(SECTIONED)
        self.assertEqual([c["section_title"] for c in chunks],
                         ["Overview of the Family", "Triangles", "Cutoff"])
        self.assertNotIn("## Section", chunks[0]["text"])    # heading line is not body text

    def test_idx_chunk_never_spans_two_sections(self):
        # Adversarial (P7): short sections would all fit in one 1,500-char chunk; each
        # must still be its own chunk, or later text carries the wrong section title.
        chunks, _ = chunks_of(SECTIONED)
        self.assertEqual(len(chunks), 3)
        self.assertIn("emotional unit", chunks[0]["text"])
        self.assertNotIn("Triangles stabilise", chunks[0]["text"])
        self.assertIn("Triangles stabilise", chunks[1]["text"])

    def test_idx_long_section_splits_into_chunks_that_all_keep_the_title(self):
        long_body = "\n\n".join(f"Paragraph {i} " + "word " * 120 for i in range(8))
        chunks, _ = chunks_of(f"## Section 1 – Long one\n\n{long_body}\n")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c["section_title"] == "Long one" for c in chunks))

    def test_idx_utf8_bom_before_the_first_heading(self):
        chunks, _ = chunks_of("﻿" + SECTIONED)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["section_title"], "Overview of the Family")

    def test_idx_crlf_line_endings(self):
        chunks, _ = chunks_of(SECTIONED.replace("\n", "\r\n"))
        self.assertEqual(len(chunks), 3)

    def test_idx_hyphen_and_em_dash_headings_are_sections(self):
        chunks, _ = chunks_of("## Section 1 - One\n\nalpha text\n\n"
                              "## Section 2 — Two\n\nbeta text\n")
        self.assertEqual([c["section_title"] for c in chunks], ["One", "Two"])

    def test_idx_adjacent_headings_keep_their_own_text_and_report_the_empty_one(self):
        chunks, stats = chunks_of("## Section 1 – A\n## Section 2 – B\nbody of b\n"
                                  "## Section 3 – C\nbody of c\n")
        self.assertEqual([c["section_title"] for c in chunks], ["B", "C"])
        self.assertNotIn("## Section", chunks[0]["text"])
        self.assertEqual((stats["headings"], stats["empty"]), (3, 1))

    def test_idx_one_line_formatting_banner_is_not_indexed_but_real_preamble_is(self):
        banner, _ = chunks_of("# Transcript Formatting - v12\n\n" + SECTIONED)
        self.assertEqual(len(banner), 3)
        real, _ = chunks_of("Opening remarks by the host.\nMore of them.\n\n" + SECTIONED)
        self.assertEqual(len(real), 4)
        self.assertEqual(real[0]["section_title"], "")
        self.assertIn("Opening remarks", real[0]["text"])

    def test_idx_plain_document_has_no_sections_and_numbered_paragraphs(self):
        chunks, stats = chunks_of("First paragraph here.\n\nSecond one.\n\nThird.")
        self.assertEqual(stats, {})
        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0]["paragraph_start"], chunks[0]["paragraph_end"]), (1, 3))
        self.assertEqual(chunks[0]["section_title"], "")


class TestParagraphChunks(unittest.TestCase):
    def test_idx_chunks_are_whole_paragraphs_with_continuous_numbering(self):
        paras_text = [f"Paragraph {i}. " + " ".join(["alpha"] * 100) for i in range(1, 13)]
        chunks, _ = chunks_of("\n\n".join(paras_text))
        self.assertGreater(len(chunks), 1)
        covered = []
        for c in chunks:
            covered += list(range(c["paragraph_start"], c["paragraph_end"] + 1))
            for part in c["text"].split("\n\n"):
                self.assertIn(part, paras_text)        # never a split paragraph
        self.assertEqual(covered, list(range(1, 13)))   # no paragraph lost or repeated


class TestStubsAndOverlongParagraphs(unittest.TestCase):
    """Found by learning-qa on the real index: 323 chunks under 20 characters (a page
    number followed by a large paragraph) and 240 chunks over 3,000 characters (sources
    with no blank lines), one of 46,703."""

    def test_idx_short_paragraph_before_a_large_one_is_not_a_stub_chunk(self):
        # A ~120-char running header, then text that fills a chunk on its own: without the
        # minimum-size rule the header is closed off as its own tiny chunk.
        header = "176 " + "Family Systems Journal " * 5
        big = ("A sentence of ordinary text. " * 52).strip()        # one ~1,500-char piece
        self.assertGreater(len(header) + len(big), B.CHUNK_CHARS)
        chunks, _ = chunks_of(header + "\n\n" + big)
        self.assertTrue(all(len(c["text"]) >= B.MIN_CHUNK_CHARS for c in chunks),
                        [len(c["text"]) for c in chunks])
        self.assertIn("176", chunks[0]["text"])               # the header text is kept

    def test_idx_stub_rule_does_not_merge_across_sections(self):
        chunks, _ = chunks_of("## Section 1 – A\n\nshort\n\n## Section 2 – B\n\n"
                              + "word " * 300)
        self.assertEqual(chunks[0]["section_title"], "A")      # section still wins
        self.assertNotIn("word word", chunks[0]["text"])

    def test_idx_overlong_paragraph_is_split_and_no_chunk_exceeds_the_limit(self):
        para = " ".join(f"Sentence number {i} says something about triangles." for i in range(400))
        self.assertGreater(len(para), 15000)
        chunks, stats = chunks_of(para)
        self.assertGreater(len(chunks), 5)
        self.assertLessEqual(max(len(c["text"]) for c in chunks), B.CHUNK_CHARS)
        self.assertEqual(stats["split"], 1)

    def test_idx_splitting_loses_no_text_and_cuts_only_at_boundaries(self):
        sentences = [f"Sentence {i} about differentiation of self." for i in range(300)]
        pieces = B.split_long(" ".join(sentences))
        self.assertEqual(" ".join(pieces), " ".join(sentences))     # nothing dropped or altered
        for piece in pieces:
            self.assertTrue(piece.endswith("."))                   # never mid-sentence

    def test_idx_text_with_no_sentence_breaks_is_cut_at_spaces_without_loss(self):
        text = " ".join(f"word{i}" for i in range(2000))             # no punctuation at all
        pieces = B.split_long(text)
        self.assertGreater(len(pieces), 1)
        self.assertLessEqual(max(map(len, pieces)), B.CHUNK_CHARS)
        self.assertEqual(" ".join(pieces).split(), text.split())

    def test_idx_one_unbroken_token_longer_than_the_limit_is_still_split(self):
        pieces = B.split_long("x" * 4000)
        self.assertEqual("".join(pieces), "x" * 4000)
        self.assertLessEqual(max(map(len, pieces)), B.CHUNK_CHARS)

    def test_idx_build_reports_how_many_paragraphs_were_split(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
            (Path(d) / "a.txt").write_text(" ".join(f"Sentence {i} here." for i in range(600)))
            (Path(d) / "b.txt").write_text(SECTIONED)
            (Path(d) / "c.txt").write_text(SECTIONED)
            log = []
            B.build(d, o, config_dir=d, log=log.append)
        self.assertTrue(any("overlong paragraphs in 1 documents were split" in m for m in log))


class TestReadText(unittest.TestCase):
    def read(self, raw):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x.txt"
            f.write_bytes(raw)
            with mock.patch("builtins.print"):
                return B.read_text(f)

    def test_idx_utf16_with_bom_is_decoded(self):
        self.assertEqual(self.read("Temperamental Categories".encode("utf-16")),
                         "Temperamental Categories")

    def test_idx_cp1252_without_bom_is_not_mistaken_for_utf16(self):
        # An even-length cp1252 byte string decodes "successfully" as BOM-less UTF-16.
        raw = "café society and family systems theory".encode("cp1252")
        self.assertEqual(len(raw) % 2, 0)
        self.assertEqual(self.read(raw), "café society and family systems theory")

    def test_idx_utf8_bom_is_stripped(self):
        self.assertEqual(self.read(b"\xef\xbb\xbfhello"), "hello")


class TestGuiRebuildContract(unittest.TestCase):
    """The desktop app's "Rebuild Index" button loads build_index.py by path and calls it.
    A rewrite of the builder once removed the API the button called."""

    def test_idx_gui_calls_an_api_that_exists_in_the_builder(self):
        gui = (Path(__file__).resolve().parent / "bowen_rag_gui.py").read_text()
        self.assertIn("bi.build(", gui)
        self.assertNotIn("DocumentIndexer", gui)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_index_via_path", str(Path(B.__file__)))
        bi = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bi)
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            for i in range(2):
                (Path(src) / f"d{i}.txt").write_text(SECTIONED)
            messages = []
            stats = bi.build(src, out, config_dir=src, log=lambda m: messages.append(m))
            self.assertEqual(stats["chunks"], 6)
            self.assertTrue(messages)       # the GUI log receives the build's messages


class TestBuild(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.root = Path(self._t.name)
        self.src, self.out, self.cfg = self.root / "src", self.root / "out", self.root / "cfg"
        for d in (self.src, self.out, self.cfg):
            d.mkdir()
        self.log = []
        # min_df=2 needs every distinguishing term in two documents
        for i in range(3):
            (self.src / f"doc{i}.txt").write_text(
                SECTIONED.replace("Cutoff", f"Cutoff{i}"), encoding="utf-8")

    def run_build(self, **kw):
        return B.build(self.src, self.out, config_dir=self.cfg, log=self.log.append, **kw)

    def test_idx_saved_files_agree_and_no_temp_files_remain(self):
        stats = self.run_build()
        meta = json.loads((self.out / "chunk_metadata.json").read_text())
        vec = json.loads((self.out / "vectorizer.json").read_text())
        matrix = sp_sparse.load_npz(str(self.out / "tfidf_matrix.npz"))
        self.assertEqual(len(meta), 9)
        self.assertEqual(matrix.shape, (len(meta), len(vec["feature_names"])))
        self.assertEqual(stats["chunks"], len(meta))
        self.assertEqual(vec["max_features"], B.TFIDF_PARAMS["max_features"])
        self.assertEqual(vec["min_df"], B.TFIDF_PARAMS["min_df"])
        self.assertEqual(list(self.out.glob("*.tmp")), [])

    def test_idx_metadata_is_resolved_into_every_chunk(self):
        (self.cfg / "headers_candidates.yml").write_text(yaml.safe_dump({"candidates": [
            {"doc_name": "doc0", "author": "Jones", "year": "2014", "title": "On Triangles",
             "container": "Family Systems Journal"}]}))
        (self.cfg / "chapter_map.yml").write_text(yaml.safe_dump({
            "books": {"b": {"doc_pattern": "doc1", "year": 1978, "title": "The Book",
                            "authors": [{"given": "Murray", "family": "Bowen"}]}},
            "chapters": [{"doc_pattern": "doc1", "chapter": 4, "title": "Ch Four"}]}))
        self.run_build()
        meta = json.loads((self.out / "chunk_metadata.json").read_text())
        d0 = next(c for c in meta if c["doc_name"] == "doc0")
        d1 = next(c for c in meta if c["doc_name"] == "doc1")
        d2 = next(c for c in meta if c["doc_name"] == "doc2")
        self.assertEqual((d0["author"], d0["date"], d0["container"]),
                         (["Jones"], "2014", "Family Systems Journal"))
        self.assertEqual((d1["author"], d1["date"], d1["chapter"], d1["container"]),
                         (["Murray Bowen"], "1978", 4, "The Book"))
        self.assertEqual((d2["author"], d2["date"]), ([], None))   # unknown, not invented

    def test_idx_malformed_config_yaml_is_an_error_not_silent_empty_metadata(self):
        (self.cfg / "headers_candidates.yml").write_text("candidates: [oops")
        with self.assertRaises(B.IndexBuildError) as cm:
            self.run_build()
        self.assertIn("headers_candidates.yml", str(cm.exception))

    def test_idx_missing_config_is_reported(self):
        self.run_build()
        self.assertTrue(any("chapter_map.yml not found" in m for m in self.log))
        self.assertTrue(any("no author/date metadata" in m for m in self.log))

    def test_idx_empty_document_is_skipped_loudly_not_silently(self):
        (self.src / "blank.txt").write_text("   \n\n  ")
        stats = self.run_build()
        self.assertEqual(stats["files_skipped"], ["blank.txt"])
        self.assertTrue(any("NOT indexed" in m for m in self.log))
        self.assertTrue(any("blank.txt: no extractable text" in m for m in self.log))

    def test_idx_pdf_without_pymupdf_is_an_error_never_indexed_as_text(self):
        (self.src / "scan.pdf").write_bytes(b"%PDF-1.4\n" + bytes(range(128, 255)) * 20)
        with mock.patch.object(B, "PDF_SUPPORT", False):
            with self.assertRaises(B.IndexBuildError) as cm:
                self.run_build()
        self.assertIn("PyMuPDF", str(cm.exception))

    def test_idx_txt_and_pdf_with_the_same_stem_use_the_txt_and_say_so(self):
        (self.src / "doc0.pdf").write_bytes(b"%PDF-1.4 not parsed")
        with mock.patch.object(B, "PDF_SUPPORT", False):    # the pdf must not even be opened
            stats = self.run_build()
        self.assertEqual(stats["files_skipped"], [])
        self.assertTrue(any("doc0: both .txt and .pdf present" in m for m in self.log))

    def test_idx_directory_with_no_documents_is_an_error(self):
        for f in self.src.iterdir():
            f.unlink()
        with self.assertRaises(B.IndexBuildError):
            self.run_build()

    def test_idx_sectioned_document_with_empty_sections_is_reported(self):
        (self.src / "partial.txt").write_text(
            "## Section 1 – A\n\n\n## Section 2 – B\n\nreal text about triangles\n")
        self.run_build()
        self.assertTrue(any("partial: 1 of 2 sections have no text" in m for m in self.log))


if __name__ == "__main__":
    unittest.main()
