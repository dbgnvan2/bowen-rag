"""
Tests for extract_headers.py. headers_candidates.yml is REGENERATED from scratch by this
script, so corrections made to it by hand are lost on the next run; the corrections that
matter are therefore tested here, at their source (the year rule and author_map.yml).

Run: python3 -m unittest test_extract_headers
"""
import tempfile
import unittest
from pathlib import Path

import yaml

import extract_headers as E


class TestFsjYear(unittest.TestCase):
    def _write(self, text):
        d = tempfile.mkdtemp()
        f = Path(d) / "FSJ 1.1 A.txt"
        f.write_text(text, encoding="utf-8")
        return f

    def test_fsj_year_is_the_copyright_line_not_a_reference_year(self):
        f = self._write("Title: X Author: Y\nREFERENCES Bowen, Murray. 2013. Origins.\n"
                        "\u00a9 Georgetown Family Center, 2017")
        self.assertEqual(E.fsj_year(f), ("2017", "copyright"))

    def test_fsj_without_a_copyright_line_gets_no_year(self):
        f = self._write("REFERENCES Bowen, Murray. 2013. Origins.")
        self.assertEqual(E.fsj_year(f), (None, "none"))


class TestYearRule(unittest.TestCase):
    def test_life_dates_are_not_a_publication_year(self):
        self.assertEqual(E.year_from("", "FSJ 12.2 Noone Jaak Panksepp (1943-2017)"), (None, "none"))

    def test_a_year_span_in_the_header_is_not_a_publication_year(self):
        self.assertEqual(E.year_from("data collected 1976-1978", "Some Paper"), (None, "none"))

    def test_a_real_year_beside_a_range_is_still_found(self):
        self.assertEqual(E.year_from("", "Noone Panksepp (1943-2017) obituary 2018"),
                         ("2018", "filename"))

    def test_a_full_date_is_not_mistaken_for_a_range(self):
        self.assertEqual(E.year_from("", "Beautiful Boy Review - Kathleen Kerr - 2023-02-16"),
                         ("2023", "filename"))

    def test_plain_filename_year_wins_over_header_text(self):
        self.assertEqual(E.year_from("written in 1999", "Paper 2004"), ("2004", "filename"))


class TestRegenerationKeepsTheCorrections(unittest.TestCase):
    """Run the whole extractor on filenames that earlier produced wrong metadata."""

    NAMES = {
        "Wisdom of the Ages Houston TX 2009 1.txt": "conference recording, speakers unknown",
        "FSJ 5.1 Caskie Bowen Theory and Health Care Costs.txt": "title says Bowen, author is Caskie",
        "Dianna and Lillie Can a Feminist stil Like Murray Bowen  Lerner 1985 article.txt": "",
        "FSJ 3.1 (3) Holt 52 -62 Gender and Bowen Theory.txt": "",
        "FSJ 12.2 Noone Jaak Panksepp (1943-2017).txt": "life dates",
        "Bowen Theory Secrets_Chapter05_Differentiation_of_Self.txt": "Kerr's book, not Bowen's",
    }

    def run_extract(self, extra=()):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
            for n in list(self.NAMES) + list(extra):
                (Path(d) / n).write_text("body text only\n", encoding="utf-8")
            out = Path(o) / "headers.yml"
            E.main(Path(d), out)
            recs = yaml.safe_load(out.read_text())["candidates"]
        return {r["doc_name"]: r for r in recs}

    def test_regenerated_authors_are_not_bowen_for_other_authors_works(self):
        recs = self.run_extract()
        self.assertEqual(recs["FSJ 5.1 Caskie Bowen Theory and Health Care Costs"]["author"], "Caskie")
        self.assertEqual(recs["FSJ 3.1 (3) Holt 52 -62 Gender and Bowen Theory"]["author"], "Holt")
        lerner = [r for n, r in recs.items() if n.startswith("Dianna and Lillie")][0]
        self.assertEqual(lerner["author"], "Lerner")

    def test_regenerated_conference_recording_has_no_author(self):
        recs = self.run_extract()
        self.assertNotIn("author", recs["Wisdom of the Ages Houston TX 2009 1"])

    def test_regenerated_life_dates_do_not_become_a_year(self):
        recs = self.run_extract()
        self.assertNotIn("year", recs["FSJ 12.2 Noone Jaak Panksepp (1943-2017)"])

    def test_one_record_per_document_when_both_txt_and_pdf_exist(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
            (Path(d) / "Scan Doc Kerr 1988.txt").write_text("ocr text\n")
            (Path(d) / "Scan Doc Kerr 1988.pdf").write_bytes(b"%PDF-1.4")
            E.main(Path(d), Path(o) / "h.yml")
            recs = yaml.safe_load((Path(o) / "h.yml").read_text())["candidates"]
        self.assertEqual([r["doc_name"] for r in recs], ["Scan Doc Kerr 1988"])


if __name__ == "__main__":
    unittest.main()
