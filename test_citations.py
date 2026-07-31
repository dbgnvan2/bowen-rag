"""
Golden-string + behaviour tests for citations.py.

Spec:  docs/spec_citation_styles.md
Run:   python3 test_citations.py         (or: python3 -m pytest test_citations.py)

Expected strings are derived from each style's published rules, NOT snapshotted
from code output — a failure means the formatter is wrong, not the test.
"""

import re
import tempfile
import unittest
from pathlib import Path

import citations as C


# ── Canonical fixture records ────────────────────────────────────────────────
BOOK_BOWEN = {
    "pattern": "Family Therapy in_Clinical_Practice",
    "type": "book",
    "authors": [{"family": "Bowen", "given": "Murray"}],
    "year": 1978,
    "title": "Family Therapy in Clinical Practice",
    "publisher": "Jason Aronson",
    "publisher_place": "New York",
    "verified": True,
}

ARTICLE_KERR = {
    "pattern": "Chronic Anxiety and Defining",
    "type": "article-journal",
    "authors": [{"family": "Kerr", "given": "Michael"}],
    "year": 1988,
    "title": "Chronic anxiety and defining a self",
    "container_title": "The Atlantic",
    "volume": "262",
    "issue": "3",
    "page": "35-58",
    "verified": True,
}

BOOK_TWO_AUTHORS = {
    "pattern": "Family Evaluation",
    "type": "book",
    "authors": [{"family": "Kerr", "given": "Michael"},
                {"family": "Bowen", "given": "Murray"}],
    "year": 1988,
    "title": "Family Evaluation",
    "publisher": "W. W. Norton",
    "publisher_place": "New York",
    "verified": True,
}


class TestReferenceGolden(unittest.TestCase):
    """M2.A — exact reference-list entries per style."""

    def test_m2a_reference_apa(self):
        self.assertEqual(
            C.format_reference(BOOK_BOWEN, "APA"),
            "Bowen, M. (1978). *Family Therapy in Clinical Practice*. Jason Aronson.")
        self.assertEqual(
            C.format_reference(ARTICLE_KERR, "APA"),
            "Kerr, M. (1988). Chronic anxiety and defining a self. "
            "*The Atlantic*, *262*(3), 35-58.")

    def test_m2a_reference_mla(self):
        self.assertEqual(
            C.format_reference(BOOK_BOWEN, "MLA"),
            "Bowen, Murray. *Family Therapy in Clinical Practice*. Jason Aronson, 1978.")
        self.assertEqual(
            C.format_reference(ARTICLE_KERR, "MLA"),
            'Kerr, Michael. "Chronic anxiety and defining a self." '
            "*The Atlantic*, vol. 262, no. 3, 1988, pp. 35-58.")

    def test_m2a_reference_chicago(self):
        self.assertEqual(
            C.format_reference(BOOK_BOWEN, "Chicago"),
            "Bowen, Murray. 1978. *Family Therapy in Clinical Practice*. "
            "New York: Jason Aronson.")
        self.assertEqual(
            C.format_reference(ARTICLE_KERR, "Chicago"),
            'Kerr, Michael. 1988. "Chronic anxiety and defining a self." '
            "*The Atlantic* 262 (3): 35-58.")

    def test_m2a_reference_harvard(self):
        self.assertEqual(
            C.format_reference(BOOK_BOWEN, "Harvard"),
            "Bowen, M. (1978) *Family Therapy in Clinical Practice*. "
            "New York: Jason Aronson.")
        self.assertEqual(
            C.format_reference(ARTICLE_KERR, "Harvard"),
            "Kerr, M. (1988) 'Chronic anxiety and defining a self', "
            "*The Atlantic*, 262(3), pp. 35-58.")

    def test_m2a_reference_vancouver(self):
        self.assertEqual(
            C.format_reference(BOOK_BOWEN, "Vancouver"),
            "Bowen M. Family Therapy in Clinical Practice. "
            "New York: Jason Aronson; 1978.")
        self.assertEqual(
            C.format_reference(ARTICLE_KERR, "Vancouver"),
            "Kerr M. Chronic anxiety and defining a self. "
            "The Atlantic. 1988;262(3):35-58.")

    def test_m2a_two_authors_join(self):
        self.assertEqual(
            C.format_reference(BOOK_TWO_AUTHORS, "APA"),
            "Kerr, M., & Bowen, M. (1988). *Family Evaluation*. W. W. Norton.")
        self.assertEqual(
            C.format_reference(BOOK_TWO_AUTHORS, "Harvard"),
            "Kerr, M. and Bowen, M. (1988) *Family Evaluation*. New York: W. W. Norton.")
        self.assertEqual(
            C.format_reference(BOOK_TWO_AUTHORS, "Vancouver"),
            "Kerr M, Bowen M. Family Evaluation. New York: W. W. Norton; 1988.")


class TestIntextGolden(unittest.TestCase):
    """M2.B — in-text citation per style, with and without a quote page."""

    def test_m2b_intext_with_page(self):
        cases = {
            "APA": "(Bowen, 1978, p. 45)",
            "Harvard": "(Bowen, 1978, p. 45)",
            "MLA": "(Bowen 45)",
            "Chicago": "(Bowen 1978, 45)",
            "Vancouver": "[1]",
        }
        for style, expected in cases.items():
            self.assertEqual(
                C.format_intext(BOOK_BOWEN, style, page="45", number=1), expected,
                f"{style} in-text (quote) mismatch")

    def test_m2b_intext_no_page(self):
        cases = {
            "APA": "(Bowen, 1978)",
            "Harvard": "(Bowen, 1978)",
            "MLA": "(Bowen)",
            "Chicago": "(Bowen 1978)",
            "Vancouver": "[1]",
        }
        for style, expected in cases.items():
            self.assertEqual(
                C.format_intext(BOOK_BOWEN, style, number=1), expected,
                f"{style} in-text (no page) mismatch")

    def test_m2b_intext_two_authors(self):
        self.assertEqual(C.format_intext(BOOK_TWO_AUTHORS, "APA"), "(Kerr & Bowen, 1988)")
        self.assertEqual(C.format_intext(BOOK_TWO_AUTHORS, "MLA", page="12"),
                         "(Kerr and Bowen 12)")


class TestOrdering(unittest.TestCase):
    """M2.C — author styles alphabetical; numbered styles keep number order."""

    def setUp(self):
        self.adams = {"authors": [{"family": "Adams", "given": "Ann"}],
                      "year": 2000, "title": "Zzz", "type": "book"}
        self.num_records = [(1, BOOK_BOWEN), (2, self.adams)]

    def test_m2c_author_style_alphabetical(self):
        ordered = C.order_references(self.num_records, "APA")
        self.assertEqual([n for n, _ in ordered], [2, 1])  # Adams before Bowen

    def test_m2c_numbered_style_by_number(self):
        ordered = C.order_references(self.num_records, "Vancouver")
        self.assertEqual([n for n, _ in ordered], [1, 2])


class TestMissingData(unittest.TestCase):
    """M2.D — missing data uses n.d./fallback; nothing is fabricated (adversarial)."""

    def test_m2d_no_year_is_ndot(self):
        rec = dict(BOOK_BOWEN); rec.pop("year")
        out = C.format_reference(rec, "APA")
        self.assertIn("(n.d.)", out)
        self.assertNotRegex(out, r"\(\d{4}\)")  # no invented year

    def test_m2d_synth_record_unknown_author_not_fabricated(self):
        rec = C.synth_record("099_Tempermental Categories", author_lookup=lambda d: "Unknown")
        self.assertTrue(rec["_fallback"])
        self.assertEqual(rec["authors"], [])
        self.assertEqual(rec["year"], "n.d.")
        self.assertEqual(rec["title"], "Tempermental Categories")  # leading '099_' stripped
        out = C.format_reference(rec, "APA")
        self.assertEqual(out, "(n.d.). *Tempermental Categories*.")

    def test_m2d_synth_record_uses_author_map(self):
        rec = C.synth_record("Bowen on Triangles", author_lookup=lambda d: "Murray Bowen")
        self.assertEqual(rec["authors"], [{"family": "Bowen", "given": "Murray"}])


class TestMarkerRewrite(unittest.TestCase):
    """M3.C — rewrite only [[...]] markers; never touch single brackets in quotes."""

    def setUp(self):
        self.num_to_record = {1: BOOK_BOWEN, 2: ARTICLE_KERR}

    def test_m3c_apa_rewrites_double_bracket(self):
        body = "Differentiation is central [[1]]. A quote follows [[2, p. 45]]."
        out = C.apply_intext_citations(body, self.num_to_record, "APA")
        self.assertIn("(Bowen, 1978).", out)
        self.assertIn("(Kerr, 1988, p. 45)", out)
        self.assertNotIn("[[", out)

    def test_m3c_single_brackets_in_quotes_survive(self):
        # A source's OWN footnote number inside a verbatim quote must NOT be rewritten,
        # even though 1 and 2 are valid reference numbers (P7 / source fidelity).
        body = 'Bowen wrote: "the family [1] is a system [2]" — see [[1]].'
        out = C.apply_intext_citations(body, self.num_to_record, "APA")
        self.assertIn('"the family [1] is a system [2]"', out)   # quote intact
        self.assertIn("see (Bowen, 1978)", out)                  # real citation styled

    def test_m3c_preserves_sic_ellipsis_year(self):
        body = "As noted [sic] and [...] and in [1978] we see [[1]]."
        out = C.apply_intext_citations(body, self.num_to_record, "APA")
        self.assertIn("[sic]", out)
        self.assertIn("[...]", out)
        self.assertIn("[1978]", out)
        self.assertIn("(Bowen, 1978)", out)

    def test_m3c_grouped_citation_apa(self):
        out = C.apply_intext_citations("Both agree [[1, 2]].", self.num_to_record, "APA")
        self.assertEqual(out, "Both agree (Bowen, 1978; Kerr, 1988).")

    def test_m3c_grouped_citation_vancouver(self):
        out = C.apply_intext_citations(
            "Both [[1, 2]]. One [[1, p. 5]].", self.num_to_record, "Vancouver")
        self.assertIn("[1, 2]", out)
        self.assertIn("One [1].", out)      # page stripped in numbered style
        self.assertNotIn("p. 5", out)

    def test_m3c_grouped_drops_unknown_keeps_known(self):
        # [[1, 9]] — 9 has no record; 1 does. Keep 1, don't lose it.
        out = C.apply_intext_citations("Mixed [[1, 9]].", {1: BOOK_BOWEN}, "APA")
        self.assertEqual(out, "Mixed (Bowen, 1978).")

    def test_m3c_unknown_number_left_untouched(self):
        out = C.apply_intext_citations("See [[9]].", {1: BOOK_BOWEN}, "APA")
        self.assertEqual(out, "See [[9]].")  # no record for 9 → not a citation


class TestRoundTrip(unittest.TestCase):
    """M3.D — producer/consumer: body markers + reference list stay consistent."""

    def test_m3d_report_roundtrip_apa(self):
        num_to_record = {1: BOOK_BOWEN, 2: ARTICLE_KERR}
        body = "One [[1]]. Two [[2]]. Quote [[1, p. 5]]. Both [[1, 2]]."
        rewritten = C.apply_intext_citations(body, num_to_record, "APA")
        refs = C.build_reference_list_md(num_to_record, "APA")
        self.assertEqual(
            rewritten,
            "One (Bowen, 1978). Two (Kerr, 1988). Quote (Bowen, 1978, p. 5). "
            "Both (Bowen, 1978; Kerr, 1988).")
        self.assertTrue(refs.startswith("Bowen, M. (1978)."))   # alphabetical
        self.assertIn("Kerr, M. (1988).", refs)

    def test_m3d_cited_numbers_includes_grouped(self):
        self.assertEqual(C.cited_numbers("Only grouped [[1, 2]].", {1, 2}), {1, 2})

    def test_m3d_single_bracket_body_yields_no_cited(self):
        # A report that used single brackets [1] (the WRONG format) registers as zero
        # cited — the signal both apps use to warn the user instead of silently listing
        # all sources unstyled (P19 loud-zero-from-non-empty).
        self.assertEqual(C.cited_numbers("Uses single [1] and [2].", {1, 2}), set())

    def test_m3d_report_roundtrip_vancouver_numbered(self):
        num_to_record = {1: BOOK_BOWEN, 2: ARTICLE_KERR}
        refs = C.build_reference_list_md(num_to_record, "Vancouver")
        self.assertTrue(refs.startswith("1. Bowen M."))
        self.assertIn("2. Kerr M.", refs)


class TestLoaderAndHelpers(unittest.TestCase):
    """M1.A/M1.C — load_sources + match_source + author parsing."""

    def test_m1a_load_sources_missing_and_valid(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(C.load_sources(Path(d)), [])  # missing → []
            (Path(d) / "sources.yml").write_text(
                "sources:\n"
                "  - pattern: 'Bowen on Triangles'\n"
                "    type: book\n"
                "    year: 1974\n"
                "    title: Bowen on Triangles\n", encoding="utf-8")
            recs = C.load_sources(Path(d))
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["pattern"], "Bowen on Triangles")

    def test_m1c_match_source_first_wins(self):
        sources = [{"pattern": "Bowen on Triangles", "title": "specific"},
                   {"pattern": "Bowen", "title": "general"}]
        self.assertEqual(
            C.match_source("Bowen on Triangles 1974", sources)["title"], "specific")
        self.assertEqual(
            C.match_source("Bowen Basic Series", sources)["title"], "general")
        self.assertIsNone(C.match_source("Guerin A Family Affair", sources))

    def test_parse_author_string(self):
        self.assertEqual(C.parse_author_string("Murray Bowen"),
                         [{"family": "Bowen", "given": "Murray"}])
        self.assertEqual(C.parse_author_string("Bowen & Kerr"),
                         [{"family": "Bowen", "given": ""},
                          {"family": "Kerr", "given": ""}])

    def test_initials_multi(self):
        self.assertEqual(C._initials("Michael E"), "M. E.")
        self.assertEqual(C._initials("Murray"), "M.")
        self.assertEqual(C._initials("Michael E", period=False, spaced=False), "ME")

    def test_normalize_style_defaults_to_apa(self):
        self.assertEqual(C.normalize_style("apa"), "APA")
        self.assertEqual(C.normalize_style("nonsense"), "APA")
        self.assertEqual(C.normalize_style(""), "APA")


if __name__ == "__main__":
    unittest.main(verbosity=2)
