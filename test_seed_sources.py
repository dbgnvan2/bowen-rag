"""
Tests for seed_sources.py (M1.B).

Run: python3 test_seed_sources.py

Key test is the round-trip: the YAML the seeder WRITES must parse cleanly back
through citations.load_sources — the producer/consumer contract (P19).
"""

import os
import tempfile
import unittest
from pathlib import Path

import citations as C
import seed_sources as S


AUTHOR_MAP = [
    {"pattern": "Bowen on Triangles", "author": "Murray Bowen"},
    {"pattern": "Bowen Family Systems Theory", "author": "Daniel Papero"},
]
AUTHORITY = [
    {"pattern": "Bowen on Triangles", "multiplier": 3.0, "note": "Bowen"},
    {"pattern": "Some Journal", "multiplier": 1.3},
]


class TestGuessers(unittest.TestCase):
    def test_m1b_guess_year_from_date(self):
        self.assertEqual(S.guess_year("Beautiful Boy Review - Kathleen Kerr - 2023-02-16"), 2023)
        self.assertEqual(S.guess_year("2011 04 Making a Difference - Michael Kerr"), 2011)
        self.assertEqual(S.guess_year("1951-TEMCY-paper"), 1951)

    def test_m1b_guess_year_none_when_absent(self):
        self.assertIsNone(S.guess_year("Bowen on Triangles"))
        self.assertIsNone(S.guess_year("099_Tempermental Categories"))  # 099 is not a year

    def test_m1b_guess_type(self):
        self.assertEqual(S.guess_type("BOWEN-KERR INTERVIEW SERIES #1"), "interview")
        self.assertEqual(S.guess_type("Bowen Basic Series Tape 1"), "speech")
        self.assertEqual(S.guess_type("FSJ 13.1 Brooks Triangles"), "article-journal")
        self.assertEqual(S.guess_type("099_Tempermental Categories"), "generic")


class TestBuildRecords(unittest.TestCase):
    def test_m1b_author_filled_from_map(self):
        recs = S.build_records(["Bowen on Triangles"], AUTHOR_MAP, AUTHORITY)
        self.assertEqual(recs[0]["authors"], [{"family": "Bowen", "given": "Murray"}])

    def test_m1b_unknown_author_is_empty_list(self):
        recs = S.build_records(["099_Tempermental Categories"], AUTHOR_MAP, AUTHORITY)
        self.assertEqual(recs[0]["authors"], [])           # not fabricated

    def test_m1b_primary_sources_sorted_first(self):
        # Triangles has boost 3.0, the other none (1.0) → Triangles must come first.
        recs = S.build_records(
            ["099_Tempermental Categories", "Bowen on Triangles"], AUTHOR_MAP, AUTHORITY)
        self.assertEqual(recs[0]["title"], "Bowen on Triangles")


class TestRoundTrip(unittest.TestCase):
    """P19 — the seeder's output must load back through citations.load_sources."""

    def test_m1b_render_then_load(self):
        recs = S.build_records(
            ["Bowen on Triangles", "099_Tempermental Categories",
             "Beautiful Boy Review - Kathleen Kerr - 2023-02-16"],
            AUTHOR_MAP, AUTHORITY)
        text = S.render_yaml(recs)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "sources.yml").write_text(text, encoding="utf-8")
            loaded = C.load_sources(Path(d))
        self.assertEqual(len(loaded), 3)                    # nothing dropped on parse
        m = C.match_source("Bowen on Triangles 1974 edition", loaded)
        self.assertIsNotNone(m)
        self.assertEqual(m["authors"], [{"family": "Bowen", "given": "Murray"}])
        # A seeded record with an apostrophe-free title still formats without error.
        self.assertIn("Bowen", C.format_reference(m, "APA"))

    def test_m1b_apostrophe_title_survives_round_trip(self):
        recs = S.build_records(["Defining a Self in One's Family"], AUTHOR_MAP, AUTHORITY)
        text = S.render_yaml(recs)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "sources.yml").write_text(text, encoding="utf-8")
            loaded = C.load_sources(Path(d))
        self.assertEqual(len(loaded), 1)
        self.assertIn("One's Family", loaded[0]["title"])   # '' un-escaped correctly


if __name__ == "__main__":
    unittest.main(verbosity=2)
