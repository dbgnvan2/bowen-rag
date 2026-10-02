"""
Data-consistency tests for the editorial config: author_map.yml and sources.yml.

These guard the real data files (not fixtures) against the wrong-attribution defect:
the bare "Bowen" catch-all in author_map.yml credited Murray Bowen with FSJ articles by
other authors whose titles mention Bowen, and with conference-recording transcripts.
Spec: docs/spec_citation_styles.md (M1.B, no fabrication)

Run: python3 -m unittest test_sources_data
"""
import re
import unittest
from pathlib import Path

import citations as C
import seed_sources as S

BASE = Path(__file__).resolve().parent
AUTHOR_MAP = S._load_yaml(BASE / "author_map.yml", "authors")
SOURCES = C.load_sources(BASE)

# "FSJ 10.2 Jones Epigenetics ..." / "FSJ 3.1 (3) Holt 52 -62 ..." -> surname token
_FSJ_AUTHOR = re.compile(r"^FSJ \d+\.\d+ (?:\(\d+\) )?([A-Za-z]+)")


def _map_author(doc_name):
    return S._first_match(doc_name, AUTHOR_MAP, "author")


class TestAuthorMap(unittest.TestCase):
    def test_m1b_fsj_article_by_other_author_is_not_credited_to_bowen(self):
        # Adversarial (P7): titles contain "Bowen", authors are someone else.
        cases = {
            "FSJ 10.2 Jones Epigenetics, Social Genomics, and Bowen": "Jones",
            "FSJ 13.2 Harrison Difference Bowen Theory Makes in": "Harrison",
            "FSJ 18.2 McKnight Bowlbys Attachment Theory and Bowens": "McKnight",
            "FSJ 18.2 Millikin Bowen Coachs Attempt at a": "Millikin",
            "FSJ 5.1 Caskie Bowen Theory and Health Care Costs": "Caskie",
            "Dianna and Lillie Can a Feminist stil Like Murray Bowen  Lerner 1985 article": "Lerner",
        }
        for doc, expect in cases.items():
            self.assertEqual(_map_author(doc), expect, doc)

    def test_m1b_genuine_bowen_fsj_articles_still_map_to_bowen(self):
        for doc in ("FSJ 12.2 Bowen Letter from Murray Bowen About",
                    "FSJ 9.2 Bowen Subjectivity, Homo Sapiens, and Science"):
            self.assertEqual(_map_author(doc), "Murray Bowen", doc)

    def test_m1b_conference_recordings_have_no_author(self):
        for doc in ("Wisdom of the Ages Houston TX 2009 1",
                    "Wisdom of the Ages Pittsburgh 1998 3b"):
            self.assertIsNone(_map_author(doc), doc)


class TestSourcesData(unittest.TestCase):
    def test_m1b_sources_yml_loaded(self):
        self.assertGreater(len(SOURCES), 300)   # a failed load would silently return []

    def test_m1b_no_fsj_record_credits_bowen_for_another_authors_article(self):
        bad = []
        for r in SOURCES:
            m = _FSJ_AUTHOR.match(r["pattern"])
            fams = [a.get("family") for a in (r.get("authors") or [])]
            if m and m.group(1) != "Bowen" and "Bowen" in fams:
                bad.append(r["pattern"])
        self.assertEqual(bad, [])

    def test_m1b_sources_records_agree_with_author_map_for_bowen(self):
        # A record naming Bowen as author must be one the author map also credits to
        # Bowen (alone or with Kerr); otherwise the two config files contradict.
        bad = []
        for r in SOURCES:
            fams = [a.get("family") for a in (r.get("authors") or [])]
            if "Bowen" in fams and "Bowen" not in (_map_author(r["pattern"]) or ""):
                bad.append(r["pattern"])
        self.assertEqual(bad, [])

    def test_m1b_conference_recording_records_have_no_author(self):
        recs = [r for r in SOURCES if r["pattern"].startswith("Wisdom of the Ages")]
        self.assertEqual(len(recs), 13)
        for r in recs:
            self.assertFalse(r.get("authors"), r["pattern"])

    def test_m1b_life_dates_are_not_a_publication_year(self):
        rec = next(r for r in SOURCES if "Panksepp" in r["pattern"])
        self.assertEqual(str(rec.get("year")), "n.d.")


if __name__ == "__main__":
    unittest.main()
