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

import sys

import citations as C
import seed_sources as S

sys.path.insert(0, str(Path(__file__).resolve().parent / "rag-document-search" / "scripts"))
import build_index as B  # noqa: E402

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


class TestChunkMetadataAgreesWithSources(unittest.TestCase):
    """The chunk side (chapter_map.yml / headers_candidates.yml, copied into every chunk)
    and sources.yml are two unverified sources for the same documents. Where both name an
    author they must agree; the extraction used to credit Murray Bowen with other authors'
    articles that merely mention Bowen in the title."""

    @classmethod
    def setUpClass(cls):
        cls.chapters, cls.books, cls.headers = B.load_metadata(BASE)

    def chunk_authors(self, doc):
        meta = B.resolve_metadata(doc, self.chapters, self.books, self.headers)
        fams = []
        for name in meta["author"]:
            fams += [a["family"].lower() for a in C.parse_author_string(name) if a.get("family")]
        return sorted(set(fams)), meta

    def test_data_headers_loaded(self):
        self.assertGreater(len(self.headers), 300)

    def test_data_no_author_disagreement_between_chunk_metadata_and_sources_yml(self):
        bad = []
        for r in SOURCES:
            doc = r["pattern"]
            chunk_f, _ = self.chunk_authors(doc)
            src_f = sorted({a.get("family", "").lower() for a in (r.get("authors") or [])
                            if a.get("family")})
            if chunk_f and src_f and chunk_f != src_f:
                bad.append((doc, chunk_f, src_f))
        self.assertEqual(bad, [])

    def test_data_conference_recording_chunks_carry_no_author(self):
        for r in SOURCES:
            if r["pattern"].startswith("Wisdom of the Ages"):
                self.assertEqual(self.chunk_authors(r["pattern"])[0], [], r["pattern"])

    def test_data_life_dates_are_not_a_chunk_year(self):
        _, meta = self.chunk_authors("FSJ 12.2 Noone Jaak Panksepp (1943-2017)")
        self.assertIsNone(meta["date"])

    def test_data_kerr_book_chapters_are_credited_to_kerr_in_both_maps(self):
        doc = "Bowen Theory Secrets_Chapter05_Differentiation_of_Self"
        self.assertEqual(S._first_match(doc, AUTHOR_MAP, "author"), "Michael Kerr")
        self.assertEqual(self.chunk_authors(doc)[0], ["kerr"])

    def test_data_every_document_with_a_header_record_resolves_to_it(self):
        # the restored document must have its own header record, not "unknown"
        doc = "Emotional Regression and Cancer - Michael Kerr - 2021-01-15"
        self.assertEqual(B.resolve_metadata(doc, self.chapters, self.books,
                                            self.headers)["kind"], "article")


if __name__ == "__main__":
    unittest.main()
