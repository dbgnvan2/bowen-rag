"""
Authority tiers and the rank-based authority boost (spec E11, option b).

Run: python3 -m unittest test_authority
"""
import ast
import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import authority as A
from test_exact_match import FixtureIndex, managers

ROOT = Path(__file__).resolve().parent
REFS = ROOT / "rag-document-search" / "references"
TIERS = A.load_tiers(ROOT / "authority_tiers.yml")


def boost(name):
    return A.boost_for(name, TIERS)


class TestTiers(unittest.TestCase):
    def test_e11_default_tiers_equal_yml(self):
        """The desktop bundle has no authority_tiers.yml and uses DEFAULT_TIERS."""
        self.assertEqual(TIERS, A.parse_tiers(A.DEFAULT_TIERS))

    def test_e11_bowen_above_kerr_equal_papero_above_everyone_else(self):
        """Dave, 2026-11-01: Bowen first; Kerr second and equal to Papero; all three above
        any other source. Real document names from the index."""
        bowen = ["Family Therapy in_Clinical_Practice_Chapter09_The_Use_of_Family_Theory",
                 "Bowen Basic Series Tape 5 SYMPTOM DEVELOPMENT",
                 "FSJ 9.2 Bowen Subjectivity, Homo Sapiens, and Science",
                 "FSJ 2.1 (1) Bowen  Psycho formulation Schiz 17-47",
                 "A Family Concept of Schizophrenia Bowen 1960"]
        kerr = ["Bowen Theory Secrets_Chapter09_Emotional_Programming",
                "Family Evaluation_Chapter05_Chronic_Anxiety",
                "FSJ 4.2 Kerr Bowen Theory and Evolutionary Theory",
                "FSJ 4.2 (3) Kerr, M. Bowen Theory and Evo Theory 119-175",
                "FSJ 4.1  Kerr  From the Editor V4 N1",
                "Emotional Regression and Cancer - Michael Kerr - 2021-01-15"]
        papero = ["Bowen Family Systems Theory_Chapter01_Bowen_Theory_in_Perspective",
                  "Family Systems Therapy Papero",
                  "FSJ 13.2. papero129-144",
                  "FSJ 3.1 (4) BR Papero 64-68"]
        others = ["FSJ 14.2 Galloway How Emotions Are Made",
                  "Bowen Theory and Therapy Friedman (Handbook of Family Therapy II) No Date",
                  "Family Center Reports Vol 6 No 3",
                  "Beautiful Boy Review - Kathleen Kerr - 2023-02-16",
                  "14.2 galloway How Emotions Are Made"]
        for n in bowen:
            self.assertEqual(boost(n), 3.0, n)
        for n in kerr + papero:
            self.assertEqual(boost(n), 2.0, n)
        top_other = max(boost(n) for n in others)
        self.assertLess(top_other, 2.0)
        self.assertEqual(boost("Beautiful Boy Review - Kathleen Kerr - 2023-02-16"), 1.0)

    def test_e11_parse_rejects_malformed_entries(self):
        for bad in ({"multiplier": 2.0}, {"pattern": "a", "regex": "a", "multiplier": 2.0},
                    {"pattern": "a", "multiplier": 0.5}, {"regex": "(", "multiplier": 2.0}):
            with self.assertRaises((ValueError, re.error), msg=bad):
                A.parse_tiers([bad])

    def test_e11_unreadable_yml_is_reported_not_silent(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "authority_tiers.yml"
            p.write_text("tiers:\n  - multiplier: 2.0\n")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                got = A.load_tiers(p)
        self.assertIn("could not load", err.getvalue())
        self.assertEqual(got, A.parse_tiers(A.DEFAULT_TIERS))


class TestRankOrder(unittest.TestCase):
    def test_e11_no_boost_is_raw_order(self):
        raw = [0.2, 0.9, 0.0, 0.5, -1.0]
        self.assertEqual(A.rank_order(raw, lambda i: 1.0), [1, 3, 0])

    def test_e11_boost_reorders_but_keeps_the_most_relevant(self):
        """Adversarial (the 'emotional programming' shape): 12 unboosted items are the most
        relevant; 36 weaker x3 items follow. Multiplying scores put 0 of the 12 in the top
        12; dividing ranks keeps the best ones."""
        raw = [100 - n for n in range(12)] + [40 - n * 0.5 for n in range(36)]
        weight = lambda i: 1.0 if i < 12 else 3.0
        multiplied = sorted(range(len(raw)), key=lambda i: -raw[i] * weight(i))[:12]
        self.assertEqual(sum(1 for i in multiplied if i < 12), 0)
        top = A.rank_order(raw, weight, 12)
        self.assertEqual(top[0], 0)                                      # rank 1 stays first
        self.assertGreaterEqual(sum(1 for i in top if i < 12), 4)
        self.assertTrue(any(i >= 12 for i in top))                       # the boost acts

    def test_e11_rank_divided_by_weight(self):
        """A x3 item at raw rank 30 ties the unboosted rank 10 and goes after it."""
        raw = [100 - n for n in range(40)]
        weight = lambda i: 3.0 if i == 29 else 1.0
        order = A.rank_order(raw, weight)
        self.assertEqual(order.index(29), 10)          # after ranks 1..10, before rank 11


class TestAppsUseRankBoost(FixtureIndex):
    SEARCHES = ("semantic_search", "bm25_search", "embedding_search", "hybrid_search",
                "top_docs_search", "keyword_search")

    def test_e11_every_boost_site_uses_rank_order(self):
        """All boost sites in both apps go through authority.rank_order (no score x boost)."""
        for path in ("streamlit_app.py", "bowen_rag_gui.py"):
            tree = ast.parse((ROOT / path).read_text())
            cls = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.ClassDef) and n.name == "IndexManager")
            methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
            for name in self.SEARCHES:
                src = ast.unparse(methods[name])
                self.assertIn("authority.rank_order(", src, f"{path}: {name}")
                self.assertNotRegex(src, r"\* *(boost_fn|authority_boost)\(", f"{path}: {name}")

    def test_e11_boosted_document_cannot_displace_the_best_matches(self):
        """self_doc made x3: its weak chunks may rise, but the raw top results stay."""
        for name, idx in self.loaded():
            module = __import__(type(idx).__module__)
            off = idx.semantic_search("emotional programming self", 14, use_boost=False)
            self_pos = min(n for n, r in enumerate(off) if r["doc_name"] == "self_doc")
            with mock.patch.object(module, "AUTHORITY_TIERS", tuple(A.parse_tiers(
                    [{"pattern": "self_doc", "multiplier": 3.0}]))):
                on = idx.semantic_search("emotional programming self", 14, use_boost=True)
                kw = idx.keyword_search("emotional", 10, use_boost=True)
            self.assertEqual([r["id"] for r in on[:3]], [r["id"] for r in off[:3]], name)
            new_pos = min(n for n, r in enumerate(on) if r["doc_name"] == "self_doc")
            # raw rank 10 / 3 = 3.33: after raw ranks 1-3, ahead of 4. Multiplying the
            # score (0.119 x 3 > 0.263) would have put it first.
            self.assertEqual((self_pos, new_pos), (9, 3), name)
            self.assertIn("×3 ★", on[new_pos]["score_label"], name)
            self.assertEqual(on[new_pos]["score"], off[self_pos]["score"], name)   # raw score
            self.assertIn("self_doc", [r["doc_name"] for r in kw], name)


def _corpus_has_bts():
    p = REFS / "chunk_metadata.json"
    if not p.exists():
        return False
    return any("Emotional_Programming" in c["doc_name"] for c in json.loads(p.read_text()))


@unittest.skipUnless(_corpus_has_bts(), "shipped index with Bowen Theory Secrets not present")
class TestE11Corpus(unittest.TestCase):
    def test_e11_boost_never_removes_all_literal_hits(self):
        """BM25 at the Chat default (k=12, boost on) returned 0 chunks containing
        'emotional programming' before E11; it must return at least one."""
        import streamlit_app
        idx = streamlit_app.IndexManager(); idx.load()
        rx = re.compile(r"emotional\s+programming", re.I)
        for mode in ("bm25_search", "semantic_search"):
            got = getattr(idx, mode)("What is emotional programming?", 12, use_boost=True)
            self.assertGreaterEqual(sum(1 for c in got if rx.search(c["text"])), 1, mode)


if __name__ == "__main__":
    unittest.main()
