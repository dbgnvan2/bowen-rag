"""
Known-terms regression (spec E12): Bowen terms the corpus is known to contain must stay
findable, and every retrieval mode the Chat and Report pages offer must hand the model at
least one passage containing the term.

The term list and minimum counts are in tests/known_terms.yml. The corpus tests run
against the shipped index in rag-document-search/references/ and skip when it is absent.

Run: python3 -m unittest test_known_terms
"""
import json
import math
import unittest
from pathlib import Path

import yaml

import exact_match as em

ROOT = Path(__file__).resolve().parent
REFS = ROOT / "rag-document-search" / "references"
KNOWN_TERMS = ROOT / "tests" / "known_terms.yml"

CHAT_K = 12     # streamlit_app.py chat "Chunks" default
REPORT_K = 30   # streamlit_app.py rpt_k default


def load_known_terms() -> list[dict]:
    return yaml.safe_load(KNOWN_TERMS.read_text(encoding="utf-8"))["terms"]


def _corpus_has_chapter09() -> bool:
    try:
        meta = json.loads((REFS / "chunk_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return any("Emotional_Programming" in c.get("doc_name", "") for c in meta)


class TestKnownTermsFile(unittest.TestCase):
    """Runs everywhere (no index needed): the file is well formed."""

    def test_e12_known_terms_file_is_consistent(self):
        terms = load_known_terms()
        self.assertGreaterEqual(len(terms), 19)
        self.assertEqual(len({t["term"] for t in terms}), len(terms), "duplicate term")
        negatives = [t for t in terms if "max_docs" in t]
        self.assertTrue(negatives, "no negative (absent) term")
        for t in terms:
            if "max_docs" in t:
                continue
            self.assertEqual(t["min_docs"], math.floor(0.9 * t["measured_docs"]), t["term"])
            self.assertGreater(t["min_docs"], 0, t["term"])


@unittest.skipUnless(_corpus_has_chapter09(), "shipped index not present (corpus volume unmounted)")
class TestE12Corpus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import streamlit_app
        cls.idx = streamlit_app.IndexManager()
        cls.idx.load(REFS)
        cls.terms = load_known_terms()

    def _modes(self, k: int) -> dict:
        """The retrieval modes the Chat and Report pages offer, called as the pages call them."""
        idx = self.idx
        modes = {
            "top-docs": lambda q: idx.top_docs_search(q, top_chunks=300, top_docs=k, use_boost=True),
            "semantic": lambda q: idx.semantic_search(q, k, use_boost=True),
            "keyword":  lambda q: idx.keyword_search(q, k, use_boost=True),
            "both":     lambda q: idx.combined_search(q, k, use_boost=True),
        }
        if idx.embed_matrix is not None:
            modes["embedding"] = lambda q: idx.embedding_search(q, k, use_boost=True)
            modes["hybrid"] = lambda q: idx.hybrid_search(q, k, use_boost=True)
        return modes

    def test_e12_known_terms_present(self):
        """Every known term is in at least min_docs documents; the negative term in none."""
        for t in self.terms:
            got = self.idx.exact_presence(t["term"], mode="substring")["doc_count"]
            if "max_docs" in t:
                self.assertLessEqual(got, t["max_docs"], t["term"])
            else:
                self.assertGreaterEqual(got, t["min_docs"], t["term"])

    def test_e12_known_terms_reach_the_model(self):
        """For each term, asked bare and as "What is <term>?", every mode at the Chat (k=12)
        and Report (k=30) defaults, after the exact-match guard, gives the model at least
        one passage containing the phrase the guard checked."""
        failures = []
        for t in self.terms:
            if "max_docs" in t:
                continue
            for q in (t["term"], f"What is {t['term']}?"):
                for k in (CHAT_K, REPORT_K):
                    for mode, run in self._modes(k).items():
                        out, phrases, reason, presence = em.exact_guard(self.idx, q, run(q), k=k)
                        # The guarantee is only meaningful if the guard found the phrase.
                        self.assertTrue(phrases and all(p["found"] for p in presence.values()),
                                        f"{q!r}: guard checked {phrases} ({reason})")
                        hit = sum(1 for c in out
                                  if any(em.count_matches(c["text"], p) for p in phrases))
                        if hit == 0 or len(out) > k:
                            failures.append(f"{mode} k={k} {q!r}: {hit}/{len(out)}")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
