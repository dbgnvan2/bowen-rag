"""
Tests for rag-document-search/scripts/semantic_search.py (the CLI search).

It used to load a stale dense tfidf_matrix.npy and refit a 500-feature vectorizer against
an 8000-feature matrix, so every query crashed. These tests build a real index with
build_index.py and search it, so the two scripts are tested against each other.

Run: python3 -m unittest test_semantic_search
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "rag-document-search" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import build_index as B           # noqa: E402
import semantic_search as SS      # noqa: E402

# Two documents per topic: build_index.py keeps only terms found in >= 2 chunks
# (min_df=2), so each distinguishing term must appear in two documents.
DOCS = {
    "tri_a": "## Section 1 – Triangles\n\nA triangle is a three person emotional system. "
             "Triangles form when anxiety rises in a two person system.\n",
    "tri_b": "## Section 1 – Triangles again\n\nTriangles stabilise a two person system "
             "when anxiety rises; the triangle shifts the anxiety.\n",
    "cut_a": "## Section 1 – Cutoff\n\nEmotional cutoff is how people manage unresolved "
             "attachment to their family of origin by distancing.\n",
    "cut_b": "## Section 1 – Cutoff again\n\nCutoff and distancing from the family of origin "
             "manage unresolved attachment.\n",
    "frq_a": "## Section 1 – Method\n\nThe method counts word frequency across the corpus. "
             "Word frequency is weighted by inverse document frequency.\n",
    "frq_b": "## Section 1 – Method again\n\nWord frequency and inverse document frequency "
             "weight each term across the corpus.\n",
}


def build(tmp_src, tmp_out):
    for name, text in DOCS.items():
        (Path(tmp_src) / f"{name}.txt").write_text(text)
    ix = B.DocumentIndexer(tmp_src)
    with contextlib.redirect_stdout(io.StringIO()):
        ix.build_index()
        ix.save_index(tmp_out)


class TestCliSearch(unittest.TestCase):
    def setUp(self):
        self._s = tempfile.TemporaryDirectory(); self._o = tempfile.TemporaryDirectory()
        self.addCleanup(self._s.cleanup); self.addCleanup(self._o.cleanup)
        build(self._s.name, self._o.name)

    def test_cli_search_finds_the_matching_document_in_a_freshly_built_index(self):
        res = SS.SemanticSearcher(self._o.name).search("emotional triangle anxiety", top_k=3)
        self.assertTrue(res[0]["doc_name"].startswith("tri_"), res[0]["doc_name"])

    def test_cli_search_ranks_the_wrong_document_lower(self):
        # Adversarial: a query about cutoff must not return a triangles doc first.
        res = SS.SemanticSearcher(self._o.name).search("cutoff distancing family of origin")
        self.assertTrue(res[0]["doc_name"].startswith("cut_"), res[0]["doc_name"])

    def test_cli_search_refuses_an_index_whose_files_disagree(self):
        # Dirty-state (P8): the metadata is from a different build than the matrix.
        meta_path = Path(self._o.name, "chunk_metadata.json")
        meta = json.loads(meta_path.read_text())
        meta_path.write_text(json.dumps(meta[:-1]))
        with self.assertRaises(RuntimeError) as cm:
            SS.SemanticSearcher(self._o.name)
        self.assertIn("out of sync", str(cm.exception))

    def test_cli_search_ignores_a_stale_dense_npy_matrix(self):
        import numpy as np
        np.save(Path(self._o.name, "tfidf_matrix.npy"), np.zeros((2, 3)))
        res = SS.SemanticSearcher(self._o.name).search("triangle")
        self.assertTrue(res[0]["doc_name"].startswith("tri_"), res[0]["doc_name"])

    def test_cli_search_missing_npz_is_a_clear_error(self):
        Path(self._o.name, "tfidf_matrix.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            SS.SemanticSearcher(self._o.name)


if __name__ == "__main__":
    unittest.main()
