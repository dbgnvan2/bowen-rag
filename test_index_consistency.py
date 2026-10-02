"""
Producer/consumer consistency of the search index.

build_index.py writes the index; streamlit_app.py, bowen_rag_gui.py and semantic_search.py
each re-fit a TF-IDF vectorizer when they load it. If their settings drift from the ones
used to build the matrix, search silently ranks the wrong chunks. These tests pin the
settings and check that every loader fails loudly on files that disagree.

Run: python3 -m unittest test_index_consistency
"""
import ast
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rag-document-search" / "scripts"))
import build_index as B  # noqa: E402

DOCS = {}
for _topic, _words in (("tri", "triangle anxiety two person system"),
                       ("cut", "cutoff distancing family of origin attachment"),
                       ("frq", "word frequency inverse document corpus")):
    for _i in "ab":
        DOCS[f"{_topic}_{_i}"] = (f"## Section 1 – {_topic}\n\n{_words} {_words}. "
                                  f"More on {_words}.\n")


def build(src, out):
    for name, text in DOCS.items():
        (Path(src) / f"{name}.txt").write_text(text)
    B.build(src, out, config_dir=src, log=lambda m: None)


def vectorizer_kwargs(path):
    """TfidfVectorizer(...) keyword arguments written literally in `path`."""
    tree = ast.parse((ROOT / path).read_text())
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "TfidfVectorizer"
                and node.keywords):
            found.append({k.arg: ast.literal_eval(k.value) for k in node.keywords})
    return found


class TestSettingsPinned(unittest.TestCase):
    def test_idx_apps_refit_with_the_settings_the_matrix_was_built_with(self):
        for path in ("streamlit_app.py", "bowen_rag_gui.py"):
            found = vectorizer_kwargs(path)
            self.assertTrue(found, f"no literal TfidfVectorizer(...) found in {path}")
            for kwargs in found:
                self.assertEqual(kwargs, B.TFIDF_PARAMS, path)


class TestLoadersFailLoudly(unittest.TestCase):
    def setUp(self):
        self._s = tempfile.TemporaryDirectory(); self._o = tempfile.TemporaryDirectory()
        self.addCleanup(self._s.cleanup); self.addCleanup(self._o.cleanup)
        build(self._s.name, self._o.name)
        self.out = Path(self._o.name)

    def managers(self):
        import streamlit_app
        yield "streamlit_app", streamlit_app.IndexManager
        try:
            import bowen_rag_gui
            yield "bowen_rag_gui", bowen_rag_gui.IndexManager
        except ImportError as e:     # tkinter missing on this machine
            self.skipTest(f"bowen_rag_gui not importable here: {e}")

    def test_idx_loaders_accept_a_consistent_index(self):
        for name, cls in self.managers():
            info = cls().load(self.out)
            self.assertEqual(info["chunks"], 6, name)

    def test_idx_loaders_keep_the_tfidf_matrix_sparse_and_score_like_dense(self):
        # A dense copy of the matrix cost ~790 MB for the real corpus; scores must not change.
        import numpy as np
        from scipy import sparse
        from sklearn.metrics.pairwise import cosine_similarity
        for name, cls in self.managers():
            mgr = cls()
            mgr.load(self.out)
            self.assertTrue(sparse.issparse(mgr.matrix), name)
            qv = mgr.vectorizer.transform(["triangle anxiety two person system"])
            np.testing.assert_allclose(cosine_similarity(qv, mgr.matrix),
                                       cosine_similarity(qv, mgr.matrix.toarray()),
                                       atol=1e-12, err_msg=name)
            self.assertTrue(len(mgr.semantic_search("triangle anxiety", 3)) > 0, name)

    def test_idx_loaders_reject_metadata_from_a_different_build(self):
        meta = json.loads((self.out / "chunk_metadata.json").read_text())
        (self.out / "chunk_metadata.json").write_text(json.dumps(meta[:-1]))
        for name, cls in self.managers():
            with self.assertRaises(RuntimeError, msg=name) as cm:
                cls().load(self.out)
            self.assertIn("out of sync", str(cm.exception))

    def test_idx_loaders_reject_a_vectorizer_json_from_a_different_build(self):
        vec = json.loads((self.out / "vectorizer.json").read_text())
        vec["chunks_sha256"] = "0" * 64      # built from other chunk text
        (self.out / "vectorizer.json").write_text(json.dumps(vec))
        for name, cls in self.managers():
            with self.assertRaises(RuntimeError, msg=name):
                cls().load(self.out)

    def test_idx_loaders_use_the_saved_features_not_a_fresh_selection(self):
        # A different scikit-learn can pick other features from the same text (this broke
        # the Railway deploy). The saved list must win: simulate a fresh fit disagreeing.
        vec = json.loads((self.out / "vectorizer.json").read_text())
        vec["feature_names"][-1] = "zzz_not_chosen_by_a_fresh_fit"
        (self.out / "vectorizer.json").write_text(json.dumps(vec))
        for name, cls in self.managers():
            m = cls()
            m.load(self.out)
            self.assertEqual(m.vectorizer.get_feature_names_out().tolist(),
                             vec["feature_names"], name)

    def test_idx_vectorizer_json_without_a_hash_still_loads(self):
        vec = json.loads((self.out / "vectorizer.json").read_text())
        del vec["chunks_sha256"]
        (self.out / "vectorizer.json").write_text(json.dumps(vec))
        for name, cls in self.managers():
            cls().load(self.out)

    def write_embeddings(self, text_for_fingerprint=None):
        import numpy as np
        import index_fingerprint as F
        meta = json.loads((self.out / "chunk_metadata.json").read_text())
        np.save(self.out / "embed_matrix.npy", np.zeros((len(meta), 384), dtype="float32"))
        if text_for_fingerprint is not False:
            chunks = meta if text_for_fingerprint is None else text_for_fingerprint
            F.write(self.out, chunks)

    def test_idx_loaders_accept_embeddings_with_a_matching_fingerprint(self):
        self.write_embeddings()
        for name, cls in self.managers():
            self.assertTrue(cls().load(self.out)["embeddings"], name)

    def test_idx_loaders_accept_embeddings_with_no_sidecar_for_older_indexes(self):
        self.write_embeddings(text_for_fingerprint=False)
        for name, cls in self.managers():
            self.assertTrue(cls().load(self.out)["embeddings"], name)

    def test_idx_loaders_reject_embeddings_built_from_different_text_same_count(self):
        # The dirty-state case the row-count check cannot see: same number of chunks,
        # different text (a rebuild after editing documents).
        meta = json.loads((self.out / "chunk_metadata.json").read_text())
        other = [dict(c, text=c["text"] + " edited") for c in meta]
        self.write_embeddings(text_for_fingerprint=other)
        for name, cls in self.managers():
            with self.assertRaises(RuntimeError, msg=name) as cm:
                cls().load(self.out)
            self.assertIn("different chunk text", str(cm.exception))

    def test_idx_loaders_do_not_fall_back_to_a_stale_dense_npy(self):
        import numpy as np
        (self.out / "tfidf_matrix.npz").unlink()
        np.save(self.out / "tfidf_matrix.npy", np.zeros((6, 3)))
        for name, cls in self.managers():
            with self.assertRaises(FileNotFoundError, msg=name):
                cls().load(self.out)


if __name__ == "__main__":
    unittest.main()
