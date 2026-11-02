"""
Tests for exact_match.py and the exact_search / exact_presence delegations in both
IndexManager classes.

Spec: docs/spec_exact_match_search_v2.md (E0-E5 here; later tiers add E6-E12).
Offline fixtures: synthetic chunk dicts for the matching rules, and a small index built
with build_index.build(src, out, config_dir=src) as test_index_consistency.py does.
Note: build_index collapses single newlines inside a paragraph to a space, so the
"across a newline" cases use synthetic chunks; a built index never contains them.

Run: python3 -m unittest test_exact_match -v
"""
import copy
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rag-document-search" / "scripts"))
import build_index as B  # noqa: E402
import citations  # noqa: E402
import exact_match as em  # noqa: E402


def _chunk(i, doc, text, **kw):
    return {"id": i, "doc_name": doc, "text": text, **kw}


# Synthetic chunks (carried over from the literal_search.py prototype's tests).
CHUNKS = [
    _chunk(0, "A", "Kerr describes emotional programming as the process by which…"),
    _chunk(1, "A", "A second mention of Emotional Programming, and emotional programming again."),
    _chunk(2, "B", "No phrase here. Lots of emotional talk about the emotional system."),
    _chunk(3, "C", "The word emotional\nprogramming is split by a line break here."),
    _chunk(4, "C", "Soft-wrapped: emotional program-\nming across a hyphenated break."),
    _chunk(5, "D", "Ligature test: the ﬁrst deﬁnition of emotional programming."),
    _chunk(6, "E", "Not a match: emotionally programmed; nor reprogramming emotional."),
    _chunk(7, "F", "Curly quotes: “emotional programming” is what Barrett calls it."),
]

# A built fixture index. "kerr_book" holds the phrase in more chunks than any top_k used
# below; "scan_doc" is OCR (via ocr_manifest.yml) and has the lost-space variant.
_FILLER = ("The family as an emotional unit shows patterns of anxiety, distance and "
           "conflict across generations, and each person responds to the others. ")


def _para(core: str) -> str:
    return core + " " + _FILLER * 12          # ~1,700 chars: one paragraph, one chunk


FIXTURE_DOCS = {
    "kerr_book": "\n\n".join(_para(f"Part {n}: emotional programming shapes the self.")
                             for n in range(8)),
    "diff_doc": "\n\n".join([
        _para("Differentiation of self is the core concept."),
        _para("Self-differentiation varies from person to person."),
        _para("A well differentiated person keeps a clear head."),
    ]),
    "self_doc": "\n\n".join([
        _para("Look after yourself, said the selfish one."),
        _para("The self is shaped in the family."),
    ]),
    "scan_doc": _para("A scanned page says thatemotional programming without a space."),
}
OCR_MANIFEST = "ocr:\n- doc_name: scan_doc\n  mean_word_confidence: 81.5\n"


def build_fixture(src: str, out: str) -> None:
    for name, text in FIXTURE_DOCS.items():
        (Path(src) / f"{name}.txt").write_text(text, encoding="utf-8")
    (Path(src) / "ocr_manifest.yml").write_text(OCR_MANIFEST, encoding="utf-8")
    B.build(src, out, config_dir=src, log=lambda m: None)


def managers(test):
    """(name, IndexManager class) for both apps; the GUI is skipped without tkinter."""
    import streamlit_app
    yield "streamlit_app", streamlit_app.IndexManager
    try:
        import bowen_rag_gui
    except ImportError as e:
        test.skipTest(f"bowen_rag_gui not importable here: {e}")
    yield "bowen_rag_gui", bowen_rag_gui.IndexManager


class FixtureIndex(unittest.TestCase):
    """Builds the fixture index once per class and loads both IndexManagers on it."""

    @classmethod
    def setUpClass(cls):
        cls._s = tempfile.TemporaryDirectory()
        cls._o = tempfile.TemporaryDirectory()
        build_fixture(cls._s.name, cls._o.name)
        cls.out = Path(cls._o.name)

    @classmethod
    def tearDownClass(cls):
        cls._s.cleanup()
        cls._o.cleanup()

    def loaded(self):
        for name, cls in managers(self):
            idx = cls()
            idx.load(self.out)
            yield name, idx


# ── E0 ────────────────────────────────────────────────────────────────────────
class TestE0Normalize(unittest.TestCase):
    def test_e0_normalize_idempotent_and_whitespace(self):
        s = "  Emotional\n\tProgramming  "
        self.assertEqual(em.normalize(s), "emotional programming")
        self.assertEqual(em.normalize(em.normalize(s)), em.normalize(s))

    def test_e0_normalize_ligature_and_soft_hyphen(self):
        self.assertEqual(em.normalize("deﬁne"), "define")
        self.assertEqual(em.normalize("de­fine"), "define")

    def test_e0_normalize_hyphen_line_break_rejoined_compound_kept(self):
        self.assertEqual(em.normalize("program-\nming"), "programming")
        self.assertEqual(em.normalize("self-differentiation"), "self-differentiation")

    def test_e0_normalize_quotes_and_dashes_to_ascii(self):
        self.assertEqual(em.normalize("“a” ‘b’ c–d e—f"), "\"a\" 'b' c-d e-f")

    def test_e0_normalize_case_flag(self):
        self.assertEqual(em.normalize("Emotional  Cutoff", case=True), "Emotional Cutoff")
        self.assertEqual(em.normalize("Emotional  Cutoff"), "emotional cutoff")

    def test_e0_index_does_not_mutate_chunks(self):
        chunks = copy.deepcopy(CHUNKS)
        ix = em.ExactIndex(chunks)
        ix.search("emotional programming")
        ix.presence("emotional programming")
        self.assertEqual(chunks, CHUNKS)
        self.assertEqual(len(ix._scan_texts), len(CHUNKS))


# ── E1 ────────────────────────────────────────────────────────────────────────
class TestE1Search(FixtureIndex):
    def test_e1_returns_all_no_truncation(self):
        for name, idx in self.loaded():
            hits = idx.exact_search("emotional programming")
            # 8 chunks in kerr_book, all returned (more than any top_k of 5), plus none
            # from scan_doc in word mode (its only occurrence lacks a space).
            self.assertEqual(len(hits), 8, name)
            self.assertEqual({h["doc_name"] for h in hits}, {"kerr_book"}, name)
            self.assertTrue(all(h["mode"] == "exact" and h["hits"] == 1 for h in hits), name)
            self.assertEqual([h["id"] for h in hits], sorted(h["id"] for h in hits), name)

    def test_e1_no_per_document_collapse(self):
        for name, idx in self.loaded():
            docs = [h["doc_name"] for h in idx.exact_search("family")]
            self.assertGreater(docs.count("kerr_book"), 1, name)

    def test_e1_case_insensitive(self):
        for name, idx in self.loaded():
            self.assertEqual(len(idx.exact_search("EMOTIONAL Programming")), 8, name)

    def test_e1_case_sensitive_opt_in(self):
        ix = em.ExactIndex(CHUNKS)
        ids = [h["id"] for h in ix.search("Emotional Programming", case=True)]
        self.assertEqual(ids, [1])

    def test_e1_sorted_by_hits_then_id(self):
        ix = em.ExactIndex(CHUNKS)
        hits = ix.search("emotional programming")
        self.assertEqual([h["id"] for h in hits], [1, 0, 3, 4, 5, 7])
        self.assertEqual(hits[0]["hits"], 2)

    def test_e1_does_not_mutate_chunks(self):
        for name, idx in self.loaded():
            before = copy.deepcopy(idx.chunks)
            hits = idx.exact_search("emotional programming")
            hits[0]["text"] = "changed"
            self.assertEqual(idx.chunks, before, name)
            self.assertNotIn("hits", idx.chunks[0], name)

    def test_e1_not_loaded_raises_instead_of_returning_empty(self):
        for name, cls in managers(self):
            with self.assertRaises(RuntimeError, msg=name):
                cls().exact_search("emotional programming")


# ── E2 ────────────────────────────────────────────────────────────────────────
class TestE2Word(FixtureIndex):
    def test_e2_word_boundary(self):
        for name, idx in self.loaded():
            # "differentiation" matches itself and "self-differentiation", not "differentiated"
            diff = [h["text"] for h in idx.exact_search("differentiation")]
            self.assertEqual(len(diff), 2, name)
            self.assertFalse(any(t.startswith("A well differentiated") for t in diff), name)
            # "self" does not match "yourself" / "selfish"
            self_hits = [h["text"] for h in idx.exact_search("self")]
            self.assertFalse(any(t.startswith("Look after yourself") for t in self_hits), name)
            self.assertTrue(any(t.startswith("The self is shaped") for t in self_hits), name)

    def test_e2_matches_across_newline(self):
        ix = em.ExactIndex(CHUNKS)
        ids = [h["id"] for h in ix.search("emotional programming")]
        self.assertIn(3, ids)        # "emotional\nprogramming"
        self.assertIn(4, ids)        # "program-\nming"
        self.assertIn(5, ids)        # ligature chunk
        self.assertIn(7, ids)        # curly quotes around the phrase

    def test_e2_finds_every_true_occurrence_and_no_false_one(self):
        ix = em.ExactIndex(CHUNKS)
        self.assertEqual(sorted(h["id"] for h in ix.search("emotional programming")),
                         [0, 1, 3, 4, 5, 7])
        # 'emotionally programmed' / 'reprogramming emotional' never match the phrase
        self.assertNotIn(6, [h["id"] for h in ix.search("emotional programming")])
        self.assertNotIn(6, [h["id"] for h in ix.search("programm")])

    def test_e2_empty_needle_raises(self):
        ix = em.ExactIndex(CHUNKS)
        for mode in em.MODES:
            with self.assertRaises(ValueError, msg=mode):
                ix.search("   ", mode=mode) if mode != "regex" else ix.search("", mode=mode)

    def test_e2_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            em.ExactIndex(CHUNKS).search("x", mode="fuzzy")


# ── E3 ────────────────────────────────────────────────────────────────────────
class TestE3Substring(unittest.TestCase):
    def test_e3_substring(self):
        ix = em.ExactIndex([_chunk(0, "A", "Fusion and fused selves."), _chunk(1, "B", "Nothing.")])
        hits = ix.search("fu", mode="substring")
        self.assertEqual([(h["id"], h["hits"]) for h in hits], [(0, 2)])
        self.assertEqual(ix.search("fu"), [])                 # word mode: no boundary match

    def test_e3_substring_hits_partial_words(self):
        ix = em.ExactIndex(CHUNKS)
        self.assertIn(6, [h["id"] for h in ix.search("programm", mode="substring")])


# ── E4 ────────────────────────────────────────────────────────────────────────
class TestE4Regex(unittest.TestCase):
    def test_e4_regex(self):
        ix = em.ExactIndex(CHUNKS)
        hits = ix.search(r"emotional(ly)? programm\w*", mode="regex")
        self.assertIn(6, [h["id"] for h in hits])
        self.assertEqual({h["id"]: h["hits"] for h in hits}[1], 2)

    def test_e4_regex_case_insensitive_by_default(self):
        ix = em.ExactIndex(CHUNKS)
        self.assertEqual(len(ix.search(r"EMOTIONAL PROGRAMMING", mode="regex")), 6)

    def test_e4_bad_pattern_raises(self):
        with self.assertRaises(re.error):
            em.ExactIndex(CHUNKS).search("(", mode="regex")


# ── E5 ────────────────────────────────────────────────────────────────────────
class TestE5Presence(FixtureIndex):
    def test_e5_presence_aggregate(self):
        for name, idx in self.loaded():
            p = idx.exact_presence("emotional programming")
            self.assertTrue(p["found"], name)
            self.assertEqual((p["total_hits"], p["chunk_count"], p["doc_count"]), (8, 8, 1), name)
            d = p["docs"][0]
            self.assertEqual((d["doc_name"], d["hits"], d["chunks"], d["ocr"]),
                             ("kerr_book", 8, 8, False), name)
            expected = [citations.passage_locator(h)
                        for h in idx.exact_search("emotional programming")]
            self.assertEqual(len(expected), 8, name)
            self.assertTrue(all(e.startswith("¶ ") for e in expected), name)
            self.assertEqual(d["locators"], expected, name)

    def test_e5_presence_synthetic_counts(self):
        p = em.ExactIndex(CHUNKS).presence("emotional programming")
        self.assertEqual((p["total_hits"], p["chunk_count"], p["doc_count"]), (7, 6, 4))
        self.assertEqual([d["doc_name"] for d in p["docs"]], ["A", "C", "D", "F"])

    def test_e5_absent_is_found_false(self):
        for name, idx in self.loaded():
            p = idx.exact_presence("emotional contact")
            self.assertFalse(p["found"], name)
            self.assertEqual((p["total_hits"], p["chunk_count"], p["doc_count"], p["docs"]),
                             (0, 0, 0, []), name)
            self.assertEqual(p["loose_total_hits"], 0, name)

    def test_e5_loose_counts_lost_space(self):
        for name, idx in self.loaded():
            p = idx.exact_presence("emotional programming")
            # word mode misses "thatemotional programming"; substring mode finds it
            self.assertEqual((p["loose_total_hits"], p["loose_doc_count"]), (9, 2), name)
            s = idx.exact_presence("emotional programming", mode="substring")
            self.assertEqual((s["total_hits"], s["doc_count"]), (9, 2), name)
            self.assertEqual((s["loose_total_hits"], s["loose_doc_count"]), (9, 2), name)
            scan = [d for d in s["docs"] if d["doc_name"] == "scan_doc"][0]
            self.assertEqual((scan["ocr"], scan["ocr_confidence"]), (True, 81.5), name)

    def test_e5_locators_omitted_when_empty(self):
        ix = em.ExactIndex(CHUNKS)
        p = ix.presence("emotional programming", locator_fn=lambda c: "")
        self.assertTrue(all(d["locators"] == [] for d in p["docs"]))


if __name__ == "__main__":
    unittest.main()
