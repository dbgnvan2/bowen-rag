"""
Tests for exact_match.py and the exact_search / exact_presence delegations in both
IndexManager classes.

Spec: docs/spec_exact_match_search_v2.md (E0-E8; later tiers add E10-E12).
Offline fixtures: synthetic chunk dicts for the matching rules, and a small index built
with build_index.build(src, out, config_dir=src) as test_index_consistency.py does.
Note: build_index collapses single newlines inside a paragraph to a space, so the
"across a newline" cases use synthetic chunks; a built index never contains them.

Run: python3 -m unittest test_exact_match -v
"""
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import subprocess
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



# ── E6a ───────────────────────────────────────────────────────────────────────
class TestE6aPhrases(unittest.TestCase):
    def test_e6a_quoted(self):
        self.assertEqual(em.phrases_for('What is "emotional programming"?'),
                         (["emotional programming"], "quoted"))
        self.assertEqual(em.phrases_for('“emotional cutoff” and "nodal events" and "Emotional Cutoff"'),
                         (["emotional cutoff", "nodal events"], "quoted"))

    def test_e6a_short_query(self):
        self.assertEqual(em.phrases_for("emotional programming"),
                         (["emotional programming"], "short-query"))
        self.assertEqual(em.phrases_for("  emotional programming?! "),
                         (["emotional programming"], "short-query"))

    def test_e6a_question_frame_stripped(self):
        """Dave, 2026-11-01: question words (question_frames.yml) are removed first, so the
        default experiment question checks the term, not the whole question."""
        cases = {
            "What is emotional programming?": "emotional programming",
            "What does Bowen mean by differentiation of self?": "differentiation of self",
            "define fusion": "fusion",
            "What is the emotional system?": "emotional system",
            "What does triangle mean?": "triangle",
            "Tell me about nodal events": "nodal events",
        }
        for q, term in cases.items():
            self.assertEqual(em.phrases_for(q), ([term], "question-term"), q)
        # term words are never frame words; a question that is only frame words stays whole
        self.assertEqual(em.phrases_for("emotional process"), (["emotional process"], "short-query"))
        self.assertEqual(em.phrases_for("What is?"), (["What is"], "short-query"))
        # frames are whole words: "whatever" is not "what"; "Theme" is not "the"
        self.assertEqual(em.phrases_for("Theme of anxiety"), (["Theme of anxiety"], "short-query"))

    def test_e6a_frames_file_missing_is_reported(self):
        em._FRAMES_CACHE.pop("/no/such/question_frames.yml", None)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            frames = em.load_question_frames(Path("/no/such/question_frames.yml"))
        self.assertIn("could not load", err.getvalue())
        self.assertEqual(em.phrases_for("What is emotional programming?", frames),
                         (["What is emotional programming"], "short-query"))
        em._FRAMES_CACHE.pop("/no/such/question_frames.yml", None)

    def test_e6a_long_query_none(self):
        self.assertEqual(em.phrases_for("How does Bowen describe the process of emotional "
                                        "programming across generations?"), ([], "none"))
        self.assertEqual(em.phrases_for("?!"), ([], "none"))
        self.assertEqual(em.phrases_for('x "  " b'), (['x "  " b'], "short-query"))


# ── E6 ────────────────────────────────────────────────────────────────────────
def _hits(ix, phrase):
    return ix.search(phrase)


class TestE6Union(unittest.TestCase):
    def setUp(self):
        self.ix = em.ExactIndex(CHUNKS)

    def contains(self, c):
        return em.count_matches(c["text"], "emotional programming") > 0

    def test_e6_auto_union(self):
        ranked = [CHUNKS[2], CHUNKS[6]]                         # retrieval found none
        out = em.union_exact(ranked, _hits(self.ix, "emotional programming"), budget=2)
        self.assertEqual(len(out), 2)
        self.assertTrue(all(self.contains(c) for c in out))
        self.assertEqual(out[0]["id"], 1)                        # most hits first
        self.assertTrue(out[0]["exact_union"])
        self.assertEqual(ranked, [CHUNKS[2], CHUNKS[6]])        # inputs untouched

    def test_e6_existing_hits_count_toward_budget(self):
        ranked = [CHUNKS[0], CHUNKS[2], CHUNKS[6]]              # one hit already present
        out = em.union_exact(ranked, _hits(self.ix, "emotional programming"), budget=2)
        self.assertEqual(len(out), 3)
        self.assertEqual(sum(1 for c in out if self.contains(c)), 2)
        self.assertIn(0, [c["id"] for c in out])

    def test_e6_trim_keeps_existing_hit_at_the_tail(self):
        ranked = [CHUNKS[2], CHUNKS[6], CHUNKS[0]]              # the one hit is last
        out = em.union_exact(ranked, _hits(self.ix, "emotional programming"), budget=2)
        self.assertEqual([c["id"] for c in out], [1, 2, 0])

    def test_e6_noop_when_absent(self):
        ranked = [CHUNKS[2]]
        self.assertEqual(em.union_exact(ranked, _hits(self.ix, "emotional contact"), 5), ranked)

    def test_e6_k_extends_a_short_ranked_list(self):
        out = em.union_exact([], _hits(self.ix, "emotional programming"), budget=2, k=4)
        self.assertEqual([c["id"] for c in out], [1, 0])
        self.assertEqual(em.union_exact([], _hits(self.ix, "emotional programming"), 2), [])

    def test_e6_budget_rule(self):
        self.assertEqual(em.union_budget(12, 5), 5)
        self.assertEqual(em.union_budget(6, 5), 3)
        self.assertEqual(em.union_budget(1, 5), 1)
        self.assertEqual(em.union_budget(30, 10), 10)
        self.assertEqual(em.union_budget(12, 0), 0)
        self.assertEqual(em.union_budget(12), min(em.EXACT_UNION_BUDGET, 6))

    def test_e6_guard_unions_and_reports(self):
        class Idx:
            exact = self.ix
            def exact_search(s, n, **kw): return self.ix.search(n, **kw)
            def exact_presence(s, n, **kw): return self.ix.presence(n, **kw)
        out, phrases, reason, pres = em.exact_guard(Idx(), "emotional programming",
                                                    [CHUNKS[2], CHUNKS[6]], k=4)
        self.assertEqual((phrases, reason), (["emotional programming"], "short-query"))
        # the question form now reaches the same term (frame words removed)
        out_q, phrases_q, reason_q, _ = em.exact_guard(Idx(), "What is emotional programming?",
                                                       [CHUNKS[2], CHUNKS[6]], k=4)
        self.assertEqual((phrases_q, reason_q), (["emotional programming"], "question-term"))
        self.assertEqual(sum(1 for c in out_q if self.contains(c)), 2)
        self.assertEqual(len(out), 4)
        self.assertEqual(sum(1 for c in out if self.contains(c)), 2)     # budget = 4 // 2
        self.assertTrue(pres["emotional programming"]["found"])
        # union=False (hand-staged) leaves the list alone but still reports presence
        out2, _, _, pres2 = em.exact_guard(Idx(), "emotional programming",
                                           [CHUNKS[2]], union=False)
        self.assertEqual(out2, [CHUNKS[2]])
        self.assertTrue(pres2["emotional programming"]["found"])
        # keep= filters candidate hits (the Chat author filter)
        out3, *_ = em.exact_guard(Idx(), "emotional programming", [CHUNKS[2], CHUNKS[6]],
                                  k=4, keep=lambda c: c["doc_name"] == "F")
        self.assertEqual([c["id"] for c in out3 if self.contains(c)], [7])

    def test_e6_guard_long_question_adds_nothing(self):
        class Idx:
            def exact_search(s, n, **kw): raise AssertionError("no search expected")
            def exact_presence(s, n, **kw): raise AssertionError("no check expected")
        q = "How does Bowen describe the process of emotional programming across generations?"
        out, phrases, reason, pres = em.exact_guard(Idx(), q, [CHUNKS[2]])
        self.assertEqual((out, phrases, reason, pres), ([CHUNKS[2]], [], "none", {}))


REFS = ROOT / "rag-document-search" / "references"


def _corpus_has_chapter09() -> bool:
    try:
        meta = json.loads((REFS / "chunk_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return any("Emotional_Programming" in c.get("doc_name", "") for c in meta)


@unittest.skipUnless(_corpus_has_chapter09(), "shipped index not present (corpus volume unmounted)")
class TestE6Corpus(unittest.TestCase):
    """Corpus-gated: runs against rag-document-search/references when it is present."""

    @classmethod
    def setUpClass(cls):
        import streamlit_app
        cls.idx = streamlit_app.IndexManager()
        cls.idx.load(REFS)

    def test_e6_emotional_programming_chapter09(self):
        q = "emotional programming"
        modes = {"bm25": lambda: self.idx.bm25_search(q, 12, use_boost=True)}
        if self.idx.embed_matrix is not None:
            modes["hybrid"] = lambda: self.idx.hybrid_search(q, 12, use_boost=True)
        for mode, run in modes.items():
            out, *_ = em.exact_guard(self.idx, q, run(), k=12)
            self.assertEqual(len(out), 12, mode)
            self.assertTrue(any("Bowen Theory Secrets_Chapter09_Emotional_Programming"
                                in c["doc_name"] for c in out), mode)
            self.assertGreaterEqual(
                sum(1 for c in out if em.count_matches(c["text"], q)), 1, mode)


# ── E7 ────────────────────────────────────────────────────────────────────────
_RUN_FIND = """
import contextlib, io, json, sys
from pathlib import Path
cases = json.loads(sys.argv[1])
sys.argv = ["bowen_ask.py"]
import bowen_ask
results = []
for refs, argv, stdin in cases:
    bowen_ask.app.REFS_DIR = Path(refs)
    sys.argv = ["bowen_ask.py"] + argv
    sys.stdin = io.StringIO(stdin)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            bowen_ask.main()
            code = 0
        except SystemExit as e:
            code = e.code
    results.append({"code": code, "out": out.getvalue(), "err": err.getvalue()})
print("@@" + json.dumps({"results": results, "anthropic": "anthropic" in sys.modules}))
"""

_FIND_CASES = {          # name: (use the fixture index?, argv, stdin)
    "found":     (True, ["--find", "emotional programming"], ""),
    "stdin":     (True, ["--find"], "emotional programming\n"),
    "absent":    (True, ["--find", "no such phrase xyz"], ""),
    "bad_regex": (True, ["--find", "(", "--mode", "regex"], ""),
    "empty":     (True, ["--find"], "  \n"),
    "bad_mode":  (True, ["--find", "x", "--mode", "fuzzy"], ""),
    "json":      (True, ["--find", "emotional programming", "--mode", "substring", "--json"], ""),
    "no_index":  (False, ["--find", "emotional programming"], ""),
}


class TestE7FindCli(FixtureIndex):
    """Every case runs through bowen_ask.main() in ONE fresh interpreter (the import of
    streamlit_app and its ML libraries costs seconds), with stdout/stderr captured per case."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        names = list(_FIND_CASES)
        cases = [(str(cls.out) if fx else "/nonexistent/refs", argv, stdin)
                 for fx, argv, stdin in _FIND_CASES.values()]
        r = subprocess.run([sys.executable, "-c", _RUN_FIND, json.dumps(cases)],
                           capture_output=True, text=True, cwd=ROOT, timeout=300)
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("@@")), None)
        if line is None:
            raise AssertionError(f"find runner failed:\n{r.stdout}\n{r.stderr}")
        data = json.loads(line[2:])
        cls.res = dict(zip(names, data["results"]))
        cls.anthropic_imported = data["anthropic"]

    def test_e7_find_cli_found_exit0(self):
        r = self.res["found"]
        self.assertEqual(r["code"], 0, r["err"])
        self.assertIn("8 occurrences in 8 passages across 1 document ", r["out"])
        self.assertIn("kerr_book", r["out"])
        self.assertIn("9 counting partial-word (loose) matches", r["out"])

    def test_e7_find_cli_reads_stdin(self):
        r = self.res["stdin"]
        self.assertEqual(r["code"], 0, r["err"])
        self.assertIn("kerr_book", r["out"])

    def test_e7_find_cli_absent_exit1(self):
        r = self.res["absent"]
        self.assertEqual(r["code"], 1, r["err"])
        self.assertIn("0 occurrences in the indexed text (0 loose)", r["out"])

    def test_e7_find_cli_bad_regex_exit2(self):
        r = self.res["bad_regex"]
        self.assertEqual(r["code"], 2)
        self.assertIn("re.error", r["err"])
        self.assertIn("missing )", r["err"])

    def test_e7_find_cli_empty_and_bad_mode_exit2(self):
        self.assertEqual(self.res["empty"]["code"], 2)
        self.assertEqual(self.res["bad_mode"]["code"], 2)
        self.assertIn("unknown match mode", self.res["bad_mode"]["err"])

    def test_e7_find_json_and_ocr_flag(self):
        r = self.res["json"]
        self.assertEqual(r["code"], 0, r["err"])
        scan = [d for d in json.loads(r["out"])["docs"] if d["doc_name"] == "scan_doc"][0]
        self.assertEqual((scan["ocr"], scan["ocr_confidence"]), (True, 81.5))

    def test_e7_find_does_not_import_anthropic(self):
        self.assertEqual(self.res["found"]["code"], 0)
        self.assertFalse(self.anthropic_imported, "the --find path imported anthropic")

    def test_e7_unreadable_index_is_not_reported_as_absent(self):
        r = self.res["no_index"]
        self.assertEqual(r["code"], 4, r["out"])
        self.assertIn("could not read the index", r["err"])


# ── E8 ────────────────────────────────────────────────────────────────────────
_OLD_RULE7 = (
    "7. Terminology precision — CRITICAL for Bowen theory: Before answering, check whether the "
    "exact term or phrase from the user's query appears verbatim in the provided excerpts. "
    "If it does NOT appear verbatim, you MUST begin your response with a clearly visible notice:\n"
    "⚠️ Note: The exact term/phrase \"[user's term]\" does not appear in the provided sources. "
    "The following is drawn from semantically related content and represents an inference, "
    "not a direct match. Bowen theory uses precise terminology — the absence of a term is "
    "significant and should not be papered over by substituting similar-sounding concepts.\n"
)


def _system_prompt(path):
    tree = ast.parse((ROOT / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SYSTEM_PROMPT":
            return ast.literal_eval(node.value)
    raise AssertionError(f"no literal SYSTEM_PROMPT in {path}")


class TestE8Verification(FixtureIndex):
    def setUp(self):
        self.ix = em.ExactIndex(CHUNKS)

    def test_e8_block_text(self):
        pres = {"emotional programming": self.ix.presence("emotional programming")}
        block = em.verification_block(pres, [CHUNKS[2], CHUNKS[0]], "short-query")
        self.assertTrue(block.startswith("EXACT-MATCH VERIFICATION"))
        self.assertIn("scanning all 8 indexed passages", block)
        self.assertIn('"emotional programming": occurs 7 times in 6 passages across '
                      "4 documents of the indexed corpus. 1 of the 2 excerpts provided "
                      "contain it.", block)

    def test_e8_block_absent_quoted(self):
        pres = {"emotional contact": self.ix.presence("emotional contact")}
        block = em.verification_block(pres, [CHUNKS[2]], "quoted")
        self.assertIn('"emotional contact": does not occur as a whole word or phrase anywhere '
                      "in the indexed corpus. 0 of the 1 excerpts provided contain it.", block)

    def test_e8_block_whole_question_miss_is_not_a_term_miss(self):
        q = "What is emotional programming"
        block = em.verification_block({q: self.ix.presence(q)}, [CHUNKS[0]], "short-query")
        self.assertIn("(the whole question, checked as one phrase)", block)
        self.assertIn("says nothing about the individual words or any shorter term", block)

    def test_e8_block_none_says_no_check_ran(self):
        block = em.verification_block({}, [CHUNKS[0]], "none")
        self.assertIn("no exact-match check was run", block)
        self.assertIn("Nothing here says whether any term occurs in the corpus", block)

    def test_e8_block_reports_loose_matches(self):
        for name, idx in self.loaded():
            p = idx.exact_presence("emotional programming")
            block = em.verification_block({"emotional programming": p}, [], "short-query")
            self.assertIn("A further 1 occurrence runs into a neighbouring word", block, name)

    def test_e8_block_note_appended(self):
        block = em.verification_block({}, [], "none", note="These excerpts were hand-staged.")
        self.assertTrue(block.endswith("These excerpts were hand-staged."))

    def test_e8_counts_text_actually_sent(self):
        pres = {"emotional programming": self.ix.presence("emotional programming")}
        sent = ["no phrase here", "a neighbour: emotional\nprogramming"]
        block = em.verification_block(pres, sent, "quoted")
        self.assertIn("1 of the 2 excerpts provided contain it.", block)
        self.assertEqual(em.count_matches("a neighbour: emotional\nprogramming",
                                          "emotional programming"), 1)

    def test_e8_chat_message_counts_neighbours(self):
        import streamlit_app
        for name, idx in self.loaded():
            if name != "streamlit_app":
                continue
            q = "emotional programming"
            # a chunk without the phrase whose ±1 neighbour has it
            target = next(c for c in idx.chunks
                          if c["doc_name"] == "kerr_book" and not em.count_matches(c["text"], q)
                          and any(em.count_matches(t, q)
                                  for t in idx.get_context_window(c["id"], window=1)))
            p = idx.exact_presence(q)
            content, block, docs = streamlit_app.build_chat_message(
                idx, [target], q, {q: p}, "short-query")
            self.assertTrue(content.startswith("EXACT-MATCH VERIFICATION"))
            sent = [t for ts in docs.values() for t in ts]
            containing = sum(1 for t in sent if em.count_matches(t, q))
            self.assertGreater(len(sent), 1)                    # the chunk and its neighbours
            self.assertGreaterEqual(containing, 1)              # only via a neighbour
            self.assertIn(f"{containing} of the {len(sent)} excerpts provided contain it.", block)
            self.assertIn(f"Question: {q}", content)

    def test_e8_prompt_contains_block(self):
        import streamlit_app
        block = em.verification_block(
            {"x": self.ix.presence("emotional contact")}, [], "quoted")
        prompt = streamlit_app.build_report_prompt("q", "ctx", "refs", 3, 500, verification=block)
        self.assertTrue(prompt.startswith(block + "\n\n---\n\n"))
        self.assertIn("## SOURCE EXCERPTS (3 passages)", prompt)
        self.assertNotIn("EXACT-MATCH", streamlit_app.build_report_prompt("q", "c", "r", 1, 1))
        # bowen_ask.build_prompt is that same function, not a copy
        tree = ast.parse((ROOT / "bowen_ask.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_prompt")
        calls = [ast.unparse(c.func) for c in ast.walk(fn) if isinstance(c, ast.Call)]
        self.assertIn("app.build_report_prompt", calls)

    def test_e8_rule7_wording(self):
        prompts = {p: _system_prompt(p) for p in ("streamlit_app.py", "bowen_rag_gui.py")}
        for path, sp in prompts.items():
            self.assertNotIn("does not appear in the provided sources", sp, path)
            self.assertIn("EXACT-MATCH VERIFICATION", sp, path)
            self.assertIn("'emotional contact'", sp, path)       # the rationale stays
        self.assertEqual(prompts["streamlit_app.py"], prompts["bowen_rag_gui.py"])

    def test_e8_session_prompt_bump(self):
        import streamlit_app
        current = streamlit_app.SYSTEM_PROMPT
        start = current.index("7. Terminology precision")
        end = current.index("This rule exists because")
        old = current[:start] + _OLD_RULE7 + current[end:]
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),
                         "0da4491b1e2398eae6f1cb5ab89e60be40ec30b57975f357c1cf26daa3fc054c")
        self.assertTrue(streamlit_app._is_superseded_system_prompt(old))
        self.assertFalse(streamlit_app._is_superseded_system_prompt(current))
        self.assertFalse(streamlit_app._is_superseded_system_prompt(old + " (edited)"))


# ── E10 ───────────────────────────────────────────────────────────────────────
# Doc "spread" mentions both words many times but never side by side; doc "exact" has the
# phrase once. Old OR-scoring of raw counts ranked "spread" first.
_E10_CHUNKS = [
    _chunk(0, "spread", "Emotional reactivity. " * 6 + "Programming of computers. " * 6),
    _chunk(1, "exact", "Kerr calls it emotional programming."),
    _chunk(2, "other", "Programming languages only."),
]


def _manager_on(cls, chunks):
    """An IndexManager with only the fields keyword_search reads (no index files)."""
    import exact_match
    m = cls()
    m.chunks = chunks
    m.exact = exact_match.ExactIndex(chunks)
    m.loaded = True
    return m


class TestE10Keyword(FixtureIndex):
    def test_e10_punctuation_stripped(self):
        """'programming?' used to be one token that matched nothing."""
        stop = frozenset({"what"})
        self.assertEqual(em.keyword_terms("What is programming?", stop),
                         ([], None, ["programming"]))
        self.assertEqual(em.keyword_terms("'programmed' anxiety, self-differentiation!", stop)[2],
                         ["programmed", "anxiety", "self-differentiation"])
        for name, idx in self.loaded():
            docs = {r["doc_name"] for r in idx.keyword_search("programming?", 10, use_boost=False)}
            self.assertEqual(docs, {"kerr_book", "scan_doc"}, name)

    def test_e10_all_terms_required(self):
        """'differentiation' (diff_doc) and 'yourself' (self_doc) never share a chunk:
        OR-scoring returned both documents; AND returns nothing."""
        for name, idx in self.loaded():
            self.assertEqual(idx.keyword_search("differentiation yourself", 10, use_boost=False),
                             [], name)
            got = idx.keyword_search("differentiated head", 10, use_boost=False)
            self.assertEqual([r["doc_name"] for r in got], ["diff_doc"], name)
            self.assertIn("clear head", got[0]["text"], name)

    def test_e10_unquoted_phrase_first(self):
        for name, cls in managers(self):
            idx = _manager_on(cls, copy.deepcopy(_E10_CHUNKS))
            got = idx.keyword_search("What is emotional programming?", 10, use_boost=False)
            self.assertEqual([r["doc_name"] for r in got], ["exact", "spread"], name)
            self.assertIn("phrase", got[0]["score_label"], name)
            self.assertNotIn("phrase", got[1]["score_label"], name)
            self.assertGreater(got[1]["score"], got[0]["score"], name)   # raw count lost
            self.assertNotIn("_phrase", got[0], name)

    def test_e10_unquoted_phrase_keeps_interior_words(self):
        stop = frozenset({"what"})
        self.assertEqual(em.keyword_terms("What is differentiation of self?", stop)[1],
                         "differentiation of self")
        self.assertIsNone(em.keyword_terms("programming", stop)[1])

    def test_e10_quoted_phrase_on_normalised_text_whole_words(self):
        """Quoted phrases match across a line break and curly quotes (E0), and as whole
        words: "self" no longer matches "yourself"/"selfish"."""
        for name, idx in self.loaded():
            docs = {r["doc_name"] for r in idx.keyword_search('\u201cself\u201d', 10, use_boost=False)}
            self.assertEqual(docs, {"diff_doc", "kerr_book", "self_doc"}, name)
            hit_ids = {i for i, _, _ in idx.exact.keyword_hits('"self"', idx._STOP, idx._stems)}
            yourself = [c["id"] for c in idx.chunks if c["text"].startswith("Look after yourself")]
            self.assertEqual(len(yourself), 1, name)
            self.assertNotIn(yourself[0], hit_ids, name)          # "yourself", "selfish"
        ix = em.ExactIndex([_chunk(0, "C", "emotional\nprogramming here")])
        self.assertEqual(ix.keyword_hits('"emotional programming"', frozenset(), lambda w: [w]),
                         [(0, 3, True)])

    def test_e10_chunks_not_mutated(self):
        for name, cls in managers(self):
            chunks = copy.deepcopy(_E10_CHUNKS)
            idx = _manager_on(cls, chunks)
            idx.keyword_search("emotional programming", 10, use_boost=True)
            self.assertEqual(chunks, _E10_CHUNKS, name)



class TestE8QuestionTerm(unittest.TestCase):
    def test_e8_absent_question_term_wording(self):
        ix = em.ExactIndex(CHUNKS)
        block = em.verification_block({"nodal events": ix.presence("nodal events")},
                                      [CHUNKS[0]], "question-term")
        self.assertIn('"nodal events" (the question with its question words removed): does not '
                      "occur as a whole word or phrase anywhere in the indexed corpus", block)
        self.assertIn("0 of the 1 excerpts provided contain it.", block)

# ── Carried over from the literal_search.py prototype ─────────────────────────
class TestCarriedOver(unittest.TestCase):
    def test_highlight(self):
        self.assertEqual(em.highlight("x Emotional programming y", "emotional programming"),
                         "x **Emotional programming** y")
        self.assertEqual(em.highlight("a emotional\nprogramming b", "emotional programming"),
                         "a **emotional\nprogramming** b")


if __name__ == "__main__":
    unittest.main()
