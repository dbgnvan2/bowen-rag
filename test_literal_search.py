"""Tests for literal_search.py — synthetic chunks, no index files needed."""
import unittest

import literal_search as ls


def _chunk(i, doc, text):
    return {"id": i, "doc_name": doc, "text": text}


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


class NormalizeTests(unittest.TestCase):
    def test_idempotent_and_whitespace(self):
        s = "  Emotional\n\tProgramming  "
        self.assertEqual(ls.normalize(s), "emotional programming")
        self.assertEqual(ls.normalize(ls.normalize(s)), ls.normalize(s))

    def test_ligature_and_soft_hyphen(self):
        self.assertEqual(ls.normalize("deﬁne"), "define")
        self.assertEqual(ls.normalize("de­fine"), "define")

    def test_hyphen_line_break_rejoined(self):
        self.assertEqual(ls.normalize("program-\nming"), "programming")
        # a real hyphenated compound on one line is kept
        self.assertEqual(ls.normalize("self-differentiation"), "self-differentiation")


class ExtractPhrasesTests(unittest.TestCase):
    def test_quoted_wins(self):
        self.assertEqual(ls.extract_phrases('What is "emotional programming"?'), ["emotional programming"])
        self.assertEqual(ls.extract_phrases('“emotional cutoff” and x'), ["emotional cutoff"])

    def test_unquoted_is_whole_query(self):
        self.assertEqual(ls.extract_phrases("emotional programming"), ["emotional programming"])


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.ix = ls.LiteralIndex(CHUNKS)

    def test_finds_every_true_occurrence_and_no_false_one(self):
        hits = self.ix.search("emotional programming")
        self.assertEqual(sorted(h.chunk["id"] for h in hits), [0, 1, 3, 4, 5, 7])
        self.assertEqual({h.chunk["id"]: h.occurrences for h in hits}[1], 2)

    def test_whole_word_boundary(self):
        # 'reprogramming emotional' / 'emotionally programmed' never match the phrase
        self.assertFalse(any(h.chunk["id"] == 6 for h in self.ix.search("emotional programming")))
        # default is substring: 'programm' hits 'programmed' / 'reprogramming'
        self.assertTrue(any(h.chunk["id"] == 6 for h in self.ix.search("programm")))
        # whole-word is opt-in and excludes them
        self.assertFalse(any(h.chunk["id"] == 6 for h in self.ix.search("programm", whole_words=True)))

    def test_ranking_by_occurrences(self):
        self.assertEqual(self.ix.search("emotional programming")[0].chunk["id"], 1)

    def test_no_topk_means_all(self):
        self.assertEqual(len(self.ix.search("emotional")), 8)
        self.assertEqual(len(self.ix.search("emotional", top_k=2)), 2)

    def test_presence(self):
        p = self.ix.presence("emotional programming")
        self.assertTrue(p.found)
        self.assertEqual((p.occurrences, p.chunks, p.documents), (7, 6, 4))   # docs A, C, D, F
        self.assertFalse(self.ix.presence("emotional contact").found)

    def test_regex(self):
        hits = self.ix.search_regex(r"emotional(ly)? programm\w*")
        self.assertIn(6, [h.chunk["id"] for h in hits])


class PipelineHelpersTests(unittest.TestCase):
    def setUp(self):
        self.ix = ls.LiteralIndex(CHUNKS)

    def test_corpus_check_line_counts_provided(self):
        line = self.ix.corpus_check_line("emotional programming", [CHUNKS[2], CHUNKS[0]])
        self.assertIn("occurs 7 times in 6 passages across 4 documents", line)
        self.assertIn("1 of the excerpts provided below contain it verbatim", line)

    def test_corpus_check_line_absent(self):
        line = self.ix.corpus_check_line("emotional contact", [CHUNKS[2]])
        self.assertIn("does not occur anywhere", line)

    def test_ensure_literal_hits_inserts_missing(self):
        results = [CHUNKS[2], CHUNKS[6]]                 # retrieval found none
        out = self.ix.ensure_literal_hits(results, "emotional programming", reserve=2)
        self.assertEqual(len(out), 2)
        self.assertTrue(all("emotional programming" in ls.normalize(c["text"]) for c in out))
        self.assertEqual(out[0]["mode"], "literal")

    def test_ensure_literal_hits_counts_existing(self):
        results = [CHUNKS[0], CHUNKS[2], CHUNKS[6]]      # one hit already present
        out = self.ix.ensure_literal_hits(results, "emotional programming", reserve=2)
        self.assertEqual(len(out), 3)
        self.assertEqual(sum(1 for c in out if "emotional programming" in ls.normalize(c["text"])), 2)
        self.assertIn(CHUNKS[0]["id"], [c["id"] for c in out])

    def test_ensure_literal_hits_noop_when_absent(self):
        results = [CHUNKS[2]]
        self.assertEqual(self.ix.ensure_literal_hits(results, "emotional contact"), results)

    def test_highlight(self):
        self.assertEqual(ls.highlight("x Emotional programming y", "emotional programming"),
                         "x **Emotional programming** y")


if __name__ == "__main__":
    unittest.main()
