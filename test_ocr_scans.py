"""
Tests for rag-document-search/scripts/ocr_scans.py.

The helpers are tested without tesseract. The end-to-end test builds the hard case the real
scans contain — a two-page spread, rotated 90 degrees, each page skewed differently — and
checks the two pages come out separate, upright and in reading order. (A plain whole-sheet
OCR of such a sheet interleaved and garbled the two pages.) It is skipped, with the reason
printed, when tesseract or a usable font is not available.

Run: python3 -m unittest test_ocr_scans
"""
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent / "rag-document-search" / "scripts"))
import ocr_scans as O  # noqa: E402


def row(level="5", block="1", par="1", line="1", conf="95", text="word"):
    return {"level": level, "block_num": block, "par_num": par, "line_num": line,
            "conf": conf, "text": text}


class TestDehyphenate(unittest.TestCase):
    def test_ocr_line_break_hyphen_is_removed_before_a_lowercase_continuation(self):
        self.assertEqual(O.dehyphenate(["a family orientation of schizo-", "phrenia is seen"]),
                         "a family orientation of schizophrenia is seen")

    def test_ocr_hyphen_is_kept_before_a_capital_or_digit(self):
        self.assertEqual(O.dehyphenate(["the Bowen-", "Kerr interviews"]), "the Bowen- Kerr interviews")
        self.assertEqual(O.dehyphenate(["in 1976-", "1978 the study"]), "in 1976- 1978 the study")

    def test_ocr_dash_not_at_line_end_is_untouched_and_blank_lines_ignored(self):
        self.assertEqual(O.dehyphenate(["well-known fact", "", "  ", "next"]),
                         "well-known fact next")

    def test_ocr_ordinary_lines_are_joined_with_a_space(self):
        self.assertEqual(O.dehyphenate(["one", "two", "three"]), "one two three")


class TestChooseOrientation(unittest.TestCase):
    """Score tables measured on real scans (the truth for each was checked by looking at
    the page) and on the synthetic spread. Tesseract reads a sheet turned 90 degrees one
    way, so the best score alone picks the wrong rotation on several of these."""

    CASES = [
        ("synthetic spread, stored sideways", {0: 8, 90: 8, 180: 113, 270: 114}, 270),
        ("upright page with illustration", {0: 344, 90: 36, 180: 60, 270: 276}, 0),
        ("upright three-column list", {0: 225, 90: 48, 180: 40, 270: 196}, 0),
        ("upright page, near tie", {0: 415, 90: 49, 180: 50, 270: 413}, 0),
        ("sideways spread (Bowen 1960)", {0: 144, 90: 364, 180: 42, 270: 76}, 90),
        ("sideways spread (Beal), near tie", {0: 373, 90: 473, 180: 82, 270: 115}, 90),
        ("upright page, dense text", {0: 652, 90: 85, 180: 89, 270: 520}, 0),
        ("upside-down page", {0: 30, 90: 20, 180: 400, 270: 120}, 180),
    ]

    def test_ocr_orientation_is_chosen_correctly_on_measured_cases(self):
        for name, scores, truth in self.CASES:
            self.assertEqual(O.choose_orientation(scores), truth, name)

    def test_ocr_a_readable_partner_scoring_slightly_higher_does_not_flip_the_answer(self):
        # Upright pages always show a strong 270-degree partner (80-99% of the winner in the
        # measured cases above), so a near-tie can go either way on any given page.
        near_tie = {0: 400, 90: 40, 180: 50, 270: 405}      # truth: upright (0)
        self.assertEqual(max(near_tie, key=near_tie.get), 270)   # best score alone: wrong
        self.assertEqual(O.choose_orientation(near_tie), 0)      # pairing rule: right
        self.assertEqual(O.choose_orientation({0: 360, 90: 380, 180: 70, 270: 90}), 90)
        self.assertEqual(O.choose_orientation({0: 60, 90: 50, 180: 410, 270: 415}), 270)

    def test_ocr_unreadable_page_gets_rotation_zero(self):
        self.assertEqual(O.choose_orientation({0: 0, 90: 0, 180: 0, 270: 0}), 0)


class TestTsvToParagraphs(unittest.TestCase):
    def test_ocr_words_lines_and_paragraphs_are_rebuilt_from_layout_data(self):
        rows = [row(text="First", par="1", line="1"), row(text="line", par="1", line="1"),
                row(text="schizo-", par="1", line="2"), row(text="phrenia", par="1", line="3"),
                row(text="Second", par="2", line="1"), row(text="paragraph", par="2", line="1")]
        paras, confs = O.tsv_to_paragraphs(rows)
        self.assertEqual(paras, ["First line schizophrenia", "Second paragraph"])
        self.assertEqual(len(confs), 6)

    def test_ocr_non_words_and_non_word_levels_are_skipped(self):
        rows = [row(level="4", text=""), row(conf="-1", text="~"), row(text="  "),
                row(text="kept", conf="88")]
        paras, confs = O.tsv_to_paragraphs(rows)
        self.assertEqual((paras, confs), (["kept"], [88.0]))

    def test_ocr_no_words_gives_nothing_not_an_empty_paragraph(self):
        self.assertEqual(O.tsv_to_paragraphs([row(level="2", text="")]), ([], []))

    def test_ocr_blocks_are_separate_paragraphs_even_with_the_same_par_number(self):
        rows = [row(block="1", par="1", text="top"), row(block="2", par="1", text="bottom")]
        self.assertEqual(O.tsv_to_paragraphs(rows)[0], ["top", "bottom"])


def text_lines_image(w=900, h=1200, lines=14):
    """White page with horizontal dark bars standing in for lines of text."""
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for i in range(lines):
        y = 120 + i * 60
        d.rectangle([80, y, w - 80, y + 18], fill=0)
    return img


class TestGeometry(unittest.TestCase):
    def test_ocr_skew_is_estimated_and_the_sign_means_rotate_counter_clockwise_to_fix(self):
        base = text_lines_image()
        self.assertLess(abs(O.estimate_skew(np.asarray(base))), 0.25)
        skewed = base.rotate(2.0, fillcolor=255)               # rotated counter-clockwise
        est = O.estimate_skew(np.asarray(skewed))
        self.assertAlmostEqual(est, -2.0, delta=0.4)
        fixed = skewed.rotate(est, fillcolor=255)
        self.assertLess(abs(O.estimate_skew(np.asarray(fixed))), 0.4)

    def test_ocr_gutter_is_found_in_the_quiet_band_between_two_pages(self):
        w, h = 2000, 1000
        img = Image.new("L", (w, h), 255)
        d = ImageDraw.Draw(img)
        d.rectangle([100, 100, 900, 900], fill=0)               # left page ink
        d.rectangle([1100, 100, 1900, 900], fill=0)             # right page ink
        d.rectangle([w - 40, 0, w, h], fill=0)                  # black scan edge
        d.rectangle([w // 2 - 2, 400, w // 2 + 2, 410], fill=0)  # a speck in the gutter
        x = O.find_gutter(np.asarray(img))
        self.assertTrue(905 <= x <= 1095, x)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return None


class TestEndToEnd(unittest.TestCase):
    LEFT = ["The family is an emotional unit and the symptom",
            "is understood as part of the family process.",
            "Differentiation of self describes the degree",
            "to which a person can separate thinking from feeling."]
    RIGHT = ["Triangles stabilise a relationship system",
             "when anxiety rises between two people.",
             "Emotional cutoff is a way of managing",
             "unresolved attachment to the family of origin."]

    def page(self, lines, skew):
        font = _font(44)
        img = Image.new("L", (1300, 1700), 255)
        d = ImageDraw.Draw(img)
        for i, ln in enumerate(lines * 3):
            d.text((90, 140 + i * 90), ln, fill=0, font=font)
        return img.rotate(skew, fillcolor=255)

    def test_ocr_rotated_two_page_spread_comes_out_as_two_upright_pages_in_order(self):
        if shutil.which("tesseract") is None:
            self.skipTest("tesseract is not installed")
        if _font(44) is None:
            self.skipTest("Pillow cannot load a scalable default font here")
        spread = Image.new("L", (2600, 1700), 255)
        spread.paste(self.page(self.LEFT, 1.5), (0, 0))
        spread.paste(self.page(self.RIGHT, -1.0), (1300, 0))
        sideways = spread.rotate(90, expand=True, fillcolor=255)    # the scan as stored
        parts = O.ocr_sheet(sideways)
        self.assertEqual([p[0] for p in parts], ["left", "right"])
        left = " ".join(parts[0][1]).lower()
        right = " ".join(parts[1][1]).lower()
        self.assertIn("differentiation of self", left)
        self.assertIn("emotional unit", left)
        self.assertNotIn("triangles", left)                # nothing from the other page
        self.assertIn("emotional cutoff", right)
        self.assertNotIn("differentiation", right)
        confs = parts[0][2] + parts[1][2]
        self.assertGreater(sum(confs) / len(confs), 80)


if __name__ == "__main__":
    unittest.main()
