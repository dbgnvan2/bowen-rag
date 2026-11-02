"""authority.py — authority tiers and the rank-based authority boost.

Purpose: one implementation of the authority boost for both apps (streamlit_app.py and
bowen_rag_gui.py each keep `authority_boost(doc_name)` and call this module).
Spec:    docs/spec_exact_match_search_v2.md#E11 (option b, chosen by Dave 2026-11-01)

The boost used to multiply raw similarity scores. Scores from different documents are
compressed into a narrow band, so a weak match from a x3 document outranked a strong
match from an unboosted one: for "emotional programming" BM25 k=12 returned 0 chunks
containing the phrase. Now the boost divides the *rank* instead: a chunk at raw rank r
(1 = most relevant) with multiplier m is placed at r / m. A x3 chunk at raw rank 30
ties an unboosted chunk at raw rank 10, so the boost reorders comparable chunks but
cannot push the most relevant ones out of a top-k.

The tiers themselves are editorial content and live in authority_tiers.yml. The copy
in DEFAULT_TIERS is used only when that file is absent (the PyInstaller bundle does not
ship it); test_authority.py fails if the two differ.

Standard library only (PyYAML is imported lazily to read the file).
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

# Mirror of authority_tiers.yml (pattern = case-insensitive substring of the document
# name, regex = case-insensitive re.search; the first match wins).
DEFAULT_TIERS = [
    # Bowen (3.0)
    {"pattern": "Family Therapy in_Clinical_Practice_Chapter", "multiplier": 3.0},
    {"pattern": "Bowen Basic Series Tape", "multiplier": 3.0},
    {"pattern": "BOWEN-KERR INTERVIEW SERIES", "multiplier": 3.0},
    {"pattern": "Bowen on Triangles", "multiplier": 3.0},
    {"pattern": "Psychotherapy Past Present Future", "multiplier": 3.0},
    {"pattern": "Erosion of Theory", "multiplier": 3.0},
    {"pattern": "Erosion Letters", "multiplier": 3.0},
    {"pattern": "A Family Concept of Schizophrenia Bowen", "multiplier": 3.0},
    {"pattern": "Interview of Bowen by", "multiplier": 3.0},
    {"regex": r"^(FSJ|Copy of) [\d.]+\.?\s+(\(\d+\)\s+)?Bowen\b", "multiplier": 3.0},
    # Kerr (2.0)
    {"pattern": "Family Evaluation", "multiplier": 2.0},
    {"pattern": "Bowen Theory Secrets", "multiplier": 2.0},
    {"pattern": "Chronic Anxiety and Defining", "multiplier": 2.0},
    {"pattern": "Cancer and the Emotional System", "multiplier": 2.0},
    {"pattern": "Family and Society Kerr", "multiplier": 2.0},
    {"pattern": "Family as a System Kerr", "multiplier": 2.0},
    {"pattern": "Family Systems and Therapy Kerr", "multiplier": 2.0},
    {"pattern": "Physical Illness as the Family Emotional", "multiplier": 2.0},
    {"pattern": "- Michael Kerr -", "multiplier": 2.0},
    {"regex": r"^(FSJ|Copy of) [\d.]+\.?\s+(\(\d+\)\s+)?Kerr\b", "multiplier": 2.0},
    # Papero (2.0)
    {"pattern": "Bowen Family Systems Theory_", "multiplier": 2.0},
    {"pattern": "Papero", "multiplier": 2.0},
    # Family Systems Journal and Family Center Reports (1.3)
    {"pattern": "FSJ ", "multiplier": 1.3},
    {"pattern": "Copy of ", "multiplier": 1.3},
    {"pattern": "Family Center Reports", "multiplier": 1.3},
    # Other named theorists (1.15)
    {"pattern": "Friedman", "multiplier": 1.15},
    {"pattern": "Fogarty", "multiplier": 1.15},
    {"pattern": "Guerin", "multiplier": 1.15},
    {"pattern": "Toman", "multiplier": 1.15},
    {"pattern": "Gilbert", "multiplier": 1.15},
    {"pattern": "Harrison", "multiplier": 1.15},
]


class Tier(tuple):
    """(matcher text, multiplier, is_regex); a tuple so lists of tiers compare by value."""

    def __new__(cls, text: str, multiplier: float, is_regex: bool = False):
        return super().__new__(cls, (text, float(multiplier), bool(is_regex)))

    def matches(self, doc_name: str) -> bool:
        text, _, is_regex = self
        if is_regex:
            return re.search(text, doc_name, re.I) is not None
        return text.lower() in doc_name.lower()


def parse_tiers(entries: list) -> list[Tier]:
    """Tier objects from yml-style dicts. A malformed entry raises ValueError.

    Purpose: one parser for the yml file and DEFAULT_TIERS.
    Spec:    docs/spec_exact_match_search_v2.md#E11
    Tests:   test_authority.py::TestTiers::test_e11_default_tiers_equal_yml
    """
    out = []
    for n, t in enumerate(entries, 1):
        if not isinstance(t, dict) or ("pattern" in t) == ("regex" in t):
            raise ValueError(f"tier {n}: needs exactly one of 'pattern' or 'regex': {t!r}")
        mult = float(t["multiplier"])
        if mult < 1.0:
            raise ValueError(f"tier {n}: multiplier {mult} is below 1.0")
        if "regex" in t:
            re.compile(t["regex"])
            out.append(Tier(t["regex"], mult, True))
        else:
            out.append(Tier(t["pattern"], mult))
    return out


def load_tiers(path: Path) -> list[Tier]:
    """Tiers from authority_tiers.yml, or DEFAULT_TIERS when the file does not exist.

    A file that exists but cannot be read is reported on stderr (not silently replaced).
    """
    if not Path(path).exists():
        return parse_tiers(DEFAULT_TIERS)
    try:
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return parse_tiers(data.get("tiers", []))
    except Exception as e:
        print(f"WARNING: could not load {path} ({e}); using the built-in authority tiers.",
              file=sys.stderr)
        return parse_tiers(DEFAULT_TIERS)


def boost_for(doc_name: str, tiers: list[Tier]) -> float:
    """Multiplier of the first tier matching `doc_name`; 1.0 when none does."""
    return _boost_cached(doc_name, tuple(tiers))


@lru_cache(maxsize=4096)
def _boost_cached(doc_name: str, tiers: tuple) -> float:
    for t in tiers:
        if t.matches(doc_name):
            return t[1]
    return 1.0


def rank_order(raw, weight, limit: int | None = None) -> list[int]:
    """Indices of the positive `raw` scores, best first, with the authority boost on ranks.

    Purpose: apply the authority boost without letting it remove the most relevant items.
    Spec:    docs/spec_exact_match_search_v2.md#E11 (option b)
    Tests:   test_authority.py::TestRankOrder

    `raw[i]` is item i's unboosted relevance; `weight(i)` its multiplier (>= 1). Items are
    ranked by raw score (rank 1 = best; ties keep index order), then sorted by
    rank / weight, ties going to the better raw rank. With every weight 1.0 this is the
    raw order. Items with raw <= 0 are left out.
    """
    order = sorted((i for i in range(len(raw)) if raw[i] > 0), key=lambda i: -raw[i])
    keyed = sorted(range(len(order)), key=lambda r: ((r + 1) / weight(order[r]), r))
    out = [order[r] for r in keyed]
    return out if limit is None else out[:limit]
