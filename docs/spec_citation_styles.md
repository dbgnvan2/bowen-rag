# Spec — Selectable Citation Styles

**Status:** awaiting approval (no code written yet)
**Date:** 2026-07-30
**Owner:** Dave Galloway

## Goal

Let the user pick a citation style in Settings. The Report then (a) lists its References in
that style and (b) renders the in-text citation at each cited point/quote in that style.

Five styles: **APA (7th), MLA (9th), Chicago (17th, author-date), Harvard (Cite Them Right),
Vancouver (numbered).** APA is the default.

## Core design principle — no fabricated bibliographic data

The index stores only `doc_name` and `page`. Full references need author/year/title/publisher,
which do not exist in the index. Per the app's no-fabrication rule:

- The **LLM never writes author names, years, or titles.** It keeps emitting numbered
  placeholders `[1]`, `[2]` exactly as today (optionally `[1, p. 45]` when quoting).
- A **deterministic post-processor** rewrites those placeholders into the chosen style and
  builds the reference list, reading **only** from a hand-verifiable bibliography file
  (`sources.yml`). Missing fields render with each style's real "unknown" convention
  (APA `n.d.`), never invented.
- Sources with no bibliography record fall back to a filename-derived title and are **counted
  and surfaced** (not silently normalised).

This extends the existing "references appended programmatically = single source of truth"
design (`streamlit_app.py:1520`).

---

## M1 — Bibliography data

### M1.A — `sources.yml` schema
Pattern-matched records (same first-match-wins substring engine as `author_map.yml`):
```yaml
sources:
  - pattern: "Family Therapy in_Clinical_Practice"   # substring of doc_name
    type: book                # book|chapter|article-journal|webpage|speech|interview|report|manuscript
    authors:
      - {family: "Bowen", given: "Murray"}
    year: 1978                # int, or "n.d."
    title: "Family Therapy in Clinical Practice"
    container_title: ""       # journal/book title for chapter|article
    publisher: "Jason Aronson"
    publisher_place: "New York"
    volume: ""
    issue: ""
    page: ""                  # page range of the WORK (distinct from chunk page)
    editor: ""
    url: ""
    doi: ""
    verified: true            # false = auto-seeded guess, not yet confirmed
```

### M1.B — Seeder script `scripts/seed_sources.py`
- Enumerates distinct `doc_name`s from `chunk_metadata.json`.
- Fills `authors` from `author_map.yml`; `year` by regex over the filename
  (`\b(19|20)\d{2}\b` and `YYYY-MM-DD`); `title` from `authority_tiers.yml` `note` when present,
  else a cleaned filename (strip leading `NNN_`, underscores→spaces).
- Everything filled is `verified: false`; unknown fields left blank.
- **Writes to `sources.seeded.yml`; refuses to overwrite an existing `sources.yml`.**
- Prints coverage: "seeded N records; author known for X, year for Y".

### M1.C — Loader/matcher `citations.load_sources()` / `match_source(doc_name)`
- Cached load of `sources.yml` (falls back to empty list if absent → all sources use filename fallback).
- `match_source` returns the first record whose `pattern` is a case-insensitive substring of `doc_name`, else `None`.

---

## M2 — Citation formatters (`citations.py`, shared by both apps)

Pure, deterministic functions. Hand-rolled (no `citeproc` dependency) — implements a pragmatic
subset correct for the common source types (book, chapter, journal article, lecture/tape,
interview), with a generic fallback for others. Multi-author rule simplified to: 1 → full;
2 → both; 3+ → first + "et al." (exact per-style thresholds documented as a known simplification).

- **M2.A `format_reference(record, style)`** → one reference-list entry string.
- **M2.B `format_intext(record, style, page=None, is_quote=False, number=None)`** →
  numbered styles return `[n]`; author-date return `(Family, Year)` / `(Family, Year, p. X)`
  for quotes; MLA returns `(Family Page)`.
- **M2.C `order_references(records, style)`** → appearance order for Vancouver; alphabetical by
  first author family for the other four.
- **M2.D fallbacks** → no `year` → `n.d.`; no record → filename title + `n.d.`, flagged.

Style reference/in-text targets (verified in tests as golden strings):

| Style | In-text (quote) | Reference entry | List order |
|---|---|---|---|
| APA | (Bowen, 1978, p. 45) | Bowen, M. (1978). *Family therapy in clinical practice*. Jason Aronson. | alpha |
| MLA | (Bowen 45) | Bowen, Murray. *Family Therapy in Clinical Practice*. Jason Aronson, 1978. | alpha |
| Chicago | (Bowen 1978, 45) | Bowen, Murray. 1978. *Family Therapy in Clinical Practice*. New York: Jason Aronson. | alpha |
| Harvard | (Bowen, 1978, p. 45) | Bowen, M. (1978) *Family therapy in clinical practice*. New York: Jason Aronson. | alpha |
| Vancouver | [1] | 1. Bowen M. Family therapy in clinical practice. New York: Jason Aronson; 1978. | appearance |

---

## M3 — Report integration

- **M3.A** Settings "Citation style" selectbox (APA default); persisted to `st.session_state`
  and savable to `.env` as `CITATION_STYLE`.
- **M3.B** Report prompt: keep numbered `[N]` markers; add one line telling the model to append
  a page token `[N, p. X]` **only** when quoting a specific passage (page taken from the supplied chunk).
- **M3.C** Post-processor rewrites markers. Regex matches **only** `[digits]` / `[digits, p. …]`
  — never arbitrary brackets like `[sic]` or `[…]`. Numbered styles leave `[N]` intact.
- **M3.D** Reference list built via `format_reference` + `order_references`; replaces `refs_md`.
  Author-date lists are alphabetical, so a `[1]`→`(Bowen, 1978)` map is used in-text while the
  list is sorted separately.
- **M3.E** Page-on-quote: `[N, p. X]` → `(…, p. X)`. Missing/garbled token degrades to `(…)`.
  **Integration-only path (LLM behaviour) — flagged untested at unit level; rendering never breaks.**

---

## M4 — Two apps + exports

- **M4.A** Streamlit report path (primary — the deployed Railway app).
- **M4.B** Desktop GUI report path (shared `citations.py`; thin UI wiring only).
- **M4.C** DOCX/PDF/Appendix: reference list and in-text are plain markdown/text → flow through
  existing `md_to_docx_bytes` / `md_to_pdf_bytes` unchanged; a round-trip test confirms.
- **M4.D** Report YAML frontmatter records the citation style used.

## M5 — Docs

`USER_GUIDE.md` (new Settings option + behaviour), `CLAUDE.md` (architecture + `sources.yml`),
`.env.example` (`CITATION_STYLE`, and the pre-existing `APP_PASSWORD` gap noted earlier).

---

## Acceptance criteria → tests

| ID | Criterion | Test |
|---|---|---|
| M1.A | Schema loads; bad/empty file → empty list, no crash | `test_m1a_load_sources_empty_and_valid` |
| M1.B | Seeder fills author from map, year from filename date, marks `verified:false`, won't clobber | `test_m1b_seed_sources_from_fixtures` |
| M1.C | First-substring-match wins; no match → None | `test_m1c_match_source_first_wins` |
| M2.A | Each of 5 styles → exact golden reference string (book, journal, lecture) | `test_m2a_reference_<style>` ×5 |
| M2.B | In-text per style; quote adds page; numbered → `[n]` | `test_m2b_intext_<style>` ×5 |
| M2.C | Author-date alphabetical; Vancouver appearance order | `test_m2c_ordering` |
| M2.D | No year → `n.d.`; no record → filename fallback **and flagged** (adversarial: nothing invented) | `test_m2d_missing_data_not_fabricated` |
| M3.C | `[1]`/`[1, p.45]` rewritten; `[sic]` and `[…]` left untouched (adversarial) | `test_m3c_marker_rewrite_ignores_nonmarkers` |
| M3.D | Round-trip: body with markers + N sources → correct in-text + list (P19 producer/consumer) | `test_m3d_report_roundtrip_apa_vancouver` |
| M4.C | Styled report → docx/pdf bytes without error; refs present | `test_m4c_export_with_citations` |
| M3.A/M4.D | Saved `CITATION_STYLE` loads; frontmatter records it | `test_m3a_style_persist_and_frontmatter` |

**Not unit-testable (flagged for human/integration review):**
- **M3.E** — the LLM actually emitting correct `[N, p. X]` tokens for quotes. Verified by a manual
  report run; code guarantees graceful fallback, not LLM correctness.
- **Seed data accuracy** — auto-seeded author/year guesses need human verification per source
  (`verified: false` until confirmed). Coverage surfaced in the Index page.

## Implementation order (dependencies)

1. M1.A schema + M1.C loader → M2 formatters (pure, fully testable first)
2. M2 tests (golden strings) — highest-regression-risk, written FIRST per testing rules
3. M1.B seeder (needs schema) → run once → hand-verify a first batch
4. M3 report integration (needs M2) — Streamlit first
5. M4.C export round-trip; M4.B GUI wiring
6. M5 docs + `learning-qa` review pass before commit

## Implementation notes & review outcomes (deltas from the plan)

Built as planned, with these refinements — several from the `learning-qa` pre-merge review:

- **Citation markers are DOUBLE brackets `[[N]]`** (not `[N]`). A single-bracket `[1]`
  can appear inside quoted source text — the corpus has 37 such footnote numbers — and a
  single-bracket scheme would rewrite them into wrong citations inside a verbatim quote
  (a fidelity bug). Double brackets can't collide; a quoted `[1]` passes through untouched.
  (Review #2, P7.) Verified by `test_m3c_single_brackets_in_quotes_survive`.
- **Grouped citations** `[[1, 3]]` are expanded and joined per style — APA `(Bowen, 1978;
  Kerr, 1988)`, Vancouver `[1, 3]`. Without this, grouped cites stayed numeric in an
  author-date report and silently dropped a source from the reference list. (Review #1,
  P19/P2.) Tests: `test_m3c_grouped_citation_*`, `test_m3d_cited_numbers_includes_grouped`.
- **Page locator only when unambiguous** — a page is offered to the model only when a
  document's retrieved chunks resolve to a single page; multi-page docs get no locator
  rather than a confidently-wrong one. (Review #3, P4/P6.)
- **Fallback/verified visibility** — both report paths show `X/M cited sources verified`
  so a reader can see how much bibliographic data is confirmed. (Review #4, P2.)
- **Post-processing is guarded** in both apps (a bad human-edited `sources.yml` record
  degrades to plain numbered references instead of crashing the page / losing the report).
  (Review #5, P5/P13.)
- **M4.D** — style is recorded via an on-screen caption + report status line (Streamlit adds
  no YAML frontmatter today; adding it would leak into docx/pdf), not frontmatter.
- **`match_source` uses longest (most-specific) pattern**, order-independent; the seeder's
  generated header documents this. (Review #6, P6.)
- **Loud-zero guard** — the double-bracket design depends on the model emitting `[[N]]`. If a
  generation reverts entirely to single `[1]`, `cited_numbers` is empty on a non-trivial body;
  both apps now detect this and show a visible warning ("no `[[N]]` markers found — citations
  unstyled, all sources listed, regenerate") instead of silently dumping all sources uncurated.
  (Re-review residual, P19 loud-zero-from-non-empty.) Signal tested by
  `test_m3d_single_bracket_body_yields_no_cited`.

## Open decisions (defaults chosen; override in approval)

- **Chicago = author-date** (parenthetical), consistent with the others. Notes-bibliography
  (footnotes) is a different in-text mechanism and a bigger lift — excluded unless requested.
- **Harvard = "Cite Them Right"** variant (one of several institutional variants).
- **Both apps** — CONFIRMED (Streamlit + desktop GUI). Shared `citations.py`; per-app UI wiring
  in each report path.
