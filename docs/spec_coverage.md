# Spec coverage — Selectable Citation Styles

Spec: [`docs/spec_citation_styles.md`](spec_citation_styles.md). Every criterion below maps to
the file that satisfies it and the automated test that proves it.

Run the tests:

```bash
python3 test_citations.py && python3 test_seed_sources.py && python3 test_report_export.py
```

40 tests total (29 + 8 + 3), all passing.

| ID | Criterion | Implementation | Test | Status |
|---|---|---|---|---|
| M1.A | `sources.yml` schema loads; missing/invalid → `[]` | `citations.load_sources` | `test_m1a_load_sources_missing_and_valid` | done |
| M1.B | Seeder: author from map, year from filename, `verified:false`, won't clobber `sources.yml` | `seed_sources.py` | `test_m1b_*` (6) | done |
| M1.C | Pattern match; longest (most-specific) wins | `citations.match_source` | `test_m1c_match_source_first_wins` | done |
| M2.A | Reference entry exact per style (book, journal, 2-author) | `citations.format_reference` | `test_m2a_reference_*` (6) | done |
| M2.B | In-text per style; quote adds page; numbered → `[n]` | `citations.format_intext` | `test_m2b_intext_*` (3) | done |
| M2.C | Author styles alphabetical; Vancouver by number | `citations.order_references` | `test_m2c_*` (2) | done |
| M2.D | No year → `n.d.`; no record → filename fallback; nothing fabricated | `citations.synth_record` | `test_m2d_*` (3) | done |
| M3.A | Settings citation-style picker; default from `CITATION_STYLE` | `streamlit_app.py` Settings; `bowen_rag_gui.py` Report tab | UI (syntax + smoke) | done |
| M3.B | Prompt instructs `[[N]]` / `[[N, p. X]]` markers | both apps' report prompt | integration-only | done (code) |
| M3.C | Rewrite `[[N]]`/grouped; ignore single-bracket quotes, `[sic]`, `[…]`, years | `citations.apply_intext_citations` | `test_m3c_*` (7) | done |
| M3.D | Round-trip: body markers + reference list consistent | `citations.build_reference_list_md` | `test_m3d_*` (3) | done |
| M3.E | Page-on-quote; offered only when unambiguous; graceful fallback | both apps `doc_page`; `_page_locator` | integration-only | partial (integration-only) |
| M4.A | Streamlit report path | `streamlit_app.py` `page_report` | smoke on real data | done |
| M4.B | Desktop GUI report path | `bowen_rag_gui.py` `_generate_report`/`_work` | syntax + logic parity | done |
| M4.C | Styled report → `.docx`/`.pdf`, citations preserved | `md_to_docx_bytes`/`md_to_pdf_bytes` | `test_m4c_*` (3) | done |
| M4.D | Chosen style recorded | on-screen caption + report status line | UI | done (caption, not frontmatter) |
| M5 | Docs: USER_GUIDE, CLAUDE.md, .env.example | those files | — | done |

## Not unit-testable (flagged for human / integration review)

- **M3.B / M3.E — LLM marker behaviour.** Whether the model reliably emits `[[N]]` (and a
  correct `[[N, p. X]]` for quotes) is an LLM-behaviour path, not unit-deterministic. The code
  guarantees graceful degradation: an unrecognised/forgotten marker is left as-is rather than
  mis-rendered, and post-processing is guarded so a bad record can't crash the report. Needs one
  live report run per app to confirm end-to-end.
- **Seed data accuracy.** Auto-seeded author/year/title are guesses (`verified: false`) until a
  human confirms them. Coverage is surfaced in Settings and in each report's status line.
- **Live UI rendering** (Streamlit placeholder streaming; tkinter widget replacement) — logic
  verified, visual rendering needs a live run.
