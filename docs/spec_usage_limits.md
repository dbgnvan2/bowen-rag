# Spec: daily DeepSeek token limits

Status: approved in chat 2026-10-01 (per-visitor cookie method, 300K/day per visitor,
2M/day global). The app has no logins, so a "visitor" is a browser cookie. Clearing
cookies or using a private window gives a fresh per-visitor budget; the **global**
cap is the real spend backstop.

| ID | Requirement | Test |
|---|---|---|
| U1 | Each visitor may use at most `DAILY_TOKEN_CAP_PER_USER` (default 300,000) DeepSeek tokens (input + output, incl. reasoning) per UTC day. A call is refused before it starts once the visitor is at the cap. | `test_usage_limit.py::test_u1_*` |
| U2 | All visitors together may use at most `DAILY_TOKEN_CAP_GLOBAL` (default 2,000,000) per UTC day. A call is refused once the global cap is reached. | `test_u2_*` |
| U3 | A Report is refused unless both the visitor and the global budget have at least `REPORT_MIN_BUDGET` (default 60,000) tokens left. Chat only needs a non-empty budget. | `test_u3_*` |
| U4 | A call reserves worst-case tokens up front (so concurrent calls cannot all pass the check), and the reservation is released when it ends. Usage is recorded from the API's reported token counts. If the API gives none or reports zero for a call that produced text, or the stream is abandoned or fails mid-way, an estimate (input estimate + at least 16,000 output tokens) is recorded and a warning logged. A call that fails before producing output records 0. | `test_u4_*` |
| U5 | Counters reset at 00:00 UTC. State is kept in memory and persisted best-effort to `USAGE_DIR` (use a Railway volume; without one it resets on redeploy). A corrupt state file is moved aside and logged, not silently trusted. A write failure is logged and the in-memory counters keep working. | `test_u5_*` |
| U6 | The visitor id is a 32-hex string read from the `bowen_uid` cookie; anything else is ignored and a new id is issued. | `test_u6_*` |
| U7 | Only the DeepSeek provider is metered. | integration-only (no automated test; `_llm_stream` provider branch) |
| U8 | The sidebar shows the visitor's remaining tokens when DeepSeek is the provider. | integration-only (UI) |

Not covered by this change: stop-reason / truncation detection (review finding M6).
Known limits: reservations are an upper-bound guess, so a call can still overshoot the cap by the difference between the reservation and its real size; hence U3. Reasoning-token accounting depends on DeepSeek's Anthropic-
compatible endpoint reporting them in `output_tokens` (unverified).
