# Spec: daily token limits (all server-keyed providers)

Status: approved in chat 2026-10-01 (per-visitor cookie method, 300K/day per visitor,
2M/day global). The app has no logins, so a "visitor" is a browser cookie. Clearing
cookies or using a private window gives a fresh per-visitor budget; the **global**
cap is the real spend backstop.

| ID | Requirement | Test |
|---|---|---|
| U1 | Each visitor may use at most `DAILY_TOKEN_CAP_PER_USER` (default 300,000) weighted tokens (input + output, incl. reasoning; see U10) per UTC day. A call is refused before it starts once the visitor is at the cap. | `test_usage_limit.py::test_u1_*` |
| U2 | All visitors together may use at most `DAILY_TOKEN_CAP_GLOBAL` (default 2,000,000) per UTC day. A call is refused once the global cap is reached. | `test_u2_*` |
| U3 | A Report is refused unless both the visitor and the global budget have at least `REPORT_MIN_BUDGET` (default 60,000) tokens left. Chat only needs a non-empty budget. | `test_u3_*` |
| U4 | A call reserves worst-case tokens up front (so concurrent calls cannot all pass the check), and the reservation is released when it ends. Usage is recorded from the API's reported token counts. If the API gives none or reports zero for a call that produced text, or the stream is abandoned or fails mid-way, an estimate (input estimate + at least 16,000 output tokens) is recorded and a warning logged. A call that fails before producing output records 0. | `test_u4_*` |
| U5 | Counters reset at 00:00 UTC. State is kept in memory and persisted best-effort to `USAGE_DIR` (use a Railway volume; without one it resets on redeploy). A corrupt state file is moved aside and logged, not silently trusted. A write failure is logged and the in-memory counters keep working. | `test_u5_*` |
| U6 | The visitor id is a 32-hex string (`fullmatch`, so a trailing newline is rejected) read from the `bowen_uid` cookie; anything else is ignored and a new id is issued. A cookie that cannot be read is logged. | `test_usage_limit.py::test_u6_*`, `test_llm_stream_limits.py::test_u6_*` |
| U7 | Claude, OpenAI and DeepSeek calls (which use the operator's server-side keys) are all metered; Ollama (self-hosted) is not. | `test_llm_stream_limits.py::test_u7_*` |
| U9 | A call is refused before any API request when its estimated cost ((input estimate + 16,000 output floor) × weight) exceeds the visitor's or the global remaining budget, or `REPORT_MIN_BUDGET`. Nothing is reserved when refused. A request larger than the whole daily limit says so and names the Report settings to lower. | `test_u9_*`, `test_llm_stream_limits.py::test_u9_*` |
| U10 | One budget covers every provider: a model's tokens are multiplied by its weight from `model_weights.yml` (deepseek-v4-flash = 1; unlisted models use `default_weight`, 10). A missing or invalid weights file is logged and every model is weighted ×10. | `test_u10_*`, `test_llm_stream_limits.py::test_u10_*` |
| U8 | The sidebar shows the visitor's remaining budget (and the model's weight when not 1) for every metered provider. | integration-only (UI; checked by hand in the browser) |

Not covered by this change: stop-reason / truncation detection (review finding M6).
Known limits:
- The cookie is not signed and one visitor can mint any number of ids by dropping cookies, so the per-visitor limit is a speed bump. Roughly 7 fresh ids reach the 2M global cap and lock out other visitors until 00:00 UTC. Signing the cookie would not change that; limiting new ids per IP would, but is not built (the client IP header on Railway has not been verified).
- The weights in `model_weights.yml` are placeholders, not prices: set them from the providers' price lists. With the default weight of 10 and a 300K daily limit, a call on any model other than deepseek-v4-flash is refused unless about 160K weighted tokens remain, and a full Report on such a model never fits.
- The reservation is an estimate (3 characters per token, pessimistic), so a call can overshoot by the difference between the estimate and its real size. Reasoning-token accounting depends on DeepSeek's Anthropic-
compatible endpoint reporting them in `output_tokens` (unverified).
