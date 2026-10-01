# LEARNINGS.md — failure-pattern playbook

Purpose: turn individual bug fixes into a reusable theory of how this project fails, and check new code against it. Prime directive: make failure loud and make negatives provable. The fuller generic catalogue (P1–P36) is in `~/.claude/standards/learnings.md`; this file holds the portable core plus project specifics.

## Patterns
- **P1 Transient recorded as permanent negative.** Ask: is this "no" really "not right now"?
- **P2 Silent drop on failure.** Ask: if this fails mid-batch, would anyone know?
- **P3 Narrow-scope assumption.** Ask: what does looking in one place exclude?
- **P4 Hardcoded constant encoding an assumption.** Ask: should this be config?
- **P5 Inconsistent robustness across sibling calls.** Ask: are ALL calls of this kind hardened?
- **P6 Trusting a status field without verifying the artifact.** Ask: does it reflect something real now?
- **P7 Gameable or over-weighted scoring proxy.** Ask: what scores high for the wrong reason?

## Review checklist
1. Could any failure written as a terminal state actually succeed on retry (P1)?
2. Does any call return empty/None on failure with no log or count (P2)?
3. Have all sources/paths been enumerated, not just one (P3)?
4. Any literal date/threshold/domain word inside logic (P4)?
5. Do sibling external calls share the same hardening (P5)?
6. Is every status/counter verified against the thing it describes, including zero/empty values (P6)?
7. What input scores high for the wrong reason (P7)?

## Open risks
(none recorded)

## Misses
(none recorded)

## Fix log
Format: Issue -> Root cause (Pn) -> What would have caught it -> Fix -> Rule. Newest first.

- 2026-10-01 usage_limit.py (found by learning-qa pre-flight, fixed before commit)
  - Concurrent calls all passed the check before any recorded spend -> check and spend were separate steps (P6) -> an interleaved-calls test -> `begin()` reserves worst-case tokens under the lock, `finish()` reconciles -> reserve resources atomically with the check.
  - API usage of 0 was trusted and recorded nothing -> a status value taken at face value (P6) -> a zero-usage test -> zero usage for a call that produced text is estimated and logged -> verify counters against the activity they describe, including zero.
  - Cookie read failure swallowed, so limits silently reset (P2) -> a log warning plus a Streamlit version pin -> never `except: pass` on a control path.
  - Abandoned or failed streams were charged almost nothing -> the estimate ignored invisible reasoning tokens -> floor raised to 16,000 output tokens.
