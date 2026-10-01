"""Daily token limits for the hosted app (per visitor and global).

No Streamlit dependency. The visitor id comes from a browser cookie, so a visitor can
reset their own budget by clearing cookies; the global cap is the spend backstop.
Spec: docs/spec_usage_limits.md
"""
import json
import logging
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

COOKIE_NAME = "bowen_uid"
_UID_RE = re.compile(r"^[0-9a-f]{32}$")
# Used only when the API reports no usage: roughly 3 characters per token is
# deliberately pessimistic for English text.
_CHARS_PER_TOKEN = 3
# Reasoning tokens are invisible to us on an abandoned or failed stream, so charge at
# least this much output; it is also what a new call reserves for itself (U4).
_MIN_ESTIMATED_OUTPUT = 16_000


class Reservation:
    """Tokens set aside for an in-flight call; released exactly once by `finish`."""

    def __init__(self, uid: str, amount: int):
        self.uid = uid
        self.amount = amount
        self.done = False


class UsageLimitError(RuntimeError):
    """Raised before an LLM call when a daily limit is reached."""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        value = int(raw) if raw else default
    except ValueError:
        log.error("%s=%r is not an integer; using %d", name, raw, default)
        return default
    return value if value > 0 else default


def caps_from_env() -> tuple:
    """Return (per_user_cap, global_cap, report_min_budget) from the environment."""
    return (
        _env_int("DAILY_TOKEN_CAP_PER_USER", 300_000),
        _env_int("DAILY_TOKEN_CAP_GLOBAL", 2_000_000),
        _env_int("REPORT_MIN_BUDGET", 60_000),
    )


def valid_uid(value) -> bool:
    """Purpose: accept only a 32-char lowercase hex visitor id.
    Spec:  docs/spec_usage_limits.md#U6
    Tests: test_usage_limit.py::test_u6_valid_uid_rejects_non_hex
    """
    return isinstance(value, str) and bool(_UID_RE.match(value))


def new_uid() -> str:
    return secrets.token_hex(16)


def estimate_tokens(chars: int) -> int:
    return chars // _CHARS_PER_TOKEN + 1


class UsageLimiter:
    def __init__(self, directory, per_user_cap: int, global_cap: int, now=None):
        self.dir = Path(directory)
        self.per_user_cap = per_user_cap
        self.global_cap = global_cap
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._day = None
        self._state = {"global": 0, "users": {}}
        self._pending_user = {}     # tokens reserved by in-flight calls, per visitor
        self._pending_global = 0

    # ── state ────────────────────────────────────────────────────────────────
    def _path(self, day: str) -> Path:
        return self.dir / f"usage-{day}.json"

    def _roll(self):
        """Load today's state if the UTC day changed. Caller holds the lock."""
        day = self._now().strftime("%Y-%m-%d")
        if day == self._day:
            return
        self._day = day
        self._state = {"global": 0, "users": {}}
        path = self._path(day)
        try:
            if path.exists():
                data = json.loads(path.read_text())
                users = {k: int(v) for k, v in data["users"].items()}
                self._state = {"global": int(data["global"]), "users": users}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            log.error("Usage file %s is unreadable (%s); moving it aside and "
                      "restarting today's counters from zero.", path, e)
            try:
                path.replace(path.with_suffix(".corrupt"))
            except OSError:
                pass
        self._prune(day)

    def _prune(self, today: str):
        try:
            for f in self.dir.glob("usage-*.json"):
                if f.stem != f"usage-{today}":
                    f.unlink(missing_ok=True)
        except OSError:
            pass

    def _persist(self):
        path = self._path(self._day)
        tmp = path.with_suffix(".tmp")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self._state))
            os.replace(tmp, path)
        except OSError as e:
            log.error("Could not persist usage to %s (%s); counters are kept in "
                      "memory only and will reset on restart.", path, e)

    # ── public API ───────────────────────────────────────────────────────────
    def remaining(self, uid: str) -> tuple:
        """Return (visitor_remaining, global_remaining) for today."""
        with self._lock:
            self._roll()
            used = self._state["users"].get(uid, 0)
            return (max(0, self.per_user_cap - used),
                    max(0, self.global_cap - self._state["global"]))

    def _refusal(self, need: int, user_rem: int, global_rem: int):
        """Return the UsageLimitError to raise, or None if the budget is enough."""
        if global_rem < need:
            return UsageLimitError(
                "The app's total daily capacity has been reached. "
                "Try again after 00:00 UTC.")
        if user_rem < need:
            if need > 1 and user_rem > 0:
                return UsageLimitError(
                    f"A report needs about {need:,} tokens and you have {user_rem:,} "
                    "left today. Your allowance resets at 00:00 UTC.")
            return UsageLimitError(
                f"You have used your daily limit of {self.per_user_cap:,} tokens. "
                "It resets at 00:00 UTC.")
        return None

    def check(self, uid: str, min_budget: int = 1):
        """Purpose: refuse a call before it starts when a daily limit is reached.
        Spec:  docs/spec_usage_limits.md#U1 #U2 #U3
        Tests: test_usage_limit.py::test_u1_*, test_u2_*, test_u3_*
        """
        if not valid_uid(uid):
            raise ValueError("invalid visitor id")
        user_rem, global_rem = self.remaining(uid)
        err = self._refusal(max(1, min_budget), user_rem, global_rem)
        if err:
            raise err

    def begin(self, uid: str, input_chars: int, min_budget: int = 1) -> Reservation:
        """Purpose: check the budget and reserve worst-case tokens for this call in one
        step, so concurrent calls cannot all pass the check before any is recorded.
        Spec:  docs/spec_usage_limits.md#U1 #U2 #U3 #U4
        Tests: test_usage_limit.py::test_u4_concurrent_calls_cannot_all_pass
        """
        if not valid_uid(uid):
            raise ValueError("invalid visitor id")
        need = max(1, min_budget)
        with self._lock:
            self._roll()
            user_rem = max(0, self.per_user_cap - self._state["users"].get(uid, 0)
                           - self._pending_user.get(uid, 0))
            global_rem = max(0, self.global_cap - self._state["global"]
                             - self._pending_global)
            err = self._refusal(need, user_rem, global_rem)
            if err:
                raise err
            amount = min(max(need, estimate_tokens(input_chars) + _MIN_ESTIMATED_OUTPUT),
                         user_rem, global_rem)
            self._pending_user[uid] = self._pending_user.get(uid, 0) + amount
            self._pending_global += amount
            return Reservation(uid, amount)

    def finish(self, res: Reservation, tokens: int):
        """Release the reservation and record the actual tokens. Safe to call twice."""
        with self._lock:
            if res.done:
                return
            res.done = True
            self._pending_user[res.uid] = max(0, self._pending_user.get(res.uid, 0)
                                              - res.amount)
            self._pending_global = max(0, self._pending_global - res.amount)
            if tokens > 0:
                self._roll()
                self._state["users"][res.uid] = (
                    self._state["users"].get(res.uid, 0) + tokens)
                self._state["global"] += tokens
                self._persist()

    def record(self, uid: str, tokens: int):
        self.finish(Reservation(uid, 0), tokens)

    def meter(self, res: Reservation, input_chars: int, tokens, usage_fn):
        """Purpose: pass tokens through and record the call's usage when it ends.
        Spec:  docs/spec_usage_limits.md#U4
        Tests: test_usage_limit.py::test_u4_*

        `usage_fn()` returns an object with input_tokens/output_tokens. If it is
        unavailable or reports zero for a call that produced text, or the consumer
        abandons the stream, an estimate is recorded. A failure before any output
        records nothing.
        """
        yielded = 0
        total = None
        try:
            for tok in tokens:
                yielded += len(tok)
                yield tok
            try:
                u = usage_fn()
                total = int(u.input_tokens or 0) + int(u.output_tokens or 0)
            except Exception as e:
                log.warning("No usage from API (%s); recording an estimate.", e)
            if total is not None and total <= 0 and yielded > 0:
                log.warning("API reported zero usage for a call that produced text; "
                            "recording an estimate.")
                total = None
        except Exception:
            if yielded == 0:
                total = 0
            raise
        finally:
            if total is None:
                total = (estimate_tokens(input_chars)
                         + max(estimate_tokens(yielded), _MIN_ESTIMATED_OUTPUT))
                log.warning("Recorded estimated usage of %d tokens.", total)
            self.finish(res, total)
