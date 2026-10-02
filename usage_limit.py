"""Daily token limits for the hosted app (per visitor and global).

No Streamlit dependency. The visitor id comes from a browser cookie, so a visitor can
reset their own budget by clearing cookies; the global cap is the spend backstop.
Spec: docs/spec_usage_limits.md
"""
import json
import logging
import math
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

COOKIE_NAME = "bowen_uid"
_UID_RE = re.compile(r"[0-9a-f]{32}")
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


# Pessimistic weight used for any model not listed in model_weights.yml, and for every
# model if that file cannot be read: over-counting is the safe direction for a spend cap.
FALLBACK_WEIGHT = 10.0


def load_weights(path) -> tuple:
    """Purpose: read per-model cost weights (a model's tokens count this many times
    against the daily budget). Returns (default_weight, {model: weight}).
    Spec:  docs/spec_usage_limits.md#U10
    Tests: test_usage_limit.py::test_u10_*

    A missing or unreadable file, or an invalid weight, is logged and falls back to the
    pessimistic FALLBACK_WEIGHT; it never makes a model cheaper than configured.
    """
    path = Path(path)
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        default = float(data.get("default_weight", FALLBACK_WEIGHT))
        models = {str(k): float(v) for k, v in (data.get("weights") or {}).items()}
        if default <= 0 or any(v <= 0 for v in models.values()):
            raise ValueError("weights must be positive")
        return default, models
    except Exception as e:
        log.error("Cannot use %s (%s); every model is weighted x%g against the daily "
                  "budget until it is fixed.", path, e, FALLBACK_WEIGHT)
        return FALLBACK_WEIGHT, {}


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
    return isinstance(value, str) and bool(_UID_RE.fullmatch(value))


def new_uid() -> str:
    return secrets.token_hex(16)


def estimate_tokens(chars: int) -> int:
    return chars // _CHARS_PER_TOKEN + 1


class UsageLimiter:
    def __init__(self, directory, per_user_cap: int, global_cap: int, now=None,
                 weights=None):
        self.dir = Path(directory)
        self.per_user_cap = per_user_cap
        self.global_cap = global_cap
        # (default_weight, {model: weight}); weight 1.0 everywhere if not given
        self._default_weight, self._weights = weights if weights else (1.0, {})
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
    def weight(self, model: str) -> float:
        """Cost weight for `model` (see load_weights)."""
        return self._weights.get(model, self._default_weight)

    def remaining(self, uid: str) -> tuple:
        """Return (visitor_remaining, global_remaining) for today, net of calls in flight."""
        with self._lock:
            self._roll()
            used = self._state["users"].get(uid, 0) + self._pending_user.get(uid, 0)
            return (max(0, self.per_user_cap - used),
                    max(0, self.global_cap - self._state["global"] - self._pending_global))

    def _refusal(self, need: int, user_rem: int, global_rem: int, est: int = 0):
        """Return the UsageLimitError to raise, or None if the budget is enough."""
        if global_rem < need:
            if global_rem > 0 and need > 1:
                return UsageLimitError(
                    f"The app's remaining daily capacity ({global_rem:,} tokens) is smaller "
                    f"than this request needs (about {need:,}). Try a smaller request, or "
                    "try again after 00:00 UTC.")
            return UsageLimitError(
                "The app's total daily capacity has been reached. "
                "Try again after 00:00 UTC.")
        if user_rem < need:
            if user_rem <= 0:
                return UsageLimitError(
                    f"You have used your daily limit of {self.per_user_cap:,} tokens. "
                    "It resets at 00:00 UTC.")
            if est > self.per_user_cap:
                return UsageLimitError(
                    f"This request is too large for the daily limit (about {est:,} tokens "
                    f"against {self.per_user_cap:,}). Lower 'Retrieve top N' or 'Chunks "
                    "per source', or pick a cheaper model.")
            return UsageLimitError(
                f"This request needs about {need:,} tokens and you have {user_rem:,} "
                "left today. Your allowance resets at 00:00 UTC.")
        return None

    def check(self, uid: str, min_budget: int = 1):
        """Purpose: refuse a call before it starts when a daily limit is reached.
        Spec:  docs/spec_usage_limits.md#U1 #U2 #U3
        Tests: test_usage_limit.py::test_u1_*, test_u2_*, test_u3_*
        """
        if not valid_uid(uid):
            raise UsageLimitError("Could not identify your browser session; reload the page.")
        user_rem, global_rem = self.remaining(uid)
        err = self._refusal(max(1, min_budget), user_rem, global_rem)
        if err:
            raise err

    def begin(self, uid: str, input_chars: int, min_budget: int = 1,
              weight: float = 1.0) -> Reservation:
        """Purpose: refuse a call that cannot fit in the remaining budget, and otherwise
        reserve its estimated cost in the same step, so neither an oversized call nor
        several concurrent calls can pass the check and then overspend.
        Spec:  docs/spec_usage_limits.md#U1 #U2 #U3 #U4 #U9
        Tests: test_usage_limit.py::test_u4_concurrent_calls_cannot_all_pass,
               test_u9_*

        The estimate is (input estimate + 16,000 output floor) x weight. A call is
        admitted only if the visitor AND the global budget both cover it (and at least
        `min_budget`); the reservation is that full amount.
        """
        if not valid_uid(uid):
            raise UsageLimitError("Could not identify your browser session; reload the page.")
        est = math.ceil((estimate_tokens(input_chars) + _MIN_ESTIMATED_OUTPUT) * weight)
        need = max(1, min_budget, est)
        with self._lock:
            self._roll()
            user_rem = max(0, self.per_user_cap - self._state["users"].get(uid, 0)
                           - self._pending_user.get(uid, 0))
            global_rem = max(0, self.global_cap - self._state["global"]
                             - self._pending_global)
            err = self._refusal(need, user_rem, global_rem, est)
            if err:
                raise err
            self._pending_user[uid] = self._pending_user.get(uid, 0) + need
            self._pending_global += need
            return Reservation(uid, need)

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

    def meter(self, res: Reservation, input_chars: int, tokens, usage_fn,
              weight: float = 1.0):
        """Purpose: pass tokens through and record the call's usage when it ends.
        Spec:  docs/spec_usage_limits.md#U4
        Tests: test_usage_limit.py::test_u4_*

        The recorded amount is raw tokens x `weight`.
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
            self.finish(res, math.ceil(total * weight))
