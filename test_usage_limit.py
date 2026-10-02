"""Tests for usage_limit.py. Spec: docs/spec_usage_limits.md"""
import json
import logging
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import usage_limit as ul

U1 = "a" * 32
U2 = "b" * 32
FLOOR = ul._MIN_ESTIMATED_OUTPUT      # 16,000: the least any call is assumed to cost


class Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.clock = Clock()

    def make(self, per_user=300_000, glob=2_000_000, weights=None):
        return ul.UsageLimiter(self.dir, per_user, glob, now=self.clock, weights=weights)


class TestCaps(Base):
    def test_u1_user_refused_at_cap_but_other_user_allowed(self):
        lim = self.make()
        lim.record(U1, 300_000)
        with self.assertRaises(ul.UsageLimitError):
            lim.check(U1)
        lim.check(U2)   # a different visitor still has a budget

    def test_u1_just_under_cap_allowed(self):
        lim = self.make()
        lim.record(U1, 299_999)
        lim.check(U1)

    def test_u1_boundary_exactly_at_cap_refused_and_one_under_allowed(self):
        lim = self.make(per_user=1000, glob=10_000)
        lim.record(U1, 999)
        lim.check(U1)
        lim.record(U1, 1)
        with self.assertRaises(ul.UsageLimitError):
            lim.check(U1)

    def test_u2_global_cap_blocks_everyone(self):
        lim = self.make(per_user=1_000_000, glob=500_000)
        lim.record(U1, 500_000)
        with self.assertRaises(ul.UsageLimitError) as cm:
            lim.check(U2)
        self.assertIn("total daily capacity", str(cm.exception))

    def test_u2_global_boundary(self):
        lim = self.make(per_user=1_000_000, glob=1000)
        lim.record(U1, 999)
        lim.check(U2)
        lim.record(U1, 1)
        with self.assertRaises(ul.UsageLimitError):
            lim.check(U2)

    def test_u3_report_needs_minimum_budget(self):
        lim = self.make()
        lim.record(U1, 250_000)                   # 50,000 left
        lim.check(U1, min_budget=1)               # chat is fine
        with self.assertRaises(ul.UsageLimitError):
            lim.check(U1, min_budget=60_000)      # report is not

    def test_u3_report_checks_global_budget_too(self):
        lim = self.make(per_user=1_000_000, glob=100_000)
        lim.record(U2, 70_000)                    # 30,000 global left
        with self.assertRaises(ul.UsageLimitError):
            lim.check(U1, min_budget=60_000)

    def test_u1_invalid_uid_is_a_usage_error_not_a_crash(self):
        # The page catches UsageLimitError; a bare ValueError would show as "LLM error".
        with self.assertRaises(ul.UsageLimitError):
            self.make().check("../../etc/passwd")
        with self.assertRaises(ul.UsageLimitError):
            self.make().begin("", 100)


class TestOversizedCalls(Base):
    """U9: a call whose own size exceeds the remaining budget is refused up front."""

    def test_u9_call_larger_than_remaining_budget_is_refused(self):
        lim = self.make(per_user=300_000)
        lim.record(U1, 240_000)                   # 60,000 left; passes the old 60K floor
        big = 3 * 200_000                         # ~200K tokens: fits the daily cap, not 60K
        with self.assertRaises(ul.UsageLimitError) as cm:
            lim.begin(U1, big, min_budget=60_000)
        self.assertIn("needs about", str(cm.exception))
        self.assertEqual(lim.remaining(U1)[0], 60_000)     # nothing was reserved

    def test_u9_chat_with_a_nearly_empty_budget_cannot_send_a_huge_prompt(self):
        lim = self.make(per_user=300_000)
        lim.record(U1, 299_999)                   # 1 token left
        with self.assertRaises(ul.UsageLimitError):
            lim.begin(U1, 3 * 150_000, min_budget=1)

    def test_u9_request_larger_than_the_whole_daily_limit_says_so(self):
        lim = self.make(per_user=300_000)
        lim.record(U1, 1_000)
        with self.assertRaises(ul.UsageLimitError) as cm:
            lim.begin(U1, 3 * 400_000)            # ~400K tokens > the 300K cap
        self.assertIn("too large for the daily limit", str(cm.exception))
        self.assertIn("Chunks per source", str(cm.exception))

    def test_u9_global_budget_must_also_cover_the_call(self):
        lim = self.make(per_user=300_000, glob=100_000)
        lim.record(U2, 90_000)                    # 10,000 global left
        with self.assertRaises(ul.UsageLimitError) as cm:
            lim.begin(U1, 300)
        # only THIS request is too big for what is left: say so, not "capacity reached"
        self.assertIn("remaining daily capacity (10,000 tokens)", str(cm.exception))
        self.assertNotIn("has been reached", str(cm.exception))

    def test_u9_exhausted_global_budget_says_capacity_reached(self):
        lim = self.make(per_user=300_000, glob=100_000)
        lim.record(U2, 100_000)
        with self.assertRaises(ul.UsageLimitError) as cm:
            lim.begin(U1, 300)
        self.assertIn("has been reached", str(cm.exception))

    def test_u9_remaining_is_net_of_calls_in_flight(self):
        # The sidebar number must not overstate what a new call could use.
        lim = self.make(per_user=300_000, glob=2_000_000)
        res = lim.begin(U1, 300)
        user_rem, global_rem = lim.remaining(U1)
        self.assertEqual(user_rem, 300_000 - res.amount)
        self.assertEqual(global_rem, 2_000_000 - res.amount)
        lim.finish(res, 0)
        self.assertEqual(lim.remaining(U1), (300_000, 2_000_000))

    def test_u9_normal_call_is_admitted_and_reserved(self):
        lim = self.make(per_user=300_000)
        res = lim.begin(U1, 30_000)               # ~10K input tokens + 16K floor
        self.assertGreaterEqual(res.amount, FLOOR)
        lim.finish(res, 0)


class TestWeights(Base):
    """U10: one budget across providers; a pricier model counts for more."""

    def test_u10_weighted_usage_is_charged_at_weight(self):
        lim = self.make(weights=(10.0, {"deepseek-v4-flash": 1.0}))
        self.assertEqual(lim.weight("deepseek-v4-flash"), 1.0)
        self.assertEqual(lim.weight("claude-sonnet-4-6"), 10.0)
        res = lim.begin(U1, 300, weight=10.0)
        list(lim.meter(res, 300, iter(["x"]),
                       lambda: SimpleNamespace(input_tokens=1_000, output_tokens=500),
                       weight=10.0))
        self.assertEqual(lim.remaining(U1)[0], 300_000 - 15_000)

    def test_u10_reservation_scales_with_weight(self):
        lim = self.make()
        cheap = lim.begin(U1, 300, weight=1.0)
        lim.finish(cheap, 0)
        pricey = lim.begin(U1, 300, weight=10.0)
        self.assertGreater(pricey.amount, cheap.amount * 9)
        lim.finish(pricey, 0)

    def test_u10_pricey_model_runs_out_of_budget_sooner_for_the_same_text(self):
        # Adversarial: same visitor, same 100K-token input; a x10 model needs 10x the budget.
        lim = self.make(per_user=300_000)
        lim.begin(U1, 3 * 100_000, weight=1.0)     # fits
        lim2 = self.make(per_user=300_000)
        with self.assertRaises(ul.UsageLimitError):
            lim2.begin(U1, 3 * 100_000, weight=10.0)

    def test_u10_weights_file_loaded(self):
        f = self.dir / "w.yml"
        f.write_text("default_weight: 7\nweights:\n  m-flash: 1\n")
        self.assertEqual(ul.load_weights(f), (7.0, {"m-flash": 1.0}))

    def test_u10_missing_or_bad_weights_file_falls_back_pessimistically_and_logs(self):
        with self.assertLogs("usage_limit", level="ERROR"):
            self.assertEqual(ul.load_weights(self.dir / "nope.yml"),
                             (ul.FALLBACK_WEIGHT, {}))
        bad = self.dir / "bad.yml"
        bad.write_text("default_weight: 5\nweights:\n  m: -1\n")
        with self.assertLogs("usage_limit", level="ERROR"):
            self.assertEqual(ul.load_weights(bad), (ul.FALLBACK_WEIGHT, {}))
        junk = self.dir / "junk.yml"
        junk.write_text("weights: [not, a, mapping")
        with self.assertLogs("usage_limit", level="ERROR"):
            self.assertEqual(ul.load_weights(junk), (ul.FALLBACK_WEIGHT, {}))

    def test_u10_shipped_model_weights_file_is_valid(self):
        default, models = ul.load_weights(Path(__file__).resolve().parent / "model_weights.yml")
        self.assertEqual(models["deepseek-v4-flash"], 1.0)
        self.assertGreater(default, 1.0)       # unlisted models are over-counted, not free


class TestRecording(Base):
    def run_meter(self, lim, tokens, usage_fn, input_chars=300):
        res = lim.begin(U1, input_chars)
        return list(lim.meter(res, input_chars, iter(tokens), usage_fn))

    def test_u4_records_api_usage(self):
        lim = self.make()
        out = self.run_meter(lim, ["a", "b"],
                             lambda: SimpleNamespace(input_tokens=100, output_tokens=40))
        self.assertEqual(out, ["a", "b"])
        self.assertEqual(lim.remaining(U1)[0], 300_000 - 140)

    def test_u4_estimates_when_api_gives_no_usage(self):
        lim = self.make()

        def boom():
            raise AttributeError("no usage")

        with self.assertLogs("usage_limit", level="WARNING"):
            self.run_meter(lim, ["abc"] * 10, boom, input_chars=300)
        used = 300_000 - lim.remaining(U1)[0]
        self.assertGreaterEqual(used, 101 + FLOOR)   # input estimate + output floor

    def test_u4_abandoned_stream_is_charged(self):
        lim = self.make()
        gen = lim.meter(lim.begin(U1, 300), 300, iter(["x", "y", "z"]),
                        lambda: SimpleNamespace(input_tokens=1, output_tokens=1))
        next(gen)
        with self.assertLogs("usage_limit", level="WARNING"):
            gen.close()                              # user navigated away
        self.assertGreaterEqual(300_000 - lim.remaining(U1)[0], FLOOR)

    def test_u4_failure_before_output_records_nothing(self):
        lim = self.make()

        def failing():
            raise RuntimeError("401 bad key")
            yield "never"

        with self.assertRaises(RuntimeError):
            list(lim.meter(lim.begin(U1, 300), 300, failing(), lambda: None))
        self.assertEqual(lim.remaining(U1)[0], 300_000)

    def test_u4_failure_after_output_is_charged_by_estimate(self):
        lim = self.make()

        def partial():
            yield "some text"
            raise RuntimeError("connection dropped")

        with self.assertRaises(RuntimeError), self.assertLogs("usage_limit", "WARNING"):
            list(lim.meter(lim.begin(U1, 300), 300, partial(), lambda: None))
        self.assertLess(lim.remaining(U1)[0], 300_000)

    def test_u4_zero_usage_from_api_is_estimated_not_free(self):
        lim = self.make()
        with self.assertLogs("usage_limit", level="WARNING"):
            self.run_meter(lim, ["some text"],
                           lambda: SimpleNamespace(input_tokens=0, output_tokens=None))
        self.assertGreaterEqual(300_000 - lim.remaining(U1)[0], FLOOR)

    def test_u4_concurrent_calls_cannot_all_pass(self):
        # Calls begin before any finishes; reservations must stop the one that no
        # longer fits.
        lim = self.make(per_user=100_000, glob=1_000_000)
        a = lim.begin(U1, 300, min_budget=40_000)
        b = lim.begin(U1, 300, min_budget=40_000)
        with self.assertRaises(ul.UsageLimitError):
            lim.begin(U1, 300, min_budget=40_000)
        lim.finish(a, 1_000)       # real usage was small: the budget is available again
        c = lim.begin(U1, 300, min_budget=40_000)
        lim.finish(b, 0)
        lim.finish(c, 0)

    def test_u4_concurrent_reservations_count_against_global_cap(self):
        lim = self.make(per_user=100_000, glob=70_000)
        lim.begin(U1, 300, min_budget=40_000)
        with self.assertRaises(ul.UsageLimitError):
            lim.begin(U2, 300, min_budget=40_000)

    def test_u4_finish_is_idempotent_and_releases_failed_open(self):
        lim = self.make(per_user=100_000, glob=100_000)
        res = lim.begin(U1, 300, min_budget=90_000)
        lim.finish(res, 0)                  # stream never opened
        lim.finish(res, 5_000)              # second call must not charge
        self.assertEqual(lim.remaining(U1)[0], 100_000)
        lim.begin(U1, 300, min_budget=90_000)   # reservation was freed

    def test_u4_finish_charges_actual_tokens_once(self):
        lim = self.make(per_user=100_000)
        res = lim.begin(U1, 300)
        lim.finish(res, 2_500)
        lim.finish(res, 2_500)
        self.assertEqual(lim.remaining(U1)[0], 97_500)


class TestDayAndPersistence(Base):
    def test_u5_resets_at_utc_midnight(self):
        lim = self.make()
        lim.record(U1, 300_000)
        with self.assertRaises(ul.UsageLimitError):
            lim.check(U1)
        self.clock.t += timedelta(hours=12)           # next UTC day
        lim.check(U1)
        self.assertEqual(lim.remaining(U1), (300_000, 2_000_000))

    def test_u5_state_survives_restart_and_old_days_pruned(self):
        (self.dir / "usage-2026-09-30.json").write_text("{}")
        lim = self.make()
        lim.record(U1, 120_000)
        lim2 = self.make()                            # simulates a restart
        self.assertEqual(lim2.remaining(U1)[0], 180_000)
        self.assertFalse((self.dir / "usage-2026-09-30.json").exists())

    def test_u5_corrupt_file_is_moved_aside_and_logged(self):
        (self.dir / "usage-2026-10-01.json").write_text("{not json")
        lim = self.make()
        with self.assertLogs("usage_limit", level="ERROR"):
            self.assertEqual(lim.remaining(U1)[0], 300_000)
        self.assertTrue((self.dir / "usage-2026-10-01.corrupt").exists())

    def test_u5_wrong_shape_file_is_treated_as_corrupt(self):
        (self.dir / "usage-2026-10-01.json").write_text(json.dumps({"global": "x"}))
        with self.assertLogs("usage_limit", level="ERROR"):
            self.make().remaining(U1)

    def test_u5_unwritable_directory_still_counts_in_memory(self):
        blocker = self.dir / "file"
        blocker.write_text("x")                       # a file where a directory is needed
        lim = ul.UsageLimiter(blocker / "sub", 300_000, 2_000_000, now=self.clock)
        with self.assertLogs("usage_limit", level="ERROR"):
            lim.record(U1, 300_000)
        with self.assertRaises(ul.UsageLimitError):
            lim.check(U1)


class TestIdsAndConfig(unittest.TestCase):
    def test_u6_valid_uid_rejects_non_hex(self):
        self.assertTrue(ul.valid_uid(ul.new_uid()))
        for bad in ("", "a" * 31, "A" * 32, "g" * 32, "a" * 33, None, "../" + "a" * 29, 5):
            self.assertFalse(ul.valid_uid(bad), bad)

    def test_u6_valid_uid_rejects_trailing_newline(self):
        # re.match with "$" accepts "<32 hex>\n"; fullmatch must not.
        self.assertFalse(ul.valid_uid("a" * 32 + "\n"))

    def test_u1_caps_default_and_bad_env_values_fall_back(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ul.caps_from_env(), (300_000, 2_000_000, 60_000))
        with mock.patch.dict(os.environ, {"DAILY_TOKEN_CAP_PER_USER": "abc",
                                          "DAILY_TOKEN_CAP_GLOBAL": "-5"}, clear=True):
            with self.assertLogs("usage_limit", level="ERROR"):
                self.assertEqual(ul.caps_from_env()[:2], (300_000, 2_000_000))


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main()
