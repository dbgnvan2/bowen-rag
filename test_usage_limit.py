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

    def make(self, per_user=300, glob=2000):
        return ul.UsageLimiter(self.dir, per_user, glob, now=self.clock)


class TestCaps(Base):
    def test_u1_user_refused_at_cap_but_other_user_allowed(self):
        lim = self.make()
        lim.record(U1, 300)
        with self.assertRaises(ul.UsageLimitError):
            lim.check(U1)
        lim.check(U2)   # a different visitor still has a budget

    def test_u1_just_under_cap_allowed(self):
        lim = self.make()
        lim.record(U1, 299)
        lim.check(U1)

    def test_u2_global_cap_blocks_everyone(self):
        lim = self.make(per_user=1000, glob=500)
        lim.record(U1, 500)
        with self.assertRaises(ul.UsageLimitError) as cm:
            lim.check(U2)
        self.assertIn("total daily capacity", str(cm.exception))

    def test_u3_report_needs_minimum_budget(self):
        lim = self.make(per_user=300, glob=2000)
        lim.record(U1, 250)                       # 50 left
        lim.check(U1, min_budget=1)               # chat is fine
        with self.assertRaises(ul.UsageLimitError):
            lim.check(U1, min_budget=60)          # report is not

    def test_u3_report_checks_global_budget_too(self):
        lim = self.make(per_user=1000, glob=100)
        lim.record(U2, 70)                        # 30 global left
        with self.assertRaises(ul.UsageLimitError):
            lim.check(U1, min_budget=60)

    def test_u1_invalid_uid_rejected(self):
        with self.assertRaises(ValueError):
            self.make().check("../../etc/passwd")


class TestRecording(Base):
    def run_meter(self, lim, tokens, usage_fn, input_chars=300):
        res = lim.begin(U1, input_chars)
        return list(lim.meter(res, input_chars, iter(tokens), usage_fn))

    def test_u4_records_api_usage(self):
        lim = self.make(per_user=100_000)
        out = self.run_meter(lim, ["a", "b"],
                             lambda: SimpleNamespace(input_tokens=100, output_tokens=40))
        self.assertEqual(out, ["a", "b"])
        self.assertEqual(lim.remaining(U1)[0], 100_000 - 140)

    def test_u4_estimates_when_api_gives_no_usage(self):
        lim = self.make(per_user=100_000)

        def boom():
            raise AttributeError("no usage")

        with self.assertLogs("usage_limit", level="WARNING"):
            self.run_meter(lim, ["abc"] * 10, boom, input_chars=300)
        used = 100_000 - lim.remaining(U1)[0]
        self.assertGreaterEqual(used, 101 + 16_000)   # input estimate + output floor

    def test_u4_abandoned_stream_is_charged(self):
        lim = self.make(per_user=100_000)
        gen = lim.meter(lim.begin(U1, 300), 300, iter(["x", "y", "z"]),
                        lambda: SimpleNamespace(input_tokens=1, output_tokens=1))
        next(gen)
        with self.assertLogs("usage_limit", level="WARNING"):
            gen.close()                              # user navigated away
        self.assertGreaterEqual(100_000 - lim.remaining(U1)[0], 16_000)

    def test_u4_failure_before_output_records_nothing(self):
        lim = self.make(per_user=100_000)

        def failing():
            raise RuntimeError("401 bad key")
            yield "never"

        with self.assertRaises(RuntimeError):
            list(lim.meter(lim.begin(U1, 300), 300, failing(), lambda: None))
        self.assertEqual(lim.remaining(U1)[0], 100_000)

    def test_u4_failure_after_output_is_charged_by_estimate(self):
        lim = self.make(per_user=100_000)

        def partial():
            yield "some text"
            raise RuntimeError("connection dropped")

        with self.assertRaises(RuntimeError), self.assertLogs("usage_limit", "WARNING"):
            list(lim.meter(lim.begin(U1, 300), 300, partial(), lambda: None))
        self.assertLess(lim.remaining(U1)[0], 100_000)

    def test_u4_zero_usage_from_api_is_estimated_not_free(self):
        lim = self.make(per_user=100_000)
        with self.assertLogs("usage_limit", level="WARNING"):
            self.run_meter(lim, ["some text"],
                           lambda: SimpleNamespace(input_tokens=0, output_tokens=None))
        self.assertGreaterEqual(100_000 - lim.remaining(U1)[0], 16_000)

    def test_u4_concurrent_calls_cannot_all_pass(self):
        # Three calls begin before any finishes; the reservation must stop the third.
        lim = self.make(per_user=100_000, glob=100_000)
        a = lim.begin(U1, 300, min_budget=40_000)
        b = lim.begin(U1, 300, min_budget=40_000)
        with self.assertRaises(ul.UsageLimitError):
            lim.begin(U1, 300, min_budget=40_000)
        # once one finishes with small real usage, the budget is available again
        lim.finish(a, 1_000)
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
        lim.record(U1, 300)
        with self.assertRaises(ul.UsageLimitError):
            lim.check(U1)
        self.clock.t += timedelta(hours=12)           # next UTC day
        lim.check(U1)
        self.assertEqual(lim.remaining(U1), (300, 2000))

    def test_u5_state_survives_restart_and_old_days_pruned(self):
        (self.dir / "usage-2026-09-30.json").write_text("{}")
        lim = self.make()
        lim.record(U1, 120)
        lim2 = self.make()                            # simulates a restart
        self.assertEqual(lim2.remaining(U1)[0], 180)
        self.assertFalse((self.dir / "usage-2026-09-30.json").exists())

    def test_u5_corrupt_file_is_moved_aside_and_logged(self):
        (self.dir / "usage-2026-10-01.json").write_text("{not json")
        lim = self.make()
        with self.assertLogs("usage_limit", level="ERROR"):
            self.assertEqual(lim.remaining(U1)[0], 300)
        self.assertTrue((self.dir / "usage-2026-10-01.corrupt").exists())

    def test_u5_wrong_shape_file_is_treated_as_corrupt(self):
        (self.dir / "usage-2026-10-01.json").write_text(json.dumps({"global": "x"}))
        with self.assertLogs("usage_limit", level="ERROR"):
            self.make().remaining(U1)

    def test_u5_unwritable_directory_still_counts_in_memory(self):
        blocker = self.dir / "file"
        blocker.write_text("x")                       # a file where a directory is needed
        lim = ul.UsageLimiter(blocker / "sub", 300, 2000, now=self.clock)
        with self.assertLogs("usage_limit", level="ERROR"):
            lim.record(U1, 300)
        with self.assertRaises(ul.UsageLimitError):
            lim.check(U1)


class TestIdsAndConfig(unittest.TestCase):
    def test_u6_valid_uid_rejects_non_hex(self):
        self.assertTrue(ul.valid_uid(ul.new_uid()))
        for bad in ("", "a" * 31, "A" * 32, "g" * 32, "a" * 33, None, "../" + "a" * 29, 5):
            self.assertFalse(ul.valid_uid(bad), bad)

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
