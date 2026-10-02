"""
Wiring tests for the daily token limits in streamlit_app.py: _llm_stream, _begin_metered
and _ensure_visitor_id. usage_limit.py is tested on its own in test_usage_limit.py; these
tests check that the app actually calls it correctly for each provider, so a wiring
mistake cannot switch the spend cap off while the unit tests stay green.
Spec: docs/spec_usage_limits.md (U6, U7, U9, U10)

All provider clients are fakes; no network and no API keys are used.
Run: python3 -m unittest test_llm_stream_limits
"""
import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import streamlit_app as app
import usage_limit as ul

UID = "c" * 32
WEIGHTS = (10.0, {"deepseek-v4-flash": 1.0})


class FakeAnthropic:
    """Stands in for anthropic.Anthropic; counts how many clients were created."""
    created = 0
    fail_open = False
    usage = SimpleNamespace(input_tokens=1_000, output_tokens=500)

    def __init__(self, **kw):
        type(self).created += 1
        self.messages = self

    def stream(self, **kw):
        if type(self).fail_open:
            raise RuntimeError("401 bad key")
        outer = type(self)

        class _S:
            text_stream = iter(["Hello", " world"])

            def __enter__(s):
                return s

            def __exit__(s, *a):
                return False

            def get_final_message(s):
                return SimpleNamespace(usage=outer.usage)

        return _S()


class FakeOpenAI:
    created = 0
    with_usage = True

    def __init__(self, **kw):
        type(self).created += 1
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        assert kw.get("stream_options") == {"include_usage": True}
        chunks = [
            SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(content="Hi"))]),
            # the final usage-only chunk has NO choices: must not crash
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2_000, completion_tokens=300),
                            choices=[]),
        ]
        if not type(self).with_usage:
            chunks = chunks[:1]

        class _S:
            def __enter__(s):
                return s

            def __exit__(s, *a):
                return False

            def __iter__(s):
                return iter(chunks)

        return _S()


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.lim = ul.UsageLimiter(self._tmp.name, 300_000, 2_000_000, weights=WEIGHTS)
        FakeAnthropic.created = 0
        FakeAnthropic.fail_open = False
        FakeOpenAI.created = 0
        FakeOpenAI.with_usage = True
        self.ss = {"provider": "deepseek", "usage_uid": UID,
                   "deepseek_key": "k", "deepseek_model": "deepseek-v4-flash",
                   "claude_key": "k", "claude_model": "claude-sonnet-4-6",
                   "openai_key": "k", "openai_model": "gpt-4o"}
        for p in (mock.patch.object(app.st, "session_state", self.ss),
                  mock.patch.object(app, "_get_limiter", lambda: self.lim),
                  mock.patch("anthropic.Anthropic", FakeAnthropic),
                  mock.patch("openai.OpenAI", FakeOpenAI)):
            p.start()
            self.addCleanup(p.stop)

    def run_stream(self, text="a question", system="sys", **kw):
        return "".join(app._llm_stream([{"role": "user", "content": text}], system, **kw))

    def used(self):
        return 300_000 - self.lim.remaining(UID)[0]


class TestMeteredProviders(Base):
    def test_u7_deepseek_call_is_recorded_at_flash_weight(self):
        self.assertEqual(self.run_stream(), "Hello world")
        self.assertEqual(self.used(), 1_500)                 # (1000 + 500) x 1

    def test_u10_claude_call_is_recorded_at_the_default_weight(self):
        self.ss["provider"] = "claude"
        self.assertEqual(self.run_stream(), "Hello world")
        self.assertEqual(self.used(), 15_000)                # (1000 + 500) x 10

    def test_u10_deepseek_pro_is_weighted_unlike_flash(self):
        # Flash has weight 1, so a dropped weight would go unnoticed there; pro is unlisted
        # in the weights and must be charged the (x10) default.
        self.ss["deepseek_model"] = "deepseek-v4-pro"
        self.assertEqual(self.run_stream(), "Hello world")
        self.assertEqual(self.used(), 15_000)

    def test_u7_openai_stream_with_usage_only_final_chunk_is_recorded(self):
        self.ss["provider"] = "openai"
        self.assertEqual(self.run_stream(), "Hi")
        self.assertEqual(self.used(), 23_000)                # (2000 + 300) x 10

    def test_u7_openai_without_reported_usage_is_estimated_not_free(self):
        self.ss["provider"] = "openai"
        FakeOpenAI.with_usage = False
        with self.assertLogs("usage_limit", level="WARNING"):
            self.assertEqual(self.run_stream(), "Hi")
        self.assertGreaterEqual(self.used(), 10 * ul._MIN_ESTIMATED_OUTPUT)   # estimate x weight 10

    def test_u7_ollama_is_not_metered(self):
        self.ss["provider"] = "ollama"
        lines = [json.dumps({"message": {"content": "ok"}, "done": False}).encode(),
                 json.dumps({"message": {"content": ""}, "done": True}).encode()]
        resp = SimpleNamespace(raise_for_status=lambda: None, iter_lines=lambda: iter(lines))
        with mock.patch("requests.post", return_value=resp):
            self.assertEqual(self.run_stream(), "ok")
        self.assertEqual(self.used(), 0)
        self.assertEqual(FakeAnthropic.created, 0)


class TestRefusalsHappenBeforeAnyApiCall(Base):
    def test_u1_visitor_at_cap_is_refused_and_no_client_is_created(self):
        self.lim.record(UID, 300_000)
        for provider in ("deepseek", "claude", "openai"):
            self.ss["provider"] = provider
            with self.assertRaises(ul.UsageLimitError):
                self.run_stream()
        self.assertEqual(FakeAnthropic.created + FakeOpenAI.created, 0)

    def test_u9_oversized_report_is_refused_before_the_api_call(self):
        self.lim.record(UID, 240_000)                        # 60,000 left
        with self.assertRaises(ul.UsageLimitError):
            self.run_stream(text="x" * 600_000, min_budget=60_000)   # ~200K tokens
        self.assertEqual(FakeAnthropic.created, 0)

    def test_u1_missing_visitor_id_is_a_usage_error_not_a_crash(self):
        self.ss["usage_uid"] = ""
        with self.assertRaises(ul.UsageLimitError):
            self.run_stream()
        self.assertEqual(FakeAnthropic.created, 0)

    def test_u10_a_pricey_model_is_refused_where_flash_is_allowed(self):
        text = "x" * 300_000                                  # ~100K tokens
        self.run_stream(text=text)                           # flash: fits
        self.ss["provider"] = "claude"
        with self.assertRaises(ul.UsageLimitError):
            self.run_stream(text=text)                       # x10: does not

    def test_u4_failed_client_open_releases_the_reservation(self):
        FakeAnthropic.fail_open = True
        with self.assertRaises(RuntimeError):
            self.run_stream(min_budget=250_000)
        FakeAnthropic.fail_open = False
        self.run_stream(min_budget=250_000)                  # reservation was freed
        self.assertEqual(self.used(), 1_500)


class TestVisitorCookie(unittest.TestCase):
    def run_ensure(self, cookies):
        ss = {}
        html = mock.Mock()
        ctx = SimpleNamespace(cookies=cookies)
        with mock.patch.object(app.st, "session_state", ss), \
             mock.patch.object(app.st, "context", ctx), \
             mock.patch("streamlit.components.v1.html", html):
            app._ensure_visitor_id()
        return ss, html

    def test_u6_valid_cookie_is_reused_and_no_cookie_script_is_emitted(self):
        ss, html = self.run_ensure({ul.COOKIE_NAME: UID})
        self.assertEqual(ss["usage_uid"], UID)
        html.assert_not_called()

    def test_u6_missing_cookie_issues_a_new_valid_id_and_sets_it(self):
        ss, html = self.run_ensure({})
        self.assertTrue(ul.valid_uid(ss["usage_uid"]))
        self.assertIn(ss["usage_uid"], html.call_args[0][0])
        self.assertIn("document.cookie", html.call_args[0][0])

    def test_u6_non_hex_or_forged_cookie_values_are_replaced(self):
        for bad in ("../../etc/passwd", "A" * 32, UID + "\n", "<script>", ""):
            ss, html = self.run_ensure({ul.COOKIE_NAME: bad})
            self.assertTrue(ul.valid_uid(ss["usage_uid"]), bad)
            self.assertNotEqual(ss["usage_uid"], bad)
            html.assert_called_once()

    def test_u6_unreadable_cookies_are_logged_not_silently_ignored(self):
        class Boom:
            @property
            def cookies(self):
                raise AttributeError("no st.context")

        ss = {}
        with mock.patch.object(app.st, "session_state", ss), \
             mock.patch.object(app.st, "context", Boom()), \
             mock.patch("streamlit.components.v1.html"), \
             self.assertLogs(app.__name__, level="WARNING"):
            app._ensure_visitor_id()
        self.assertTrue(ul.valid_uid(ss["usage_uid"]))


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main()
