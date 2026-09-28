"""Thinking effort and total API output budget are independent of short answers."""

import unittest

from deepseek_client import DEFAULT_NONTHINKING_MAX_TOKENS, DEFAULT_THINKING_MAX_TOKENS, DeepSeekClient
from test_deepseek_rollout import KEY, response
from unittest.mock import Mock


class ThinkingTests(unittest.TestCase):
    def test_max_effort_is_sent_with_enabled_type(self):
        client = DeepSeekClient("deepseek-flash", KEY, thinking="enabled", reasoning_effort="max")
        client._post = Mock(return_value=(response(reasoning="fixture reasoning"), 1))
        client.create([], "auto")
        payload = client._post.call_args.args[0]
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertEqual(payload["max_tokens"], DEFAULT_THINKING_MAX_TOKENS)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertIs(payload["tools"][0]["function"]["strict"], True)
        self.assertEqual(client.calls[0]["reasoning_effort"], "max")

    def test_non_thinking_default_budget_has_no_effort_override(self):
        client = DeepSeekClient("deepseek-flash", KEY)
        client._post = Mock(return_value=(response(), 1))
        client.create([], "auto")
        payload = client._post.call_args.args[0]
        self.assertEqual(payload["max_tokens"], DEFAULT_NONTHINKING_MAX_TOKENS)
        self.assertNotIn("reasoning_effort", payload)

    def test_enabled_default_high_and_explicit_budget(self):
        client = DeepSeekClient("deepseek-flash", KEY, thinking="enabled", max_tokens=8192)
        self.assertEqual(client.reasoning_effort, "high")
        self.assertEqual(client.max_tokens, 8192)

    def test_invalid_type_effort_and_conflicting_settings_fail_locally(self):
        for kwargs in ({"thinking": "max"}, {"thinking": "enabled", "reasoning_effort": "unknown"},
                       {"thinking": "disabled", "reasoning_effort": "max"}, {"max_tokens": 32769}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DeepSeekClient("deepseek-flash", KEY, **kwargs)


if __name__ == "__main__":
    unittest.main()
