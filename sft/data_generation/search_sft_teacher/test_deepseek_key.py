import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deepseek_key


class DeepSeekKeyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "private/key"
        self.key = "unit-test-key-not-real"

    def test_file_round_trip_without_environment(self):
        deepseek_key.save_key_file(self.key, self.path)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(deepseek_key.load_key_file(self.path), self.key)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o700)

    def test_existing_file_is_not_overwritten(self):
        deepseek_key.save_key_file(self.key, self.path)
        with self.assertRaisesRegex(deepseek_key.KeyFileError, "refusing to overwrite"):
            deepseek_key.save_key_file("replacement-test-key", self.path)
        self.assertEqual(deepseek_key.load_key_file(self.path), self.key)

    def test_symlink_is_rejected(self):
        deepseek_key.save_key_file(self.key, self.path)
        link = self.path.parent / "link"
        link.symlink_to(self.path)
        with self.assertRaises(deepseek_key.KeyFileError):
            deepseek_key.load_key_file(link)

    def test_permissive_file_and_directory_are_rejected(self):
        deepseek_key.save_key_file(self.key, self.path)
        self.path.chmod(0o644)
        with self.assertRaisesRegex(deepseek_key.KeyFileError, "600"):
            deepseek_key.load_key_file(self.path)
        self.path.chmod(0o600)
        self.path.parent.chmod(0o755)
        with self.assertRaisesRegex(deepseek_key.KeyFileError, "700"):
            deepseek_key.load_key_file(self.path)

    def test_invalid_key_error_does_not_reveal_value(self):
        value = "private-value\nmalicious-header"
        with self.assertRaises(deepseek_key.KeyFileError) as caught:
            deepseek_key.save_key_file(value, self.path)
        self.assertNotIn("private-value", str(caught.exception))
        self.assertFalse(self.path.exists())

    def test_cli_check_does_not_print_key(self):
        deepseek_key.save_key_file(self.key, self.path)
        output = io.StringIO()
        with patch.object(deepseek_key, "KEY_FILE", self.path), patch.object(sys, "argv", ["deepseek_key.py", "--check"]), patch("sys.stdout", output):
            deepseek_key.main()
        self.assertIn("Key file readable", output.getvalue())
        self.assertNotIn(self.key, output.getvalue())

    def test_cli_configures_from_existing_shell_variable(self):
        output = io.StringIO()
        with patch.object(deepseek_key, "KEY_FILE", self.path), patch.dict(os.environ, {"DEEPSEEK_API_KEY": self.key}), patch.object(sys, "argv", ["deepseek_key.py", "--configure"]), patch("sys.stdout", output), patch.object(deepseek_key.getpass, "getpass") as prompt:
            deepseek_key.main()
        prompt.assert_not_called()
        self.assertEqual(deepseek_key.load_key_file(self.path), self.key)
        self.assertNotIn(self.key, output.getvalue())

    def test_cli_refuses_noninteractive_input_without_environment_key(self):
        errors = io.StringIO()
        with patch.object(deepseek_key, "KEY_FILE", self.path), patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", ["deepseek_key.py", "--configure"]), patch("sys.stdin", io.StringIO(self.key)), patch("sys.stderr", errors), patch.object(deepseek_key.getpass, "getpass") as prompt:
            with self.assertRaises(SystemExit):
                deepseek_key.main()
        prompt.assert_not_called()
        self.assertFalse(self.path.exists())
        self.assertNotIn(self.key, errors.getvalue())

    def test_missing_file_does_not_fall_back_to_environment_or_network(self):
        self.path.parent.mkdir(mode=0o700)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": self.key}), patch.object(deepseek_key, "post_completion") as network:
            with self.assertRaises(deepseek_key.KeyFileError):
                deepseek_key.load_key_file(self.path)
        network.assert_not_called()

    def test_probe_only_dispatches_retrieve_and_reads_tool_result(self):
        captured = []

        def fake_post(key, payload):
            captured.append(json.loads(json.dumps(payload)))
            self.assertEqual(key, self.key)
            self.assertNotIn(self.key, json.dumps(payload))
            if len(captured) == 1:
                return {"choices": [{"finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{"id": "call1", "type": "function", "function": {
                        "name": "retrieve", "arguments": json.dumps({"query": "verification marker"})}}]}}], "usage": {"total_tokens": 10}}
            return {"choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": payload["messages"][-1]["content"]}}], "usage": {"total_tokens": 5}}

        with patch.object(deepseek_key, "post_completion", side_effect=fake_post):
            result = deepseek_key.probe(self.key, "test-model")
        self.assertTrue(result["probe_passed"])
        self.assertFalse(result["hybrid_rag_tested"])
        self.assertFalse(result["training_data_created"])
        self.assertEqual(result["total_tokens"], 15)
        self.assertEqual([p["tool_choice"] for p in captured], ["required", "none"])
        self.assertEqual(captured[0]["thinking"], {"type": "disabled"})
        self.assertEqual([t["function"]["name"] for t in captured[0]["tools"]], ["retrieve"])

    def test_unauthorized_tool_stops_before_second_call(self):
        response = {"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "tool_calls": [{"id": "bad", "type": "function", "function": {
                "name": "web_search", "arguments": "{}"}}]}}]}
        with patch.object(deepseek_key, "post_completion", return_value=response) as mocked:
            with self.assertRaisesRegex(RuntimeError, "unauthorized tool"):
                deepseek_key.probe(self.key, "test-model")
        self.assertEqual(mocked.call_count, 1)

    def test_probe_rejects_nonempty_and_nontext_content_before_second_call(self):
        for content in (" ", "\n", "duplicate action", 0, False, [], {}):
            raw = {"choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": content, "tool_calls": [{"id": "call1", "type": "function", "function": {
                    "name": "retrieve", "arguments": json.dumps({"query": "verification marker"})}}]}}]}
            with self.subTest(content=content), patch.object(deepseek_key, "post_completion", return_value=raw) as mocked:
                with self.assertRaisesRegex(RuntimeError, "mixed output"):
                    deepseek_key.probe(self.key, "test-model")
                self.assertEqual(mocked.call_count, 1)


if __name__ == "__main__":
    unittest.main()
