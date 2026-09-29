"""Direct-answer generation protocol tests; no network calls are made."""

import unittest

from controlled_rollout import canonical_answer
from build_sft_2600_release import normalized_overlap_counts
from generate_direct_answer_sft import (CandidateRejected, MAX_API_TOKENS,
                                         THINKING, parse_response, payload_for)


def response(content: str, **message_fields):
    return {
        "id": "response-1",
        "model": "deepseek-flash",
        "choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": content, **message_fields,
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


class DirectAnswerGenerationTests(unittest.TestCase):
    def test_payload_disables_thinking_and_exposes_no_tools(self):
        payload = payload_for("Who wrote Hamlet?", "deepseek-flash")
        self.assertEqual(payload["thinking"], {"type": THINKING})
        self.assertEqual(payload["max_tokens"], MAX_API_TOKENS)
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertNotIn("Shakespeare", payload["messages"][1]["content"])

    def test_exact_gold_matched_action_is_normalized_for_training(self):
        parsed = parse_response(
            response("<think>I know this established literary fact.</think><answer>William Shakespeare.</answer>"),
            ["William Shakespeare"],
        )
        self.assertEqual(parsed["training_answer"], "William Shakespeare")
        self.assertEqual(
            parsed["training_action"],
            "<think>Reliable prior knowledge is sufficient to answer.</think>"
            "<answer>William Shakespeare</answer>",
        )

    def test_wrong_answer_search_and_reasoning_content_are_rejected(self):
        cases = [
            response("<think>I know it.</think><answer>Christopher Marlowe</answer>"),
            response("<think>I should search.</think><search>Hamlet author</search>"),
            response("<think>I know it.</think><answer>William Shakespeare</answer>",
                     reasoning_content="private reasoning"),
            response("<think>I know it.</think><answer>William Shakespeare</answer>",
                     tool_calls=[{"type": "function"}]),
        ]
        for raw in cases:
            with self.assertRaises(CandidateRejected):
                parse_response(raw, ["William Shakespeare"])

    def test_semantic_acronyms_remain_uppercase(self):
        for value in ("BP", "CNN", "BCE"):
            self.assertEqual(canonical_answer(value), value)

    def test_release_manifest_uses_trajectory_branch_names(self):
        self.assertEqual(
            normalized_overlap_counts(
                {"nq": {"existing_sft": 2, "existing_dpo": 3, "rl_train": 4}}
            ),
            {"nq": {"retrieval_trajectories": 2, "dpo": 3, "rl_train": 4}},
        )
        self.assertEqual(
            normalized_overlap_counts(
                {"nq": {"retrieval_trajectories": 5, "dpo": 6, "rl_train": 7}}
            ),
            {"nq": {"retrieval_trajectories": 5, "dpo": 6, "rl_train": 7}},
        )


if __name__ == "__main__":
    unittest.main()
