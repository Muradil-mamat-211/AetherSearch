#!/usr/bin/env python3
"""Paid, bounded Chat continuation diagnostics. No corpus or training data."""

from __future__ import annotations

import copy
import json
import secrets
import time

from controlled_rollout import PROMPT, Rejected, evidence_citations, parse_action, response_items
from deepseek_client import DeepSeekClient, history_manifest
from deepseek_key import load_key_file
from token_budget import student_budget


def probe(key: str, *, thinking: str, inject_content: bool) -> dict:
    client = DeepSeekClient("deepseek-flash", key, thinking=thinking,
                            reasoning_effort="max" if thinking == "enabled" else None,
                            timeout=120, retries=0, max_requests=4)
    tag, first, second = (secrets.token_hex(4).upper() for _ in range(3))
    question = (f"This is an isolated protocol diagnostic, not a factual QA or training sample. "
                f"For fixture record DIAG-{tag}, what are its directory code and verification label, in that order? "
                "The diagnostic tool returns the directory entry first; the verification label is in a separate "
                "entry identified by that directory. These private random values require tool lookup.")
    history = [{"role": "user", "content": PROMPT.format(question=question, max_searches=2)}]
    real_post, requests, received_reasoning, sources = client._post, [], [], []
    injected = False

    def traced(payload):
        nonlocal injected
        # Full messages remain in memory only, to verify exact replay.
        requests.append(copy.deepcopy(payload))
        raw, attempts = real_post(payload)
        message = raw["choices"][0]["message"]
        received_reasoning.append(message.get("reasoning_content"))
        if inject_content and len(requests) == 1:
            calls = message.get("tool_calls") or []
            if calls:
                action = json.loads(calls[0]["function"]["arguments"])["action"]
            else:
                action = message.get("content", "")
            # Deliberate test-only envelope fault, never represented as a raw provider receipt.
            message["content"] = action + "</invoke>"
            message.pop("tool_calls", None)
            raw["choices"][0]["finish_reason"] = "stop"
            injected = True
        return raw, attempts

    client._post = traced
    started, checks = time.perf_counter(), []
    for turn in range(3):
        result = client.create(history, "auto" if turn < 2 else "none")
        sent = requests[-1]["messages"]
        if turn:
            if sent[:len(requests[-2]["messages"])] != requests[-2]["messages"]:
                raise RuntimeError("live_history_prefix_mismatch")
            previous = [m for m in sent if m["role"] == "assistant"]
            if thinking == "enabled" and [m.get("reasoning_content") for m in previous] != received_reasoning[:-1]:
                raise RuntimeError("live_reasoning_replay_mismatch")
        checks.append({"turn": turn + 1, "messages": len(sent),
                       "reasoning_messages": sum("reasoning_content" in m for m in sent),
                       "request_manifest_matches": client.calls[-1]["request_messages"] == history_manifest(sent)})
        call, final = response_items(result)
        if call is None:
            reason, answer = parse_action(final or "", "answer")
            if turn != 2 or first not in answer or second not in answer:
                raise Rejected("diagnostic_final_did_not_recover_both_turns")
            refs = evidence_citations(reason, 2)
            if {n for n, _ in refs} != {1, 2}:
                raise Rejected("diagnostic_final_missing_prior_turn_citation")
            return {"status": "pass", "thinking": thinking, "reasoning_effort": client.reasoning_effort,
                    "content_fault_injected": injected, "search_sources": sources,
                    "http_requests": client.request_count, "seconds": round(time.perf_counter() - started, 3),
                    "checks": checks, "final_action": final, "training_data_created": False,
                    "real_retriever_executed": False, "private_reasoning_logged": False}
        if turn >= 2:
            raise Rejected("diagnostic_extra_search")
        sources.append(client.calls[-1]["raw_action_source"])
        docs = [{"doc_id": str(i), "title": f"Diagnostic fixture {turn + 1}/{i}",
                 "text": (f"Record DIAG-{tag} directory code: {first}. Verification label is in entry FOLLOW-{tag}."
                          if turn == 0 else f"Entry FOLLOW-{tag} verification label: {second}.")
                 if i == 1 else "Test-only placeholder; no additional facts."} for i in range(1, 4)]
        info = student_budget().information(docs)["information"]
        client.append_tool_result(history, result, call, info)
    raise RuntimeError("diagnostic_no_final")


def main() -> None:
    key = load_key_file()
    failures = 0
    for thinking, injected in (("disabled", False), ("enabled", False), ("enabled", True)):
        try:
            report = probe(key, thinking=thinking, inject_content=injected)
        except Exception as exc:
            failures += 1
            report = {"status": "failed", "thinking": thinking, "content_fault_injection": injected,
                      "error_type": type(exc).__name__, "reason": str(exc), "training_data_created": False}
        print(json.dumps(report, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
