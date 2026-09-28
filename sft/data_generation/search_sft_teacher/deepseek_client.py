#!/usr/bin/env python3
"""Bounded DeepSeek Chat Completions adapter for real, controller-driven rollout."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import threading
import urllib.error
import urllib.request
from typing import Any

from controlled_rollout import (ACTION_GUIDANCE, ANSWER_REQUIREMENT, CONTINUATION_POLICY_VERSION, INSTRUCTIONS, QUERY_POLICY_VERSION, SEARCH_ACTION_SOURCE,
                                TOOL, TOOL_POLICY_VERSION, Rejected, extract_search_action, parse_action, response_items)
from token_budget import INFO_PATTERN, MAX_INFORMATION_TOKENS, MAX_SEARCH_TURNS, student_budget
from deepseek_key import NoRedirect, validate_key


API_URL = "https://api.deepseek.com/beta/chat/completions"
MAX_ANSWER_REPAIRS = 1
DEFAULT_NONTHINKING_MAX_TOKENS = 500
DEFAULT_THINKING_MAX_TOKENS = 16384
MAX_API_OUTPUT_TOKENS = 32768
DEFAULT_MAX_CUMULATIVE_REASONING_TOKENS = 32768
MAX_API_REQUEST_BYTES = 1_000_000
SEARCH_EXHAUSTED_INSTRUCTION = "Search budget exhausted. Use FINAL without a tool call. Answer only if reliable from the information already available. The controller rejects unsupported answers; do not invent an answer or evidence."
ANSWER_REPAIR_INSTRUCTION = (
    "Your previous complete final action exceeded the training action budget. Rewrite it more concisely using the same question "
    "and information already available. " + ANSWER_REQUIREMENT + ACTION_GUIDANCE +
    "Return exactly <think>brief basis</think><answer>minimal answer</answer>. "
    "Preserve valid evidence citations only in the brief think basis if you used retrieval, never inside answer. "
    "Do not add leading, trailing, or inter-tag whitespace. "
    "No new searches are permitted during this format correction. Do not invent evidence or an unsupported answer."
)


class APIError(RuntimeError):
    """An infrastructure failure, with no credential or response body attached."""


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def message_manifest(message: dict[str, Any]) -> dict[str, Any]:
    """Audit exact text without persisting the private reasoning itself."""
    public = {k: v for k, v in message.items() if k != "reasoning_content"}
    reasoning = message.get("reasoning_content")
    return {"role": message["role"], "public_sha256": fingerprint(public),
            "reasoning": {"sha256": fingerprint(reasoning), "characters": len(reasoning)}
            if isinstance(reasoning, str) else None}


def history_manifest(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message_manifest(message) for message in history]


def check_information(information: str) -> None:
    match = INFO_PATTERN.fullmatch(information) if isinstance(information, str) else None
    if not match or any("<" in group or ">" in group for group in match.groups()):
        raise Rejected("invalid_continuation_information")
    if student_budget().count(information) > MAX_INFORMATION_TOKENS:
        raise Rejected("information_token_budget_exceeded")


def validate_history(history: list[dict[str, Any]], *, thinking: str) -> set[str]:
    """Check a completed Chat history, never fix or truncate it implicitly."""
    if not isinstance(history, list):
        raise Rejected("invalid_continuation_history")
    seen, pending, previous_role = set(), None, None
    for message in history:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise Rejected("invalid_continuation_message")
        role = message.get("role")
        allowed = {"role", "content"}
        if role == "assistant":
            allowed |= {"tool_calls", "reasoning_content"}
        elif role == "tool":
            allowed.add("tool_call_id")
        elif role != "user":
            raise Rejected("unauthorized_history_role")
        if set(message) - allowed:
            raise Rejected("unexpected_history_fields")
        if ((previous_role is None and role != "user")
                or (role == "assistant" and previous_role not in {"user", "tool"})):
            raise Rejected("invalid_history_order")
        if pending is not None and role != "tool":
            raise Rejected("missing_tool_result")
        if role == "tool":
            if pending is None or message.get("tool_call_id") != pending:
                raise Rejected("unpaired_tool_result")
            check_information(message["content"])
            pending = None
        elif role == "assistant":
            if thinking == "enabled" and not isinstance(message.get("reasoning_content"), str):
                raise Rejected("missing_history_reasoning_content")
            if "reasoning_content" in message and not isinstance(message["reasoning_content"], str):
                raise Rejected("invalid_history_reasoning_content")
            if "tool_calls" in message:
                search = extract_search_message(message, finish_reason="tool_calls", tool_choice="auto", response_id="history")
                if (search is None or message["content"] != "" or not search["native_tool_call"]
                        or message["tool_calls"] != [search["normalized_tool_call"]]):
                    raise Rejected("noncanonical_search_history")
                pending = search["tool_call_id"]
                if pending in seen:
                    raise Rejected("duplicate_tool_call_id")
                seen.add(pending)
            else:
                try:
                    parse_action(message["content"], "answer")
                except Rejected as exc:
                    if str(exc) != "answer_action_token_budget_exceeded":
                        raise
        previous_role = role
    if pending is not None:
        raise Rejected("missing_tool_result")
    if len(seen) > MAX_SEARCH_TURNS:
        raise Rejected("search_limit_exceeded")
    if previous_role == "assistant":
        raise Rejected("incomplete_continuation_history")
    return seen


def extract_search_message(message: dict[str, Any], *, finish_reason: str,
                           tool_choice: str, response_id: str) -> dict[str, Any] | None:
    """Resolve public search text; synthetic envelopes are explicitly controller-owned."""
    if tool_choice not in {"auto", "required", "none"} or not isinstance(response_id, str) or not response_id:
        raise Rejected("invalid_search_action_receipt")
    calls = message.get("tool_calls", [])
    calls = [] if calls is None else calls
    if not isinstance(calls, list) or len(calls) > 1:
        raise Rejected("unauthorized_or_parallel_tool_call")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise Rejected("invalid_model_output")
    arguments = None
    if calls:
        if tool_choice == "none":
            raise Rejected("tool_call_after_search_budget")
        if finish_reason != "tool_calls":
            raise Rejected("invalid_tool_call")
        call = calls[0]
        if (not isinstance(call, dict) or call.get("type") != "function"
                or not isinstance(call.get("id"), str) or not call["id"]):
            raise Rejected("invalid_tool_call")
        function = call.get("function")
        if not isinstance(function, dict) or function.get("name") != "retrieve":
            raise Rejected("unauthorized_tool")
        arguments = function.get("arguments")
        if not isinstance(arguments, str) or len(arguments) > 16000:
            raise Rejected("invalid_tool_arguments")
        try:
            parsed = json.loads(arguments, object_pairs_hook=unique_object)
        except (ValueError, TypeError):
            raise Rejected("invalid_tool_arguments") from None
        if not isinstance(parsed, dict) or set(parsed) != {"action"} or not isinstance(parsed["action"], str):
            raise Rejected("invalid_tool_arguments")
        raw_action, source, call_id = parsed["action"], "tool_calls[].function.arguments.action", call["id"]
    elif isinstance(content, str) and "<search>" in content:
        if tool_choice == "none":
            raise Rejected("tool_call_after_search_budget")
        if finish_reason != "stop":
            raise Rejected("invalid_model_output")
        raw_action, source = content, "assistant.content"
        call_id = "call_extracted_" + hashlib.sha256((response_id + content).encode()).hexdigest()[:24]
    else:
        return None
    extraction = extract_search_action(raw_action)
    action = extraction["action"]
    if calls and content:
        if "<answer>" in content or "</answer>" in content:
            raise Rejected("mixed_search_final_output")
        if "<search>" in content:
            if extract_search_action(content)["action"] != action:
                raise Rejected("ambiguous_search_action")
    normalized_call = {"id": call_id, "type": "function", "function": {
        "name": "retrieve", "arguments": json.dumps({"action": action}, ensure_ascii=False)}}
    return {"action": action, "raw_action": raw_action, "raw_action_source": source,
            "extraction": extraction, "native_tool_call": bool(calls),
            "tool_call_id": call_id, "function_arguments": arguments,
            "normalized_tool_call": normalized_call}


class SharedRequestBudget:
    def __init__(self, limit: int):
        if type(limit) is not int or limit < 1:
            raise ValueError("invalid_shared_api_request_limit")
        self.limit = limit
        self.count = 0
        self.aborted = False
        self._lock = threading.Lock()

    def reserve(self) -> None:
        with self._lock:
            if self.aborted:
                raise APIError("api_requests_aborted")
            if self.count >= self.limit:
                raise APIError("api_request_budget_exhausted")
            self.count += 1

    def abort(self) -> None:
        with self._lock:
            self.aborted = True


class DeepSeekClient:
    provider = "deepseek"

    def __init__(self, model: str, key: str, *, thinking: str = "disabled",
                 max_tokens: int | None = None, timeout: float = 90, retries: int = 2,
                 max_requests: int = 100, reasoning_effort: str | None = None,
                 max_cumulative_reasoning_tokens: int = DEFAULT_MAX_CUMULATIVE_REASONING_TOKENS,
                 request_budget: SharedRequestBudget | None = None):
        if model not in {"deepseek-flash", "deepseek-v4-pro"}:
            raise ValueError("Use an explicit supported DeepSeek model ID; no model fallback is allowed")
        if thinking not in {"enabled", "disabled"}:
            raise ValueError("thinking must be enabled or disabled")
        if reasoning_effort is not None and (thinking != "enabled" or reasoning_effort not in {"low", "high", "max"}):
            raise ValueError("reasoning_effort requires enabled thinking and low, high or max")
        if max_tokens is None:
            max_tokens = DEFAULT_THINKING_MAX_TOKENS if thinking == "enabled" else DEFAULT_NONTHINKING_MAX_TOKENS
        if (not 128 <= max_tokens <= MAX_API_OUTPUT_TOKENS or not math.isfinite(timeout)
                or timeout <= 0 or retries not in range(4) or max_requests < 1
                or type(max_cumulative_reasoning_tokens) is not int or max_cumulative_reasoning_tokens < 1):
            raise ValueError("Invalid API limits")
        self.model, self.thinking = model, thinking
        self.reasoning_effort = (reasoning_effort or "high") if thinking == "enabled" else None
        self._key = validate_key(key)
        self.max_tokens, self.timeout, self.retries = max_tokens, timeout, retries
        self.max_requests, self.request_count = max_requests, 0
        self.request_budget = request_budget
        self.max_cumulative_reasoning_tokens = max_cumulative_reasoning_tokens
        self.cumulative_reasoning_tokens = 0
        self.cumulative_prompt_usage_units = 0
        self.calls: list[dict[str, Any]] = []
        self._expected_history: list[dict[str, Any]] | None = None
        self._pending_response: dict[str, Any] | None = None
        self._last_wire_history: list[dict[str, Any]] = []
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def begin_trajectory(self) -> None:
        """Start an independent question; the client is serial, not thread-safe."""
        self._expected_history = None
        self._pending_response = None
        self._last_wire_history = []
        self.cumulative_reasoning_tokens = 0
        self.cumulative_prompt_usage_units = 0

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        for attempt in range(self.retries + 1):
            if self.request_count >= self.max_requests:
                raise APIError("api_request_budget_exhausted")
            if self.request_budget is not None:
                self.request_budget.reserve()
            self.request_count += 1
            request = urllib.request.Request(
                API_URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
                headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
            )
            try:
                with self.opener.open(request, timeout=self.timeout) as stream:
                    raw = stream.read(2_097_153)
            except urllib.error.HTTPError as exc:
                code = exc.code
                exc.close()
                if code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise APIError(f"deepseek_http_{code}") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == self.retries:
                    raise APIError("deepseek_connection_failed") from None
            else:
                if len(raw) > 2_097_152:
                    raise APIError("deepseek_response_too_large")
                try:
                    response = json.loads(raw, object_pairs_hook=unique_object)
                except (ValueError, UnicodeDecodeError):
                    raise APIError("deepseek_invalid_json") from None
                if not isinstance(response, dict):
                    raise APIError("deepseek_invalid_response")
                return response, attempt + 1
            time.sleep(min(2 ** attempt, 8))
        raise APIError("deepseek_retry_limit")

    def create(self, history: list[dict[str, Any]], choice: str) -> dict[str, Any]:
        result = self._create_once(history, choice)
        call, final_text = response_items(result)
        if call is not None:
            return result
        try:
            parse_action(final_text or "", "answer")
        except Rejected as exc:
            if str(exc) != "answer_action_token_budget_exceeded":
                return result
        else:
            return result
        # The model rewrites its answer; the discarded text never becomes a training event.
        original = self.calls[-1]
        original["discarded_reason"] = "answer_action_token_budget_exceeded"
        # Include controller instructions that were actually sent on the wire.
        correction_history = copy.deepcopy(self._last_wire_history)
        correction_history.extend([result["assistant_message"], {"role": "user", "content": ANSWER_REPAIR_INSTRUCTION}])
        self._expected_history = history_manifest(correction_history)
        repaired = self._create_once(correction_history, "none", answer_repair=True)
        self.calls[-1]["repair_of_response_id"] = original["id"]
        _, repaired_text = response_items(repaired)
        parse_action(repaired_text or "", "answer")
        return repaired

    def _create_once(self, history: list[dict[str, Any]], choice: str, *, answer_repair: bool = False) -> dict[str, Any]:
        if choice not in {"required", "auto", "none"}:
            raise ValueError("Unsupported tool choice")
        student_budget()
        seen_ids = validate_history(history, thinking=self.thinking)
        supplied_manifest = history_manifest(history)
        if self._expected_history is not None and supplied_manifest != self._expected_history:
            raise Rejected("continuation_history_changed")
        self._pending_response = None
        # Required is only used by the isolated tool-protocol smoke test.
        # Adaptive rollout uses auto, then none when its search budget is spent.
        wire_choice = "auto" if self.thinking == "enabled" and choice == "required" else choice
        messages = [{"role": "system", "content": INSTRUCTIONS}] + copy.deepcopy(history)
        if choice == "none" and not answer_repair:
            messages.append({"role": "user", "content": SEARCH_EXHAUSTED_INSTRUCTION})
        payload = {"model": self.model, "messages": messages,
                   "tools": [{"type": "function", "function": copy.deepcopy({k: v for k, v in TOOL.items() if k != "type"})}],
                   "tool_choice": wire_choice, "thinking": {"type": self.thinking},
                   "max_tokens": self.max_tokens, "stream": False}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        request_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        if request_bytes > MAX_API_REQUEST_BYTES:
            raise Rejected("api_request_byte_budget_exceeded")
        self._last_wire_history = copy.deepcopy(messages[1:])
        request_audit = {"continuation_policy": CONTINUATION_POLICY_VERSION,
                         "request_messages": history_manifest(messages),
                         "request_sha256": fingerprint(payload), "request_bytes": request_bytes,
                         "replayed_reasoning_messages": sum("reasoning_content" in m for m in messages),
                         "replayed_reasoning_characters": sum(len(m.get("reasoning_content", "")) for m in messages)}
        started = time.perf_counter()
        try:
            raw, attempts = self._post(payload)
        except APIError as exc:
            self.calls.append({"status": str(exc), "seconds": time.perf_counter() - started,
                               "endpoint": API_URL, "tool_policy": TOOL_POLICY_VERSION,
                               "query_policy": QUERY_POLICY_VERSION, "strict_requested": True, **request_audit})
            raise
        self.calls.append({"id": raw.get("id"), "model": raw.get("model"),
                           "system_fingerprint": raw.get("system_fingerprint"),
                           "usage": raw.get("usage") or {}, "http_attempts": attempts,
                           "seconds": time.perf_counter() - started, "status": "received",
                           "endpoint": API_URL, "tool_policy": TOOL_POLICY_VERSION,
                           "query_policy": QUERY_POLICY_VERSION, "strict_requested": True,
                           "tool_choice": wire_choice, "thinking": self.thinking,
                           "reasoning_effort": self.reasoning_effort, "max_tokens": self.max_tokens, **request_audit})
        usage = raw.get("usage") or {}
        reported_prompt = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if type(reported_prompt) is int and reported_prompt >= 0:
            prompt_charge, prompt_source = reported_prompt, "provider_usage"
        else:
            prompt_charge, prompt_source = request_bytes, "utf8_request_byte_upper_bound"
        self.cumulative_prompt_usage_units += prompt_charge
        self.calls[-1].update(prompt_usage_units=prompt_charge, prompt_usage_source=prompt_source,
                              cumulative_prompt_usage_units=self.cumulative_prompt_usage_units)
        if any(not isinstance(raw.get(k), str) or not raw[k].strip() for k in ("id", "model")):
            raise Rejected("missing_api_receipt")
        choices = raw.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise Rejected("invalid_model_response")
        item = choices[0]
        message = item.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise Rejected("invalid_model_response")
        reason = item.get("finish_reason")
        if reason not in {"stop", "tool_calls"}:
            if reason in {"aborted", "insufficient_system_resource"}:
                raise APIError("deepseek_generation_interrupted")
            raise Rejected("incomplete_model_output")
        calls = message.get("tool_calls", [])
        calls = [] if calls is None else calls
        if not isinstance(calls, list) or len(calls) > 1:
            raise Rejected("unauthorized_or_parallel_tool_call")
        content = message.get("content")
        self.calls[-1]["assistant_content"] = copy.deepcopy(content)
        self.calls[-1].update(raw_tool_calls=copy.deepcopy(calls), finish_reason=reason)
        normalized = {"id": raw.get("id"), "model": raw.get("model") or self.model,
                      "status": "completed", "usage": raw.get("usage") or {}, "output": []}
        search = extract_search_message(message, finish_reason=reason, tool_choice=choice, response_id=raw["id"])
        if search is not None:
            if self.thinking == "enabled" and not isinstance(message.get("reasoning_content"), str):
                raise Rejected("missing_reasoning_content")
            call = search["normalized_tool_call"]
            if call["id"] in seen_ids:
                raise Rejected("duplicate_tool_call_id")
            normalized["output"] = [{"type": "function_call", "name": "retrieve", "call_id": call["id"], "arguments": call["function"]["arguments"]}]
            self.calls[-1].update(search, action_type="search", tool_name="retrieve", action_source=SEARCH_ACTION_SOURCE)
            assistant = {"role": "assistant", "content": "", "tool_calls": [copy.deepcopy(call)]}
            if self.thinking == "enabled":
                assistant["reasoning_content"] = message["reasoning_content"]
            normalized["assistant_message"] = assistant
        else:
            if reason != "stop" or not isinstance(content, str) or not content.strip():
                raise Rejected("missing_final_answer")
            normalized["output"] = [{"type": "message", "content": [{"type": "output_text", "text": content}]}]
            self.calls[-1].update(action_type="final", action=content)
            assistant = {"role": "assistant", "content": content}
            if self.thinking == "enabled":
                if not isinstance(message.get("reasoning_content"), str):
                    raise Rejected("missing_reasoning_content")
                assistant["reasoning_content"] = message["reasoning_content"]
            normalized["assistant_message"] = assistant
        if self.thinking == "enabled":
            details = (raw.get("usage") or {}).get("completion_tokens_details") or {}
            reported = details.get("reasoning_tokens") if isinstance(details, dict) else None
            if type(reported) is int and reported >= 0:
                charge, source = reported, "provider_usage"
            else:
                charge, source = len(assistant["reasoning_content"].encode("utf-8")), "utf8_byte_upper_bound"
        else:
            charge, source = 0, "thinking_disabled"
        self.cumulative_reasoning_tokens += charge
        self.calls[-1].update(reasoning_budget_charge=charge, reasoning_budget_source=source,
                              cumulative_reasoning_tokens=self.cumulative_reasoning_tokens,
                              reasoning_budget_limit=self.max_cumulative_reasoning_tokens)
        if self.cumulative_reasoning_tokens > self.max_cumulative_reasoning_tokens:
            raise Rejected("cumulative_reasoning_budget_exceeded")
        self.calls[-1]["response_message"] = message_manifest(assistant)
        self._pending_response = {"id": normalized["id"], "input": supplied_manifest,
                                  "message": message_manifest(assistant), "output": fingerprint(normalized["output"])}
        return normalized

    def append_tool_result(self, history: list[dict[str, Any]], response: dict[str, Any],
                           call: dict[str, Any], information: str) -> None:
        validate_history(history, thinking=self.thinking)
        if not isinstance(response, dict) or not isinstance(call, dict):
            raise Rejected("continuation_receipt_mismatch")
        pending = self._pending_response
        assistant = response.get("assistant_message")
        if (pending is None or response.get("id") != pending["id"] or not isinstance(assistant, dict)
                or assistant.get("role") != "assistant"
                or message_manifest(assistant) != pending["message"]
                or history_manifest(history) != pending["input"]
                or fingerprint(response.get("output")) != pending["output"]):
            raise Rejected("continuation_receipt_mismatch")
        actual_call, final = response_items(response)
        if actual_call is None or final is not None or actual_call != call:
            raise Rejected("continuation_tool_call_mismatch")
        check_information(information)
        addition = [copy.deepcopy(assistant), {"role": "tool", "tool_call_id": call["call_id"], "content": information}]
        candidate = history + addition
        validate_history(candidate, thinking=self.thinking)
        # Commit both messages together only after every boundary check succeeds.
        self._expected_history = history_manifest(candidate)
        history.extend(addition)
        self._pending_response = None
