#!/usr/bin/env python3
"""Configure a private DeepSeek key file and run a test-only tool-call probe."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import secrets
import stat
import sys
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any


KEY_FILE = Path.home() / ".config/search-r1/deepseek_api_key"
API_URL = "https://api.deepseek.com/chat/completions"


class KeyFileError(ValueError):
    """A credential cannot be used safely; messages never contain its value."""


def validate_key(value: str) -> str:
    key = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,512}", key):
        raise KeyFileError("Key must be a single ASCII token, 8-512 characters.")
    return key


def check_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError:
        raise KeyFileError("Private key directory is missing or inaccessible.") from None
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise KeyFileError("Key directory must be a real directory owned by this user.")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise KeyFileError("Key directory must have permissions 700.")


def save_key_file(value: str, path: Path = KEY_FILE) -> None:
    key = validate_key(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    check_directory(path.parent)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        raise KeyFileError("Key file already exists; refusing to overwrite it.") from None
    except OSError:
        raise KeyFileError("Cannot create the private key file.") from None
    with os.fdopen(fd, "w", encoding="ascii") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(key + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_key_file(path: Path = KEY_FILE) -> str:
    check_directory(path.parent)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise KeyFileError("Key file is missing, inaccessible, or a symlink.") from None
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            raise KeyFileError("Key file must be a regular file owned by this user.")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise KeyFileError("Key file must have permissions 600.")
        if not 0 < info.st_size <= 514:
            raise KeyFileError("Key file is empty or has an invalid size.")
        raw = stream.read(515)
    try:
        return validate_key(raw.decode("ascii"))
    except UnicodeDecodeError:
        raise KeyFileError("Key file must contain a single ASCII token.") from None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def post_completion(key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        API_URL, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    # Do not forward the authorization header to any redirected endpoint.
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(request, timeout=90) as response:
            raw = response.read(1_048_577)
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        raise RuntimeError(f"DeepSeek API HTTP {code}; response body omitted.") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError("DeepSeek API connection failed; sensitive details omitted.") from None
    if len(raw) > 1_048_576:
        raise RuntimeError("DeepSeek API response exceeded the probe size limit.")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise RuntimeError("DeepSeek API returned invalid JSON.") from None
    if not isinstance(value, dict):
        raise RuntimeError("DeepSeek API returned an unexpected response.")
    return value


def message_from(response: dict[str, Any], finish_reason: str) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("Expected exactly one model response choice.")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != finish_reason:
        raise RuntimeError("Model response was incomplete or had an unexpected finish reason.")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise RuntimeError("Model response did not contain an assistant message.")
    return message


def probe(key: str, model: str) -> dict[str, Any]:
    tools = [{"type": "function", "function": {
        "name": "retrieve", "description": "Get a test-only marker from the local controller.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"], "additionalProperties": False},
    }}]
    messages = [
        {"role": "system", "content": "Call retrieve for the verification marker. After receiving the tool result, reply with exactly the marker, and nothing else."},
        {"role": "user", "content": "Request the verification marker using retrieve."},
    ]
    payload = {"model": model, "messages": messages, "tools": tools,
               "thinking": {"type": "disabled"}, "tool_choice": "required", "max_tokens": 128}
    first = post_completion(key, payload)
    message = message_from(first, "tool_calls")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1 or message.get("content") not in (None, ""):
        raise RuntimeError("Probe rejected mixed output or multiple tool calls.")
    call = calls[0]
    if (not isinstance(call, dict) or call.get("type") != "function"
            or not isinstance(call.get("id"), str) or not call["id"]):
        raise RuntimeError("Probe rejected an invalid tool call.")
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != "retrieve":
        raise RuntimeError("Probe rejected an unauthorized tool.")
    try:
        arguments = json.loads(function.get("arguments", ""))
    except (ValueError, TypeError):
        raise RuntimeError("Probe rejected invalid tool arguments.") from None
    if (not isinstance(arguments, dict) or set(arguments) != {"query"}
            or not isinstance(arguments["query"], str) or not arguments["query"].strip()
            or len(arguments["query"]) > 300):
        raise RuntimeError("Probe rejected invalid tool arguments.")
    marker = "TEST_ONLY_" + secrets.token_hex(8)
    messages.append({"role": "assistant", "content": None, "tool_calls": calls})
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": marker})
    second = post_completion(key, {**payload, "tool_choice": "none", "max_tokens": 64})
    final = message_from(second, "stop")
    if final.get("tool_calls") or not isinstance(final.get("content"), str) or final["content"].strip() != marker:
        raise RuntimeError("Probe failed: model did not return the exact tool result.")
    total_tokens = 0
    for response in (first, second):
        usage = response.get("usage")
        count = usage.get("total_tokens") if isinstance(usage, dict) else None
        if isinstance(count, int) and count >= 0:
            total_tokens += count
    return {"probe_passed": True, "registered_tools": ["retrieve"], "api_requests": 2,
            "model": model, "total_tokens": total_tokens,
            "hybrid_rag_tested": False, "training_data_created": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--configure", action="store_true", help="Save the current shell's DEEPSEEK_API_KEY, or prompt without echo")
    mode.add_argument("--check", action="store_true", help="Check the key file without printing its value or calling the API")
    mode.add_argument("--probe", action="store_true", help="Make two small API calls; not a real RAG or training-data test")
    parser.add_argument("--model", help="Explicit DeepSeek model ID for --probe")
    args = parser.parse_args()
    if args.probe and not args.model:
        parser.error("--model is required for --probe")
    try:
        if args.configure:
            if KEY_FILE.exists() or KEY_FILE.is_symlink():
                raise KeyFileError("Key file already exists; refusing to overwrite it.")
            value = os.getenv("DEEPSEEK_API_KEY", "")
            if not value.strip():
                if not sys.stdin.isatty():
                    raise KeyFileError("Run --configure in your terminal; no key input will be echoed.")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    try:
                        value = getpass.getpass("New DeepSeek API key (hidden): ")
                    except getpass.GetPassWarning:
                        raise KeyFileError("Cannot disable input echo; no key was read.") from None
            save_key_file(value, KEY_FILE)
            print(f"Key file saved: {KEY_FILE} (file 600, directory 700; value not displayed)")
        elif args.check:
            load_key_file(KEY_FILE)
            print(f"Key file readable: {KEY_FILE} (file 600, directory 700; value not displayed)")
        else:
            print(json.dumps(probe(load_key_file(KEY_FILE), args.model), sort_keys=True))
    except (KeyFileError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except (EOFError, KeyboardInterrupt):
        parser.exit(1, "Key input cancelled; no key was saved.\n")


if __name__ == "__main__":
    main()
