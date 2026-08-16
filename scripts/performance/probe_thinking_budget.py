#!/home/ducct/repos/vllm/.venv/bin/python
"""Verify that vLLM forces Qwen out of thinking before workload release."""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


def find_subsequence(values: list[int], target: list[int], start: int = 0) -> int:
    for index in range(start, len(values) - len(target) + 1):
        if values[index : index + len(target)] == target:
            return index
    return -1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--thinking-token-budget", type=int, choices=(6144,), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.workload.read_text().splitlines() if line]
    record = min(records, key=lambda item: item["request"]["prompt_tokens"])
    payload = {
        "model": args.model,
        "messages": record["request"]["messages"],
        "max_completion_tokens": args.thinking_token_budget + 512,
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": 260815,
        "stream": False,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "thinking_token_budget": args.thinking_token_budget,
    }
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        body = json.loads(response.read())

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=False
    )
    start_ids = tokenizer.encode("<think>", add_special_tokens=False)
    end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    prompt_ids = list(body.get("prompt_token_ids") or [])
    choice = body["choices"][0]
    output_ids = list(choice.get("token_ids") or [])
    full_ids = prompt_ids + output_ids
    start_index = find_subsequence(full_ids, start_ids)
    end_index = find_subsequence(full_ids, end_ids, start_index + len(start_ids))
    reasoning_tokens = (
        end_index - start_index - len(start_ids)
        if start_index >= 0 and end_index >= 0
        else None
    )
    message = choice.get("message") or {}
    content = message.get("content") or ""
    passed = (
        reasoning_tokens is not None
        and 0 <= reasoning_tokens <= args.thinking_token_budget
        and bool(content.strip())
        and choice.get("finish_reason") in {"stop", "length"}
    )
    report = {
        "schema_version": "rivf26.thinking_budget_probe.v1",
        "status": "PASS" if passed else "FAIL",
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "request_id": record["request_id"],
        "prompt_tokens": record["request"]["prompt_tokens"],
        "max_completion_tokens": payload["max_completion_tokens"],
        "enable_thinking": True,
        "thinking_token_budget": args.thinking_token_budget,
        "measured_reasoning_tokens": reasoning_tokens,
        "answer_chars": len(content),
        "answer_nonempty": bool(content.strip()),
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
