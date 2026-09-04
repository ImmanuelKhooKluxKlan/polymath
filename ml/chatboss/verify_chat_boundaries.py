"""Verify completion-only token boundaries before spending GPU time on SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--cache-dir", default="/runpod-volume/chatboss/huggingface-cache")
    return parser.parse_args()


def template_tokens(tokenizer, messages: list[dict], kwargs: dict, generation_prompt: bool) -> list[int]:
    options = {
        "tokenize": True,
        "add_generation_prompt": generation_prompt,
        **kwargs,
    }
    try:
        encoded = tokenizer.apply_chat_template(messages, **options)
    except TypeError:
        options.pop("enable_thinking", None)
        encoded = tokenizer.apply_chat_template(messages, **options)
    # Qwen3.5's tokenizer returns a BatchEncoding even when return_dict was not
    # explicitly requested. Comparing len(BatchEncoding) compares its field
    # count, not the token sequence.
    if hasattr(encoded, "get") and encoded.get("input_ids") is not None:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    checked = 0
    failures = []
    inputs = []
    for input_name in args.input:
        path = Path(input_name).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs.append({"path": str(path), "sha256": digest})
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row["prompt"]
                completion = row["completion"]
                kwargs = row.get("chat_template_kwargs", {})
                prompt_tokens = template_tokens(tokenizer, prompt, kwargs, True)
                combined_tokens = template_tokens(tokenizer, [*prompt, *completion], kwargs, False)
                checked += 1
                if combined_tokens[:len(prompt_tokens)] != prompt_tokens:
                    failures.append({
                        "path": str(path),
                        "line": line_number,
                        "promptTokens": len(prompt_tokens),
                        "combinedTokens": len(combined_tokens),
                    })
    report = {
        "schema": "polymath-chatboss-token-boundary-check-v1",
        "model": args.base_model,
        "revision": args.revision,
        "examplesChecked": checked,
        "mismatches": len(failures),
        "inputs": inputs,
        "failures": failures[:20],
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("Training blocked: prompt/completion token boundaries do not match.")


if __name__ == "__main__":
    main()
