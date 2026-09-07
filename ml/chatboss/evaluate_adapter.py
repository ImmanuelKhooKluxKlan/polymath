"""Run deterministic, auditable ChatBoss behavior checks on a base or LoRA model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


WORD_RE = re.compile(r"[a-z0-9']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--adapter")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="/runpod-volume/chatboss/huggingface-cache")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def lexical_f1(output: str, reference: str) -> float:
    predicted = set(normalized_words(output))
    expected = set(normalized_words(reference))
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score_case(case: dict, output: str) -> dict:
    lowered = output.lower()
    required_groups = case.get("requiredAny", [])
    required_results = [any(term.lower() in lowered for term in group) for group in required_groups]
    forbidden_hits = [term for term in case.get("forbidden", []) if term.lower() in lowered]
    forbidden_pattern_hits = [
        pattern
        for pattern in case.get("forbiddenRegex", [])
        if re.search(pattern, output, flags=re.I)
    ]
    word_count = len(normalized_words(output))
    length_ok = word_count <= int(case.get("maxWords", 120))
    denominator = len(required_results) + 2
    safe = not forbidden_hits and not forbidden_pattern_hits
    earned = sum(required_results) + int(safe) + int(length_ok)
    return {
        "score": earned / denominator,
        "requiredPassed": required_results,
        "forbiddenHits": forbidden_hits,
        "forbiddenPatternHits": forbidden_pattern_hits,
        "lengthOk": length_ok,
        "wordCount": word_count,
        "lexicalF1": lexical_f1(output, case.get("reference", "")),
        "safetyPassed": safe,
    }


def render_prompt(tokenizer, messages: list[dict]):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )


def main() -> None:
    args = parse_args()
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_config = AutoConfig.from_pretrained(
        args.base_model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    architectures = set(getattr(model_config, "architectures", []) or [])
    uses_conditional_wrapper = any(name.endswith("ForConditionalGeneration") for name in architectures)
    model_loader = AutoModelForImageTextToText if uses_conditional_wrapper else AutoModelForCausalLM
    loading_mode = "image-text-conditional" if uses_conditional_wrapper else "causal-lm"
    model = model_loader.from_pretrained(
        args.base_model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
        # The 4-bit 35B-A3B checkpoint fits one 46 GB L40. Automatic mapping
        # overestimates some non-quantized modules and attempts a CPU spill,
        # which bitsandbytes rejects unless slower FP32 offload is enabled.
        device_map={"": 0},
        quantization_config=quantization,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    cases_path = Path(args.cases)
    cases = load_jsonl(cases_path)
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[eval] {index}/{len(cases)} {case['id']}", flush=True)
        inputs = render_prompt(tokenizer, case["messages"])
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        else:
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_ids = inputs["input_ids"]
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(generated[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
        result = {
            "id": case["id"],
            "role": case["role"],
            "category": case["category"],
            "output": output,
            **score_case(case, output),
        }
        results.append(result)
        print(f"[eval] score={result['score']:.3f} safety={result['safetyPassed']}", flush=True)

    role_scores = {}
    for role in sorted({item["role"] for item in results}):
        selected = [item for item in results if item["role"] == role]
        role_scores[role] = sum(item["score"] for item in selected) / len(selected)
    report = {
        "schema": "polymath-chatboss-behavior-eval-v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": args.base_model,
        "revision": args.revision,
        "adapter": args.adapter or None,
        "loadingMode": loading_mode,
        "modelClass": type(model).__name__,
        "casesSha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "caseCount": len(results),
        "overallScore": sum(item["score"] for item in results) / len(results),
        "meanLexicalF1": sum(item["lexicalF1"] for item in results) / len(results),
        "safetyPassRate": sum(item["safetyPassed"] for item in results) / len(results),
        "roleScores": role_scores,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "caseCount", "overallScore", "meanLexicalF1", "safetyPassRate", "roleScores"
    )}, indent=2))


if __name__ == "__main__":
    main()
