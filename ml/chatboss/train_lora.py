"""Train a rollback-safe QLoRA adapter for the Polymath ChatBoss checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="/runpod-volume/chatboss/huggingface-cache")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--minimum-train-examples", type=int, default=100)
    parser.add_argument("--allow-small-dataset", action="store_true", help="Experiment only; never auto-promotes")
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    validation_path = Path(args.validation)
    output = Path(args.output)
    train_count = count_jsonl(train_path)
    validation_count = count_jsonl(validation_path)
    if train_count < args.minimum_train_examples and not args.allow_small_dataset:
        raise SystemExit(
            f"Training blocked: {train_count} examples found; at least "
            f"{args.minimum_train_examples} reviewed examples are required."
        )
    if validation_count < 2:
        raise SystemExit("Training blocked: validation must contain at least 2 holdout examples.")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("json", data_files={
        "train": str(train_path),
        "validation": str(validation_path),
    })
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
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=str(output),
        model_init_kwargs={
            "revision": args.revision,
            "cache_dir": args.cache_dir,
            "dtype": torch.bfloat16,
            "quantization_config": quantization,
        },
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=args.max_length,
        completion_only_loss=True,
        gradient_checkpointing=True,
        bf16=True,
        tf32=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        report_to="none",
        seed=67,
        data_seed=67,
    )
    trainer = SFTTrainer(
        model=args.base_model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    adapter_root = output / "adapter"
    trainer.save_model(str(adapter_root))
    tokenizer.save_pretrained(str(adapter_root))
    metrics = {**result.metrics, **trainer.evaluate()}
    adapter_weights = adapter_root / "adapter_model.safetensors"
    (output / "training_manifest.json").write_text(json.dumps({
        "schema": "polymath-chatboss-training-v1",
        "status": "candidate-not-approved",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "baseModel": args.base_model,
        "revision": args.revision,
        "trainExamples": train_count,
        "validationExamples": validation_count,
        "trainSha256": sha256(train_path),
        "validationSha256": sha256(validation_path),
        "loraRank": args.rank,
        "loraAlpha": args.alpha,
        "epochs": args.epochs,
        "learningRate": args.learning_rate,
        "maxLength": args.max_length,
        "seed": 67,
        "quantization": "4-bit NF4 double-quant",
        "adapterWeightsBytes": adapter_weights.stat().st_size,
        "adapterWeightsSha256": sha256(adapter_weights),
        "metrics": metrics,
    }, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
