"""Train a ROCmPilot LoRA adapter with reproducible validation and checkpoints."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


SYSTEM_PROMPT = (
    "You are ROCmPilot, a code migration assistant that helps AI developers "
    "adapt CUDA-first AI workloads to AMD ROCm environments. Provide practical, "
    "structured migration guidance using the requested Markdown sections."
)

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _add_argument(parser: argparse.ArgumentParser, name: str, **kwargs: Any) -> None:
    """Accept modern kebab-case and historical underscore CLI spellings."""
    historical_name = "--" + name[2:].replace("-", "_")
    parser.add_argument(name, historical_name, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_argument(parser, "--model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    _add_argument(parser, "--train-file", default="data/clean_v2/train.jsonl")
    _add_argument(parser, "--val-file", default="data/clean_v2/val.jsonl")
    _add_argument(parser, "--output-dir", default="outputs/rocmpilot-clean-v2-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    _add_argument(parser, "--learning-rate", type=float, default=2e-4)
    _add_argument(parser, "--batch-size", type=int, default=1)
    _add_argument(parser, "--eval-batch-size", type=int, default=1)
    _add_argument(parser, "--gradient-accumulation-steps", type=int, default=8)
    _add_argument(parser, "--max-seq-length", type=int, default=768)
    _add_argument(
        parser,
        "--max-steps",
        type=int,
        default=-1,
        help="Positive values override --epochs; -1 trains for the requested epochs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    _add_argument(parser, "--logging-steps", type=int, default=1)
    _add_argument(parser, "--eval-steps", type=int, default=10)
    _add_argument(parser, "--save-steps", type=int, default=10)
    _add_argument(
        parser,
        "--evaluation-strategy",
        choices=("no", "steps", "epoch"),
        default="epoch",
    )
    _add_argument(
        parser, "--save-strategy", choices=("no", "steps", "epoch"), default="epoch"
    )
    _add_argument(parser, "--save-total-limit", type=int, default=2)
    _add_argument(parser, "--warmup-ratio", type=float, default=0.0)
    _add_argument(parser, "--weight-decay", type=float, default=0.0)
    _add_argument(parser, "--lr-scheduler-type", default="linear")
    _add_argument(parser, "--lora-r", type=int, default=16)
    _add_argument(parser, "--lora-alpha", type=int, default=32)
    _add_argument(parser, "--lora-dropout", type=float, default=0.05)
    _add_argument(
        parser,
        "--target-modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated PEFT target module names.",
    )
    _add_argument(
        parser,
        "--load-best-model-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_argument(
        parser,
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    _add_argument(
        parser,
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.evaluation_strategy == "no" and args.load_best_model_at_end:
        raise ValueError(
            "--load-best-model-at-end requires validation; use "
            "--no-load-best-model-at-end with --evaluation-strategy no"
        )
    if args.load_best_model_at_end and args.save_strategy != args.evaluation_strategy:
        raise ValueError(
            "Best-model selection requires matching --save-strategy and "
            "--evaluation-strategy"
        )
    if (
        args.load_best_model_at_end
        and args.evaluation_strategy == "steps"
        and args.save_steps % args.eval_steps != 0
    ):
        raise ValueError("--save-steps must be a multiple of --eval-steps")


def format_example(example: dict[str, Any]) -> str:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output_text = str(example.get("output", "")).strip()
    return (
        f"### System\n{SYSTEM_PROMPT}\n\n"
        f"### Instruction\n{instruction}\n\n"
        f"### Input\n{input_text}\n\n"
        f"### Response\n{output_text}"
    )


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "transformers", "datasets", "peft", "accelerate"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    dataset_sizes: dict[str, int],
    status: str,
    train_metrics: dict[str, Any] | None = None,
) -> None:
    metadata = {
        "schema_version": 1,
        "status": status,
        "historical_adapter_warning": (
            "These settings describe this run only. The published hackathon adapter used "
            "the historical data/*.jsonl files and the configuration documented in "
            "docs/experiment_history.md."
        ),
        "arguments": vars(args),
        "dataset_sizes": dataset_sizes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_hip_version": getattr(torch.version, "hip", None),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
        },
        "train_metrics": train_metrics,
    }
    path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from {args.train_file}")
    print(f"Loading validation data from {args.val_file}")
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )
    dataset_sizes = {name: len(values) for name, values in dataset.items()}
    metadata_path = output_path / "experiment_metadata.json"
    _write_metadata(
        metadata_path,
        args=args,
        dataset_sizes=dataset_sizes,
        status="prepared",
    )

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_gpu = torch.cuda.is_available()
    dtype = (
        torch.bfloat16
        if use_gpu and torch.cuda.is_bf16_supported()
        else torch.float16
        if use_gpu
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if use_gpu:
        model = model.to("cuda")
    else:
        print("Warning: no GPU backend detected; training on CPU will be very slow.")
    model.config.use_cache = False

    target_modules = [value.strip() for value in args.target_modules.split(",") if value.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_one(example: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            format_example(example),
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    train_dataset = dataset["train"].map(
        tokenize_one, remove_columns=dataset["train"].column_names
    )
    val_dataset = dataset["validation"].map(
        tokenize_one, remove_columns=dataset["validation"].column_names
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss" if args.load_best_model_at_end else None,
        greater_is_better=False if args.load_best_model_at_end else None,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        data_seed=args.seed,
        bf16=use_gpu and dtype == torch.bfloat16,
        fp16=use_gpu and dtype == torch.float16,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting LoRA fine-tuning. Validation and checkpoint policy are recorded in metadata.")
    train_result = trainer.train()
    trainer.save_state()
    print(f"Saving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    _write_metadata(
        metadata_path,
        args=args,
        dataset_sizes=dataset_sizes,
        status="completed",
        train_metrics=train_result.metrics,
    )
    print(f"Training complete. Adapter and metadata saved to {args.output_dir}")


if __name__ == "__main__":
    main()
