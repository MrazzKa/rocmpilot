"""Run the same ROCm migration benchmark prompts on a base model and LoRA adapter."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

try:
    from training.benchmark_metrics import SCALAR_METRICS, aggregate_results, evaluate_generation
except ModuleNotFoundError:  # Allows ``python training/evaluate_benchmark.py``.
    from benchmark_metrics import SCALAR_METRICS, aggregate_results, evaluate_generation


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_ADAPTER = "MrazzKa/rocmpilot-qwen25-coder-lora"
SYSTEM_PROMPT = (
    "You are ROCmPilot, a code migration assistant that helps AI developers "
    "adapt CUDA-first AI workloads to AMD ROCm environments. Provide practical, "
    "structured migration guidance. Distinguish verified facts from version-dependent "
    "advice and state limitations. Use exactly these Markdown H2 sections: Summary; "
    "Detected ROCm issue; Recommended fix; Corrected code or config; Verification "
    "commands; ROCm readiness score; Cursor prompt; Notes and limitations."
)


def load_examples(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            example = json.loads(line)
            if not isinstance(example, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            if not example.get("instruction"):
                raise ValueError(f"{path}:{line_number} has no instruction")
            example.setdefault("id", f"example-{line_number:04d}")
            example.setdefault("category", "unknown")
            examples.append(example)
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No benchmark examples found in {path}")
    return examples


def format_prompt(example: dict[str, Any]) -> str:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    return (
        f"### System\n{SYSTEM_PROMPT}\n\n"
        f"### Instruction\n{instruction}\n\n"
        f"### Input\n{input_text}\n\n"
        "### Response\n"
    )


def _resolve_device(requested: str, torch: Any) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is false")
        if requested == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("--device mps was requested but MPS is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _model_dtype(device: str, torch: Any) -> Any:
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    seed: int,
    torch: Any,
) -> str:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    model_device = next(model.parameters()).device
    inputs = {name: value.to(model_device) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def render_markdown_report(results: dict[str, Any]) -> str:
    if results.get("status") != "completed" or results.get("number_of_examples", 0) < 1:
        raise ValueError("A benchmark report requires actual completed predictions")
    experiment = results["experiment"]
    overall = results["overall"]
    lines = [
        "# ROCmPilot Base-vs-LoRA Evaluation",
        "",
        "This report was generated from saved model generations. It does not use an LLM judge, "
        "and lexical rubric checks should not be interpreted as proof of technical correctness.",
        "",
        "## Setup",
        "",
        f"- Base model: `{experiment['base_model']}`",
        f"- Adapter: `{experiment['adapter']}`",
        f"- Dataset: `{experiment['dataset']}`",
        f"- Examples: {results['number_of_examples']}",
        f"- Device: `{experiment['device']}`",
        f"- Deterministic decoding: `do_sample=False`, max new tokens {experiment['max_new_tokens']}, seed {experiment['seed']}",
        "",
        "## Overall results",
        "",
        "| Metric | Base | LoRA adapter | Difference (adapter − base) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in SCALAR_METRICS:
        lines.append(
            f"| {metric.replace('_', ' ')} | {overall['base'][metric]:.4f} | "
            f"{overall['adapter'][metric]:.4f} | "
            f"{overall['difference_adapter_minus_base'][metric]:+.4f} |"
        )

    intervals = overall.get("bootstrap_95_percent_ci_for_difference", {})
    if intervals:
        lines.extend(
            [
                "",
                "Paired percentile bootstrap intervals are descriptive because this benchmark "
                "is small; they should not be presented as a broad significance claim.",
                "",
                "| Metric | 95% bootstrap interval for difference |",
                "| --- | ---: |",
            ]
        )
        for metric, interval in intervals.items():
            lines.append(f"| {metric.replace('_', ' ')} | [{interval['low']:+.4f}, {interval['high']:+.4f}] |")

    lines.extend(
        [
            "",
            "## Per-category results",
            "",
            "Each category may contain few examples. Report the sample count with every score.",
            "",
        ]
    )
    for category, summary in results["by_category"].items():
        lines.extend(
            [
                f"### {category} (n={summary['n']})",
                "",
                "| Metric | Base | Adapter | Difference |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for metric in SCALAR_METRICS:
            lines.append(
                f"| {metric.replace('_', ' ')} | {summary['base'][metric]:.4f} | "
                f"{summary['adapter'][metric]:.4f} | "
                f"{summary['difference_adapter_minus_base'][metric]:+.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Metric limitations",
            "",
        ]
    )
    for metric, note in results["metric_notes"].items():
        lines.append(f"- **{metric.replace('_', ' ')}:** {note}")
    lines.extend(
        [
            "",
            "Raw prompts, references, generations, and per-example metric details are stored in "
            "`benchmark_predictions.jsonl`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_predictions(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(results: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["scope", "category", "n", "metric", "base", "adapter", "difference_adapter_minus_base"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        groups = [("overall", "all", results["overall"])] + [
            ("category", category, summary)
            for category, summary in results["by_category"].items()
        ]
        for scope, category, summary in groups:
            for metric in SCALAR_METRICS:
                writer.writerow(
                    {
                        "scope": scope,
                        "category": category,
                        "n": summary["n"],
                        "metric": metric,
                        "base": summary["base"][metric],
                        "adapter": summary["adapter"][metric],
                        "difference_adapter_minus_base": summary["difference_adapter_minus_base"][metric],
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--dataset", type=Path, default=Path("data/challenge_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1")

    try:
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Evaluation dependencies are missing. Install requirements-training.txt first."
        ) from exc

    examples = load_examples(args.dataset, args.limit)
    device = _resolve_device(args.device, torch)
    if device == "cpu":
        print(
            "Warning: CPU inference for a 1.5B-parameter model can be slow and memory-intensive.",
            file=sys.stderr,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=_model_dtype(device, torch),
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    ).to(device)
    base_model.eval()

    prompts = [format_prompt(example) for example in examples]
    base_generations = []
    print(f"Running {len(examples)} base-model generations on {device}...")
    for index, prompt in enumerate(prompts):
        base_generations.append(
            generate_one(
                base_model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed + index,
                torch=torch,
            )
        )

    adapter_model = PeftModel.from_pretrained(base_model, args.adapter).to(device)
    adapter_model.eval()
    adapter_generations = []
    print(f"Running {len(examples)} adapter generations with identical decoding settings...")
    for index, prompt in enumerate(prompts):
        adapter_generations.append(
            generate_one(
                adapter_model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed + index,
                torch=torch,
            )
        )

    prediction_rows = []
    for example, prompt, base_generation, adapter_generation in zip(
        examples, prompts, base_generations, adapter_generations
    ):
        prediction_rows.append(
            {
                "id": example["id"],
                "category": example["category"],
                "prompt": prompt,
                "reference_answer": example.get("reference_answer", example.get("output", "")),
                "rubric": {
                    "required_concepts": example.get("required_concepts", []),
                    "forbidden_or_incorrect_concepts": example.get("forbidden_or_incorrect_concepts", []),
                    "consistency_terms": example.get("consistency_terms", []),
                    "expected_sections": example.get("expected_sections"),
                    "sources": example.get("sources", example.get("source", [])),
                    "notes": example.get("notes"),
                },
                "base_generation": base_generation,
                "adapter_generation": adapter_generation,
                "metrics": {
                    "base": evaluate_generation(example, base_generation),
                    "adapter": evaluate_generation(example, adapter_generation),
                },
            }
        )

    results = aggregate_results(
        prediction_rows, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    results["experiment"] = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "dataset": args.dataset.as_posix(),
        "device": device,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "do_sample": False,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "benchmark_predictions.jsonl"
    results_path = args.output_dir / "benchmark_results.json"
    csv_path = args.output_dir / "benchmark_results.csv"
    report_path = args.output_dir / "eval_results.md"
    _write_predictions(prediction_rows, predictions_path)
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(results, csv_path)
    report_path.write_text(render_markdown_report(results), encoding="utf-8")
    print(f"Wrote predictions and reports to {args.output_dir}.")


if __name__ == "__main__":
    main()
