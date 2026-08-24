"""Generate a deterministic, consistency-checked synthetic ROCmPilot dataset.

The original hackathon data in ``data/*.jsonl`` is intentionally left untouched.
By default this script writes a versioned dataset to ``data/clean_v2``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TEMPLATES: list[dict[str, Any]] = [
    {
        "template_id": "hardcoded_device_v1",
        "category": "device_selection",
        "parameter_name": "cuda_device",
        "values": [
            "cuda:0",
            "cuda:1",
            "cuda",
            "cuda:2",
            "cuda:3",
            "cuda:4",
            "cuda:5",
            "cuda:6",
            "cuda:7",
        ],
        "instruction": (
            "Analyze this PyTorch code for ROCm migration issues:\n"
            "```python\nimport torch\ndevice = '{cuda_device}'\nmodel.to(device)\n```"
        ),
        "output": (
            "## Summary\nThe code hardcodes a GPU device string.\n\n"
            "## Detected ROCm issue\nHardcoding `{cuda_device}` assumes a particular "
            "device selection. PyTorch on ROCm intentionally uses the `torch.cuda` "
            "namespace, but availability and the desired device still need to be checked.\n\n"
            "## Recommended fix\nKeep the PyTorch `cuda` device name for the HIP "
            "backend, but validate availability and the requested index before use.\n\n"
            "## Corrected code or config\n```python\nimport torch\n"
            "if not torch.cuda.is_available():\n"
            "    device = torch.device('cpu')\n"
            "else:\n"
            "    requested = torch.device('{cuda_device}')\n"
            "    if requested.index is not None and requested.index >= torch.cuda.device_count():\n"
            "        raise ValueError(f'GPU index {{requested.index}} is unavailable')\n"
            "    device = requested\n"
            "model.to(device)\n```\n\n"
            "## Verification commands\n```bash\npython -c \"import torch; "
            "print(torch.cuda.is_available(), torch.version.hip)\"\n```\n\n"
            "## ROCm readiness score\n80/100. The namespace is portable to ROCm, but "
            "the selected device must exist.\n\n"
            "## Cursor prompt\n\"Replace the hardcoded `{cuda_device}` selection with "
            "an availability-aware PyTorch device selection.\"\n\n"
            "## Notes and limitations\nThis is migration guidance. Test the change in the "
            "target ROCm environment."
        ),
    },
    {
        "template_id": "nvidia_smi_v1",
        "category": "monitoring",
        "parameter_name": "smi_cmd",
        "values": [
            "nvidia-smi",
            "/usr/bin/nvidia-smi",
            "watch nvidia-smi",
            "nvidia-smi -l 1",
        ],
        "instruction": (
            "I have a health check script:\n```bash\n{smi_cmd} "
            "--query-gpu=memory.used --format=csv\n```\nWill this work on AMD?"
        ),
        "output": (
            "## Summary\nThe health check invokes the NVIDIA management CLI.\n\n"
            "## Detected ROCm issue\n`{smi_cmd}` and its query flags are NVIDIA-specific; "
            "the command is not a portable AMD telemetry interface.\n\n"
            "## Recommended fix\nUse a management tool installed with the target ROCm "
            "release (AMD SMI is the current interface; older environments may provide "
            "ROCm SMI) and adapt the parser to that tool's output.\n\n"
            "## Corrected code or config\n```bash\namd-smi metric --help\n```\n\n"
            "## Verification commands\n```bash\namd-smi list\nrocminfo\n```\n\n"
            "## ROCm readiness score\n0/100. The original command depends on NVIDIA drivers.\n\n"
            "## Cursor prompt\n\"Replace `{smi_cmd}` with an AMD SMI health check and "
            "update parsing for AMD SMI output; retain a documented fallback only if the "
            "deployed ROCm version uses ROCm SMI.\"\n\n"
            "## Notes and limitations\nCLI availability and flags vary by ROCm release. "
            "Validate them on the deployment image."
        ),
    },
    {
        "template_id": "cuda_base_image_v1",
        "category": "containers",
        "parameter_name": "base_image",
        "values": [
            "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
            "nvidia/cuda:12.1.0-base-ubuntu20.04",
            "nvcr.io/nvidia/pytorch:23.10-py3",
            "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
        ],
        "instruction": (
            "Here is my Dockerfile:\n```dockerfile\nFROM {base_image}\n"
            "RUN pip install torch\n```\nMake it AMD ROCm compatible."
        ),
        "output": (
            "## Summary\nThe Dockerfile starts from a CUDA/NVIDIA image.\n\n"
            "## Detected ROCm issue\n`{base_image}` does not provide an AMD ROCm user-space "
            "stack. A compatible host driver, image, framework build, and GPU device "
            "passthrough must be selected together.\n\n"
            "## Recommended fix\nChoose an official ROCm image tag compatible with the "
            "host and required PyTorch version; do not copy a floating example tag without "
            "checking the current compatibility documentation.\n\n"
            "## Corrected code or config\n```dockerfile\n"
            "FROM rocm/pytorch:<verified-rocm-pytorch-tag>\n```\n\n"
            "## Verification commands\n```bash\ndocker run --rm --device=/dev/kfd "
            "--device=/dev/dri <image> rocminfo\n```\n\n"
            "## ROCm readiness score\n0/100. The original image targets the NVIDIA stack.\n\n"
            "## Cursor prompt\n\"Replace `{base_image}` with a host-compatible official "
            "ROCm PyTorch image and add the required AMD device passthrough.\"\n\n"
            "## Notes and limitations\nPin a currently published image tag only after checking "
            "the target host, GPU, ROCm, Python, and PyTorch compatibility."
        ),
    },
    {
        "template_id": "cuda_dependency_v1",
        "category": "dependencies",
        "parameter_name": "dependency",
        "values": [
            "bitsandbytes==0.41.1",
            "bitsandbytes<0.43",
            "bitsandbytes==0.39.0",
            "flash-attn==2.3.1.post1",
            "xformers==0.0.22",
        ],
        "instruction": (
            "My requirements.txt has this:\n```text\n{dependency}\n```\n"
            "Will this work out of the box on MI300X?"
        ),
        "output": (
            "## Summary\nThe pinned package may contain CUDA-specific compiled code.\n\n"
            "## Detected ROCm issue\nCompatibility of `{dependency}` cannot be inferred from "
            "the package name alone; the exact package release, ROCm/PyTorch versions, "
            "wheel platform, and GPU architecture matter.\n\n"
            "## Recommended fix\nCheck the package's official compatibility matrix for "
            "the target ROCm stack. Use a documented ROCm wheel or a reproducible source "
            "build, and test imports plus a representative operation.\n\n"
            "## Corrected code or config\nNo universal replacement can be specified without the "
            "target version matrix.\n\n"
            "## Verification commands\n```bash\npython -m pip show {dependency_name}\n"
            "python -c \"import torch; print(torch.__version__, torch.version.hip)\"\n```\n\n"
            "## ROCm readiness score\n10/100. Compatibility is unverified for this exact pin.\n\n"
            "## Cursor prompt\n\"Verify `{dependency}` against the official support "
            "matrix for the pinned ROCm and PyTorch versions; do not guess a replacement "
            "version.\"\n\n"
            "## Notes and limitations\nPackage support changes over time, so this answer "
            "intentionally avoids claiming that a particular newer version is sufficient."
        ),
        "derived_parameters": {
            "dependency_name": lambda value: value.split("=")[0].split("<")[0],
        },
    },
    {
        "template_id": "hip_oom_v1",
        "category": "memory",
        "parameter_name": "allocation_size",
        "values": ["512.00 MiB", "1.00 GiB", "256.00 MiB", "2.50 GiB"],
        "instruction": (
            "I got this error:\n`torch.cuda.OutOfMemoryError: HIP out of memory. "
            "Tried to allocate {allocation_size}`"
        ),
        "output": (
            "## Summary\nThe process failed while requesting `{allocation_size}` of "
            "additional accelerator memory.\n\n"
            "## Detected ROCm issue\nThe HIP backend has run out of usable GPU memory. The "
            "message alone does not distinguish live tensor memory from allocator-reserved "
            "memory.\n\n"
            "## Recommended fix\nMeasure allocated and reserved memory, then reduce live "
            "tensor memory (for example batch size or activation storage) before considering "
            "allocator diagnostics.\n\n"
            "## Corrected code or config\nNo code change can be prescribed from this message alone.\n\n"
            "## Verification commands\n```bash\npython -c \"import torch; "
            "print(torch.cuda.memory_summary() if torch.cuda.is_available() else 'no GPU')\"\n"
            "```\n\n"
            "## ROCm readiness score\n50/100. ROCm is active, but the workload exceeds usable "
            "memory or needs memory-lifetime changes.\n\n"
            "## Cursor prompt\n\"Profile live and reserved PyTorch GPU memory around the "
            "allocation of {allocation_size}; propose a measured reduction rather than an "
            "unverified allocator setting.\"\n\n"
            "## Notes and limitations\nA single OOM string is insufficient to diagnose the root "
            "cause. Capture workload shape and memory statistics on the target system."
        ),
    },
]


def _balanced_values(values: list[str], count: int, rng: random.Random) -> list[str]:
    """Return randomized values while ensuring each value is used before repeats."""
    sampled: list[str] = []
    while len(sampled) < count:
        cycle = list(values)
        rng.shuffle(cycle)
        sampled.extend(cycle[: count - len(sampled)])
    return sampled


def generate_examples(
    *, seed: int, examples_per_template: int = 0, include_metadata: bool = True
) -> list[dict[str, Any]]:
    """Generate examples with every parameter sampled once and reused everywhere."""
    if examples_per_template < 0:
        raise ValueError("examples_per_template cannot be negative")

    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for template in TEMPLATES:
        count = examples_per_template or len(template["values"])
        sampled_values = _balanced_values(
            template["values"], count, rng
        )
        for index, sampled_value in enumerate(sampled_values):
            parameters = {template["parameter_name"]: sampled_value}
            for name, derive in template.get("derived_parameters", {}).items():
                parameters[name] = derive(sampled_value)

            example: dict[str, Any] = {
                "instruction": template["instruction"].format(**parameters),
                "input": "",
                "output": template["output"].format(**parameters),
            }
            if include_metadata:
                example = {
                    "id": f"{template['template_id']}-{index:03d}",
                    "category": template["category"],
                    "template_id": template["template_id"],
                    "parameters": {template["parameter_name"]: sampled_value},
                    **example,
                }
            examples.append(example)
    return examples


def split_examples(
    examples: list[dict[str, Any]], *, seed: int, strategy: str
) -> dict[str, list[dict[str, Any]]]:
    """Split examples randomly or by held-out parameter values."""
    rng = random.Random(seed)
    if strategy == "random":
        shuffled = list(examples)
        rng.shuffle(shuffled)
        train_end = int(len(shuffled) * 0.8)
        val_end = train_end + int(len(shuffled) * 0.1)
        return {
            "train": shuffled[:train_end],
            "validation": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

    if strategy != "parameter_holdout":
        raise ValueError(f"Unknown split strategy: {strategy}")
    if any("parameters" not in example for example in examples):
        raise ValueError("parameter_holdout requires metadata")

    assignments: dict[tuple[str, str], str] = {}
    for template in TEMPLATES:
        values = list(template["values"])
        rng.shuffle(values)
        train_count = min(len(values) - 2, max(1, int(len(values) * 0.8)))
        remaining = values[train_count:]
        val_count = max(1, len(remaining) // 2)
        for value in values[:train_count]:
            assignments[(template["template_id"], value)] = "train"
        for value in remaining[:val_count]:
            assignments[(template["template_id"], value)] = "validation"
        for value in remaining[val_count:]:
            assignments[(template["template_id"], value)] = "test"

    splits = {"train": [], "validation": [], "test": []}
    for example in examples:
        value = next(iter(example["parameters"].values()))
        split = assignments[(example["template_id"], value)]
        splits[split].append(example)
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset(
    splits: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
    seed: int,
    examples_per_template: int,
    split_strategy: str,
    include_metadata: bool,
    overwrite: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "val.jsonl",
        "test": output_dir / "test.jsonl",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite an existing dataset: " + ", ".join(existing)
        )
    for split, path in paths.items():
        _write_jsonl(splits[split], path)

    limitations = [
        "Synthetic examples still come from five narrow template families.",
        "Parameter holdout reduces exact split overlap but does not remove template-level similarity.",
        "This dataset is separate from the manually curated challenge benchmark.",
        "The published hackathon adapter was trained on data/*.jsonl, not this version.",
    ]
    if examples_per_template > 0:
        limitations.insert(
            1, "Repeated parameterized examples are not independent observations."
        )
    manifest = {
        "dataset_version": "clean_v2",
        "generator": "training/generate_dataset.py",
        "seed": seed,
        "examples_per_template": examples_per_template,
        "split_strategy": split_strategy,
        "include_metadata": include_metadata,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "category_counts": {
            name: dict(Counter(row.get("category", "unknown") for row in rows))
            for name, rows in splits.items()
        },
        "limitations": limitations,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/clean_v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--examples-per-template",
        type=int,
        default=0,
        help=(
            "0 emits each available parameter value once (default); positive values "
            "allow repeated synthetic records and are mainly for controlled experiments"
        ),
    )
    parser.add_argument(
        "--split-strategy",
        choices=("parameter_holdout", "random"),
        default="parameter_holdout",
        help="parameter_holdout prevents the same sampled value crossing splits",
    )
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_metadata = not args.no_metadata
    if args.split_strategy == "parameter_holdout" and not include_metadata:
        raise SystemExit("--no-metadata is incompatible with parameter_holdout")
    examples = generate_examples(
        seed=args.seed,
        examples_per_template=args.examples_per_template,
        include_metadata=include_metadata,
    )
    splits = split_examples(examples, seed=args.seed, strategy=args.split_strategy)
    write_dataset(
        splits,
        output_dir=args.output_dir,
        seed=args.seed,
        examples_per_template=args.examples_per_template,
        split_strategy=args.split_strategy,
        include_metadata=include_metadata,
        overwrite=args.overwrite,
    )
    counts = ", ".join(f"{name}={len(rows)}" for name, rows in splits.items())
    print(f"Generated {len(examples)} examples in {args.output_dir} ({counts}).")
    print("Historical data/*.jsonl files were not modified.")


if __name__ == "__main__":
    main()
