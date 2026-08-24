"""Audit ROCmPilot JSONL splits for duplication, leakage, and consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


INFERRED_FAMILIES = [
    (
        "device_selection",
        "hardcoded_device_v1",
        re.compile(r"device\s*=\s*['\"]([^'\"]+)['\"]"),
        re.compile(r"Hardcoding\s+`([^`]+)`", re.IGNORECASE),
    ),
    (
        "monitoring",
        "nvidia_smi_v1",
        re.compile(r"```bash\n(.+?)\s+--query-gpu", re.DOTALL),
        re.compile(r"`([^`]*nvidia-smi[^`]*)`\s+is installed", re.IGNORECASE),
    ),
    (
        "containers",
        "cuda_base_image_v1",
        re.compile(r"FROM\s+([^\s]+)"),
        re.compile(r"The\s+`([^`]+)`\s+contains CUDA", re.IGNORECASE),
    ),
    (
        "dependencies",
        "cuda_dependency_v1",
        re.compile(r"```text\n([^\n]+)"),
        re.compile(r"`([^`]+)`\s+historically", re.IGNORECASE),
    ),
    (
        "memory",
        "hip_oom_v1",
        re.compile(r"Tried to allocate\s+([^`]+)"),
        re.compile(r"requesting\s+`([^`]+)`", re.IGNORECASE),
    ),
]


def normalize_text(value: Any) -> str:
    """Normalize only whitespace so semantic changes remain visible."""
    return re.sub(r"\s+", " ", str(value or "").strip())


def content_key(record: dict[str, Any], *, normalized: bool = False) -> str:
    fields = [record.get("instruction", ""), record.get("input", ""), record.get("output", "")]
    if normalized:
        fields = [normalize_text(field) for field in fields]
    return json.dumps(fields, ensure_ascii=False, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            value["_audit_line"] = line_number
            rows.append(value)
    return rows


def infer_family(record: dict[str, Any]) -> tuple[str, str]:
    if record.get("category") or record.get("template_id"):
        return str(record.get("category", "unknown")), str(
            record.get("template_id", "unknown")
        )
    instruction = str(record.get("instruction", ""))
    for category, template_id, instruction_pattern, _ in INFERRED_FAMILIES:
        if instruction_pattern.search(instruction):
            return category, template_id
    return "unknown", "unknown"


def check_parameter_consistency(record: dict[str, Any]) -> dict[str, Any]:
    instruction = str(record.get("instruction", ""))
    output = str(record.get("output", ""))
    parameters = record.get("parameters")
    if isinstance(parameters, dict) and parameters:
        missing = []
        for name, value in parameters.items():
            value_text = str(value)
            locations = {
                "instruction": value_text in instruction,
                "output": value_text in output,
            }
            if not all(locations.values()):
                missing.append({"name": name, "value": value_text, "present": locations})
        return {
            "status": "consistent" if not missing else "inconsistent",
            "method": "metadata",
            "details": missing,
        }

    for category, template_id, instruction_pattern, output_pattern in INFERRED_FAMILIES:
        instruction_match = instruction_pattern.search(instruction)
        if not instruction_match:
            continue
        output_match = output_pattern.search(output)
        if output_match is None:
            return {
                "status": "not_checkable",
                "method": "inferred",
                "category": category,
                "template_id": template_id,
                "instruction_value": instruction_match.group(1).strip(),
                "reason": "the historical output does not expose the parameter in a detectable form",
            }
        instruction_value = normalize_text(instruction_match.group(1))
        output_value = normalize_text(output_match.group(1))
        return {
            "status": "consistent" if instruction_value == output_value else "inconsistent",
            "method": "inferred",
            "category": category,
            "template_id": template_id,
            "instruction_value": instruction_value,
            "output_value": output_value,
        }
    return {"status": "not_checkable", "method": "none"}


def _duplicate_summary(values: Iterable[str], max_examples: int) -> dict[str, Any]:
    counts = Counter(values)
    repeated = [(value, count) for value, count in counts.items() if count > 1]
    repeated.sort(key=lambda pair: (-pair[1], pair[0]))
    return {
        "duplicate_groups": len(repeated),
        "duplicate_instances": sum(count - 1 for _, count in repeated),
        "examples": [
            {"count": count, "value": value[:500]} for value, count in repeated[:max_examples]
        ],
    }


def _record_label(split: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": split,
        "line": record.get("_audit_line"),
        "id": record.get("id"),
        "instruction": str(record.get("instruction", ""))[:500],
    }


def _cross_split_duplicates(
    left_name: str,
    left: list[dict[str, Any]],
    right_name: str,
    right: list[dict[str, Any]],
    *,
    normalized: bool,
    max_examples: int,
) -> dict[str, Any]:
    left_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    right_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in left:
        left_groups[content_key(record, normalized=normalized)].append(record)
    for record in right:
        right_groups[content_key(record, normalized=normalized)].append(record)
    shared = sorted(set(left_groups) & set(right_groups))
    examples = []
    for key in shared[:max_examples]:
        examples.append(
            {
                "left": _record_label(left_name, left_groups[key][0]),
                "right": _record_label(right_name, right_groups[key][0]),
                "left_occurrences": len(left_groups[key]),
                "right_occurrences": len(right_groups[key]),
            }
        )
    return {
        "shared_groups": len(shared),
        "cross_pairs": sum(len(left_groups[key]) * len(right_groups[key]) for key in shared),
        "examples": examples,
    }


def _near_instruction_leakage(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    threshold: float,
    max_examples: int,
) -> dict[str, Any]:
    train_unique: dict[str, dict[str, Any]] = {}
    test_unique: dict[str, dict[str, Any]] = {}
    for record in train:
        train_unique.setdefault(normalize_text(record.get("instruction", "")), record)
    for record in test:
        test_unique.setdefault(normalize_text(record.get("instruction", "")), record)

    matches = []
    for left_text, left_record in train_unique.items():
        for right_text, right_record in test_unique.items():
            if not left_text or not right_text or left_text == right_text:
                continue
            ratio = SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()
            if ratio >= threshold:
                matches.append(
                    {
                        "similarity": round(ratio, 4),
                        "train": _record_label("train", left_record),
                        "test": _record_label("test", right_record),
                    }
                )
    matches.sort(key=lambda item: -item["similarity"])
    return {
        "threshold": threshold,
        "unique_instruction_pairs": len(matches),
        "examples": matches[:max_examples],
        "method": "difflib.SequenceMatcher over whitespace-normalized unique instructions",
    }


def audit_splits(
    splits: dict[str, list[dict[str, Any]]],
    *,
    near_threshold: float = 0.9,
    max_examples: int = 10,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "near_duplicate_method": "SequenceMatcher is a lightweight heuristic, not semantic equivalence.",
        "splits": {},
        "cross_split": {},
        "parameter_consistency": {},
        "possible_train_test_leakage": {},
        "suspicious_records": [],
    }

    for split_name, rows in splits.items():
        categories = Counter()
        templates = Counter()
        consistency = Counter()
        inconsistent_examples = []
        for record in rows:
            category, template_id = infer_family(record)
            categories[category] += 1
            templates[template_id] += 1
            check = check_parameter_consistency(record)
            consistency[check["status"]] += 1
            if check["status"] == "inconsistent":
                item = {**_record_label(split_name, record), "check": check}
                inconsistent_examples.append(item)
                if len(result["suspicious_records"]) < max_examples:
                    result["suspicious_records"].append(item)
        result["splits"][split_name] = {
            "examples": len(rows),
            "exact_duplicates": _duplicate_summary(
                (content_key(row) for row in rows), max_examples
            ),
            "normalized_duplicates": _duplicate_summary(
                (content_key(row, normalized=True) for row in rows), max_examples
            ),
            "repeated_instructions": _duplicate_summary(
                (str(row.get("instruction", "")) for row in rows), max_examples
            ),
            "repeated_outputs": _duplicate_summary(
                (str(row.get("output", "")) for row in rows), max_examples
            ),
            "category_distribution": dict(sorted(categories.items())),
            "template_distribution": dict(sorted(templates.items())),
        }
        result["parameter_consistency"][split_name] = {
            "counts": dict(sorted(consistency.items())),
            "inconsistent_examples": inconsistent_examples[:max_examples],
        }

    names = list(splits)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            pair_name = f"{left_name}__{right_name}"
            result["cross_split"][pair_name] = {
                "exact": _cross_split_duplicates(
                    left_name,
                    splits[left_name],
                    right_name,
                    splits[right_name],
                    normalized=False,
                    max_examples=max_examples,
                ),
                "normalized": _cross_split_duplicates(
                    left_name,
                    splits[left_name],
                    right_name,
                    splits[right_name],
                    normalized=True,
                    max_examples=max_examples,
                ),
                "shared_instructions": len(
                    {str(row.get("instruction", "")) for row in splits[left_name]}
                    & {str(row.get("instruction", "")) for row in splits[right_name]}
                ),
                "shared_outputs": len(
                    {str(row.get("output", "")) for row in splits[left_name]}
                    & {str(row.get("output", "")) for row in splits[right_name]}
                ),
            }

    if "train" in splits and "test" in splits:
        result["possible_train_test_leakage"] = _near_instruction_leakage(
            splits["train"],
            splits["test"],
            threshold=near_threshold,
            max_examples=max_examples,
        )
    return result


def render_markdown(audit: dict[str, Any], sources: dict[str, Any]) -> str:
    lines = [
        "# ROCmPilot Dataset Audit",
        "",
        "This report was generated by `training/audit_dataset.py` from the files listed below. "
        "Counts are descriptive; the similarity check is a heuristic and is not evidence of semantic equivalence.",
        "",
        "## Inputs",
        "",
        "| Split | Path | SHA-256 |",
        "| --- | --- | --- |",
    ]
    for split, source in sources.items():
        lines.append(f"| {split} | `{source['path']}` | `{source['sha256']}` |")

    lines.extend(
        [
            "",
            "## Split summary",
            "",
            "| Split | Examples | Exact duplicate instances | Normalized duplicate instances | Repeated instruction instances | Repeated output instances |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, split in audit["splits"].items():
        lines.append(
            f"| {name} | {split['examples']} | "
            f"{split['exact_duplicates']['duplicate_instances']} | "
            f"{split['normalized_duplicates']['duplicate_instances']} | "
            f"{split['repeated_instructions']['duplicate_instances']} | "
            f"{split['repeated_outputs']['duplicate_instances']} |"
        )

    lines.extend(["", "## Cross-split overlap", "", "| Pair | Exact groups | Normalized groups | Shared instructions | Shared outputs |", "| --- | ---: | ---: | ---: | ---: |"])
    for pair, values in audit["cross_split"].items():
        lines.append(
            f"| {pair.replace('__', ' ↔ ')} | {values['exact']['shared_groups']} | "
            f"{values['normalized']['shared_groups']} | {values['shared_instructions']} | "
            f"{values['shared_outputs']} |"
        )

    lines.extend(["", "## Parameter consistency", ""])
    for split, values in audit["parameter_consistency"].items():
        counts = values["counts"]
        lines.append(
            f"- **{split}:** {counts.get('consistent', 0)} consistent, "
            f"{counts.get('inconsistent', 0)} inconsistent, "
            f"{counts.get('not_checkable', 0)} not checkable."
        )

    lines.extend(["", "## Category and template distribution", ""])
    for split, values in audit["splits"].items():
        lines.append(f"### {split}")
        lines.append("")
        lines.append("- Categories: " + json.dumps(values["category_distribution"], ensure_ascii=False))
        lines.append("- Templates: " + json.dumps(values["template_distribution"], ensure_ascii=False))
        lines.append("")

    leakage = audit.get("possible_train_test_leakage", {})
    lines.extend(
        [
            "## Possible train/test leakage",
            "",
            f"The lightweight check found **{leakage.get('unique_instruction_pairs', 0)}** "
            f"non-exact train/test instruction pairs at or above similarity "
            f"{leakage.get('threshold', 'N/A')}. Exact overlaps are reported separately above.",
            "",
        ]
    )
    if leakage.get("examples"):
        lines.extend(["| Similarity | Train instruction | Test instruction |", "| ---: | --- | --- |"])
        for match in leakage["examples"]:
            left = normalize_text(match["train"]["instruction"]).replace("|", "\\|")[:180]
            right = normalize_text(match["test"]["instruction"]).replace("|", "\\|")[:180]
            lines.append(f"| {match['similarity']:.4f} | {left} | {right} |")
        lines.append("")

    suspicious = audit.get("suspicious_records", [])
    lines.extend(["## Suspicious examples", ""])
    if not suspicious:
        lines.append("No parameter inconsistencies were detected by the implemented checks.")
    else:
        for item in suspicious:
            check = item["check"]
            lines.append(
                f"- `{item['split']}:{item['line']}` — instruction value "
                f"`{check.get('instruction_value', 'metadata')}`, output value "
                f"`{check.get('output_value', check.get('details', 'missing'))}`."
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Exact or near overlap means a random synthetic test split should not be treated as "
            "strong evidence of generalization. Parameter checks are intentionally conservative: "
            "`not_checkable` means the audit could not recover both values, not that the record is correct.",
            "",
        ]
    )
    return "\n".join(lines)


def _source_metadata(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        name: {
            "path": path.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in paths.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--validation", "--val", dest="validation", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/data_audit.md"))
    parser.add_argument("--near-threshold", type=float, default=0.9)
    parser.add_argument("--max-examples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.near_threshold <= 1:
        raise SystemExit("--near-threshold must be between 0 and 1")
    paths = {"train": args.train, "validation": args.validation, "test": args.test}
    splits = {name: load_jsonl(path) for name, path in paths.items()}
    sources = _source_metadata(paths)
    audit = audit_splits(
        splits, near_threshold=args.near_threshold, max_examples=args.max_examples
    )
    payload = {"sources": sources, "audit": audit}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(audit, sources), encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}.")


if __name__ == "__main__":
    main()
