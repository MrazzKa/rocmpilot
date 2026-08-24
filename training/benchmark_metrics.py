"""Interpretable, dependency-free metrics for the ROCmPilot challenge benchmark."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable


DEFAULT_SECTIONS = [
    "Summary",
    "Detected ROCm issue",
    "Recommended fix",
    "Corrected code or config",
    "Verification commands",
    "ROCm readiness score",
    "Cursor prompt",
    "Notes and limitations",
]

SCALAR_METRICS = [
    "structural_compliance",
    "required_concept_coverage",
    "forbidden_concept_avoidance",
    "input_output_consistency",
    "rouge_l_f1",
]


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _concept_parts(concept: Any) -> tuple[str, list[str]]:
    if isinstance(concept, str):
        return concept, [concept]
    if not isinstance(concept, dict):
        raise TypeError(f"Concept must be a string or object, got {type(concept).__name__}")
    terms = concept.get("any_of", [])
    if isinstance(terms, str):
        terms = [terms]
    name = str(concept.get("name") or " / ".join(str(term) for term in terms))
    return name, [str(term) for term in terms]


def match_concepts(text: str, concepts: Iterable[Any]) -> dict[str, Any]:
    normalized_text = _normalized(text)
    matched: list[str] = []
    missing: list[str] = []
    for concept in concepts:
        name, terms = _concept_parts(concept)
        if terms and any(_normalized(term) in normalized_text for term in terms):
            matched.append(name)
        else:
            missing.append(name)
    total = len(matched) + len(missing)
    return {
        "score": len(matched) / total if total else 1.0,
        "matched": matched,
        "missing": missing,
        "total": total,
    }


def structural_compliance(text: str, sections: Iterable[str]) -> dict[str, Any]:
    headings = {
        _normalized(match.group(1))
        for match in re.finditer(r"(?m)^##\s+(.+?)\s*$", text)
    }
    expected = list(sections)
    present = [section for section in expected if _normalized(section) in headings]
    missing = [section for section in expected if _normalized(section) not in headings]
    return {
        "score": len(present) / len(expected) if expected else 1.0,
        "present": present,
        "missing": missing,
        "total": len(expected),
    }


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute token-level ROUGE-L F1 as a secondary descriptive metric."""
    predicted = _normalized(prediction).split()
    expected = _normalized(reference).split()
    if not predicted or not expected:
        return 1.0 if predicted == expected else 0.0
    previous = [0] * (len(expected) + 1)
    for predicted_token in predicted:
        current = [0]
        for index, expected_token in enumerate(expected, 1):
            if predicted_token == expected_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(predicted)
    recall = lcs / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_generation(example: dict[str, Any], generation: str) -> dict[str, Any]:
    sections = structural_compliance(
        generation, example.get("expected_sections", DEFAULT_SECTIONS)
    )
    required = match_concepts(generation, example.get("required_concepts", []))
    forbidden_hits = match_concepts(
        generation, example.get("forbidden_or_incorrect_concepts", [])
    )
    consistency = match_concepts(generation, example.get("consistency_terms", []))
    forbidden_avoidance = 1.0 - forbidden_hits["score"] if forbidden_hits["total"] else 1.0
    reference = str(example.get("reference_answer", example.get("output", "")))
    return {
        "structural_compliance": sections["score"],
        "required_concept_coverage": required["score"],
        "forbidden_concept_avoidance": forbidden_avoidance,
        "input_output_consistency": consistency["score"],
        "rouge_l_f1": rouge_l_f1(generation, reference),
        "details": {
            "sections": sections,
            "required_concepts": required,
            "forbidden_concepts": {
                "matched": forbidden_hits["matched"],
                "not_matched": forbidden_hits["missing"],
                "total": forbidden_hits["total"],
            },
            "consistency_terms": consistency,
        },
    }


def _mean_metrics(rows: list[dict[str, Any]], model_key: str) -> dict[str, float]:
    if not rows:
        return {metric: 0.0 for metric in SCALAR_METRICS}
    return {
        metric: fmean(float(row["metrics"][model_key][metric]) for row in rows)
        for metric in SCALAR_METRICS
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]], *, samples: int, seed: int
) -> dict[str, dict[str, float]]:
    if samples <= 0 or len(rows) < 2:
        return {}
    rng = random.Random(seed)
    intervals: dict[str, dict[str, float]] = {}
    for metric in SCALAR_METRICS:
        differences = [
            float(row["metrics"]["adapter"][metric])
            - float(row["metrics"]["base"][metric])
            for row in rows
        ]
        bootstrap = []
        for _ in range(samples):
            bootstrap.append(fmean(rng.choice(differences) for _ in differences))
        bootstrap.sort()
        low = bootstrap[int(0.025 * (samples - 1))]
        high = bootstrap[int(0.975 * (samples - 1))]
        intervals[metric] = {
            "low": low,
            "high": high,
            "method": "paired percentile bootstrap",
            "samples": samples,
        }
    return intervals


def aggregate_results(
    rows: list[dict[str, Any]], *, bootstrap_samples: int = 0, seed: int = 42
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate benchmark results without predictions")

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        base = _mean_metrics(group, "base")
        adapter = _mean_metrics(group, "adapter")
        return {
            "n": len(group),
            "base": base,
            "adapter": adapter,
            "difference_adapter_minus_base": {
                metric: adapter[metric] - base[metric] for metric in SCALAR_METRICS
            },
        }

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category", "unknown"))].append(row)
    overall = summarize(rows)
    overall["bootstrap_95_percent_ci_for_difference"] = _paired_bootstrap(
        rows, samples=bootstrap_samples, seed=seed
    )
    return {
        "status": "completed",
        "number_of_examples": len(rows),
        "metric_scale": "0 to 1; higher is better",
        "overall": overall,
        "by_category": {
            category: summarize(group) for category, group in sorted(by_category.items())
        },
        "metric_notes": {
            "structural_compliance": "Fraction of expected Markdown sections present.",
            "required_concept_coverage": "Lexical rubric coverage; synonyms must be declared in the dataset.",
            "forbidden_concept_avoidance": "Fraction of declared incorrect concepts not detected.",
            "input_output_consistency": "Coverage of item-specific terms that the answer should preserve.",
            "rouge_l_f1": "Secondary token-overlap description; not a correctness measure for open-ended migration guidance.",
        },
    }
