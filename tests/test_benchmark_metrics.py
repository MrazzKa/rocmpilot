import json
from pathlib import Path

import pytest

from training.benchmark_metrics import aggregate_results, evaluate_generation, rouge_l_f1
from training.evaluate_benchmark import load_examples, render_markdown_report


def test_metric_functions_on_mock_generation():
    example = {
        "required_concepts": [{"name": "HIP detection", "any_of": ["torch.version.hip"]}],
        "forbidden_or_incorrect_concepts": ["install CUDA toolkit"],
        "consistency_terms": ["fused_bias.cu"],
        "expected_sections": ["Summary", "Recommended fix"],
        "reference_answer": "Use torch.version.hip for fused_bias.cu.",
    }
    generation = (
        "## Summary\nUse torch.version.hip for fused_bias.cu.\n\n"
        "## Recommended fix\nTest the HIP build."
    )
    metrics = evaluate_generation(example, generation)
    assert metrics["structural_compliance"] == 1.0
    assert metrics["required_concept_coverage"] == 1.0
    assert metrics["forbidden_concept_avoidance"] == 1.0
    assert metrics["input_output_consistency"] == 1.0
    assert 0.0 < metrics["rouge_l_f1"] <= 1.0
    assert rouge_l_f1("a b c", "a b c") == 1.0


def test_report_requires_actual_results_and_uses_measured_values():
    with pytest.raises(ValueError):
        render_markdown_report({"status": "pending", "number_of_examples": 0})

    row = {
        "category": "mock",
        "metrics": {
            "base": {
                "structural_compliance": 0.0,
                "required_concept_coverage": 0.25,
                "forbidden_concept_avoidance": 0.5,
                "input_output_consistency": 0.75,
                "rouge_l_f1": 0.1,
            },
            "adapter": {
                "structural_compliance": 1.0,
                "required_concept_coverage": 0.5,
                "forbidden_concept_avoidance": 1.0,
                "input_output_consistency": 1.0,
                "rouge_l_f1": 0.2,
            },
        },
    }
    results = aggregate_results([row])
    results["experiment"] = {
        "base_model": "base",
        "adapter": "adapter",
        "dataset": "mock.jsonl",
        "device": "cpu",
        "max_new_tokens": 8,
        "seed": 1,
    }
    report = render_markdown_report(results)
    assert "0.0000" in report
    assert "+1.0000" in report
    assert "wins" not in report.lower()


def test_challenge_jsonl_is_valid_and_has_rubrics():
    path = Path("data/challenge_eval.jsonl")
    rows = load_examples(path)
    assert len(rows) >= 8
    for row in rows:
        assert row["reference_answer"]
        assert row["required_concepts"]
        assert row["sources"]
        reference_metrics = evaluate_generation(row, row["reference_answer"])
        assert reference_metrics["structural_compliance"] == 1.0
        assert reference_metrics["required_concept_coverage"] == 1.0
        assert reference_metrics["forbidden_concept_avoidance"] == 1.0
        assert reference_metrics["input_output_consistency"] == 1.0
        for source in row["sources"]:
            assert source["url"].startswith("https://")
