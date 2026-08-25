# ROCmPilot Base-vs-LoRA Evaluation

This report was generated from saved model generations. It does not use an LLM judge, and lexical rubric checks should not be interpreted as proof of technical correctness.

## Setup

- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Adapter: `MrazzKa/rocmpilot-qwen25-coder-lora`
- Dataset: `data/challenge_eval.jsonl`
- Examples: 8
- Device: `cuda`
- Deterministic decoding: `do_sample=False`, max new tokens 512, seed 42

## Overall results

| Metric | Base | LoRA adapter | Difference (adapter − base) |
| --- | ---: | ---: | ---: |
| structural compliance | 0.1250 | 0.7969 | +0.6719 |
| required concept coverage | 0.2083 | 0.2083 | +0.0000 |
| forbidden concept avoidance | 1.0000 | 0.8750 | -0.1250 |
| input output consistency | 0.9375 | 0.8750 | -0.0625 |
| rouge l f1 | 0.1722 | 0.1768 | +0.0046 |

Paired percentile bootstrap intervals are descriptive because this benchmark is small; they should not be presented as a broad significance claim.

| Metric | 95% bootstrap interval for difference |
| --- | ---: |
| structural compliance | [+0.3125, +0.9219] |
| required concept coverage | [-0.0938, +0.0938] |
| forbidden concept avoidance | [-0.3750, +0.0000] |
| input output consistency | [-0.2500, +0.1250] |
| rouge l f1 | [-0.0143, +0.0249] |

## Per-category results

Each category may contain few examples. Report the sample count with every score.

### custom_kernels (n=2)

| Metric | Base | Adapter | Difference |
| --- | ---: | ---: | ---: |
| structural compliance | 0.5000 | 1.0000 | +0.5000 |
| required concept coverage | 0.0000 | 0.1250 | +0.1250 |
| forbidden concept avoidance | 1.0000 | 1.0000 | +0.0000 |
| input output consistency | 0.7500 | 0.7500 | +0.0000 |
| rouge l f1 | 0.1926 | 0.1736 | -0.0190 |

### deployment (n=2)

| Metric | Base | Adapter | Difference |
| --- | ---: | ---: | ---: |
| structural compliance | 0.0000 | 0.5000 | +0.5000 |
| required concept coverage | 0.2500 | 0.1250 | -0.1250 |
| forbidden concept avoidance | 1.0000 | 0.5000 | -0.5000 |
| input output consistency | 1.0000 | 1.0000 | +0.0000 |
| rouge l f1 | 0.1673 | 0.1973 | +0.0300 |

### distributed (n=1)

| Metric | Base | Adapter | Difference |
| --- | ---: | ---: | ---: |
| structural compliance | 0.0000 | 1.0000 | +1.0000 |
| required concept coverage | 0.6667 | 0.6667 | +0.0000 |
| forbidden concept avoidance | 1.0000 | 1.0000 | +0.0000 |
| input output consistency | 1.0000 | 1.0000 | +0.0000 |
| rouge l f1 | 0.2012 | 0.1677 | -0.0335 |

### monitoring (n=1)

| Metric | Base | Adapter | Difference |
| --- | ---: | ---: | ---: |
| structural compliance | 0.0000 | 1.0000 | +1.0000 |
| required concept coverage | 0.0000 | 0.0000 | +0.0000 |
| forbidden concept avoidance | 1.0000 | 1.0000 | +0.0000 |
| input output consistency | 1.0000 | 1.0000 | +0.0000 |
| rouge l f1 | 0.1331 | 0.1533 | +0.0203 |

### pytorch_runtime (n=2)

| Metric | Base | Adapter | Difference |
| --- | ---: | ---: | ---: |
| structural compliance | 0.0000 | 0.6875 | +0.6875 |
| required concept coverage | 0.2500 | 0.2500 | +0.0000 |
| forbidden concept avoidance | 1.0000 | 1.0000 | +0.0000 |
| input output consistency | 1.0000 | 0.7500 | -0.2500 |
| rouge l f1 | 0.1616 | 0.1758 | +0.0142 |

## Metric limitations

- **structural compliance:** Fraction of expected Markdown sections present.
- **required concept coverage:** Lexical rubric coverage; synonyms must be declared in the dataset.
- **forbidden concept avoidance:** Fraction of declared incorrect concepts not detected.
- **input output consistency:** Coverage of item-specific terms that the answer should preserve.
- **rouge l f1:** Secondary token-overlap description; not a correctness measure for open-ended migration guidance.

Raw prompts, references, generations, and per-example metric details are stored in `benchmark_predictions.jsonl`.
