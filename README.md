# ROCmPilot

[![tests](https://github.com/MrazzKa/rocmpilot/actions/workflows/tests.yml/badge.svg)](https://github.com/MrazzKa/rocmpilot/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ROCmPilot is an experimental CUDA-to-AMD-ROCm migration assistant created for the
AMD Developer Hackathon by lablab.ai (Track 2, team **AMDeus Ex Machina**). It
combines a Qwen2.5-Coder base model, a historical PEFT LoRA adapter, a Gradio demo,
and a reproducible research pipeline for auditing data and comparing model outputs.

> **Research result:** the held-out benchmark shows that the historical LoRA learned
> the requested response structure strongly, but did **not** demonstrate better ROCm
> technical knowledge or generalization than the base model.

- [Live Gradio demo](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/ROCmPilot)
- [Published LoRA adapter](https://huggingface.co/MrazzKa/rocmpilot-qwen25-coder-lora)
- [Experiment history](docs/experiment_history.md)
- [Post-hackathon research audit](docs/research_audit.md)

## Experiment 1: held-out base vs. LoRA

Experiment 1 ran once on a Tesla T4 with 14.56 GiB VRAM. It compared
`Qwen/Qwen2.5-Coder-1.5B-Instruct` with the historical ROCmPilot LoRA on eight
evaluation-only challenge examples using deterministic decoding (`do_sample=False`,
seed 42, maximum 512 new tokens).

| Metric | Base | Historical LoRA | LoRA - Base |
| --- | ---: | ---: | ---: |
| Structural compliance | 0.1250 | 0.7969 | +0.6719 |
| Required-concept coverage | 0.2083 | 0.2083 | 0.0000 |
| Forbidden-concept avoidance | 1.0000 | 0.8750 | -0.1250 |
| Input/output consistency | 0.9375 | 0.8750 | -0.0625 |
| ROUGE-L F1 | 0.1722 | 0.1768 | +0.0046 |

Manual technical review: **Base 2, LoRA 0, Tie 3, Unclear 3**. The automatic
metrics capture formatting and declared lexical concepts; they are not correctness
judges.

Experiment artifacts are immutable and versioned under
[`reports/runs/colab-t4-2026-08-25/`](reports/runs/colab-t4-2026-08-25/):

- [generated evaluation report](reports/runs/colab-t4-2026-08-25/eval_results.md);
- [manual review of all eight pairs](reports/runs/colab-t4-2026-08-25/manual_review.md);
- [raw base and LoRA generations](reports/runs/colab-t4-2026-08-25/benchmark_predictions.jsonl);
- [machine-readable metrics](reports/runs/colab-t4-2026-08-25/benchmark_results.json);
- [hardware and package metadata](reports/runs/colab-t4-2026-08-25/evaluation_environment.txt).

## Evidence status

| Item | Status | Evidence or limitation |
| --- | --- | --- |
| Historical LoRA training | Recorded project result | A 15-step run on a reported AMD Instinct MI300X is preserved in `reports/training_log.txt`; validation was disabled and exact package versions were not pinned. |
| Published adapter | Available | The PEFT adapter is hosted on Hugging Face. |
| Historical synthetic corpus | Preserved and audited | The 250 records in `data/*.jsonl` contain duplicates, cross-split leakage, and instruction/target mismatches. |
| Clean synthetic v2 | Generated, not trained | The corrected 26-record corpus under `data/clean_v2/` is useful for pipeline tests but remains narrow and synthetic. |
| Held-out challenge set | Executed | Eight manually curated, evaluation-only examples with declared rubrics and official sources. |
| Base-vs-LoRA comparison | Completed | Raw generations, metrics, environment metadata, and manual verdicts are committed. |

## Repository map

| Path | Purpose |
| --- | --- |
| `app.py`, `src/` | Gradio interface, deterministic demo engine, and optional live-model loader |
| `training/evaluate_benchmark.py` | Reproducible base-vs-LoRA generation and reporting pipeline |
| `training/audit_dataset.py` | Duplicate, overlap, consistency, and leakage audit |
| `training/generate_dataset.py` | Deterministic corrected synthetic-data generator |
| `training/train_lora.py` | Revised training entry point for future experiments; not a reproduction of the historical adapter |
| `data/challenge_eval.jsonl` | Evaluation only; it must never be used for training |
| `reports/runs/colab-t4-2026-08-25/` | Immutable Experiment 1 results |
| `notebooks/rocmpilot_benchmark_colab.ipynb` | Colab workflow for executing the GPU benchmark protocol |
| `docs/` | Historical evidence boundaries, audit findings, and future protocol |

## Quick start: deterministic demo

The default application does not download model weights. It runs a deterministic,
rule-based demonstration suitable for a CPU machine or a quick project review.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python app.py
```

Set `USE_LIVE_MODEL=true` only after installing `requirements-training.txt` and when
the machine has enough memory for the base model plus adapter. If live loading fails,
the application explicitly falls back to the labelled rule-based demo rather than
presenting the base model as the fine-tuned system.

## Reproduce audits

Generate clean v2 without modifying the historical JSONL files:

```bash
python training/generate_dataset.py --seed 42
```

Audit the historical corpus:

```bash
python training/audit_dataset.py \
  --train data/train.jsonl \
  --validation data/val.jsonl \
  --test data/test.jsonl \
  --output-json reports/data_audit.json \
  --output-md reports/data_audit.md
```

Generated reports are available as
[`reports/data_audit.md`](reports/data_audit.md) and
[`reports/data_audit_clean_v2.md`](reports/data_audit_clean_v2.md).

## Evaluation protocol

The completed Experiment 1 must not be overwritten or presented as a larger study.
The evaluation script remains available for an explicitly named future replication,
and the [Colab notebook](notebooks/rocmpilot_benchmark_colab.ipynb) provides GPU,
metadata, smoke-test, confirmation, full-run, review, and result-download cells.

The protocol:

1. runs identical prompts and decoding settings for base and adapter;
2. saves all prompts, references, generations, rubrics, and per-example metrics;
3. keeps `data/challenge_eval.jsonl` permanently separate from training;
4. treats lexical scores as descriptive and requires manual technical review.

Use a new output directory for any future replication. Never replace the committed
Experiment 1 files after inspecting model outputs.

## Historical training context

The published adapter was trained on 200 historical training records from five
synthetic template families. The log records 18,464,768 trainable parameters and
decreasing training loss, but no validation metrics. A later audit found only 85
unique train records, cross-split overlap, and many parameter mismatches.

The corrected `data/clean_v2/` corpus was created after the hackathon and was not used
for the published adapter. Training on it would be a new experiment and is not part
of the committed result.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check app.py src training tests
python -m compileall -q app.py src training tests
```

Tests cover deterministic data generation, parameter consistency, split separation,
audit findings, challenge-set validity, benchmark metrics, and report generation.
GitHub Actions runs the same checks on pushes and pull requests.

## Limitations

- Experiment 1 has only eight examples.
- Many generations reached the 512-token limit and showed repetition or truncated
  endings.
- At least one lexical forbidden-concept result was a false positive because the
  model quoted an incorrect concept while rejecting it.
- The historical adapter was trained on flawed, repetitive synthetic data.
- The benchmark does not establish statistical significance or broad
  generalization.
- Generated migration advice requires official-documentation checks, execution, and
  expert review on the target ROCm environment.

## License

MIT License. See [LICENSE](LICENSE).
