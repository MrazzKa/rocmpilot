# ROCmPilot

ROCmPilot is a CUDA-to-AMD-ROCm migration assistant built during the AMD Developer
Hackathon by lablab.ai (Track 2, team **AMDeus Ex Machina**). The project combines a
Qwen2.5-Coder base model, a PEFT LoRA adapter, and a Gradio application with a
deterministic rule-based demo mode.

This repository now separates the historical hackathon result from post-hackathon
data-quality and evaluation work. The published adapter exists and the training run is
logged; a defensible base-vs-adapter challenge evaluation has **not yet been run**.

## Evidence status

| Item | Status | Evidence / limitation |
| --- | --- | --- |
| LoRA training on AMD Instinct MI300X | Historical project result | A 15-step run and adapter save are recorded in `reports/training_log.txt`; exact environment versions were not pinned. |
| Published PEFT adapter | Available | [`MrazzKa/rocmpilot-qwen25-coder-lora`](https://huggingface.co/MrazzKa/rocmpilot-qwen25-coder-lora) |
| Historical synthetic corpus | Preserved | 250 records in `data/*.jsonl`; it contains duplicates, template leakage, and instruction/target mismatches. |
| Clean synthetic v2 | Generated | 26 consistent parameter variants under `data/clean_v2/`; parameter values are held out across splits, but all examples still use five narrow template families. |
| Independent challenge benchmark | Curated, not executed | 8 evaluation-only scenarios with rubrics and official AMD/PyTorch sources in `data/challenge_eval.jsonl`. |
| Base-vs-LoRA results | Pending | The real pipeline is implemented, but no model inference was run during this update. |

## What was done during the hackathon

- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Fine-tuning method: PEFT LoRA
- Reported compute: AMD Instinct MI300X on AMD Developer Cloud
- Historical corpus: 200 train, 25 validation, and 25 test records generated from
  five synthetic template families
- Logged run: 15 optimizer steps and 18,464,768 trainable parameters
- Output: a LoRA adapter published on Hugging Face
- Application: a Gradio interface with code, error-log, and environment analysis tabs

The training log shows decreasing **training loss**. Validation was loaded by the
historical script but disabled, so the run did not record validation loss. See
[`docs/experiment_history.md`](docs/experiment_history.md) for the exact distinction
between supported facts and unknown historical details.

## Post-hackathon audit

The audit confirmed that the original generator sampled template parameters
independently for instructions and targets. Among the checks that could recover both
values, it found 125 inconsistent train records, 20 inconsistent validation records,
and 14 inconsistent test records. The train split contains only 85 unique content
records out of 200; train and test share 15 exact content groups and 16 distinct
instruction strings.

The old `training/evaluate.py` also wrote a static comparison instead of loading the
models. Its claims are retained as historical, unmeasured claims in
[`reports/historical_hackathon_eval.md`](reports/historical_hackathon_eval.md), not as
benchmark results.

Full findings:

- Human-readable audit: [`reports/data_audit.md`](reports/data_audit.md)
- Machine-readable audit: [`reports/data_audit.json`](reports/data_audit.json)
- Pre-change interpretation: [`docs/research_audit.md`](docs/research_audit.md)

## Data layout

```text
data/
├── train.jsonl, val.jsonl, test.jsonl  # historical adapter data; preserved
├── clean_v2/                           # corrected synthetic data + manifest
└── challenge_eval.jsonl                # evaluation only; never training data
```

`training/generate_dataset.py` samples every parameter once per record and reuses it
in the instruction and target. A fixed seed makes generation deterministic. The
default emits each available parameter value once rather than inflating the corpus
with exact repeats. Its parameter-holdout split prevents the same parameter value
from crossing splits.

This does not solve template-level similarity: clean v2 still has only five families.
Its validation split is useful for pipeline diagnostics, not a strong generalization
claim. The separate challenge set tests more varied behavior such as PyTorch HIP
backend detection, distributed backend naming, container device exposure, HIPIFY
limitations, GPU isolation, telemetry parser migration, allocator reasoning, and C++
version guards.

## Real base-vs-LoRA evaluation

`training/evaluate_benchmark.py` runs the same prompts with deterministic decoding
(`do_sample=False`) first on the base model and then on the published adapter. It
saves every prompt, rubric, reference, generation, and per-example metric breakdown.

Metrics are reported overall and by category:

- expected Markdown-section compliance;
- required-concept coverage;
- forbidden/incorrect-concept avoidance;
- item-specific input/output consistency;
- ROUGE-L as a secondary overlap description only;
- optional paired bootstrap intervals for adapter-minus-base differences.

These lexical metrics are interpretable but imperfect. They do not replace expert
review or execution of generated code. The report generator refuses to create a
completed report without actual predictions. Current status:
[`reports/eval_results.md`](reports/eval_results.md).

An evaluation run writes:

```text
reports/benchmark_predictions.jsonl
reports/benchmark_results.json
reports/benchmark_results.csv
reports/eval_results.md
```

## Setup and demo

```bash
python -m venv .venv
# Activate the environment using the command for your shell.
python -m pip install -r requirements.txt
python app.py
```

The default app mode is deterministic and rule-based so it can run on a CPU-hosted
Space. To load the adapter in the application, set `USE_LIVE_MODEL=true`. Live mode
downloads model artifacts and can be slow on CPU.

- [Hugging Face Space](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/ROCmPilot)
- [Published adapter](https://huggingface.co/MrazzKa/rocmpilot-qwen25-coder-lora)

## Reproducible data and audit commands

Generate clean v2 without touching the historical JSONL files:

```bash
python training/generate_dataset.py --seed 42
```

The generator refuses to overwrite an existing clean dataset unless `--overwrite` is
explicitly supplied.

Audit the historical corpus:

```bash
python training/audit_dataset.py \
  --train data/train.jsonl \
  --validation data/val.jsonl \
  --test data/test.jsonl \
  --output-json reports/data_audit.json \
  --output-md reports/data_audit.md
```

Audit clean v2 by replacing the three input paths and output names with the
`data/clean_v2/*` and `reports/data_audit_clean_v2.*` paths.

## Run the challenge evaluation

Install model-training/evaluation dependencies first:

```bash
python -m pip install -r requirements-training.txt
python training/evaluate_benchmark.py \
  --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --adapter MrazzKa/rocmpilot-qwen25-coder-lora \
  --dataset data/challenge_eval.jsonl \
  --output-dir reports \
  --device auto \
  --max-new-tokens 512 \
  --seed 42
```

This downloads the base model and adapter if they are not cached. CPU execution is
supported but expected to be slow. Use `--limit 1` for a pipeline smoke test; do not
present that smoke test as the benchmark.

## Future training on clean v2

The following starts a **new experiment** with validation and checkpointing; it does
not reproduce or overwrite the published adapter:

```bash
python training/train_lora.py \
  --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --train-file data/clean_v2/train.jsonl \
  --val-file data/clean_v2/val.jsonl \
  --output-dir outputs/rocmpilot-clean-v2-lora \
  --epochs 3 \
  --max-steps -1 \
  --evaluation-strategy epoch \
  --save-strategy epoch \
  --load-best-model-at-end \
  --seed 42
```

The output directory receives checkpoints, trainer state, tokenizer/adapter files,
and `experiment_metadata.json` with arguments, dataset sizes, package versions, and
backend information. For a serious new training run, create a larger independently
reviewed training corpus and lock the environment on the target ROCm host; clean v2
is intentionally small and narrow.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover deterministic/consistent generation, parameter-holdout splitting,
duplicate and leakage detection, challenge JSONL validity, benchmark metrics, and the
requirement that completed reports derive from actual result objects.

## Limitations

- No measured base-vs-LoRA result is currently committed.
- The published adapter was trained on flawed, highly repetitive synthetic data.
- The challenge benchmark is small, and rubric matching can miss valid paraphrases or
  fail to detect subtle technical errors.
- ROCm interfaces and package support change over time; benchmark references are
  stored per item and guidance must be rechecked for the deployment version.
- Generated migrations require execution and expert review on the target hardware.

## License

MIT License. See [`LICENSE`](LICENSE).
