# Experiment history and future protocol

## Historical hackathon run

The following facts are supported by the committed code, training log, and project
history. They are not a reconstruction of every package version or random state.

- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Method: PEFT LoRA with rank 16, alpha 32, dropout 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj`
- Historical files: `data/train.jsonl` (200 records) and `data/val.jsonl` (25 records)
- Logged run: 15 optimizer steps, batch size 1, gradient accumulation 8,
  learning rate `2e-4`, BF16 when a PyTorch GPU backend was available
- Trainable parameters: 18,464,768
- Hardware reported by the project: AMD Instinct MI300X on AMD Developer Cloud
- Saved adapter: `MrazzKa/rocmpilot-qwen25-coder-lora`

Important caveats:

- Historical dataset generation had no fixed seed and contained instruction/target
  mismatches and cross-split duplicates.
- Validation was loaded but disabled. The log contains training loss only.
- Exact library versions are not pinned in the historical requirements file.
- Re-running the current script is a new experiment, not a byte-for-byte reproduction
  of the published adapter.

## Experiment 1: held-out base-vs-LoRA benchmark

Experiment 1 was run once on a Tesla T4 with 14.56 GiB VRAM. It compared
`Qwen/Qwen2.5-Coder-1.5B-Instruct` with the historical published ROCmPilot LoRA on
all eight held-out examples in `data/challenge_eval.jsonl`. Decoding was
deterministic (`do_sample=False`, seed 42, maximum 512 new tokens) and used identical
settings for both model passes.

| Metric | Base | Historical LoRA |
| --- | ---: | ---: |
| Structural compliance | 0.1250 | 0.7969 |
| Required-concept coverage | 0.2083 | 0.2083 |

Manual review assigned 2 examples to the base model, 0 to LoRA, 3 ties, and 3
unclear verdicts. The historical LoRA learned the requested response structure
strongly, but did not demonstrate improved ROCm technical knowledge or held-out
generalization. Raw outputs, environment metadata, automatic results, and the full
manual review are preserved under `reports/runs/colab-t4-2026-08-25/`.

Limitations:

- The benchmark contains only eight examples.
- Many generations reached the 512-token limit.
- The lexical forbidden-concept metric produced at least one false positive by
  matching an incorrect concept that the answer quoted while rejecting it.
- These results do not establish statistical significance or broad generalization
  claims.

## Recommended future protocol

Future runs should use an explicit seed, versioned inputs, active validation,
checkpoint saving, and run-specific `experiment_metadata.json`. The revised training
script defaults to epoch-based validation and checkpointing and can select the lowest
validation-loss checkpoint.

The cleaned synthetic data remains narrow and repetitive. Its validation loss can be
used for training diagnostics, not as the main generalization claim. Final comparison
should use the permanently separate, manually curated `data/challenge_eval.jsonl` and
save all raw generations.

Never train on `data/challenge_eval.jsonl`.
