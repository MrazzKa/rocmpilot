# Post-hackathon research audit

This document records findings made before the research-oriented changes in this
repository. It distinguishes inspection results from new experiments.

## Confirmed findings

1. The original `training/generate_dataset.py` called `random.choice()` separately
   when formatting the instruction and target. The instruction and output could
   therefore refer to different devices, commands, images, or dependencies.
2. The historical corpus contains 250 records generated from five template families
   (50 records per family) and was randomly split into 200 train, 25 validation, and
   25 test records.
3. Content-level inspection found only 85 unique full records in train, 22 in
   validation, and 22 in test. Exact full-record overlap was 16 groups between train
   and validation, 15 between train and test, and 3 between validation and test.
4. The original test split shared 16 distinct instruction strings with train. It is
   therefore not a defensible generalization benchmark.
5. Parameter checks found 125 detectable mismatches in train, 20 in validation, and
   14 in test. The historical OOM template did not repeat allocation size in its
   output and is conservatively classified as not checkable by the automated audit.
6. The original `training/evaluate.py` did not load a model or adapter. It wrote a
   fixed Markdown comparison containing preselected winners and absolute claims.
7. The original training script loaded and tokenized validation data but configured
   `eval_strategy="no"`. It also used `save_strategy="no"`, so no validation loss or
   checkpoints were produced by that script.
8. `reports/training_log.txt` supports a historical 15-step LoRA run with 200 train
   examples, 25 validation examples loaded, 18,464,768 trainable parameters, and
   decreasing training loss. It does not contain validation metrics or a base-model
   comparison.

The machine-readable audit in `reports/data_audit.json` and its Markdown rendering
are generated from the preserved historical JSONL files. Counts in those reports
should be preferred if the data changes in a future commit.

## Consequences

- The published adapter remains a valid historical hackathon artifact, but its
  existing evidence does not establish held-out generalization or superiority over
  the base model.
- `data/*.jsonl` remains unchanged. Corrected synthetic data is versioned under
  `data/clean_v2/` and must not be described as the published adapter's training data.
- `data/challenge_eval.jsonl` is evaluation-only and must never be included in a
  training command.

