# ROCmPilot Base-vs-LoRA Evaluation

**Evaluation framework implemented; benchmark results pending.**

No base-vs-adapter inference has been run as part of this repository update. This
machine does not currently have the model dependencies installed, and downloading
the base model plus running two sets of generations would be a substantial operation.

Run `training/evaluate_benchmark.py` to create this report from actual generations.
The same run also writes:

- `reports/benchmark_predictions.jsonl`
- `reports/benchmark_results.json`
- `reports/benchmark_results.csv`

The previous static, non-measured comparison is retained for transparency in
`reports/historical_hackathon_eval.md`; it is not benchmark evidence.
