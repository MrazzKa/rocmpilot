"""Backward-compatible entry point for the real base-vs-LoRA benchmark.

This file no longer writes a static, pre-judged report. See
``training/evaluate_benchmark.py`` for the implementation and CLI options.
"""

try:
    from training.evaluate_benchmark import main
except ModuleNotFoundError:  # Allows ``python training/evaluate.py``.
    from evaluate_benchmark import main


if __name__ == "__main__":
    main()
