# Experiment 1 manual review

This review covers the eight held-out challenge examples in the completed Colab T4
run. Technical correctness and usefulness take priority over lexical overlap. The
raw generations and automatic metric details remain unchanged in
`benchmark_predictions.jsonl`.

## Summary

| Verdict | Count |
| --- | ---: |
| Base | 2 |
| LoRA | 0 |
| Tie | 3 |
| Unclear | 3 |

The historical LoRA learned the requested response structure much more strongly
than the base model, but this run does not show improved ROCm technical knowledge or
held-out generalization.

## Per-example assessment

| Example | Verdict | Reasoning |
| --- | --- | --- |
| `challenge-pytorch-backend-detection-001` | Unclear | The base answer incorrectly treats Apple MPS as ROCm/HIP. The LoRA answer uses `torch.cuda.is_available()` but does not distinguish CUDA from HIP with `torch.version.cuda` and `torch.version.hip`, then degrades into prompt repetition. Both miss the documented reuse of the `torch.cuda` namespace on ROCm. |
| `challenge-distributed-backend-001` | Unclear | Both preserve the public PyTorch backend name `nccl`, but neither clearly explains that RCCL is the underlying ROCm implementation or supplies the requested `gloo` CPU fallback. The base makes a misleading MI300X support claim; the LoRA uses an NVIDIA verification command and then produces a long list of unsupported claims. |
| `challenge-compose-device-passthrough-001` | Tie | Both answers fail the requested ROCm device exposure fix. The base recommends NVIDIA drivers and `/dev/nvidia0`; the LoRA recommends a Docker `-gpus all` invocation. Neither adds `/dev/kfd` and `/dev/dri` to the Compose `devices` section. |
| `challenge-hipify-custom-kernel-001` | Unclear | The base incorrectly keeps `nvcc` and adds an NVIDIA `sm_75` target. The LoRA recognizes that renaming is insufficient and mentions a HIP compiler, but its corrected command still uses `nvcc` with CUDA flags. Neither provides the required HIPIFY scoping, manual review, and correctness-testing plan. |
| `challenge-gpu-isolation-001` | Tie | The base invents `AMD_VISIBLE_DEVICES`; the LoRA proposes discovering a UUID and passing it to an unspecified `--gpus` option. Neither recommends `ROCR_VISIBLE_DEVICES` for UUID selection or `CUDA_VISIBLE_DEVICES` as the portable index-only alternative. |
| `challenge-monitoring-parser-001` | Tie | The base proposes parsing `rocminfo`, while the LoRA performs the blind `nvidia-smi` to `rocm-smi` substitution the prompt warns against. Neither proposes AMD SMI, a vendor-neutral telemetry schema, version-aware field mapping, and fixture-based parser tests. |
| `challenge-pytorch-empty-cache-001` | Base | The base correctly identifies retained tensor references as the underlying problem and states that the list still needs further work, although its proposed code remains incomplete. The LoRA repeats the insufficient `empty_cache()` fix and later drifts into unrelated generated content. |
| `challenge-version-guard-001` | Base | The base at least attempts separate CUDA and HIP conditions, although it incorrectly treats `__HIPCC__` as a comparable version value. The LoRA replaces the CUDA branch with an invented `AMDGPU_VERSION` condition rather than keeping separate supported implementations, so it is further from the requested reasoning. |

## Metric interpretation

- Structural compliance improved from `0.1250` to `0.7969`.
- Required-concept coverage remained `0.2083` for both models.
- Forbidden-concept avoidance decreased from `1.0000` to `0.8750`.
- Input/output consistency decreased from `0.9375` to `0.8750`.
- ROUGE-L F1 changed from `0.1722` to `0.1768`; this small lexical-overlap
  difference is not evidence of correctness.

The forbidden-concept metric produced at least one false positive: the LoRA answer
quoted `NVIDIA_VISIBLE_DEVICES` while labelling it incorrect, but the lexical matcher
still counted the phrase as a forbidden hit.

## Limitations

- The benchmark contains only eight examples.
- Many generations reached the 512-token limit and contain repetition or truncated
  endings; the saved output does not include a model `finish_reason`.
- Lexical rubric metrics cannot reliably distinguish endorsement from a quoted or
  negated incorrect concept.
- The small benchmark and descriptive bootstrap intervals do not establish
  statistical significance or broad generalization claims.

## Artifact identity

The source archive SHA-256 is
`6BCA44DF69129A8B510A756E9CE0DE3B9E752C531EAE49F594D515BAFA608CC1`.
