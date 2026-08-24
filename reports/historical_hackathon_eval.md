# Historical hackathon evaluation claims

This file preserves the substance of the original static `reports/eval_results.md`
for historical transparency. The original script did not run inference or calculate
metrics; it wrote the comparison below from a hard-coded string. These statements
must not be cited as measured benchmark results.

## Original claimed setup

- Hardware: AMD Instinct MI300X
- Base model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- Fine-tuned adapter: ROCmPilot LoRA
- Dataset: 240+ synthetic instruction pairs

## Original qualitative table

| Test case | Claimed base-model weakness | Claimed ROCmPilot improvement | Original winner |
| --- | --- | --- | --- |
| `nvidia-smi` | General or hallucinated advice | Identified `rocm-smi` and verification commands | ROCmPilot |
| `cuda:0` | Left hardcoding or used a CPU fallback | Explained PyTorch ROCm namespace behavior | ROCmPilot |
| NVIDIA CUDA Docker base | Suggested NVIDIA-driver changes | Suggested a ROCm PyTorch image | ROCmPilot |
| HIP out of memory | Generic advice | Suggested ROCm-specific memory guidance | ROCmPilot |

The original prose further claimed that the adapter reliably followed the requested
Markdown structure and never hallucinated `nvcc`. No raw predictions or executable
comparison accompanied those statements, so they remain unverified.

