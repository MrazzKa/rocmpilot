---
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
  - base_model:adapter:Qwen/Qwen2.5-Coder-1.5B-Instruct
  - lora
  - transformers
  - rocm
---

# ROCmPilot Qwen2.5-Coder LoRA adapter

This PEFT LoRA adapter was created during the AMD Developer Hackathon by lablab.ai
to explore structured CUDA-to-AMD-ROCm migration guidance. It is an experimental
hackathon artifact, not a validated migration or code-correctness system.

## Model details

- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Method: PEFT LoRA, rank 16, alpha 32, dropout 0.05
- Target modules: attention projections and MLP projections
- Trainable parameters reported in the training log: 18,464,768
- Hardware reported by the project: AMD Instinct MI300X
- Logged training run: 15 optimizer steps

## Training data

The adapter was trained on the repository's original synthetic dataset: 200 training
records drawn from five narrow template families. A post-hackathon audit found exact
duplicates, cross-split template leakage, and instruction/target parameter mismatches.
The corrected `data/clean_v2/` dataset in the repository was created later and was
**not** used to train this published adapter.

## Evaluation status

The original repository contained a static qualitative comparison, not executable
base-vs-adapter inference. A reproducible challenge pipeline has since been added, but
results are pending until the base model and adapter are actually run and raw
generations are saved. Do not claim measured improvement from this model card.

## Intended use

- Research or educational exploration of LoRA adaptation and ROCm migration prompts
- Drafting migration checklists that will be reviewed and tested by a developer
- Reproducing the repository's benchmark protocol

## Out-of-scope use

- Unreviewed production migration
- Hardware, driver, dependency, or security guarantees
- Treating generated commands as safe for every ROCm version and host configuration
- Using the challenge evaluation set as training data

## Risks and limitations

- The training data is synthetic, repetitive, and affected by a confirmed consistency
  bug.
- ROCm, PyTorch, containers, and third-party package compatibility change over time.
- The adapter can emit plausible but incorrect commands, versions, flags, or API
  substitutions.
- The logged run contains training loss but no validation loss.

Every suggestion should be checked against official documentation for the deployed
versions and executed in a controlled target environment.

## Loading the adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_id = "MrazzKa/rocmpilot-qwen25-coder-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForCausalLM.from_pretrained(base_id)
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()
```

See the ROCmPilot repository for prompt formatting, the historical training log,
dataset audits, challenge rubrics, and the real base-vs-LoRA evaluation script.
