---
title: ROCmPilot
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.36.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# ROCmPilot

**Team:** AMDeus Ex Machina  
**Hackathon:** AMD Developer Hackathon by lablab.ai  
**Track:** Track 2: Fine-Tuning on AMD GPUs  

---

## Description
**ROCmPilot** is a specialized, fine-tuned AI code migration assistant designed to help developers, ML engineers, and MLOps/DevOps professionals seamlessly migrate CUDA-first AI workloads to **AMD ROCm** environments. 

When migrating from NVIDIA GPUs to AMD Instinct accelerators (like the MI300X), developers often encounter hidden hurdles: hardcoded `cuda:0` devices, CUDA-specific Docker base images, NVIDIA-only PyTorch extensions, and opaque runtime errors. 

General LLMs often provide generic, unhelpful answers when dealing with niche ROCm issues. ROCmPilot is fine-tuned specifically on a dataset of CUDA-to-ROCm migration scenarios to provide **concrete fixes, verification commands, and environment-specific debugging steps.**

---

## Problem & Solution
**The Problem:** The AI ecosystem currently has a strong bias towards CUDA. Moving workloads to AMD hardware is highly cost-effective and performant, but developers often lack the domain knowledge to troubleshoot PyTorch ROCm builds, vLLM serving configurations, or Docker container migrations.

**The Solution:** ROCmPilot analyzes Python code, runtime error logs, and Dockerfiles to pinpoint CUDA assumptions and provides actionable ROCm-compatible replacements. 

---

## Track 2: Fine-Tuning on AMD GPUs
We built a synthetic instruction dataset of 240+ common CUDA-to-ROCm migration issues. We then used **PEFT LoRA** to fine-tune `Qwen/Qwen2.5-Coder-1.5B-Instruct` directly on an **AMD Instinct MI300X** instance provided via the AMD Developer Cloud. This model is explicitly adapted to understand ROCm ecosystem quirks that general base models frequently miss.

---

## Architecture & Tech Stack
- **Compute Target:** AMD Instinct MI300X (AMD Developer Cloud)
- **Base Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Fine-Tuning Method:** LoRA / PEFT using Hugging Face `transformers` and `trl`.
- **Framework:** PyTorch (ROCm build)
- **Web App:** Gradio (hosted on Hugging Face Spaces)

---

## Links
- **Hugging Face Space (Live Demo):** [https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/ROCmPilot](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/ROCmPilot)
- **Model Adapter:** [https://huggingface.co/MrazzKa/rocmpilot-qwen25-coder-lora](https://huggingface.co/MrazzKa/rocmpilot-qwen25-coder-lora)
- **GitHub Repository:** [https://github.com/MrazzKa/rocmpilot](https://github.com/MrazzKa/rocmpilot)

---

## Demo Modes
To ensure the Hugging Face Space remains reliable on free CPU tiers, ROCmPilot includes two modes:
1. **Demo Mode (Default):** A deterministic, rule-based inference engine that detects common ROCm migration patterns and outputs realistic, structured guidance. This guarantees the app is always fast and functional for hackathon evaluation.
2. **Live Model Mode:** Set `USE_LIVE_MODEL=true` in the environment variables to load the actual Qwen2.5-Coder model with our LoRA adapter for dynamic generation.

---

## Dataset & Evaluation
- **Dataset:** `data/train.jsonl`, `val.jsonl`, `test.jsonl` containing 240+ self-authored, high-quality instruction pairs covering:
  - Hardcoded device usage
  - NVIDIA Docker base images
  - `nvidia-smi` vs `rocm-smi`
  - Quantization & dependency issues (`bitsandbytes`, `flash-attn`)
  - ROCm PyTorch memory errors
- **Evaluation:** We benchmarked the base model against the fine-tuned adapter. The fine-tuned model provides more accurate ROCm verification commands (e.g., using `rocm-smi` instead of hallucinating CUDA APIs) and recognizes ROCm-specific Docker images. See `reports/eval_results.md` for details.

---

## How to Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/MrazzKa/rocmpilot.git
cd rocmpilot
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start the Gradio App:**
```bash
python app.py
```

*Note: By default, the app runs in `DEMO_MODE`. To use the live Hugging Face model, run:*
```bash
USE_LIVE_MODEL=true python app.py
```

---

## How to Reproduce Fine-Tuning

1. Install training dependencies:
```bash
pip install -r requirements-training.txt
```

2. Generate the synthetic dataset:
```bash
python training/generate_dataset.py
```

3. Run the LoRA fine-tuning script (Optimized for AMD MI300X):
```bash
python training/train_lora.py \
    --model_name "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
    --train_file "data/train.jsonl" \
    --val_file "data/val.jsonl" \
    --output_dir "outputs/rocmpilot-qwen25-coder-lora" \
    --epochs 1 \
    --learning_rate 2e-4
```

---

## Limitations & Future Work
- **Limitations:** The model provides migration guidance and cannot guarantee 100% bug-free migrations. PyTorch ROCm environments evolve rapidly, so dependencies may change.
- **Future Work:**
  - Automated pull request generation via GitHub API.
  - Expand the dataset to include multi-node Ray cluster migration on AMD GPUs.
  - Integration into VS Code / Cursor as a native extension.

---

## License
MIT License. See `LICENSE` for details.
