# ROCmPilot Pitch Deck Outline

> **Historical artifact:** this was the hackathon pitch outline. Statements on slide 6
> were not backed by saved base-vs-adapter generations or computed metrics. They are
> retained to preserve project history and must not be cited as measured results; see
> `reports/historical_hackathon_eval.md` and the current root README.

**Team:** AMDeus Ex Machina
**Track:** Track 2: Fine-Tuning on AMD GPUs

---

## Slide 1: Title
- **ROCmPilot**
- Fine-Tuned Code Migration Assistant for AMD ROCm
- Team: AMDeus Ex Machina
- "Navigating the leap from CUDA to ROCm seamlessly."

## Slide 2: The Problem
- The AI ecosystem has a heavy CUDA bias.
- When developers move workloads to cost-effective, high-performance AMD GPUs (like MI300X), they face hidden hurdles.
- General LLMs give generic answers. They hallucinate `nvcc` and `nvidia-smi` commands even when told the target is AMD.
- Result: Frustrating debugging sessions with Docker, PyTorch, and `bitsandbytes`.

## Slide 3: The Solution - ROCmPilot
- A specialized AI assistant specifically fine-tuned for CUDA-to-ROCm migration.
- **Inputs:** Code snippets, runtime error logs, Dockerfiles.
- **Outputs:** Concrete ROCm fixes, safe verification commands (e.g., `rocm-smi`), and ready-to-use Cursor prompts.

## Slide 4: Product Demo
- Screenshot / GIF of the Gradio interface.
- 3 Tabs: Code Migration, Error Explainer, Environment Fixer.
- Show an example: Inputting a Dockerfile with `FROM nvidia/cuda` and outputting the correct `rocm/pytorch` base image.

## Slide 5: Fine-Tuning on AMD MI300X
- **Base Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Method:** PEFT LoRA (Low-Rank Adaptation)
- **Dataset:** 240+ synthetic instruction pairs detailing ROCm-specific quirks, memory errors, and dependency conflicts.
- **Hardware:** Trained directly on the AMD Developer Cloud using the MI300X accelerator.

## Slide 6: Evaluation & Results
- Compared base Qwen vs. ROCmPilot.
- **Result:** ROCmPilot completely eliminates NVIDIA tool hallucinations. It reliably prescribes `PYTORCH_HIP_ALLOC_CONF` for memory issues and identifies unportable PyTorch/CUDA wheels in `requirements.txt`.
- Evaluated on precision of verification commands.

## Slide 7: Business Value & Roadmap
- **Value:** Accelerates onboarding of enterprise AI workloads to AMD hardware. Lowers the barrier to entry for developers switching from NVIDIA.
- **Roadmap:**
  - VS Code / Cursor Extension integration.
  - Automated PR generation for repository-wide ROCm migration.
  - Expansion to Ray cluster migration.
