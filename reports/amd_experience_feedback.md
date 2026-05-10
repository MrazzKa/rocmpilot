# AMD Developer Cloud & ROCm Experience Feedback

**Team:** AMDeus Ex Machina
**Track:** Track 2: Fine-Tuning on AMD GPUs

## What Worked Well
- **Raw Performance:** The AMD Instinct MI300X instances provided phenomenal compute throughput. Token generation and batch processing during dataset generation were lightning fast.
- **PyTorch Integration:** Using the official ROCm PyTorch wheels (e.g., `torch==2.3.0+rocm6.1`) was largely seamless. Existing PyTorch code utilizing `torch.cuda` aliases cleanly to HIP in most standard use cases.
- **Hugging Face Ecosystem:** The `transformers` and `peft` libraries worked perfectly out of the box with ROCm. Fine-tuning Qwen2.5-Coder using LoRA did not require any AMD-specific modifications to the training script loop.

## What Was Confusing
- **NVIDIA Remnants:** It's confusing for beginners that ROCm still uses the `torch.cuda` namespace in PyTorch. Developers expect a `torch.rocm` or `torch.hip` namespace.
- **Memory Profiling:** While `rocm-smi` is great, some advanced memory profiling tools built for CUDA are difficult to replace on ROCm. Setting environment variables like `PYTORCH_HIP_ALLOC_CONF` requires reading through deeply nested documentation.
- **Quantization:** Older versions of libraries like `bitsandbytes` require jumping through hoops. While newer versions natively support AMD, finding the correct wheel matrix for specific ROCm versions can be challenging.

## Suggested Improvements
- **Clearer Documentation for Migration:** A centralized "CUDA-to-ROCm" translation guide for common tools (e.g., `nvcc` -> `hipcc`, `nvidia-smi` -> `rocm-smi`, `nsys` -> `rocprof`) would be invaluable. This was the exact inspiration for ROCmPilot!
- **Container Registry:** Make the `rocm/pytorch` docker images more discoverable with clearer tags mapping to PyTorch versions.

## AMD Developer Cloud Notes
- Connecting via SSH and setting up the environment was smooth.
- Disk I/O was fast enough for loading multi-gigabyte models without bottlenecks.
- Accessing the MI300X directly without queue times was a massive advantage for iterative debugging.

## Hugging Face Deployment Notes
- Deploying ROCmPilot to a Hugging Face Space was straightforward using Gradio. We implemented a `DEMO_MODE` to ensure the Space remains responsive on free CPU tiers, allowing judges to test the UX without waiting for a GPU instance to spin up.
