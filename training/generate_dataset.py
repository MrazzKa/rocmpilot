import os
import json
import random

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Templates for synthetic generation
TEMPLATES = [
    # 1. Hardcoded device
    {
        "instruction": "Analyze this PyTorch code for ROCm migration issues:\n```python\nimport torch\ndevice = '{cuda_device}'\nmodel.to(device)\n```",
        "input": "",
        "output": "## Summary\nThe code hardcodes a CUDA device string.\n\n## Detected ROCm issue\nHardcoding `{cuda_device}` is not portable. While PyTorch ROCm aliases `cuda` to HIP, it's better to use dynamic checking.\n\n## Recommended fix\nUse `torch.cuda.is_available()` to dynamically assign the device.\n\n## Corrected code or config\n```python\nimport torch\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nmodel.to(device)\n```\n\n## Verification commands\n```bash\npython -c \"import torch; print(torch.cuda.is_available())\"\n```\n\n## ROCm readiness score\n80/100. PyTorch handles this under the hood, but dynamic is better.\n\n## Cursor prompt\n\"Remove the hardcoded `{cuda_device}` and replace it with a dynamic device selection check.\"\n\n## Notes and limitations\nThis output is migration guidance. Always test generated changes in your target ROCm environment."
    },
    # 2. nvidia-smi
    {
        "instruction": "I have a health check script:\n```bash\n{smi_cmd} --query-gpu=memory.used --format=csv\n```\nWill this work on AMD?",
        "input": "",
        "output": "## Summary\nThe script uses an NVIDIA-specific GPU monitoring command.\n\n## Detected ROCm issue\n`{smi_cmd}` is installed with NVIDIA drivers and will fail with a command not found error on AMD systems.\n\n## Recommended fix\nReplace `{smi_cmd}` with `rocm-smi`.\n\n## Corrected code or config\n```bash\nrocm-smi --showmemuse\n```\n\n## Verification commands\n```bash\nrocm-smi\n```\n\n## ROCm readiness score\n0/100. The original command will fail.\n\n## Cursor prompt\n\"Update this bash command to use `rocm-smi` for AMD GPU monitoring instead of `{smi_cmd}`.\"\n\n## Notes and limitations\nThis output is migration guidance. Always test generated changes in your target ROCm environment."
    },
    # 3. Docker Base Image
    {
        "instruction": "Here is my Dockerfile:\n```dockerfile\nFROM {base_image}\nRUN pip install torch\n```\nMake it AMD ROCm compatible.",
        "input": "",
        "output": "## Summary\nThe Dockerfile uses an NVIDIA CUDA base image.\n\n## Detected ROCm issue\nThe `{base_image}` contains CUDA toolkits and NVIDIA libraries that will not interact with AMD GPUs.\n\n## Recommended fix\nSwitch to the official ROCm PyTorch base image.\n\n## Corrected code or config\n```dockerfile\nFROM rocm/pytorch:rocm6.1.2_ubuntu22.04_py3.10_pytorch_2.1.2\nRUN pip install torch\n```\n\n## Verification commands\n```bash\ndocker run -it --device=/dev/kfd --device=/dev/dri rocm/pytorch:rocm6.1.2_ubuntu22.04_py3.10_pytorch_2.1.2 rocm-smi\n```\n\n## ROCm readiness score\n0/100. The original image will not recognize AMD GPUs.\n\n## Cursor prompt\n\"Replace the NVIDIA base image in this Dockerfile with the official ROCm PyTorch image.\"\n\n## Notes and limitations\nThis output is migration guidance. Always test generated changes in your target ROCm environment."
    },
    # 4. Bitsandbytes
    {
        "instruction": "My requirements.txt has this:\n```text\n{dep}\n```\nWill this work out of the box on MI300X?",
        "input": "",
        "output": "## Summary\nThe requirements file lists a dependency heavily tied to CUDA.\n\n## Detected ROCm issue\n`{dep}` historically requires CUDA compilation or specific ROCm branches. Older versions will fail to load on AMD.\n\n## Recommended fix\nUpdate to the latest version of the library that supports ROCm natively, or use the AMD-specific fork/wheel.\n\n## Corrected code or config\n```text\nbitsandbytes>=0.43.0\n```\n\n## Verification commands\n```bash\npython -c \"import bitsandbytes; print(bitsandbytes.__version__)\"\n```\n\n## ROCm readiness score\n10/100. Older versions lack AMD support.\n\n## Cursor prompt\n\"Update this dependency to a version that officially supports ROCm (e.g., bitsandbytes 0.43.0+).\"\n\n## Notes and limitations\nThis output is migration guidance. Always test generated changes in your target ROCm environment."
    },
    # 5. OOM Error
    {
        "instruction": "I got this error:\n`torch.cuda.OutOfMemoryError: HIP out of memory. Tried to allocate {alloc_size}`",
        "input": "",
        "output": "## Summary\nThe application ran out of GPU memory on the AMD accelerator.\n\n## Detected ROCm issue\nThis is the HIP equivalent of the CUDA OOM error. PyTorch's memory allocator is fragmenting or simply exhausted.\n\n## Recommended fix\nReduce batch size, use mixed precision, or tune the PyTorch HIP memory allocator.\n\n## Corrected code or config\nNo code change required. Adjust environment or hyperparameters.\n\n## Verification commands\n```bash\nexport PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128\n```\n\n## ROCm readiness score\n50/100. The environment works but needs tuning.\n\n## Cursor prompt\n\"How do I configure `PYTORCH_HIP_ALLOC_CONF` to reduce memory fragmentation on AMD GPUs?\"\n\n## Notes and limitations\nThis output is migration guidance. Always test generated changes in your target ROCm environment."
    }
]

# Variations to expand the dataset to >240 examples
cuda_devices = ["cuda:0", "cuda:1", "cuda", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]
smi_cmds = ["nvidia-smi", "/usr/bin/nvidia-smi", "watch nvidia-smi", "nvidia-smi -l 1"]
base_images = [
    "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
    "nvidia/cuda:12.1.0-base-ubuntu20.04",
    "nvcr.io/nvidia/pytorch:23.10-py3",
    "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
]
deps = ["bitsandbytes==0.41.1", "bitsandbytes<0.43", "bitsandbytes==0.39.0", "flash-attn==2.3.1.post1", "xformers==0.0.22"]
alloc_sizes = ["512.00 MiB", "1.00 GiB", "256.00 MiB", "2.50 GiB"]

dataset = []

# Generate variations
for _ in range(50):
    dataset.append({
        "instruction": TEMPLATES[0]["instruction"].format(cuda_device=random.choice(cuda_devices)),
        "input": "",
        "output": TEMPLATES[0]["output"].format(cuda_device=random.choice(cuda_devices))
    })

for _ in range(50):
    dataset.append({
        "instruction": TEMPLATES[1]["instruction"].format(smi_cmd=random.choice(smi_cmds)),
        "input": "",
        "output": TEMPLATES[1]["output"].format(smi_cmd=random.choice(smi_cmds))
    })

for _ in range(50):
    dataset.append({
        "instruction": TEMPLATES[2]["instruction"].format(base_image=random.choice(base_images)),
        "input": "",
        "output": TEMPLATES[2]["output"].format(base_image=random.choice(base_images))
    })

for _ in range(50):
    dataset.append({
        "instruction": TEMPLATES[3]["instruction"].format(dep=random.choice(deps)),
        "input": "",
        "output": TEMPLATES[3]["output"].format(dep=random.choice(deps))
    })

for _ in range(50):
    dataset.append({
        "instruction": TEMPLATES[4]["instruction"].format(alloc_size=random.choice(alloc_sizes)),
        "input": "",
        "output": TEMPLATES[4]["output"].format(alloc_size=random.choice(alloc_sizes))
    })

# Shuffle dataset
random.shuffle(dataset)

# Split into train/val/test
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)

train_data = dataset[:train_size]
val_data = dataset[train_size:train_size+val_size]
test_data = dataset[train_size+val_size:]

def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

save_jsonl(train_data, "data/train.jsonl")
save_jsonl(val_data, "data/val.jsonl")
save_jsonl(test_data, "data/test.jsonl")

print(f"Generated {len(dataset)} total synthetic examples.")
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
print("Dataset successfully saved to data/ directory.")
