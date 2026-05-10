# examples.py

CODE_EXAMPLES = [
    [
        "PyTorch script with hardcoded device = 'cuda:0'",
        """import torch
import torch.nn as nn

device = "cuda:0"
model = nn.Linear(10, 10).to(device)
x = torch.randn(5, 10).to(device)
output = model(x)
print(output)
"""
    ],
    [
        "Training loop using CUDA specific memory APIs",
        """import torch
        
def train_step(model, data):
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = output.sum()
    
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    return loss
"""
    ],
    [
        "Inference script assuming nvidia-smi",
        """import os
import subprocess

def check_gpu():
    try:
        subprocess.run(["nvidia-smi"], check=True)
        print("NVIDIA GPU is ready for inference.")
    except Exception:
        print("No GPU found.")

check_gpu()
"""
    ]
]

ERROR_EXAMPLES = [
    [
        "RuntimeError: No HIP GPUs are available",
        """RuntimeError: No HIP GPUs are available
Traceback (most recent call last):
  File "train.py", line 12, in <module>
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
    ],
    [
        "HIP out of memory",
        """torch.cuda.OutOfMemoryError: HIP out of memory. Tried to allocate 512.00 MiB (GPU 0; 192.00 GiB total capacity; 180.00 GiB already allocated; 256.00 MiB free; 181.00 GiB reserved in total by PyTorch)"""
    ],
    [
        "ImportError related to bitsandbytes",
        """ImportError: libcudart.so.11.0: cannot open shared object file: No such file or directory
  File "/opt/conda/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py", line 156, in evaluate_cuda_setup
"""
    ]
]

DOCKER_EXAMPLES = [
    [
        "Dockerfile FROM nvidia/cuda",
        """FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip
COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
"""
    ],
    [
        "requirements.txt with NVIDIA-specific deps",
        """torch==2.0.1+cu118
torchvision==0.15.2+cu118
bitsandbytes==0.41.1
flash-attn==2.3.1.post1
xformers==0.0.22
"""
    ],
    [
        "vLLM setup instructions assuming CUDA",
        """# Install vLLM
pip install vllm
# Run the server
python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-v0.1 --tensor-parallel-size 4
"""
    ]
]
