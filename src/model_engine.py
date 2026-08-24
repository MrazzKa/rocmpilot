import os
import logging
from src.demo_engine import analyze_code_demo, analyze_error_demo, analyze_env_demo
from src.prompts import build_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USE_LIVE_MODEL = os.environ.get("USE_LIVE_MODEL", "false").lower() == "true"
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
ADAPTER_MODEL_NAME = os.environ.get("ADAPTER_MODEL_NAME", "MrazzKa/rocmpilot-qwen25-coder-lora")

model = None
tokenizer = None

def init_model():
    global model, tokenizer
    if not USE_LIVE_MODEL:
        logger.info("USE_LIVE_MODEL is false. Skipping model loading (Demo Mode).")
        return False

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch

        logger.info(f"Loading base model: {BASE_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )

        logger.info(f"Loading LoRA adapter: {ADAPTER_MODEL_NAME}")
        try:
            model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_NAME)
            logger.info("Successfully loaded fine-tuned ROCmPilot model.")
        except Exception as peft_e:
            logger.error(f"Could not load PEFT adapter ({peft_e}).")
            logger.warning(
                "Falling back to the explicitly labeled rule-based Demo Mode; "
                "the base model will not be presented as the fine-tuned adapter."
            )
            model = None
            return False
        
        return True
    except Exception as e:
        logger.error(f"Failed to load Hugging Face model: {e}")
        logger.warning("Gracefully falling back to rule-based Demo Mode.")
        return False

# Attempt to initialize on startup
model_loaded = init_model()

def generate_live_response(user_input: str, task_type: str) -> str:
    prompt = build_prompt(user_input, task_type)
    
    try:
        import torch
        inputs = tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.1
            )
            
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response
    except Exception as e:
        logger.error(f"Error during model generation: {e}")
        return f"Error during live model generation: {e}\n\nPlease check logs or disable USE_LIVE_MODEL."

def process_code(text: str) -> str:
    if model_loaded:
        return generate_live_response(text, "Code Migration")
    return analyze_code_demo(text)

def process_error(text: str) -> str:
    if model_loaded:
        return generate_live_response(text, "Error Analysis")
    return analyze_error_demo(text)

def process_env(text: str) -> str:
    if model_loaded:
        return generate_live_response(text, "Environment / Dockerfile Migration")
    return analyze_env_demo(text)
