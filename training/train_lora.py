import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Command example:
# python training/train_lora.py --model_name "Qwen/Qwen2.5-Coder-1.5B-Instruct" --train_file "data/train.jsonl" --val_file "data/val.jsonl" --output_dir "outputs/rocmpilot-qwen25-coder-lora" --epochs 1 --learning_rate 2e-4 --batch_size 2 --gradient_accumulation_steps 8

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ROCmPilot using PEFT LoRA on AMD MI300X")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Base model")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl", help="Training dataset JSONL")
    parser.add_argument("--val_file", type=str, default="data/val.jsonl", help="Validation dataset JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/rocmpilot-qwen25-coder-lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    return parser.parse_args()

def format_instruction(example):
    # Formatting for Qwen2.5-Coder (ChatML-style or simple prompt)
    system_prompt = "You are ROCmPilot, a highly specialized code migration assistant. Your goal is to help developers migrate CUDA-first workloads to AMD ROCm environments."
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
    return {"text": prompt}

def main():
    args = parse_args()
    
    print(f"Loading dataset from {args.train_file}")
    dataset = load_dataset("json", data_files={"train": args.train_file, "val": args.val_file})
    
    # Format dataset
    dataset = dataset.map(format_instruction, remove_columns=["instruction", "input", "output"])
    
    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # NOTE: PyTorch ROCm builds still expose many GPU APIs through `torch.cuda` naming.
    # Therefore, we use device_map="auto" and torch.bfloat16 (supported on MI300X natively).
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    print("Starting LoRA fine-tuning on AMD MI300X...")
    trainer.train()
    
    print(f"Saving fine-tuned adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()
