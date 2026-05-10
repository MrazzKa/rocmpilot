import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


SYSTEM_PROMPT = (
    "You are ROCmPilot, a code migration assistant that helps AI developers "
    "adapt CUDA-first AI workloads to AMD ROCm environments. Provide practical, "
    "structured migration guidance."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--train_file", default="data/train.jsonl")
    parser.add_argument("--val_file", default="data/val.jsonl")
    parser.add_argument("--output_dir", default="outputs/rocmpilot-qwen25-coder-lora")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--max_steps", type=int, default=15)
    return parser.parse_args()


def format_example(example):
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output_text = str(example.get("output", "")).strip()

    return (
        f"### System\n{SYSTEM_PROMPT}\n\n"
        f"### Instruction\n{instruction}\n\n"
        f"### Input\n{input_text}\n\n"
        f"### Response\n{output_text}"
    )


def main():
    args = parse_args()

    print(f"Loading dataset from {args.train_file}")
    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.val_file,
        },
    )

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_one(example):
        text = format_example(example)
        return tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    train_dataset = dataset["train"].map(
        tokenize_one,
        remove_columns=dataset["train"].column_names,
    )

    val_dataset = dataset["validation"].map(
        tokenize_one,
        remove_columns=dataset["validation"].column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting LoRA fine-tuning on AMD ROCm GPU...")
    trainer.train()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete.")
    print(f"Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
