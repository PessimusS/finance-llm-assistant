import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    default_data_collator,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "Qwen/Qwen2.5-3B"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT_ROOT / "data" / "train.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "output" / "lora-finance"
MAX_LENGTH = 512


def main():

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model with 4bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    print("Applying LoRA...")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    print("Loading dataset:", TRAIN_FILE)

    ds = load_dataset(
        "json",
        data_files={"train": str(TRAIN_FILE)}
    )["train"]

    print("Dataset size:", len(ds))

    print("Tokenizing dataset...")

    def tokenize_fn(example):

        prompt_ids = tokenizer(
            example["prompt"],
            add_special_tokens=False,
        )["input_ids"]
        completion_ids = tokenizer(
            example["completion"] + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = (prompt_ids + completion_ids)[:MAX_LENGTH]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:MAX_LENGTH]
        attention_mask = [1] * len(input_ids)

        pad_length = MAX_LENGTH - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_length
        labels += [-100] * pad_length
        attention_mask += [0] * pad_length

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    ds = ds.map(
        tokenize_fn,
        remove_columns=ds.column_names
    )

    print("Preparing training arguments...")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,

        learning_rate=2e-4,

        num_train_epochs=1,

        logging_steps=20,

        save_strategy="epoch",

        bf16=False,
        fp16=True,

        report_to="none"
    )

    print("Initializing Trainer...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=default_data_collator
    )

    print("Starting training...")

    trainer.train()

    print("Saving LoRA adapter...")

    model.save_pretrained(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training completed!")


if __name__ == "__main__":
    main()
