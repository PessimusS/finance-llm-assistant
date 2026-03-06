import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)

from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-3B"
TRAIN_FILE = "../data/train.jsonl"
OUTPUT_DIR = "../output/lora-finance"


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
        data_files={"train": TRAIN_FILE}
    )["train"]

    print("Dataset size:", len(ds))

    print("Tokenizing dataset...")

    def tokenize_fn(example):

        text = example["instruction"] + "\n" + example["output"]

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length"
        )

        tokens["labels"] = tokens["input_ids"].copy()

        return tokens

    ds = ds.map(
        tokenize_fn,
        remove_columns=ds.column_names
    )

    print("Preparing training arguments...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print("Initializing Trainer...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator
    )

    print("Starting training...")

    trainer.train()

    print("Saving LoRA adapter...")

    model.save_pretrained(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training completed!")


if __name__ == "__main__":
    main()