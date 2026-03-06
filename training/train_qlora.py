# file: training/train_qlora.py
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "Qwen/Qwen2.5-3B"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "finance-qlora")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# hyperparams (可按需调整)
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 512
SAVE_STEPS = 500

def make_prompt(example):
    return example["prompt"] + example["completion"]  # we will shift labels accordingly in tokenization

def main():
    print("Loading tokenizer (trust_remote_code=True)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)

    print("Loading model in 4bit (requires bitsandbytes)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # prepare model for kbit training (必要)
    model = prepare_model_for_kbit_training(model)

    # PEFT LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # load dataset as jsonl via datasets
    print("Loading dataset from", TRAIN_FILE)
    ds = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]

    # tokenization
    def tokenize_fn(examples):
        text = examples["prompt"] + examples["completion"]
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
        # labels: we want the model to learn to generate the completion,
        # so mask the prompt tokens in labels (set to -100)
        input_ids = tokenized["input_ids"]
        labels = []
        for ids in input_ids:
            # find the index where completion starts
            # heuristic: find first token of completion by tokenizing prompt alone len
            # We'll tokenize prompt separately to get prompt_len
            prompt_enc = tokenizer(examples["prompt"], truncation=True, max_length=MAX_LENGTH, padding=False)
            prompt_len = len(prompt_enc["input_ids"])
            lab = ids.copy()
            for i in range(prompt_len):
                lab[i] = -100
            labels.append(lab)
        tokenized["labels"] = labels
        return tokenized

    print("Tokenizing dataset (this may take a while)...")
    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,  # QLoRA uses 4bit + float16 compute dtype above
        logging_steps=50,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none"  # disable wandb by default
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator
    )

    print("Starting training...")
    trainer.train()
    print("Training finished. Saving adapter...")
    model.save_pretrained(OUTPUT_DIR)
    print("Saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()