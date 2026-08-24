# file: inference/chat_local.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-3B"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = PROJECT_ROOT / "output" / "lora-finance"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=False)

print("Loading base model (may download)...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", trust_remote_code=True)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

print("Ready. Type 'exit' to quit.")
while True:
    q = input("User: ")
    if q.strip().lower() in ("exit", "quit"):
        break
    if not q.strip():
        continue
    inputs = tokenizer(q, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200)
    generated = out[0][inputs["input_ids"].shape[1]:]
    print("Assistant:", tokenizer.decode(generated, skip_special_tokens=True))
