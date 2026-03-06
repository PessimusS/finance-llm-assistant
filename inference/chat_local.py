# file: inference/chat_local.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-3B"
ADAPTER_PATH = "./models/finance-qlora"

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
    print("Assistant:", tokenizer.decode(out[0], skip_special_tokens=True))