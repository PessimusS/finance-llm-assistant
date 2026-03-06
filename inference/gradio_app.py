# file: inference/gradio_app.py
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B"
ADAPTER_PATH = "../models/finance-qlora"  # 相对 training 脚本输出位置

def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=False)
    print("Loading base model (may download, large)...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", trust_remote_code=True)
    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def respond(prompt, max_new_tokens=200, temperature=0.1):
    input_text = prompt if prompt.endswith("\n") else prompt + "\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=int(max_new_tokens), temperature=float(temperature))
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

with gr.Blocks() as demo:
    gr.Markdown("# Finance LLM Assistant (QLoRA)")
    with gr.Row():
        inp = gr.Textbox(lines=4, placeholder="输入你的金融问题，例如：解释 GARCH(1,1) 模型")
        out = gr.Textbox(lines=8)
    with gr.Row():
        max_tok = gr.Slider(32, 512, value=200, label="max_new_tokens")
        temp = gr.Slider(0.0, 1.0, value=0.1, label="temperature")
        btn = gr.Button("生成")
    btn.click(fn=lambda q, m, t: respond(q, m, t), inputs=[inp, max_tok, temp], outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)