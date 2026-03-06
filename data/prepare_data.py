# file: data/prepare_data.py
import os
import json
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

# 可调：如果想先跑小样本，把 SAMPLE_SIZE 设小；要跑全量设 None
SAMPLE_SIZE = 20000  # 推荐首次 smoke-test: 20000；生产可设 None 或 177597

def normalize_sample(item):
    # 将 dataset 的不同字段规范成 instruction/input/output
    # dataset 中可能已经是 instruction/input/output
    inst = item.get("instruction") or item.get("prompt") or ""
    inp = item.get("input") or ""
    out = item.get("output") or item.get("response") or ""
    return {"instruction": inst.strip(), "input": inp.strip(), "output": out.strip()}

def build_prompt(ins, inp):
    # 统一 prompt 模板（便于模型学习“assistant-style”）
    if inp:
        return f"User: {ins}\nInput: {inp}\nAssistant:"
    else:
        return f"User: {ins}\nAssistant:"

def main():
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")
    total = len(ds)
    print(f"Raw total samples: {total}")

    if SAMPLE_SIZE is not None:
        ds = ds.select(range(min(SAMPLE_SIZE, total)))
        print(f"Selected SAMPLE_SIZE = {len(ds)}")

    samples = []
    for item in ds:
        s = normalize_sample(item)
        if not s["instruction"] or not s["output"]:
            continue
        prompt = build_prompt(s["instruction"], s["input"])
        completion = " " + s["output"].strip()
        samples.append({"prompt": prompt, "completion": completion})

    print(f"After cleaning: {len(samples)} samples")

    # split train/dev/test = 80/10/10
    n = len(samples)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)
    train = samples[:n_train]
    dev = samples[n_train:n_train + n_dev]
    test = samples[n_train + n_dev:]

    def dump(list_data, path):
        with open(path, "w", encoding="utf-8") as f:
            for rec in list_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    train_path = os.path.join(OUT_DIR, "train.jsonl")
    dev_path = os.path.join(OUT_DIR, "dev.jsonl")
    test_path = os.path.join(OUT_DIR, "test.jsonl")
    dump(train, train_path)
    dump(dev, dev_path)
    dump(test, test_path)
    print("Saved:", train_path, dev_path, test_path)

if __name__ == "__main__":
    main()