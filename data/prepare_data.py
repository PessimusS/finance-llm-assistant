# file: data/prepare_data.py
"""
将 sujet-ai/Sujet-Finance-Instruct-177k 中的 QA 类示例抽取并格式化为
训练用的 prompt/completion JSONL。
输出文件：
  data/train.jsonl
  data/dev.jsonl
  data/test.jsonl

使用说明：
  - 初次 smoke-run 建议 SAMPLE_SIZE=5000 或 2000（节省时间 / 成本）
  - 若要跑更多把 SAMPLE_SIZE 设为 None 或更大值
"""

from datasets import load_dataset
import os, json, random, re

OUT_DIR = "data"
TRAIN_PATH = os.path.join(OUT_DIR, "train.jsonl")
DEV_PATH = os.path.join(OUT_DIR, "dev.jsonl")
TEST_PATH = os.path.join(OUT_DIR, "test.jsonl")

# 配置：首次跑用小样本验证流程（设 None 表示不截断）
SAMPLE_SIZE = 5000

# 哪些 task_type 我们认为是“问答相关”，字符串匹配（小写）
QA_KEYWORDS = ["qa", "question", "conversation", "yes_no"]

def clean_text(s):
    if s is None:
        return ""
    # 去掉多余换行首尾空白，统一空格
    t = re.sub(r"\r\n|\r", "\n", str(s)).strip()
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t

def build_prompt_completion(row):
    """
    尽量从 row 中构造 (prompt, completion)
    优先逻辑：
      1) 如果 row['task_type'] 明确是 QA 类 -> 使用 user_prompt 作为问题, inputs/system_prompt 作为上下文
      2) 否则如果有 conversation 字段（list）则把对话拼成 prompt
      3) 否则，如果 user_prompt+answer 存在则使用二者
      4) 否则，如果 inputs 字段看起来像 question+Answer 模版（包含 'Answer:'）则做相应截取
    返回 (prompt, completion) 或 (None, None) 表示不能构造
    """
    task_type = (row.get("task_type") or "").lower()
    inputs = clean_text(row.get("inputs") or "")
    system_prompt = clean_text(row.get("system_prompt") or "")
    user_prompt = clean_text(row.get("user_prompt") or "")
    answer = clean_text(row.get("answer") or "")

    # helper to create assistant ending
    def wrap_prompt(system, user, ctx):
        parts = []
        if system:
            parts.append(f"System: {system}")
        if ctx:
            parts.append(f"Context: {ctx}")
        if user:
            parts.append(f"User: {user}")
        parts.append("Assistant:")
        return "\n".join(parts)

    # 1) 优先 task_type 匹配 QA 类
    if task_type and any(k in task_type for k in QA_KEYWORDS):
        # prefer user_prompt as question; use inputs as extra context if it is not identical
        ctx = ""
        if inputs and inputs != user_prompt:
            ctx = inputs
        prompt = wrap_prompt(system_prompt, user_prompt, ctx)
        completion = answer
        if user_prompt and answer:
            return prompt, completion

    # 2) conversation-like (some rows may have conversation_id, or inputs containing multiple turns)
    # Try to detect simple conversation in 'inputs' if it contains markers like 'User:' or 'Assistant:'
    if inputs and ("User:" in inputs or "Assistant:" in inputs or "user:" in inputs):
        # try to remove trailing 'Answer:' if exists and set completion from answer
        prompt = inputs
        if not prompt.strip().endswith("Assistant:"):
            # remove the trailing 'Answer:' part so assistant will generate
            prompt = re.split(r"Answer:|ANSWER:|answer:", prompt)[0].strip()
            if not prompt.endswith("Assistant:"):
                prompt = prompt + "\nAssistant:"
        if answer:
            completion = answer
            return prompt, completion
        else:
            # let model generate without a provided completion (skip)
            return None, None

    # 3) fallback: if user_prompt + answer exist
    if user_prompt and answer:
        prompt = wrap_prompt(system_prompt, user_prompt, "")
        completion = answer
        return prompt, completion

    # 4) fallback: inputs with an inline question pattern
    # e.g. inputs contains 'Question: ... Answer: ...'
    if inputs and "Answer:" in inputs:
        parts = re.split(r"Answer:|ANSWER:|answer:", inputs)
        qpart = parts[0].strip()
        apart = parts[1].strip() if len(parts) > 1 else ""
        # construct prompt from qpart
        prompt = qpart
        # ensure ends with Assistant:
        if not prompt.strip().endswith("Assistant:"):
            prompt = prompt + "\nAssistant:"
        completion = apart or answer
        if completion:
            return prompt, completion

    # 无法构造
    return None, None

def main():
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")
    print("Total raw samples:", len(ds))

    samples = []
    for row in ds:
        prompt, completion = build_prompt_completion(row)
        if prompt and completion:
            samples.append({"prompt": prompt, "completion": completion})

    print("Collected QA-like samples:", len(samples))
    if len(samples) == 0:
        print("No QA samples found. Please inspect dataset fields.")
        return

    # optionally truncate for quick smoke-run
    if SAMPLE_SIZE is not None:
        samples = samples[:SAMPLE_SIZE]
        print("Truncated to SAMPLE_SIZE:", len(samples))

    random.shuffle(samples)

    # split 90/5/5
    n = len(samples)
    n_train = int(n * 0.9)
    n_dev = int(n * 0.05)
    train = samples[:n_train]
    dev = samples[n_train:n_train + n_dev]
    test = samples[n_train + n_dev:]

    print("Final counts -> train:", len(train), "dev:", len(dev), "test:", len(test))

    os.makedirs(OUT_DIR, exist_ok=True)
    def dump(path, arr):
        with open(path, "w", encoding="utf-8") as f:
            for r in arr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(TRAIN_PATH, train)
    dump(DEV_PATH, dev)
    dump(TEST_PATH, test)
    print("Saved files:", TRAIN_PATH, DEV_PATH, TEST_PATH)

if __name__ == "__main__":
    main()