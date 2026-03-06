# Financial LLM Assistant (Finance‑Instruct + Qwen2.5‑3B)

一个从 **0 到 1 完成的大模型微调项目 Demo**。

项目目标是使用 **Qwen2.5‑3B + LoRA 微调**，基于 **Sujet‑Finance‑Instruct‑177k 数据集**训练一个可以回答金融问题的 **金融研究助手模型**。

该项目设计为：

* 个人电脑 → 连接云GPU
* HuggingFace 下载模型
* HuggingFace 下载数据集
* LoRA 微调
* 构建金融问答助手

该项目结构清晰、成本低，非常适合作为：

* AI项目作品
* GitHub展示项目
* AI产品经理 / AI工程方向简历项目

---

# 一、项目效果

训练完成后，你将获得一个可以回答如下问题的金融助手：

示例：

```
Q: What is Value at Risk (VaR)?

A: Value at Risk (VaR) is a statistical measure used to estimate the
maximum potential loss of an investment portfolio over a specified
period at a given confidence level.
```

```
Q: What is the difference between bonds and stocks?
```

```
Q: How does inflation affect interest rates?
```

---

# 二、技术栈

本项目使用的核心技术：

| 技术                       | 用途      |
| ------------------------ | ------- |
| Qwen2.5‑3B               | 基础大语言模型 |
| LoRA                     | 低成本模型微调 |
| HuggingFace Transformers | 模型加载    |
| HuggingFace Datasets     | 数据集加载   |
| PEFT                     | LoRA微调  |
| PyTorch                  | 训练框架    |
| 云GPU                     | 模型训练    |

---

# 三、项目结构

```
finance-llm-assistant/

README.md
requirements.txt

```

torch==2.2.2
transformers==4.40.2
datasets==2.19.1
accelerate==0.30.1
peft==0.10.0
bitsandbytes==0.43.1

```

文件说明：

| 文件               | 作用       |
| ---------------- | -------- |
| train\_lora.py   | 训练LoRA模型 |
| chat.py          | 加载模型进行对话 |
| prepare\_data.py | 处理数据集    |
| requirements.txt | Python依赖 |

---

# 四、运行环境

推荐云GPU配置：

| 硬件     | 推荐              |
| ------ | --------------- |
| GPU    | RTX4090 / A5000 |
| 显存     | >= 12GB         |
| 系统     | Ubuntu 20+      |
| Python | 3.10            |

预算：

```

约 30 ~ 80 元

```

---

# 五、环境安装

SSH 登录云服务器后执行：

```

pip install torch
pip install transformers
pip install datasets
pip install accelerate
pip install peft
pip install bitsandbytes

```

或者直接：

```

pip install -r requirements.txt

```

requirements.txt

```

torch
transformers
datasets
accelerate
peft
bitsandbytes

```

---

# 六、下载数据集

使用 HuggingFace datasets 下载：

```

python prepare_data.py

````

prepare\_data.py

```python
from datasets import load_dataset

print("Downloading dataset...")

# 下载金融指令数据集
dataset = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k")

# 保存为json

dataset["train"].to_json("finance_dataset.json")

print("Dataset saved to finance_dataset.json")
````

数据量：

```
177,597 samples
```

---

# 七、模型微调

运行训练脚本：

````
python train_lora.py（QLoRA版本，显存占用更低，推荐使用）

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen2.5-3B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model (4bit)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)

print("Loading dataset...")
dataset = load_dataset("json", data_files="finance_dataset.json")

def tokenize(example):

    text = example["instruction"] + " " + example.get("input", "") + " " + example["output"]

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


dataset = dataset.map(tokenize)

print("Applying LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(

    output_dir="./finance-lora",

    per_device_train_batch_size=2,

    gradient_accumulation_steps=4,

    num_train_epochs=1,

    logging_steps=10,

    save_steps=200,

    learning_rate=2e-4,

    fp16=True
)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"]

)

print("Start training...")

trainer.train()

print("Training finished")

model.save_pretrained("finance-lora")
````

```

---

# 八、模型测试

训练完成后运行：

```

python chat.py

````

chat.py

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "finance-lora"

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

while True:

    question = input("User: ")

    inputs = tokenizer(question, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    answer = tokenizer.decode(outputs[0])

    print("Assistant:", answer)
````

---

# 九、项目升级路线

Version 1

```
Qwen2.5‑3B
+ Finance‑Instruct
+ LoRA
```

Version 2

```
Qwen2.5‑7B
+ FinQA
```

Version 3

```
RAG金融助手
+ 财报数据库
+ 向量检索
```

---

# 十、简历描述

可以写为：

```
Fine‑tuned the Qwen2.5‑3B large language model using LoRA on the
Finance‑Instruct‑177k dataset to build a financial research assistant
capable of answering finance‑related questions.

Implemented the training pipeline with HuggingFace Transformers,
Datasets, and PEFT, and deployed the model on cloud GPU infrastructure.
```

---

# 十一、项目成本

预计训练成本：

| 资源    | 费用        |
| ----- | --------- |
| GPU租用 | 30~80 RMB |
| 数据集   | 免费        |
| 模型    | 免费        |

总成本：

```
< 100 RMB
```

---

# 十二、许可证

MIT License



1) data/prepare_data.py用法：
cd finance-llm-assistant
python data/prepare_data.py
# 生成 data/train.jsonl, data/dev.jsonl, data/test.jsonl

2) training/train_qlora.py
注意/建议：
per_device_train_batch_size=1 + gradient_accumulation_steps=8 是通用的显存节约策略（会模拟 larger batch）。你可依据显存增减这两个值。
若报错（例如 device_map / bitsandbytes 问题），先用小样本（data/prepare_data.py 中把 SAMPLE_SIZE 设为 500）做 smoke-test。
trust_remote_code=True 对某些模型是必需的（Qwen 系列通常有自定义实现）。
运行：
python training/train_qlora.py

3) inference/gradio_app.py
运行：
python inference/gradio_app.py
# 打开浏览器访问 http://<服务器IP>:7860

4) inference/chat_local.py