import json

import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

MODEL_NAME = "Qwen/Qwen-0.5B"
DATA_PATH = "data/yijing_train_data_1000.json"
SAVE_MODEL_DIR = "./models/qwen-0.5b-yijing"
DEVICE = "cpu"
EPOCHS = 8
BATCH_SIZE = 4
MAX_SEQ_LEN = 256

BAGUA_LIST = ["qian", "kun", "zhen", "xun", "kan", "li", "gen", "dui"]
WUXING_LIST = ["jin", "tu", "mu", "huo", "shui"]
SANCAI_LIST = ["tien", "ren", "di"]

# 加载训练数据
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)


def format_sample(item):
    input_text = f"内容：{item['content']} | 记忆类型：{item['memory_type']}"
    label_text = f"{item['hexagram']}|{item['bagua_type']}|{item['wuxing']}|{item['sancai_layer']}"
    return {"text": input_text, "label": label_text}


train_list = [format_sample(d) for d in raw_data]
ds = Dataset.from_list(train_list)

# 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, trust_remote_code=True, num_labels=len(train_list), torch_dtype=torch.float32
).to(DEVICE)


def tokenize_func(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_SEQ_LEN)


token_ds = ds.map(tokenize_func, batched=True)
token_ds = token_ds.train_test_split(test_size=0.1)

# 训练参数
train_args = TrainingArguments(
    output_dir="./train_log",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=False,
    use_cpu=True,
    learning_rate=5e-5,
    weight_decay=0.01,
)

trainer = Trainer(model=model, args=train_args, train_dataset=token_ds["train"], eval_dataset=token_ds["test"])

if __name__ == "__main__":
    print("===== 易经取象专属小模型 开始微调 =====")
    trainer.train()
    model.save_pretrained(SAVE_MODEL_DIR)
    tokenizer.save_pretrained(SAVE_MODEL_DIR)
    print(f"✅ 训练完成，模型已保存至：{SAVE_MODEL_DIR}")
