import os
import yaml
import random
from huggingface_hub import login
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    LlamaForSequenceClassification,
    LlamaTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model

# 0. Reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 1. Load the YAML config
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

hf_token = cfg["huggingface"]["token"]
if not hf_token:
    raise RuntimeError("No Hugging Face token found in config.yaml under huggingface.token")

# 2. (Optional) Log in via the Hub API
login(token=hf_token)

# 1. Load and prepare data
df = pd.read_csv("label_update_6_12.csv")
assert {"question", "Label"}.issubset(df.columns), "CSV must have 'question' and 'label' columns"
df = df.rename(columns={"label": "Label"})  # unify name

# 2. Encode labels
labels = sorted(df["Label"].unique())
label2id = {lbl: idx for idx, lbl in enumerate(labels)}
id2label = {idx: lbl for lbl, idx in label2id.items()}
df["label_id"] = df["Label"].map(label2id)

# 3. Train/validation split (stratified)
train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    # stratify=df["label_id"],
    random_state=42,
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# 4. Dataset class
MAX_LENGTH = 256
tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-3.1-8B", use_auth_token=hf_token)

tokenizer.pad_token = tokenizer.eos_token

class QADataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH):
        self.texts = df["question"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",  # or remove for dynamic batching
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_dataset = QADataset(train_df, tokenizer)
eval_dataset  = QADataset(val_df,   tokenizer)

# 5. Load model + apply LoRA
model = LlamaForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.float16,    # use 'auto' if you have multiple dtypes
    # device_map="auto",            # or remove if single-GPU
    use_auth_token=hf_token,
)

model.config.pad_token_id = model.config.eos_token_id

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(model, lora_cfg)

# 6. TrainingArguments + metrics
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

training_args = TrainingArguments(
    output_dir="./lora_llama3_cls",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    eval_steps=200,
    logging_steps=200,
    save_steps=1000,
    save_total_limit=3,
    learning_rate=2e-5,
    num_train_epochs=3,
    bf16=True,  
    # fp16=True,
    deepspeed="deepspeed_config.json",
    max_grad_norm=1.0,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 7. Train & save
trainer.train()
trainer.save_model("final_llama3_lora_cls")
tokenizer.save_pretrained("final_llama3_lora_cls")
