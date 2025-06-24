import os
import random
import numpy as np
import torch
import pandas as pd
from peft import PeftModel

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    LlamaForSequenceClassification,
    LlamaTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# 0. Reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 1. Load label data & mapping
df = pd.read_csv("label_update_6_12.csv")
df = df.rename(columns={"label": "Label"})
labels = sorted(df["Label"].unique())
label2id = {lbl: idx for idx, lbl in enumerate(labels)}
id2label = {idx: lbl for lbl, idx in label2id.items()}
df["label_id"] = df["Label"].map(label2id)

# 2. Train/validation split
_, val_df = train_test_split(
    df, test_size=0.1, random_state=42
)
val_df = val_df.reset_index(drop=True)

# 3. Dataset class
MAX_LENGTH = 256
tokenizer = LlamaTokenizerFast.from_pretrained(
    "meta-llama/Llama-3.1-8B", use_auth_token=True
)
# use EOS token as padding
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
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

eval_dataset = QADataset(val_df, tokenizer)

# 4. Metrics function
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

# 5. Shared TrainingArguments for eval
training_args = TrainingArguments(
    output_dir="eval_out",
    per_device_eval_batch_size=4,
    bf16=True,
    logging_dir="./logs",
    do_train=False,
    do_eval=True,
    dataloader_pin_memory=True,
)

# 6. Load & evaluate base model
base_model = LlamaForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=True,
)
# ensure the model knows the padding token
base_model.config.pad_token_id = tokenizer.eos_token_id

base_trainer = Trainer(
    model=base_model,
    args=training_args,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

print("⏳ Evaluating ORIGINAL base model…")
orig_metrics = base_trainer.evaluate()
print(orig_metrics)
# save metrics JSON
base_trainer.save_metrics("eval_base", orig_metrics)

# 7. Load & evaluate LoRA-tuned model
# Load same base architecture for LoRA
base_for_tuning = LlamaForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=True,
)
base_for_tuning.config.pad_token_id = tokenizer.eos_token_id

# Attach LoRA adapters
lora_model = PeftModel.from_pretrained(
    base_for_tuning,
    "final_llama3_lora_cls",
    torch_dtype=torch.float16,
)
lora_model.eval()

lora_trainer = Trainer(
    model=lora_model,
    args=training_args,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

print("⏳ Evaluating LoRA-TUNED model…")
tuned_metrics = lora_trainer.evaluate()
print(tuned_metrics)
# save metrics JSON
lora_trainer.save_metrics("eval_lora", tuned_metrics)

# 8. List contents of eval_out and show summary
to_list = os.listdir("eval_out")
print("\nFiles in eval_out:", to_list)

comparison = {
    "model": ["base", "lora_tuned"],
    "accuracy": [orig_metrics["eval_accuracy"], tuned_metrics["eval_accuracy"]],
}
df_comp = pd.DataFrame(comparison)
print("\n=== Performance Comparison ===")
print(df_comp.to_string(index=False))
