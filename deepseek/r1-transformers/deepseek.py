# deepseek.py

from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import os

# 1. Load the dataset (select 10% of samples for training and validation)
ds = load_dataset("nvidia/HelpSteer")
train_ds = ds["train"].shuffle(seed=42).select(range(int(0.1 * len(ds["train"]))))
split_datasets = train_ds.train_test_split(test_size=0.05, seed=42)
train_ds = split_datasets["train"]
test_ds = split_datasets["test"]

val_ds = ds["validation"].shuffle(seed=42).select(range(int(0.1 * len(ds["validation"]))))
print("Train samples:", len(train_ds), "Validation samples:", len(val_ds), "Test samples:", len(test_ds))

# 2. Set maximum sequence length and load model/tokenizer
max_length = 512
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)

# 3. Configure and apply LoRA
lora_config = LoraConfig(
    r=8,                # Low-rank adaptation rank
    lora_alpha=16,      # Scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,   # Dropout rate for LoRA layers
    bias="none"         # No bias used
)
model = get_peft_model(model, lora_config)

# 4. Define the tokenization function with proper padding and truncation
def tokenize_function(examples):
    return tokenizer(
        examples["prompt"],
        padding="max_length",   # Pad all sequences to max_length
        truncation=True,        # Truncate sequences longer than max_length
        max_length=max_length
    )

tokenized_train = train_ds.map(tokenize_function, batched=True)
tokenized_val = val_ds.map(tokenize_function, batched=True)

# 5. Remove extra columns that are not required (like the raw text fields)
columns_to_remove = ["prompt", "response", "helpfulness", "correctness", "coherence", "complexity", "verbosity"]
tokenized_train = tokenized_train.remove_columns(
    [col for col in columns_to_remove if col in tokenized_train.column_names]
)
tokenized_val = tokenized_val.remove_columns(
    [col for col in columns_to_remove if col in tokenized_val.column_names]
)

# 6. Use a data collator for language modeling.
# For causal LM, DataCollatorForLanguageModeling with mlm=False automatically creates labels.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 7. Set up TrainingArguments (ensure deepspeed_config.json is available)
training_args = TrainingArguments(
    fp16=True,
    output_dir="./output",
    eval_strategy="steps",  # Note: evaluation_strategy is deprecated in future versions; use eval_strategy
    learning_rate=2e-5,
    gradient_accumulation_steps=8,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    deepspeed="deepspeed_config.json",  # Path to your DeepSpeed config file
)

# 8. Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,  # Warning: this will be deprecated in future versions
    data_collator=data_collator,
)

# 9. Start training
trainer.train()


model.save_pretrained("final_model-r1")
tokenizer.save_pretrained("final_model-r1")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("final_model-r1")
tokenizer = AutoTokenizer.from_pretrained("final_model-r1")

from transformers import pipeline

# Create a text classification pipeline using your loaded model and tokenizer
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Define a function to run prediction on a single example (adjust the key 'text' if necessary)
def predict_example(example):
    # The pipeline returns a list of predictions, so we can extract the first one.
    prediction = classifier(example["prompt"])[0]
    return {"label": prediction["label"], "score": prediction["score"]}

# Apply the prediction function to each example in the test dataset
predicted_test_ds = test_ds.map(predict_example)

# Optionally, view the first few predictions
print(test_ds[:5])
print(predicted_test_ds[:5])



# model.push_to_hub("your-model-name")
# tokenizer.push_to_hub("your-model-name")
