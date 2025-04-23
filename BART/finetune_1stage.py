import numpy as np
import evaluate
import torch
import pandas as pd
from datasets import load_from_disk, Dataset, load_dataset
from transformers import (
    BartForSequenceClassification,
    BartTokenizerFast,
    BartConfig,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
import json

dataset = load_dataset(
    "json", data_files="/mnt/workspace/luyiheng/ECE3280/CSVs/00_97.json"
)['train']


tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large-mnli")


def expand_descriptions(batch):
    descriptions = []
    labels = []
    ids = []
    batch_size = len(batch["id"])
    for idx in range(batch_size):
        for i in range(1, 11):
            desc = batch.get(f"description {i}", [""] * batch_size)[idx]
            if desc and desc.strip():
                descriptions.append(desc)
                labels.append(batch["1st_stage_label"][idx])
                ids.append(batch["id"][idx])
    return {"description": descriptions, "1st_stage_label": labels, "id": ids}


expanded_dataset = dataset.map(
    expand_descriptions, batched=True, remove_columns=dataset.column_names
)

# 分词
tokenized_dataset = expanded_dataset.map(
    lambda x: tokenizer(
        x["description"], truncation=True, padding="max_length", max_length=256
    ),
    batched=True,
)

# 1. 构建 label2id 和 id2label 字典
labels = list(set(tokenized_dataset["1st_stage_label"]))
label2id = {label: idx for idx, label in enumerate(sorted(labels))}
id2label = {idx: label for label, idx in label2id.items()}

print(f"label2id: {label2id}")
print(f"id2label: {id2label}")
# 2. 正确添加 labels 字段
def add_label(example):
    example["labels"] = label2id[example["1st_stage_label"]]
    return example


tokenized_dataset = tokenized_dataset.map(add_label)


# 划分训练/验证集
split_tokenized = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_tokenized["train"]
eval_dataset = split_tokenized["test"]

train_dataset.set_format("torch")
eval_dataset.set_format("torch")


config = BartConfig.from_pretrained(
    "facebook/bart-large-mnli",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)


# Load the model
model = BartForSequenceClassification.from_pretrained(
    "facebook/bart-large-mnli", config=config, ignore_mismatched_sizes=True
)

learning_rate = 5e-5

batch_size = 8
eval_batch_size = 8

gradient_accumulation_steps = int(16 / batch_size)

# Define training arguments
training_args = TrainingArguments(
    learning_rate=learning_rate,  # The initial learning rate for Adam
    output_dir=f"/mnt/data4/luyiheng/BARTfinetune/lr_{learning_rate}",  # Output directory
    num_train_epochs=5,  # Total number of training epochs
    per_device_train_batch_size=batch_size,  # Batch size per device during training
    per_device_eval_batch_size=eval_batch_size,  # Batch size for evaluation
    gradient_accumulation_steps=gradient_accumulation_steps,
    eval_accumulation_steps=eval_batch_size,
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    logging_dir=f"/mnt/data4/luyiheng/BARTfinetune/logs/lr_{learning_rate}",  # Directory for storing logs
    logging_steps=10,  # log results every x steps
    evaluation_strategy="steps",
    eval_steps=100,  # evaluate every x steps
    save_strategy="steps",
    save_steps=100,  # save model every x steps
)


# Define the compute_metrics function for evaluation
def compute_metrics(p: EvalPrediction):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)[
        "accuracy"
    ]
    result["f1"] = metric_f1.compute(
        predictions=preds, references=p.label_ids, average="macro"
    )["f1"]
    return result


# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
