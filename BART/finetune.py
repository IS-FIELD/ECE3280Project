import numpy as np
import evaluate
import torch
import pandas as pd
from datasets import load_from_disk, Dataset, load_dataset
from transformers import (
    BartForSequenceClassification,
    BartTokenizerFast,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
import json

dataset = load_dataset(
    "json", data_files="/mnt/workspace/luyiheng/ECE3280/CSVs/00_97.json"
)['train']

split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]


tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large-mnli")


def expand_descriptions(example):
    new_examples = []
    for i in range(1, 11):
        desc = example.get(f"description {i}", "")
        if desc and desc.strip():
            new_examples.append(
                {
                    "description": desc,
                    "1st_stage_label": example["1st_stage_label"],
                    "2nd_stage_label": example["2nd_stage_label"],
                    "id": example["id"],
                }
            )
    return new_examples


expanded_dataset = dataset.map(expand_descriptions, batched=False).flatten_indices()

# 分词
tokenized_dataset = expanded_dataset.map(
    lambda x: tokenizer(
        x["description"], truncation=True, padding="max_length", max_length=256
    ),
    batched=True,
)


train_dataset.set_format("torch")
# test_dataset.set_format("torch")

# Load the model
model = BartForSequenceClassification.from_pretrained(
    "facebook/bart-large-mnli", num_labels=2
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
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
