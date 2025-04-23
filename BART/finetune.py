import numpy as np
import evaluate
import torch
import pandas as pd
from datasets import load_from_disk
from transformers import (
    BartForSequenceClassification,
    BartTokenizerFast,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
import json
with open ('/mnt/workspace/luyiheng/ECE3280/CSVs/template.json', 'r') as f:
    dataset = json.load(f)

for item in dataset:
    f_stage_label = item['1st_stage_label']
    s_stage_label = item['2nd_stage_label']
    description = item['description']


dataset = pd.DataFrame(dataset)
labels = dataset["1st_stage_label"].tolist()
descriptions = dataset["description"].tolist()


# from convert_classified_feedback_to_zsc_training_data import (
#     dataset_output as balanced_dataset_file,
# )

# Load the balanced dataset
# balanced_dataset = load_from_disk(balanced_dataset_file)

# Assuming balanced_dataset is your preprocessed and balanced dataset
# train_test_split = balanced_dataset.train_test_split(
#     test_size=0.1, shuffle=True, seed=42
# )
# train_dataset = train_test_split["train"]
# test_dataset = train_test_split["test"]

# Initialize the tokenizer
tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large-mnli")


# Function to encode the dataset
def encode_examples(examples):
    # 对 description 进行分词
    encoding = tokenizer(
        examples["description"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    # 构建标签映射
    label2id = {
        "Computing methodologies": 0,
        # 可以根据你的实际标签继续添加
        "Network": 1,  # 空标签示例
    }

    # 将标签转为数字
    encoding["labels"] = [
        label2id.get(label, 1) for label in examples["1st_stage_label"]
    ]

    return encoding


def encode_examples_stage2(examples):
    # 拼接 description 和 1st_stage_label
    prompt = [
        f"{label}: {desc}"
        for label, desc in zip(examples["1st_stage_label"], examples["description"])
    ]
    encoding = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    # 构建二级标签映射
    label2id = {
        "Modeling and simulation": 0,
        "P2P": 1,
        # ...根据你的2nd_stage_label补全...
    }
    encoding["labels"] = [
        label2id.get(label, 0) for label in examples["2nd_stage_label"]
    ]
    return encoding


# print the first record from each dataset
# print(train_dataset[0])
# print(test_dataset[0])

# Encode the full dataset
train_dataset = dataset.map(
    encode_examples, batched=True
)
# test_dataset = test_dataset.map(
#     encode_examples, batched=True
# )

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
