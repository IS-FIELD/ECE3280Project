from transformers import BartForSequenceClassification, BartTokenizerFast

model = BartForSequenceClassification.from_pretrained(
    "/mnt/data4/luyiheng/BARTfinetune/lr_5e-05/checkpoint-195"
)
tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large-mnli")

text = "It prevent machine from attack"
inputs = tokenizer(
    text, return_tensors="pt", truncation=True, padding="max_length", max_length=256
)
outputs = model(**inputs)
probs = outputs.logits.softmax(dim=-1)  # 这里就是所有以及标签的置信度


fst_label_map = {
    "0": "Applied computing",
    "1": "Computer systems organization",
    "2": "Computing methodologies",
    "3": "General and reference",
    "4": "Hardware",
    "5": "Human-centered computing",
    "6": "Information systems",
    "7": "Mathematics of computing",
    "8": "Networks",
    "9": "Security and privacy",
    "10": "Theory of computation",
}

# 取前5个id
probs = probs.squeeze(0)  # shape: [num_labels]

# 获取前5个概率最大的索引
top_5_indices = probs.topk(5).indices.tolist()  # 返回前5个最大值的索引
top_5_ids = [str(i) for i in top_5_indices]
top_5_labels = [fst_label_map[id] for id in top_5_ids]
print(f"top_5_labels: {top_5_labels}")

most_prob_id = probs.argmax(dim=-1).item()
print(f"most probably label: {fst_label_map[str(most_prob_id)]}")
