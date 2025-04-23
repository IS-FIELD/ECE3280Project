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
probs = outputs.logits.softmax(dim=-1)
pred_label_id = probs.argmax(dim=-1).item()
print(pred_label_id)
