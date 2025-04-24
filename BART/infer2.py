from transformers import BartForSequenceClassification, BartTokenizerFast
from datasets import load_dataset


pre_1st_stage_labels = input("Previous stage label: ") #这里输入第一个阶段，用户选完的标签

model = BartForSequenceClassification.from_pretrained(
    "/mnt/data4/luyiheng/AcBART2/lr_5e-05/checkpoint-195"
) #这个模型地址就是整个打包好的文件夹

tokenizer = BartTokenizerFast.from_pretrained(
    "/mnt/data4/luyiheng/AcBART2/lr_5e-05/checkpoint-195"
) #跟model一样的地址

dataset = load_dataset(
    "json", data_files="/mnt/workspace/luyiheng/ECE3280/CSVs/00_97.json"
)[
    "train"
]  # json文件就是00_97.son


def Align_1st_2nd_stage_labels(example, pre_1st_stage_labels):
    fst_stage_labels = example["1st_stage_label"]
    snd_stage_labels = example["2nd_stage_label"]
    if fst_stage_labels == pre_1st_stage_labels:
        return {"limited_2nd_stage_label": snd_stage_labels}
    else:
        return {"limited_2nd_stage_label": None}


# 这样map后会得到一个新字段"limited_2nd_stage_label"
limited_2nd_stage_labels_dataset = dataset.map(
    lambda example: Align_1st_2nd_stage_labels(example, pre_1st_stage_labels)
)

# 提取所有非None的二级标签
limited_2nd_stage_labels = [
    x["limited_2nd_stage_label"]
    for x in limited_2nd_stage_labels_dataset
    if x["limited_2nd_stage_label"] is not None
]


text = "It prevent machine from attack"
inputs = tokenizer(
    text, return_tensors="pt", truncation=True, padding="max_length", max_length=256
)
outputs = model(**inputs)
probs = outputs.logits.softmax(dim=-1)

# 这个labels_map下面是所有耳机标签的字典
labels_map = {
    "0": "Accessibility",
    "1": "Architectures",
    "2": "Artificial intelligence",
    "3": "Arts and humanities",
    "4": "Collaborative and social computing",
    "5": "Communication hardware, interfaces and storage",
    "6": "Computational complexity and cryptography",
    "7": "Computers in other domains",
    "8": "Continuous mathematics",
    "9": "Cross-computing tools and techniques",
    "10": "Cryptography",
    "11": "Data management systems",
    "12": "Database and storage security",
    "13": "Dependable and fault-tolerant systems and networks",
    "14": "Design and analysis of algorithms",
    "15": "Discrete mathematics",
    "16": "Distributed computing methodologies",
    "17": "Document management and text processing",
    "18": "Document types",
    "19": "Education",
    "20": "Electronic commerce",
    "21": "Electronic design automation",
    "22": "Embedded and cyber-physical systems",
    "23": "Emerging technologies",
    "24": "Enterprise computing",
    "25": "Formal languages and automata theory",
    "26": "Formal methods and theory of security",
    "27": "Hardware test",
    "28": "Hardware validation",
    "29": "Human and societal aspects of security and privacy",
    "30": "Human computer interaction (HCI)",
    "31": "Information retrieval",
    "32": "Information storage systems",
    "33": "Information systems applications",
    "34": "Information theory",
    "35": "Integrated circuits",
    "36": "Interaction design",
    "37": "Intrusion/-+anomaly detection and malware mitigation",
    "38": "Logic",
    "39": "Machine learning",
    "40": "Mathematical analysis",
    "41": "Mathematical software",
    "42": "Modeling and simulation",
    "43": "Models of computation",
    "44": "Network algorithms",
    "45": "Network architectures",
    "46": "Network components",
    "47": "Network performance evaluation",
    "48": "Network properties",
    "49": "Network protocols",
    "50": "Network security",
    "51": "Network services",
    "52": "Network types",
    "53": "Parallel computing methodologies",
    "54": "Power and energy",
    "55": "Probability and statistics",
    "56": "Randomness, geometry and discrete structures",
    "57": "Real-time systems",
    "58": "Robustness",
    "59": "Security in hardware",
    "60": "Security services",
    "61": "Semantics and reasoning",
    "62": "Software and application security",
    "63": "Symbolic and algebraic manipulation",
    "64": "Systems security",
    "65": "Theory and algorithms for application domains",
    "66": "Ubiquitous and mobile computing",
    "67": "Very large scale integration design",
    "68": "Visualization",
    "69": "World Wide Web",
}

print(
    f"probs: {probs}"
)  # 这个probs是一个tensor，shape是[1, num_labels]，表示labels_map中每个标签的置信度


most_probably_id = probs.argmax(dim=-1).item() # 取出置信度最大的二级标签的id，不管在不在一级标签下
print(f"most_probably_label: {labels_map[str(most_probably_id)]}")


def take_2nd_id(limited_2nd_stage_labels, labels_map):
    snd_ids = []
    for (
        snd_id,
        snd_label,
    ) in labels_map.items():  
        if snd_label in limited_2nd_stage_labels:
            snd_ids.append(snd_id)
    return snd_ids


snd_ids = take_2nd_id(limited_2nd_stage_labels, labels_map)



#这里是取出用户选定的一级标签下前五的二级标签
def take_most_prob_in_2nd_id(snd_ids, probs, topk=5):
    # probs: tensor shape [1, num_labels]
    probs = probs.squeeze(0)  # shape: [num_labels]
    # 只保留snd_ids对应的概率
    snd_ids = [int(i) for i in snd_ids]  # 保证是int
    snd_probs = [(i, probs[i].item()) for i in snd_ids]
    # 按概率降序排序
    snd_probs_sorted = sorted(snd_probs, key=lambda x: x[1], reverse=True)
    # 取前topk个id
    top_ids = [str(i[0]) for i in snd_probs_sorted[:topk]]
    return top_ids


top5_ids = take_most_prob_in_2nd_id(snd_ids, probs)
print(f"top5_ids: {top5_ids}")
print(f"top 5 keyword: {', '.join([labels_map[id] for id in top5_ids])}")

if most_probably_id not in top5_ids: # 如果置信度最大的二级标签不在前五个中，说明可能用户选的一级标签是错误的
    print(f"Maybe you should try another key: {labels_map[str(most_probably_id)]}")
