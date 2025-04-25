from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset
import os
import warnings
import json
from ollama import chat, Client
import asyncio
from fastapi.concurrency import run_in_threadpool

app = FastAPI()

client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestData(BaseModel):
    text: str
    infer: str
    context: Optional[str] = None

id2label_1st = {
    0: "Applied computing",
    1: "Computer systems organization",
    2: "Computing methodologies",
    3: "General and reference",
    4: "Hardware",
    5: "Human-centered computing",
    6: "Information systems",
    7: "Mathematics of computing",
    8: "Networks",
    9: "Security and privacy",
    10: "Theory of computation"
}

id2label_2nd = {
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


model_1st = None
tokenizer_1st = None
model_2nd = None
tokenizer_2nd = None
dataset = None

@app.on_event("startup")
async def load_models():
    global model_1st, tokenizer_1st, model_2nd, tokenizer_2nd, dataset
    
    config_1st = AutoConfig.from_pretrained(
        "IsField/AcBART",
        num_labels=11,
        id2label=id2label_1st,
        label2id={v: k for k, v in id2label_1st.items()}
    )
    model_1st = AutoModelForSequenceClassification.from_pretrained(
        "IsField/AcBART",
        config=config_1st,
        ignore_mismatched_sizes=True
    )
    tokenizer_1st = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    model_1st.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    config_2nd = AutoConfig.from_pretrained(
        "IsField/AcBART2",
        num_labels=70,
        id2label=id2label_2nd,
        label2id={v: k for k, v in id2label_2nd.items()}
    )
    model_2nd = AutoModelForSequenceClassification.from_pretrained(
        "IsField/AcBART2",
        config=config_2nd,
        ignore_mismatched_sizes=True
    )
    tokenizer_2nd = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    model_2nd.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        dataset = load_dataset("json", data_files="C:\source\google-scholar-assistant\ECE3280Project-model\CSVs\\00_97.json")["train"]
    except Exception as e:
        print(f"Error loading label mapping: {e}")

async def get_ollama_explanation(label: str) -> str:
    try:
        response = await asyncio.wait_for(
            client.chat(
                model="deepseek-llm:7b",
                messages=[{
                    "role": "user",
                    "content": f"Explain '{label}' in academic terms within 20 words."
                }]
            ),
            timeout=5.0
        )
        return response["message"]["content"]
    except asyncio.TimeoutError:
        print(f"Ollama timeout for label: {label}")
        return "ollama_no_response"
    except Exception as e:
        print(f"Ollama error for {label}: {str(e)}")
        return "ollama_no_response"

async def process_labels_with_explanations(labels: List[str], scores: List[float]) -> List[Dict]:
    tasks = [get_ollama_explanation(label) for label in labels]
    explanations = await asyncio.gather(*tasks)
    
    return [
        {"label": label, "score": score, "explanation": explanation}
        for label, score, explanation in zip(labels, scores, explanations)
    ]

async def predict_1st(text: str) -> Dict:
    def sync_inference():
        inputs = tokenizer_1st(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        ).to(model_1st.device)
        with torch.no_grad():
            outputs = model_1st(**inputs)
        return outputs
    
    outputs = await run_in_threadpool(sync_inference)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    labels = [id2label_1st[idx.item()] for idx in top5_indices]
    scores = [round(prob.item(), 4) for prob in top5_probs]
    
    return {
        "labels": await process_labels_with_explanations(labels, scores),
        "sequence": text[:50] + "..."
    }

async def predict_2nd(text: str, context: str) -> Dict:
    global dataset
    
    def align_labels(example):
        return {"limited_2nd_label": example["2nd_stage_label"] 
                if example["1st_stage_label"].lower() == context.lower() 
                else None}
    
    filtered_dataset = dataset.map(align_labels)
    valid_2nd_labels = [x["limited_2nd_label"] for x in filtered_dataset if x["limited_2nd_label"] is not None]
    
    snd_ids = [str(k) for k, v in id2label_2nd.items() if v in valid_2nd_labels]
    # print(f"Valid secondary labels: {valid_2nd_labels}")
    
    if not snd_ids:
        return {"error": "No valid secondary labels found"}
    
    def sync_inference():
        inputs = tokenizer_2nd(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        ).to(model_2nd.device)
        with torch.no_grad():
            outputs = model_2nd(**inputs)
        return outputs
    
    outputs = await run_in_threadpool(sync_inference)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
    
    snd_ids_int = [int(id) for id in snd_ids]
    filtered_probs = [(i, probs[i].item()) for i in snd_ids_int]
    sorted_probs = sorted(filtered_probs, key=lambda x: x[1], reverse=True)[:5]
    
    labels = [id2label_2nd[str(id)] for id, _ in sorted_probs]
    scores = [round(prob, 4) for _, prob in sorted_probs]
    
    return {
        "labels": await process_labels_with_explanations(labels, scores),
        "sequence": f"{context}: {text[:30]}..."
    }


@app.post("/classify")
async def classify(data: RequestData):
    try:
        print("\n=== Received Request Data ===")
        print(f"Text length: {len(data.text)} characters")
        print(f"Infer type: {data.infer}")
        print(f"Context: {data.context}")
        print("=============================\n")
        if data.infer == "infer_1":
            result = await predict_1st(data.text)
            return {**result, "type": "infer_1"}
        
        if data.infer == "infer_2" and data.context:
            result = await predict_2nd(data.text, data.context)
            if "error" in result:
                return {"error": result["error"]}
            return {**result, "type": "infer_2"}
        
        
    
        return {"error": "Invalid request parameters"}
    

    except Exception as e:
        print(f"Classification error: {str(e)}")
        return {"error": "Internal server error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
