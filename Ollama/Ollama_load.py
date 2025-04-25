from ollama import chat
from ollama import ChatResponse


first_stage_labels = [
    "Applied computing",
    "Computer systems organization",
    "Computing methodologies",
]
first_stage_explains = []
for first_stage_label in first_stage_labels:

    response: ChatResponse = chat(
        model="deepseek-llm:7b",
        messages=[
            {
                "role": "user",
                "content": f"Explain the {first_stage_label} in less than 20 words.",
            },
        ],
    )
    first_stage_explains.append(response["message"]["content"])

print(first_stage_explains)
