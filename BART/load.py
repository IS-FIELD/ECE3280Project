from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sequence_to_classify = "Modeling and simulation is a subfield of computing methodologies that focuses on creating abstract models of complex systems and simulating their behavior over time. This area encompasses a wide range of techniques, including discrete event simulation, agent-based modeling, and system dynamics. It is widely used in various domains such as engineering, economics, biology, and social sciences to analyze and predict the behavior of systems under different conditions."


candidate_labels = [
    "Transformer",
    "Modeling and simulation",
    "Network",
    "Optimization",
]
print(classifier(sequence_to_classify, candidate_labels, multi_label=True))
