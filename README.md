我们要用的预训练模型BART：https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/bart#transformers.BartForSequenceClassification


数据集规范 MNLI ：https://huggingface.co/datasets/nyu-mll/multi_nli


插件：交互式的缩小范围
示例

1：用户输入description "Modeling and simulation is a subfield of computing methodologies that focuses on creating abstract models of complex systems and simulating their behavior over time. This area encompasses a wide range of techniques, including discrete event simulation, agent-based modeling, and system dynamics. It is widely used in various domains such as engineering, economics, biology, and social sciences to analyze and predict the behavior of systems under different conditions."

2: 模型给出置信度前3的一级标题并给出一级标题的解释，让用户选择
[
{'Computing methodologies':"This field focuses on the study and development of computational methods and techniques, including algorithms, programming models, computer architectures, and computational efficiency. It emphasizes improving system performance, scalability, and exploring new computational approaches such as parallel computing, distributed computing, and quantum computing."}, 

{'Network':"The network field explores computer networks and related technologies, including data transmission, network protocols, network architecture, and network security. It covers networks of different scales, such as Local Area Networks (LAN), Wide Area Networks (WAN), and the internet, with a focus on efficient, reliable, and secure data communication."},

{'Integrated Circuit':"An integrated circuit (IC) refers to the technology of embedding multiple electronic components, such as transistors, resistors, capacitors, and others, onto a single small semiconductor chip. ICs are used in a wide range of electronic devices, including computers, smartphones, and home appliances. The design and manufacturing of ICs involve circuit design, material science, and microelectronics."}
]

3: 在用户选择完一级标题后给出前5的二级标题，格式和第二步一样，给出解释让用户选择

4：在用户选择完二级标题后，按google scholar（一级，二级）的搜索结果给出默认排名前10的文章
