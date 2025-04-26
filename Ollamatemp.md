# ollama: 显示系统所提供选项的准确描述

# 第一次解释
当用户输入描述后，infer1.py 让系统返回一级标题，这时就将所有的一级标题放到ollama中输出每个的解释
## Prompt 格式：
“Explain keyword specifically, what's the difference between those fields, and the connections with my description” 
这样是调用一次把所有一级标题一次性输入，但可以尝试调用多次ollama，每次只输一个一级标题看看效果怎么样

# 第二次解释
当用户选完一级标题后，infer2.py 让系统返回二级标题，这是再将所有的二级标题让ollama依次解释，prompt格式和第一次相似
