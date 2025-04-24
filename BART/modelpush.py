from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
# 先创建仓库（如果已存在不会报错）

api.upload_folder(
    folder_path="/mnt/data4/luyiheng/AcBART2/lr_5e-05/checkpoint-195",
    repo_id="IsField/AcBART2",
    repo_type="model",
)
