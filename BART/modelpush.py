from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/mnt/data4/luyiheng/BARTfinetune/lr_5e-05/checkpoint-195",
    repo_id="IsField/AcBART",
    repo_type="model",
)
