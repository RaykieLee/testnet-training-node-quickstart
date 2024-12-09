# 导入必要的库
import json
import os
import time

import requests
import yaml
from loguru import logger
from huggingface_hub import HfApi

# 导入自定义模块
from demo import LoraTrainingArguments, train_lora
from utils.constants import model2base_model, model2size
from utils.flock_api import get_task, submit_task
from utils.gpu_utils import get_gpu_type

# 从环境变量获取 HuggingFace 用户名
HF_USERNAME = os.environ["HF_USERNAME"]

if __name__ == "__main__":
    # 从环境变量获取任务ID
    task_id = os.environ["TASK_ID"]
    
    # 获取当前文件所在目录
    current_folder = os.path.dirname(os.path.realpath(__file__))
    # 加载训练参数配置文件
    with open(f"{current_folder}/training_args.yaml", "r") as f:
        all_training_args = yaml.safe_load(f)

    # 获取任务信息
    task = get_task(task_id)
    # 记录任务信息到日志
    logger.info(json.dumps(task, indent=4))
    
    # 从任务数据中获取必要参数
    data_url = task["data"]["training_set_url"]  # 训练数据集URL
    context_length = task["data"]["context_length"]  # 上下文长度
    max_params = task["data"]["max_params"]  # 最大参数量限制

    # 根据最大参数量限制筛选可用模型
    model2size = {k: v for k, v in model2size.items() if v <= max_params}
    all_training_args = {k: v for k, v in all_training_args.items() if k in model2size}
    logger.info(f"Models within the max_params: {all_training_args.keys()}")
    
    # 分块下载训练数据
    response = requests.get(data_url, stream=True)
    with open("demo_data.jsonl", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # 遍历所有可用模型进行训练
    for model_id in all_training_args.keys():
        # 获取GPU类型信息
        gpu_type = get_gpu_type()
        logger.info(f"gpu_type is {gpu_type}")
        logger.info(f"Start to train the model {model_id}...")
        # 尝试训练模型，如果内存溢出则跳过当前模型
        try:
            train_lora(
                model_id=model_id,
                context_length=context_length,
                training_args=LoraTrainingArguments(**all_training_args[model_id]),
                task_id=task_id,
            )
        except RuntimeError as e:
            logger.error(f"Error: {e}")
            logger.info("Proceed to the next model...")
            continue


        try:
            logger.info("Start to push the lora weight to the hub...")
            # 初始化HuggingFace API客��端
            api = HfApi(token=os.environ["HF_TOKEN"])
            # 生成模型仓库名称
            repo_name = f"{HF_USERNAME}/task-{task_id}-{model_id.replace('/', '-')}"
            
            # 尝试创建仓库，如果已存在则继续使用
            try:
                api.create_repo(
                    repo_name,
                    exist_ok=False,
                    repo_type="model",
                )
            except Exception as e:
                logger.info(f"Repo {repo_name} already exists. Will commit the new version.")

            # 上传模型文件夹到HuggingFace
            commit_message = api.upload_folder(
                folder_path="outputs",
                repo_id=repo_name,
                repo_type="model",
            )
            # 获取提交哈希值
            commit_hash = commit_message.oid
            logger.info(f"Commit hash: {commit_hash}")
            logger.info(f"Repo name: {repo_name}")
            
            # 提交任务结果
            submit_task(
                task_id, repo_name, model2base_model[model_id], gpu_type, commit_hash
            )
            logger.info("Task submitted successfully")
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.info("Proceed to the next model...")
        finally:
            # 清理临时文件和目录
            os.system("rm -rf merged_model")
            os.system("rm -rf outputs")
            continue
