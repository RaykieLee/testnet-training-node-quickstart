import os
from dataclasses import dataclass
from huggingface_hub import list_repo_refs, HfApi
from typing import Optional
from loguru import logger  # 添加这行导入

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from dataset import SFTDataCollator, SFTDataset
from merge import merge_lora_to_base_model
from utils.constants import model2template


@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: int


def get_last_trained_model(task_id: str, model_id: str, username: str) -> Optional[str]:
    """
    查找该任务的上一次训练模型
    
    Args:
        task_id: 任务ID
        model_id: 基础模型ID
        username: HuggingFace用户名
    
    Returns:
        如果找到则返回模型仓库ID，否则返回None
    """
    try:
        # 构建可能的仓库名称
        repo_name = f"{username}/task-{task_id}-{model_id.replace('/', '-')}"
        
        # 检查仓库是否存在
        api = HfApi(token=os.environ["HF_TOKEN"])
        refs = list_repo_refs(repo_name)
        
        if refs.branches:  # 如果仓库存在且有分支
            logger.info(f"找到上一次训练的模型: {repo_name}")
            return repo_name
            
    except Exception as e:
        logger.info(f"未找到上一次训练的模型: {e}")
    
    return None


def train_lora(
    model_id: str, 
    context_length: int, 
    training_args: LoraTrainingArguments,
    task_id: str = None,  # 新增参数
):
    assert model_id in model2template, f"model_id {model_id} not supported"
    
    # 配置LoRA参数
    lora_config = LoraConfig(
        r=training_args.lora_rank,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # 配置4-bit量化参数
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 查找上一次训练的模型
    base_model_id = model_id
    if task_id:
        last_model = get_last_trained_model(
            task_id=task_id,
            model_id=model_id,
            username=os.environ["HF_USERNAME"]
        )
        if last_model:
            base_model_id = last_model
            logger.info(f"将基于上一次训练的模型继续训练: {base_model_id}")
        else:
            logger.info(f"未找到上一次训练的模型，将使用基础模型: {base_model_id}")

    training_args = SFTConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=context_length,
    )
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        use_fast=True,
        token=os.environ["HF_TOKEN"],
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ["HF_TOKEN"],
    )

    # Load dataset
    dataset = SFTDataset(
        file="demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )

    # Define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    # Train model
    trainer.train()

    # save model
    trainer.save_model("outputs")

    # remove checkpoint folder
    os.system("rm -rf outputs/checkpoint-*")

    # upload lora weights and tokenizer
    print("Training Completed.")
