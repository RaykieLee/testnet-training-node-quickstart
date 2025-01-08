import torch
from pathlib import Path
import argparse
from pprint import pprint

def load_training_args(file_path: str):
    """
    加载并解析 training_args.bin 文件
    
    Args:
        file_path: training_args.bin 文件的路径
    """
    try:
        # 加载训练参数
        training_args = torch.load(file_path)
        
        # 获取所有属性
        args_dict = {
            key: value for key, value in training_args.__dict__.items() 
            if not key.startswith('__')
        }
        
        # 打印主要训练参数
        print("\n=== 训练参数摘要 ===")
        important_params = {
            'learning_rate': args_dict.get('learning_rate'),
            'num_train_epochs': args_dict.get('num_train_epochs'),
            'per_device_train_batch_size': args_dict.get('per_device_train_batch_size'),
            'gradient_accumulation_steps': args_dict.get('gradient_accumulation_steps'),
            'max_steps': args_dict.get('max_steps'),
            'warmup_steps': args_dict.get('warmup_steps'),
            'weight_decay': args_dict.get('weight_decay'),
            'lora_rank': args_dict.get('lora_rank'),
            'lora_alpha': args_dict.get('lora_alpha'),
            'lora_dropout': args_dict.get('lora_dropout'),
        }
        pprint(important_params)
        
        print("\n=== 完整参数 ===")
        pprint(args_dict)
        
        return args_dict
        
    except Exception as e:
        print(f"解析文件时出错: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='解析 training_args.bin 文件')
    parser.add_argument('file_path', type=str, help='training_args.bin 文件的路径')
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.file_path).exists():
        print(f"错误: 文件 '{args.file_path}' 不存在")
        exit(1)
        
    load_training_args(args.file_path) 