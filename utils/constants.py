qwen_template = {
    "system_format": "<|im_start|>system\n{content}<|im_end|>\n",
    "user_format": "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "assistant_format": "{content}<|im_end|>\n",
    "system": "You are a helpful assistant.",
}

gemma_template = {
    "system_format": "<bos>",
    "user_format": "<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
    "assistant_format": "{content}<eos>\n",
    "system": None,
}

model2template = {
    "Qwen/Qwen1.5-0.5B": qwen_template,
    "Qwen/Qwen1.5-1.8B": qwen_template,
    "Qwen/Qwen1.5-7B": qwen_template,
    "google/gemma-2b": gemma_template,
    "google/gemma-7b": gemma_template,
    "Qwen/Qwen2.5-0.5B": qwen_template,
    "Qwen/Qwen2.5-0.5B-Instruct": qwen_template,
    "Qwen/Qwen2.5-1.5B": qwen_template,
    "Qwen/Qwen2.5-1.5B-Instruct": qwen_template,
    "Qwen/Qwen2.5-3B": qwen_template,
    "google/gemma-2-2b": gemma_template,
    "google/gemma-2-9b": gemma_template,
}

model2size = {
    "Qwen/Qwen1.5-0.5B": 620_000_000,
    "Qwen/Qwen1.5-1.8B": 1_840_000_000,
    "Qwen/Qwen1.5-7B": 7_720_000_000,
    "google/gemma-2b": 2_510_000_000,
    "google/gemma-7b": 8_540_000_000,
    "Qwen/Qwen2.5-0.5B": 500_000_000,
    "Qwen/Qwen2.5-0.5B-Instruct": 500_000_000,
    "Qwen/Qwen2.5-1.5B": 1_500_000_000,
    "Qwen/Qwen2.5-1.5B-Instruct": 1_500_000_000,
    "Qwen/Qwen2.5-3B": 3_000_000_000,
    "google/gemma-2-2b": 2_500_000_000,  # 25亿参数
    "google/gemma-2-9b": 9_000_000_000,  # 90亿参数
}

model2base_model = {
    "Qwen/Qwen1.5-0.5B": "qwen1.5",
    "Qwen/Qwen1.5-1.8B": "qwen1.5",
    "Qwen/Qwen1.5-7B": "qwen1.5",
    "google/gemma-2b": "gemma",
    "google/gemma-7b": "gemma",
    "Qwen/Qwen2.5-0.5B": "qwen2.5",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5",
    "Qwen/Qwen2.5-1.5B": "qwen2.5",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5",
    "Qwen/Qwen2.5-3B": "qwen2.5",
    "google/gemma-2-2b": "gemma",
    "google/gemma-2-9b": "gemma",
}
