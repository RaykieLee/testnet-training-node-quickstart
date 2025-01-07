qwen_template = {
    "system_format": "<|im_start|>system\n{content}<|im_end|>\n",
    "user_format": "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "assistant_format": "{content}<|im_end|>\n",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<|im_start|>tool\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "system": "You are a helpful assistant.",
}

gemma_template = {
    "system_format": "<bos>",
    "user_format": "<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
    "assistant_format": "{content}<eos>\n",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<start_of_turn>tool\n{content}<end_of_turn>\n<start_of_turn>model\n",
    "system": None,
}

phi_template = {
    "system_format": None,
    "user_format": "<|user|>\n{content}<|end|>\n<|assistant|>",
    "assistant_format": "{content}<|end|>\n",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<|tool|>\n{content}<|end|>\n<|assistant|>",
    "system": None,
}

yi_template = {
    "system_format": "<|im_start|>system\n{content}<|im_end|>\n",
    "user_format": "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "assistant_format": "{content}<|im_end|>\n",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<|im_start|>tool\n{content}<im_end>\n<|im_start|>assistant\n",
    "system": None,
}

zephyr_template = {
    "system_format": "<|system|>\n{content}</s>",
    "user_format": "<|user|>\n{content}</s>\n<|assistant|>\n",
    "assistant_format": "{content}</s>\n",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<|tool|>\n{content}</s>\n<|assistant|>\n",
    "system": None,
}

mistral_template = {
    "system_format": "<s>",
    "user_format": "[INST]{content}[/INST]",
    "assistant_format": "{content}</s>",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "{content}",
    "system": "",
}

mixtral_template = {
    "system_format": "<s>",
    "user_format": "[INST]{content}[/INST]",
    "assistant_format": "{content}</s>",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "{content}",
    "system": "",
}

llama2_template = {
    "system_format": "<<SYS>>\n{content}\n<</SYS>>\n\n",
    "user_format": "[INST]{content}[/INST]",
    "assistant_format": "{content} </s>",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "{content}",
    "system": "You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, "
    "racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, "
    "explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information.",
}

llama3_template = {
    "system_format": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
    "user_format": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "assistant_format": "{content}<|eot_id|>",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<|start_header_id|>tool<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "system": None,
}

model2template = {
    "Qwen/Qwen1.5-1.8B": qwen_template,
    "Qwen/Qwen1.5-7B": qwen_template,
    "google/gemma-2b": gemma_template,
    "google/gemma-7b": gemma_template,
    "microsoft/Phi-3.5-mini-instruct": phi_template,
    "microsoft/Phi-3-mini-4k-instruct": phi_template,
    "01-ai/Yi-6B": yi_template,
    "HuggingFaceH4/zephyr-7b-beta": zephyr_template,
    "mistralai/Mistral-7B-v0.1": mistral_template,
    "mistralai/Mixtral-8x7B-v0.1": mixtral_template,
    "meta-llama/Llama-2-7b-chat-hf": llama2_template,
    "meta-llama/Llama-3-7b-chat-hf": llama3_template,
}

model2size = {
    "Qwen/Qwen1.5-1.8B": 1_840_000_000,
    "Qwen/Qwen1.5-7B": 7_720_000_000,
    "google/gemma-2b": 2_510_000_000,
    "google/gemma-7b": 8_540_000_000,
    "microsoft/Phi-3.5-mini-instruct": 3_500_000_000,
    "microsoft/Phi-3-mini-4k-instruct": 3_800_000_000,
    "01-ai/Yi-6B": 6_000_000_000,
    "HuggingFaceH4/zephyr-7b-beta": 7_000_000_000,
    "mistralai/Mistral-7B-v0.1": 7_000_000_000,
    "mistralai/Mixtral-8x7B-v0.1": 47_000_000_000,
    "meta-llama/Llama-2-7b-chat-hf": 7_000_000_000,
    "meta-llama/Llama-3-7b-chat-hf": 7_000_000_000,
}

model2base_model = {
    "Qwen/Qwen1.5-1.8B": "qwen1.5",
    "Qwen/Qwen1.5-7B": "qwen1.5",
    "google/gemma-2b": "gemma",
    "google/gemma-7b": "gemma",
    "microsoft/Phi-3.5-mini-instruct": "phi",
    "microsoft/Phi-3-mini-4k-instruct": "phi",
    "01-ai/Yi-6B": "yi",
    "HuggingFaceH4/zephyr-7b-beta": "zephyr",
    "mistralai/Mistral-7B-v0.1": "mistral",
    "mistralai/Mixtral-8x7B-v0.1": "mixtral",
    "meta-llama/Llama-2-7b-chat-hf": "llama2",
    "meta-llama/Llama-3-7b-chat-hf": "llama3",
}
    # "microsoft/Phi-3-small-8k-instruct",
# 
# 