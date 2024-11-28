# FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation

## Overview

FlagEvalMM is an open-source evaluation framework designed to comprehensively assess multimodal models. It provides a standardized way to evaluate models that work with multiple modalities (text, images, video) across various tasks and metrics.

## Key Features

- **Flexible Architecture**: Support for multiple multimodal models and evaluation tasks, including: VQA, image retrieval, text-to-image, etc.
- **Comprehensive Benchmarks and Metrics**: Support new and commonly used benchmarks and metrics.
- **Extensive Model Support**: The model_zoo provides inference support for a wide range of popular multimodal models including QWenVL and LLaVA. Additionally, it offers seamless integration with API-based models such as GPT, Claude, and HuanYuan.
- **Extensible Design**: Easily extendable to incorporate new models, benchmarks, and evaluation metrics.

## Installation

```bash
git clone https://github.com/flageval-baai/FlagEvalMM.git
cd FlagEvalMM
pip install -e .
```

## Usage

FlagevalMM supports one-key evaluation:

An example of llava with vllm as backend:

```bash
flagevalmm --tasks tasks/mmmu/mmmu_val.py \
        --exec model_zoo/vlm/http_api/model_adapter.py \
        --model llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
        --num-workers 8 \
        --output-dir ./results/llava-onevision-qwen2-7b-ov-chat-hf \
        --backend vllm \
        --extra-args "--limit-mm-per-prompt image=10 --max-model-len 32768"
```

`--tasks` is the path of the task you want to evaluate, witch supports multiple tasks.

`--exec` is the script to adapt the model.

`--model` can be the model name in huggingface or your own model path. It is recommended to download the model from huggingface in advance.

`--extra-args` are the parameters for the vllm server.

For large models like Qwen2-VL-72B that use vllm, you can enable multi-GPU inference with the `--tensor-parallel-size` parameter:

```bash
flagevalmm --tasks tasks/mmmu_pro/mmmu_pro_standard_test.py tasks/ocrbench/ocrbench_test.py \
        --exec model_zoo/vlm/http_api/model_adapter.py \
        --model Qwen/Qwen2-VL-72B-Instruct \
        --num-workers 8 \
        --output-dir ./results/Qwen2-VL-72B-Instruct \
        --backend vllm \
        --extra-args "--limit-mm-per-prompt image=18 --tensor-parallel-size 4 --max-model-len 32768 --trust-remote-code --mm-processor-kwargs '{\"max_dynamic_patch\":4}'"
```

Since the parameters can be quite complex, it's recommended to use a JSON config file instead. Here's an example:

Create a config file named `qwen2_vl_72b_instruct.json`:

```json
{
    "model_name": "Qwen/Qwen2-VL-72B-Instruct",
    "api_key": "EMPTY",
    "output_dir": "./results/Qwen2-VL-72B-Instruct",
    "min_image_hw": 28,
    "num_workers": 8,
    "backend": "vllm",
    "extra_args": "--limit-mm-per-prompt image=18 --tensor-parallel-size 4 --max-model-len 32768 --trust-remote-code --mm-processor-kwargs '{\"max_dynamic_patch\":4}'"
}
```

This simplifies your evaluation command to:

```bash
flagevalmm --tasks tasks/mmmu_pro/mmmu_pro_standard_test.py tasks/ocrbench/ocrbench_test.py \
        --exec model_zoo/vlm/http_api/model_adapter.py \
        --cfg qwen2_vl_72b_instruct.json
```

Example of evaluating models without vllm (using transformers instead):

```bash
flagevalmm --tasks tasks/mmmu/mmmu_val.py \
        --exec model_zoo/vlm/llama-vision/model_adapter.py \
        --model /share/project/huggingface/models/Meta-Llama-3.2-11B-Vision-Instruct \
        --output-dir ./results/Meta-Llama-3.2-11B-Vision-Instruct
```

For models using transformers directly, the `--backend` and `--extra-args` parameters are not required. Additional model examples can be found in the `model_zoo/vlm/` directory.

Example of evaluating gpt-style models:

```bash
flagevalmm --tasks tasks/mmmu/mmmu_val.py \
        --exec model_zoo/vlm/http_api/model_adapter.py \
        --model gpt-4o-mini \
        --num-workers 4 \
        --url https://api.openai.com/v1/chat/completions \
        --api-key $OPENAI_API_KEY \
        --output-dir ./results/gpt-4o-mini \
        --use-cache
```

`--use-cache` is optional, it will cache the model outputs, the same question with the same model setting will get results from cache.

## About API keys

If you want to evaluate some tasks by GPT (like charxiv, math_verse, etc.), you need to set the following environment variables:

```bash
export FLAGEVAL_API_KEY=$YOUR_OPENAI_API_KEY
export FLAGEVAL_BASE_URL="https://api.openai.com/v1"
```