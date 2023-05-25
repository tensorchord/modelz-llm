<div align="center">

# Modelz LLM

</div>

<p align=center>
<a href="https://discord.gg/KqswhpVgdU"><img alt="discord invitation link" src="https://dcbadge.vercel.app/api/server/KqswhpVgdU?style=flat"></a>
<a href="https://twitter.com/TensorChord"><img src="https://img.shields.io/twitter/follow/tensorchord?style=social" alt="trackgit-views" /></a>
</p>

Modelz LLM is an inference server that facilitates the utilization of open source large language models (LLMs), such as FastChat, LLaMA, and ChatGLM, on either **local or cloud-based** environments with **OpenAI compatible API**.

## Features

- **OpenAI compatible API**: Modelz LLM provides an OpenAI compatible API for LLMs, which means you can use the OpenAI python SDK to interact with the model.
- **Self-hosted**: Modelz LLM can be easily deployed on either local or cloud-based environments.
- **Open source LLMs**: Modelz LLM supports open source LLMs, such as FastChat, LLaMA, and ChatGLM.
- **Cloud native**: We provide docker images for different LLMs, which can be easily deployed on Kubernetes, or other cloud-based environments (e.g. [Modelz](https://docs.modelz.ai))

## Quick Start

### Install

```bash
pip install modelz-llm[gpu]
# or install from source
pip install git+https://github.com/tensorchord/modelz-llm.git[gpu]
```

### Run the self-hosted API server

Please first start the self-hosted API server by following the instructions:

```bash
modelz-llm -m "THUDM/chatglm-6b-int4"
```

Currently, we support the following models:

| Model Name | Huggingface Model | Docker Image |
| ---------- | ----------- | ---------------- |
| Vicuna 7B Delta V1.1  | `lmsys/vicuna-7b-delta-v1.1` | [modelzai/llm-vicuna-7b](https://hub.docker.com/repository/docker/modelzai/llm-vicuna-7b/general) |
| LLaMA 7B    | `decapoda-research/llama-7b-hf` | [modelzai/llm-llama-7b](https://hub.docker.com/repository/docker/modelzai/llm-llama-7b/general) |
| ChatGLM 6B INT4    | `THUDM/chatglm-6b-int4` | [modelzai/llm-chatglm-6b-int4](https://hub.docker.com/repository/docker/modelzai/llm-chatglm-6b-int4/general) |
| ChatGLM 6B  | `THUDM/chatglm-6b` | [modelzai/llm-chatglm-6b](https://hub.docker.com/repository/docker/modelzai/llm-chatglm-6b/general) |
| Bloomz 560M | `bigscience/bloomz-560m` | |
| Bloomz 1.7B | `bigscience/bloomz-1b7` | |
| Bloomz 3B | `bigscience/bloomz-3b` | |
| Bloomz 7.1B | `bigscience/bloomz-7b1` | |

<!-- | FastChat T5 3B V1.0  | `lmsys/fastchat-t5-3b-v1.0` | `lmsys/fastchat-t5-3b-v1.0` | -->

### Use OpenAI python SDK

Then you can use the OpenAI python SDK to interact with the model:

```python
import openai
openai.api_base="http://localhost:8000"
openai.api_key="any"

# create a chat completion
chat_completion = openai.ChatCompletion.create(model="any", messages=[{"role": "user", "content": "Hello world"}])
```
