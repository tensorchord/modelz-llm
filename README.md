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
- **Modelz integration**: Modelz LLM can be easily integrated with [Modelz](https://docs.modelz.ai), which is a serverless inference platform for LLMs and other foundation models.

## Quick Start

### Run the self-hosted API server

Please first start the self-hosted API server by following the instructions:

```bash
export MODELZ_MODEL="THUDM/chatglm-6b-int4"
export MODELZ_TOKENIZER="THUDM/chatglm-6b-int4"

python main.py
```

Currently, we support the following models:

| Model Name | Model (`MODELZ_MODEL`) | Tokenizer (`MODELZ_TOKENIZER`) |
| ---------- | ------------ | ---------------- |
| Vicuna 7B Delta V1.1    | `lmsys/vicuna-7b-delta-v1.1` | `lmsys/vicuna-7b-delta-v1.1` |
| LLaMA 7B    | `decapoda-research/llama-7b-hf` | `decapoda-research/llama-7b-hf` |
<!-- | FastChat T5 3B V1.0  | `lmsys/fastchat-t5-3b-v1.0` | `lmsys/fastchat-t5-3b-v1.0` | -->
| ChatGLM 6B INT4    | `THUDM/chatglm-6b-int4` | `THUDM/chatglm-6b-int4` |
| ChatGLM 6B    | `THUDM/chatglm-6b` | `THUDM/chatglm-6b` |

You could set the `MODELZ_MODEL` and `MODELZ_TOKENIZER` environment variables to specify the model and tokenizer.

### Use OpenAI python SDK

Then you can use the OpenAI python SDK to interact with the model:

```python
import openai
openai.api_base="http://localhost:8000"
openai.api_key="any"

# create a chat completion
chat_completion = openai.ChatCompletion.create(model="any", messages=[{"role": "user", "content": "Hello world"}])
```
