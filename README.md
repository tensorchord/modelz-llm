<div align="center">

# Modelz LLM

</div>

<p align=center>
<a href="https://discord.gg/KqswhpVgdU"><img alt="discord invitation link" src="https://dcbadge.vercel.app/api/server/KqswhpVgdU?style=flat"></a>
<a href="https://twitter.com/TensorChord"><img src="https://img.shields.io/twitter/follow/tensorchord?style=social" alt="trackgit-views" /></a>
</p>

Modelz LLM provides an OpenAI compatible API for using open source large language models (LLMs) like [ChatGLM](https://github.com/THUDM/ChatGLM-6B), [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/), etc.

## Usage

### Run the self-hosted API server

Please first start the self-hosted API server by following the instructions:

```bash
export MODELZ_MODEL="THUDM/chatglm-6b-int4"
export MODELZ_TOKENIZER="THUDM/chatglm-6b-int4"

python main.py
```

### Use OpenAI python SDK

Then you can use the OpenAI python SDK to interact with the model:

```python
import openai
openai.api_base="http://localhost:8000"

# create a chat completion
chat_completion = openai.ChatCompletion.create(messages=[{"role": "user", "content": "Hello world"}])
```

## Deploy with [Modelz](https://docs.modelz.ai)

To Be Added.

## Supported Models

| Model Name | Model (`MODELZ_MODEL`) | Tokenizer (`MODELZ_TOKENIZER`) |
| ---------- | ------------ | ---------------- |
| ChatGLM 6B INT4    | THUDM/chatglm-6b-int4 | THUDM/chatglm-6b-int4 |
