# Modelz LLM

Modelz LLM provides an OpenAI compatible API for using open source large language models (LLMs) like [ChatGLM](https://github.com/THUDM/ChatGLM-6B), [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/), etc.

## Usage

### OpenAI Python SDK

Please first start the self-hosted API server by following the instructions:

```bash
export MODELZ_MODEL="THUDM/chatglm-6b-int4"
export MODELZ_TOKENIZER="THUDM/chatglm-6b-int4"
export OPENAI_API_BASE="http://localhost:8080"

python main.py
```

Then you can use the OpenAI python SDK to interact with the model:

```python
import openai
# create a chat completion
chat_completion = openai.ChatCompletion.create(model="self-hosted", messages=[{"role": "user", "content": "Hello world"}])
```

### Deploy with [Modelz](https://docs.modelz.ai)

To Be Added.

## Supported Models

| Model Name | Model (`MODELZ_MODEL`) | Tokenizer (`MODELZ_TOKENIZER`) |
| ---------- | ------------ | ---------------- |
| ChatGLM 6B INT4    | THUDM/chatglm-6b-int4 | THUDM/chatglm-6b-int4 |
