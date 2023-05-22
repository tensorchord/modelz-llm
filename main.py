import logging
import os
import sys
from datetime import datetime
from typing import List

import falcon
import msgspec
import torch  # type: ignore
from falcon.asgi import App, Request, Response
from llmspec import (
    ChatChoice,
    ChatCompletionRequest,
    ChatMessage,
    CompletionResponse,
    PromptCompletionRequest,
    Role,
    TokenUsage,
    LanguageModels,
)
import transformers

DEFAULT_MODEL = "THUDM/chatglm-6b-int4"
TOKENIZER = os.environ.get("MODELZ_TOKENIZER", DEFAULT_MODEL)
MODEL = os.environ.get("MODELZ_MODEL", DEFAULT_MODEL)


formatter = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s"
logging.basicConfig(stream=sys.stdout, format=formatter, level=logging.INFO)


class LLM:
    def __init__(self, model_name: str, tokenizer_name: str) -> None:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        model_cls = getattr(transformers, LanguageModels.transformer_cls(model_name))
        self.model = model_cls.from_pretrained(model_name, trust_remote_code=True)
        self.device = (
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )
        if torch.cuda.is_available():
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.float()
        self.model.eval()

    def encode(self, text: str):
        tokens = self.tokenizer(text, return_tensors="pt")
        return tokens.input_ids

    def decode(self, token: List[int]):
        text = self.tokenizer.decode(token, skip_special_tokens=True)
        return text

    def generate(self, tokens, max_length=30):
        inputs = tokens.to(self.device)
        outputs = self.model.generate(inputs, max_length=max_length).tolist()
        return outputs


llm = LLM(MODEL, TOKENIZER)


class Ping:
    async def on_get(self, req: Request, resp: Response):
        resp.content_type = falcon.MEDIA_TEXT
        resp.text = "Modelz LLM service"


class ChatCompletions:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def on_post(self, req: Request, resp: Response):
        buf = await req.stream.readall()
        try:
            chat_req = ChatCompletionRequest.from_bytes(buf=buf)
        except msgspec.ValidationError as err:
            resp.status = falcon.HTTP_422
            resp.media = {"error": err}
            return

        tokens = llm.encode(chat_req.get_prompt(self.model_name))
        input_length = len(tokens[0])
        outputs = llm.generate(tokens=tokens)[0]
        res = outputs[input_length:]
        msg = llm.decode(res)
        completion = CompletionResponse(
            id=self.model_name,
            object="chat",
            created=datetime.now(),
            choices=[
                ChatChoice(message=ChatMessage(content=msg, role=Role.ASSISTANT)),
            ],
            usage=TokenUsage(
                prompt_tokens=input_length,
                completion_tokens=len(res),
                total_tokens=input_length + len(res),
            ),
        )
        resp.data = completion.to_json()


class Completions:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def on_post(self, req: Request, resp: Response):
        buf = await req.stream.readall()
        try:
            prompt_req = PromptCompletionRequest.from_bytes(buf=buf)
        except msgspec.ValidationError as err:
            resp.status = falcon.HTTP_422
            resp.media = {"error": err}
            return

        tokens = llm.encode(prompt_req.prompt)
        input_length = len(tokens[0])
        outputs = llm.generate(tokens=tokens)[0]
        msg = llm.decode(outputs)
        completion = CompletionResponse(
            id=self.model_name,
            object="chat",
            created=datetime.now(),
            choices=[
                ChatChoice(
                    message=ChatMessage(content=msg, role=Role.ASSISTANT),
                    finish_reason="length",
                ),
            ],
            usage=TokenUsage(
                prompt_tokens=input_length,
                completion_tokens=len(outputs),
                total_tokens=input_length + len(outputs),
            ),
        )
        resp.data = completion.to_json()


app = App()
app.add_route("/", Ping())
app.add_route("/completions", Completions(model_name=MODEL))
app.add_route("/chat/completions", ChatCompletions(model_name=MODEL))
# refer to https://platform.openai.com/docs/api-reference/chat
# make it fully compatible with the current OpenAI API endpoints
app.add_route("/v1/completions", Completions(model_name=MODEL))
app.add_route("/v1/chat/completions", ChatCompletions(model_name=MODEL))
