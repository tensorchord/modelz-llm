import logging
import os
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
    CompletionChoice,
    CompletionResponse,
    PromptCompletionRequest,
    Role,
    TokenUsage,
    LanguageModels,
    ErrorResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)


# todo: make this importable from top level
# from llmspec.llmspec import EmbeddingData
# temporary fix: embedding attr type
class EmbeddingData(msgspec.Struct):
    embedding: List[float] | List[List[float]]
    index: int
    object: str = "embedding"


import transformers
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "THUDM/chatglm-6b-int4"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER = os.environ.get("MODELZ_TOKENIZER", DEFAULT_MODEL)
MODEL = os.environ.get("MODELZ_MODEL", DEFAULT_MODEL)
EMBEDDING_MODEL = os.environ.get("MODELZ_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(process)d - %(levelname)s - %(filename)s:%(lineno)s - %(message)s"
)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


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

    def __str__(self) -> str:
        return f"LLM(model={self.model}, tokenizer={self.tokenizer})"

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
            logger.info(f"Failed to parse request: {err}")
            # return 400 otherwise the client will retry
            resp.status = falcon.HTTP_400
            resp.data = ErrorResponse.from_validation_err(err, str(buf)).to_json()
            return

        tokens = llm.encode(chat_req.get_prompt(self.model_name))
        input_length = len(tokens[0])
        outputs = llm.generate(tokens=tokens)[0]
        res = outputs[input_length:]
        msg = llm.decode(res)
        completion = CompletionResponse(
            id=self.model_name,
            object="chat",
            model=self.model_name,
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
            logger.info(f"Failed to parse request: {err}")
            # return 400 otherwise the client will retry
            resp.status = falcon.HTTP_400
            resp.data = ErrorResponse.from_validation_err(err, str(buf)).to_json()
            return

        tokens = llm.encode(prompt_req.get_prompt())
        input_length = len(tokens[0])
        outputs = llm.generate(tokens=tokens)[0]
        msg = llm.decode(outputs)
        completion = CompletionResponse(
            id=self.model_name,
            object="chat",
            model=self.model_name,
            created=datetime.now(),
            choices=[
                CompletionChoice(
                    text=msg,
                ),
            ],
            usage=TokenUsage(
                prompt_tokens=input_length,
                completion_tokens=len(outputs),
                total_tokens=input_length + len(outputs),
            ),
        )
        resp.data = completion.to_json()


class Embeddings:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def on_post(self, req: Request, resp: Response):
        buf = await req.stream.readall()
        try:
            # todo: llmspec hasn't implemented from_bytes for EmbeddingRequest
            embedding_req = msgspec.json.decode(buf, type=EmbeddingRequest)
        except msgspec.ValidationError as err:
            logger.info(f"Failed to parse request: {err}")
            resp.status = falcon.HTTP_400
            resp.status = falcon.HTTP_400
            resp.data = ErrorResponse.from_validation_err(err, str(buf)).to_json()
            return

        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(embedding_req.input)
        # convert embeddings of type list[Tensor] | ndarray to list[float]
        if isinstance(embeddings, list):
            embeddings = [e.tolist() for e in embeddings]
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.tolist()
        else:
            embeddings = embeddings.tolist()

        embedding_data = EmbeddingData(embedding=embeddings, index=0)
        embedding_resp = EmbeddingResponse(
            data=embedding_data,
            model=self.model_name,
            usage=TokenUsage(
                prompt_tokens=0,  # No prompt tokens, only embeddings generated.
                completion_tokens=0,  # No completions performed, only embeddings generated.
                total_tokens=len(embeddings),
            ),
        )
        # todo: llmspec hasn't implemented to_json for EmbeddingResponse
        resp.data = msgspec.json.encode(embedding_resp)


class EmbeddingsEngineRouteWrapper(Embeddings):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)

    async def on_post(self, req: Request, resp: Response, engine: str):
        await super().on_post(req, resp)


app = App()
app.add_route("/", Ping())
app.add_route("/completions", Completions(model_name=MODEL))
app.add_route("/chat/completions", ChatCompletions(model_name=MODEL))
app.add_route("/embeddings", Embeddings(model_name=EMBEDDING_MODEL))
app.add_route(
    "/engines/{engine}/embeddings".format(EMBEDDING_MODEL),
    EmbeddingsEngineRouteWrapper(model_name=EMBEDDING_MODEL),
)
# refer to https://platform.openai.com/docs/api-reference/chat
# make it fully compatible with the current OpenAI API endpoints
app.add_route("/v1/completions", Completions(model_name=MODEL))
app.add_route("/v1/chat/completions", ChatCompletions(model_name=MODEL))
app.add_route("/v1/embeddings", Embeddings(model_name=EMBEDDING_MODEL))
app.add_route(
    "/v1/engines/{engine}/embeddings".format(EMBEDDING_MODEL),
    EmbeddingsEngineRouteWrapper(model_name=EMBEDDING_MODEL),
)


if __name__ == "__main__":
    print(llm)
