import logging
import os
from datetime import datetime
from typing import List, Union

import falcon
import msgspec
import torch  # type: ignore
import torch.nn.functional as F
import transformers
from falcon.asgi import App, Request, Response
from llmspec import (
    ChatChoice,
    ChatCompletionRequest,
    ChatMessage,
    CompletionChoice,
    CompletionResponse,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    LanguageModels,
    PromptCompletionRequest,
    Role,
    TokenUsage,
)

DEFAULT_MODEL = "bigscience/bloomz-560m"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER = os.environ.get("MODELZ_TOKENIZER", DEFAULT_MODEL)
MODEL = os.environ.get("MODELZ_MODEL", DEFAULT_MODEL)
EMBEDDING_MODEL = os.environ.get("MODELZ_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(process)d - %(levelname)s - %(filename)s:%(lineno)s - %(message)s"
)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


class LLM:
    def __init__(self, model_name: str, tokenizer_name: str) -> None:
        # Use the same tokenizer as the model
        if tokenizer_name is None:
            tokenizer_name = model_name

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
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.device = (
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )
        if torch.cuda.is_available():
            self.model = self.model.half().to(self.device)
        self.model.eval()

    # copied from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#usage-huggingface-transformers
    def get_embedding_with_token_count(self, sentences: Union[str, List[str]]):
        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            # First element of model_output contains all token embeddings
            token_embeddings = model_output[0]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = encoded_input.to(self.device)
        token_count = inputs["attention_mask"].sum(dim=1).tolist()[0]

        # Compute token embeddings
        model_output = self.model(**inputs)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, inputs["attention_mask"])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return token_count, sentence_embeddings

    async def on_post(self, req: Request, resp: Response, engine: str = ""):
        if engine:
            logger.info("received emb req with engine: %s", engine)

        buf = await req.stream.readall()
        try:
            embedding_req = EmbeddingRequest.from_bytes(buf=buf)
        except msgspec.ValidationError as err:
            logger.info(f"Failed to parse request: {err}")
            resp.status = falcon.HTTP_400
            resp.data = ErrorResponse.from_validation_err(err, str(buf)).to_json()
            return

        token_count, embeddings = self.get_embedding_with_token_count(
            embedding_req.input
        )
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
                prompt_tokens=token_count,
                # No completions performed, only embeddings generated.
                completion_tokens=0,
                total_tokens=token_count,
            ),
        )
        resp.data = embedding_resp.to_json()


embeddings = Embeddings(EMBEDDING_MODEL)

app = App()
app.add_route("/", Ping())
app.add_route("/completions", Completions(model_name=MODEL))
app.add_route("/chat/completions", ChatCompletions(model_name=MODEL))
app.add_route("/embeddings", embeddings)
app.add_route("/engines/{engine}/embeddings", embeddings)
# refer to https://platform.openai.com/docs/api-reference/chat
# make it fully compatible with the current OpenAI API endpoints
app.add_route("/v1/completions", Completions(model_name=MODEL))
app.add_route("/v1/chat/completions", ChatCompletions(model_name=MODEL))
app.add_route("/v1/embeddings", embeddings)
app.add_route("/v1/engines/{engine}/embeddings", embeddings)


if __name__ == "__main__":
    print(llm)
