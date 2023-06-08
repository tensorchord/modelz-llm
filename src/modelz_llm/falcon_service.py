import argparse
import multiprocessing as mp

import falcon
import msgspec
from falcon.asgi import App, Request, Response
from huggingface_hub import snapshot_download
from llmspec import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ErrorResponse,
    PromptCompletionRequest,
)

from modelz_llm.emb import Emb
from modelz_llm.log import logger
from modelz_llm.model import LLM
from modelz_llm.uds import Client, run_server

UNIX_DOMAIN_SOCKET_PATH = "/tmp/modelz_llm/{}.sock"


class Ping:
    async def on_get(self, req: Request, resp: Response):
        resp.content_type = falcon.MEDIA_TEXT
        resp.text = "Modelz LLM service"


class ChatCompletions:
    def __init__(self, client: Client) -> None:
        self.client = client

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

        comp = await self.client.request(chat_req)
        logger.info(comp)
        if isinstance(comp, Exception):
            resp.status = falcon.HTTP_500
            resp.data = ErrorResponse.from_validation_err(
                comp, "internal error"
            ).to_json()
            return
        resp.data = comp.to_json()


class Completions:
    def __init__(self, client: Client) -> None:
        self.client = client

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

        comp = await self.client.request(prompt_req)
        logger.info(comp)
        if isinstance(comp, Exception):
            resp.status = falcon.HTTP_500
            resp.data = ErrorResponse.from_validation_err(
                comp, "internal error"
            ).to_json()
            return
        resp.data = comp.to_json()


class Embeddings:
    def __init__(self, client: Client) -> None:
        self.client = client

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

        emb = await self.client.request(embedding_req)
        if isinstance(emb, Exception):
            resp.status = falcon.HTTP_500
            resp.data = ErrorResponse.from_validation_err(
                emb, "internal error"
            ).to_json()
            return
        resp.data = emb.to_json()


def build_falcon_app(args: argparse.Namespace):
    if args.dry_run:
        snapshot_download(repo_id=args.model)
        snapshot_download(repo_id=args.emb_model)
        return

    llm_uds_path = UNIX_DOMAIN_SOCKET_PATH.format("llm")
    emb_uds_path = UNIX_DOMAIN_SOCKET_PATH.format("emb")
    barrier = mp.get_context("spawn").Barrier(3)
    run_server(llm_uds_path, barrier, LLM, model_name=args.model, device=args.device)
    run_server(
        emb_uds_path, barrier, Emb, model_name=args.emb_model, device=args.device
    )
    barrier.wait()

    llm_client = Client(llm_uds_path)
    completion = Completions(llm_client)
    chat_completion = ChatCompletions(llm_client)
    emb_client = Client(emb_uds_path)
    embeddings = Embeddings(emb_client)

    app = App()
    app.add_route("/", Ping())
    app.add_route("/completions", completion)
    app.add_route("/chat/completions", chat_completion)
    app.add_route("/embeddings", embeddings)
    app.add_route("/engines/{engine}/embeddings", embeddings)

    # refer to https://platform.openai.com/docs/api-reference/chat
    # make it fully compatible with the current OpenAI API endpoints
    app.add_route("/v1/completions", completion)
    app.add_route("/v1/chat/completions", chat_completion)
    app.add_route("/v1/embeddings", embeddings)
    app.add_route("/v1/engines/{engine}/embeddings", embeddings)
    return app
