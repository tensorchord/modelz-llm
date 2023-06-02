import argparse
import base64
import gc
import logging
from typing import Iterable, List, Union

import falcon
import msgspec
import numpy as np
import torch  # type: ignore
import torch.nn.functional as F
import transformers
from falcon.asgi import App, Request, Response
from huggingface_hub import snapshot_download
from llmspec import (
    ChatCompletionRequest,
    ChatResponse,
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

from modelz_llm.utils import (
    MIN_TEMPERATURE,
    MIN_TOP_P,
    partial_stop,
    prepare_logits_processor,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(process)d - %(levelname)s - %(filename)s:%(lineno)s - %(message)s"
)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


CONTEXT_LEN = 2048


class LLM:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.model_spec = LanguageModels.find(model_name).value
        tokenizer_cls = getattr(transformers, self.model_spec.tokenizer_cls)

        self.tokenizer = tokenizer_cls.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=self.model_spec.low_cpu_mem_usage,
        )
        model_cls = getattr(transformers, self.model_spec.transformer_model_cls)
        self.model = model_cls.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=self.model_spec.low_cpu_mem_usage,
        )
        if device == "auto":
            self.device = (
                torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()

    def __str__(self) -> str:
        return f"LLM(model={self.model}, tokenizer={self.tokenizer})"

    def token_encode(self, text: str):
        """Encode with tokenizer."""
        tokens = self.tokenizer(text, return_tensors="pt")
        return tokens.input_ids

    def token_decode(self, token: List[int]):
        """Decode with tokenizer."""
        text = self.tokenizer.decode(token, skip_special_tokens=True)
        return text

    def generate(self, tokens, **kwargs):
        """Generate content with model."""
        inputs = tokens.to(self.device)
        outputs = self.model.generate(inputs, **kwargs).tolist()
        return outputs

    def step_generate(self, req: ChatCompletionRequest, echo=False, stream_interval=1):
        """Ref to FastChat.

        https://github.com/lm-sys/FastChat/blob/8e38141ff5dd15f3138ccfd312dd73a471e986a1/fastchat/serve/inference.py#L58
        """
        prompt = req.get_prompt(self.model_name)
        input_ids = self.token_encode(prompt)
        input_length = len(input_ids[0])
        logits_processor = prepare_logits_processor(
            req.temperature,
            req.repetition_penalty,
            req.top_p,
            req.top_k,
        )

        if not req.stop:
            stop_token_ids = []
        elif isinstance(req.stop, list):
            stop_token_ids = req.stop
        else:
            stop_token_ids = self.token_encode(req.stop).tolist()[0]
        stop_token_ids.append(self.tokenizer.eos_token_id)
        stop_token = req.stop or self.tokenizer.eos_token

        output_ids = input_ids.tolist()[0]

        # encoding
        if self.model_spec.is_encoder_decoder:
            max_src_len = CONTEXT_LEN
            input_ids = input_ids[-max_src_len:]
            encoder_output = self.model.encoder(
                input_ids=torch.as_tensor(input_ids, device=self.device)
            )[0]
            start_ids = torch.as_tensor(
                [[self.model.generation_config.decoder_start_token_id]],
                dtype=torch.int64,
                device=self.device,
            )
        else:
            max_src_len = CONTEXT_LEN - req.max_tokens - 8

        past_key_values = out = token = None
        for i in range(req.max_tokens):
            if i == 0:
                if self.model_spec.is_encoder_decoder:
                    out = self.model.decoder(
                        input_ids=start_ids,
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                    )
                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(
                        torch.as_tensor(input_ids, device=self.device), use_cache=True
                    )
                    logits = out.logits
                past_key_values = out.past_key_values
            else:
                if self.model_spec.is_encoder_decoder:
                    out = self.model.decoder(
                        input_ids=torch.as_tensor([[token]], device=self.device),
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )

                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(
                        input_ids=torch.as_tensor([[token]], device=self.device),
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    logits = out.logits
                past_key_values = out.past_key_values

            if req.repetition_penalty > 1:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]

            if req.temperature < MIN_TEMPERATURE or req.top_p < MIN_TOP_P:  # greedy
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)
            stopped = token in stop_token_ids

            if i % stream_interval == 0 or i == req.max_tokens - 1 or stopped:
                if echo:
                    tmp_output_ids = output_ids
                    rfind_start = input_length
                else:
                    tmp_output_ids = output_ids[input_length:]
                    rfind_start = 0

                output = self.tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )

                partially_stopped = False
                if stop_token:
                    if isinstance(stop_token, str):
                        pos = output.rfind(stop_token, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                        else:
                            partially_stopped = partial_stop(output, stop_token)
                    elif isinstance(stop_token, Iterable):
                        for each_stop in stop_token:
                            pos = output.rfind(each_stop, rfind_start)
                            if pos != -1:
                                output = output[:pos]
                                stopped = True
                                break
                            else:
                                partially_stopped = partial_stop(output, each_stop)
                                if partially_stopped:
                                    break
                    else:
                        raise ValueError("Invalid stop field type.")

                # prevent yielding partial stop sequence
                if not partially_stopped:
                    pass
                    # yield ChatResponse.from_message(
                    #     output,
                    #     Role.ASSISTANT,
                    #     self.model_name,
                    #     None,
                    #     input_length,
                    #     i,
                    # )

            if stopped:
                break

        # finish stream event, which contains finish reason
        if i == req.max_tokens - 1:
            finish_reason = "length"
        elif stopped:
            finish_reason = "stop"
        else:
            finish_reason = None

        yield ChatResponse.from_message(
            output,
            Role.ASSISTANT,
            self.model_name,
            finish_reason,
            input_length,
            i,
        )

        # clean
        del past_key_values, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Ping:
    async def on_get(self, req: Request, resp: Response):
        resp.content_type = falcon.MEDIA_TEXT
        resp.text = "Modelz LLM service"


class ChatCompletions:
    def __init__(self, model: LLM) -> None:
        self.model = model
        self.model_name = model.model_name

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

        for comp in self.model.step_generate(chat_req):
            logger.info(comp)
            resp.data = comp.to_json()

        # tokens = self.model.token_encode(chat_req.get_prompt(self.model_name))
        # input_length = len(tokens[0])
        # outputs = self.model.generate(tokens=tokens)[0]
        # res = outputs[input_length:]
        # msg = self.model.token_decode(res)
        # completion = ChatResponse.from_message(
        #     msg, Role.ASSISTANT, self.model_name, None, input_length, len(res)
        # )
        # resp.data = completion.to_json()


class Completions:
    def __init__(self, model: LLM) -> None:
        self.model = model
        self.model_name = model.model_name

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

        tokens = self.model.token_encode(prompt_req.get_prompt())
        input_length = len(tokens[0])
        outputs = self.model.generate(tokens=tokens)[0]
        msg = self.model.token_decode(outputs)
        completion = CompletionResponse.from_message(
            msg, self.model_name, None, input_length, len(outputs)
        )
        resp.data = completion.to_json()


class Embeddings:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModel.from_pretrained(model_name)
        if device == "auto":
            self.device = (
                torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model = self.model.to(self.device)
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
        embeddings = embeddings.detach()
        if self.device != "cpu":
            embeddings = embeddings.cpu()
        embeddings = embeddings.numpy()
        if embedding_req.encoding_format == "base64":
            embeddings = [
                base64.b64encode(emb.astype(np.float32).tobytes()).decode("utf-8")
                for emb in embeddings
            ]
        else:
            embeddings = [emb.tolist() for emb in embeddings]

        embedding_resp = EmbeddingResponse(
            data=[
                EmbeddingData(embedding=emb, index=i)
                for i, emb in enumerate(embeddings)
            ],
            model=self.model_name,
            usage=TokenUsage(
                prompt_tokens=token_count,
                # No completions performed, only embeddings generated.
                completion_tokens=0,
                total_tokens=token_count,
            ),
        )
        resp.data = embedding_resp.to_json()


def build_falcon_app(args: argparse.Namespace):
    if args.dry_run:
        snapshot_download(repo_id=args.model)
        return
    llm = LLM(args.model, args.device)
    embeddings = Embeddings(args.emb_model, args.device)
    completion = Completions(llm)
    chat_completion = ChatCompletions(llm)
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
