import argparse
import logging
from datetime import datetime
from typing import List, Union, Iterable

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

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(process)d - %(levelname)s - %(filename)s:%(lineno)s - %(message)s"
)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


class LLM:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, low_cpu_mem_usage=True
        )
        model_cls = getattr(transformers, LanguageModels.transformer_cls(model_name))
        self.model = model_cls.from_pretrained(
            model_name, trust_remote_code=True, low_cpu_mem_usage=True
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


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if 1e-5 <= temperature < 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


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
        
        temperature: float = 0.7
        repetition_penalty: float = 1.2
        top_p: float = 1.0
        top_k: int = -1
        logits_processor = prepare_logits_processor(
            temperature, repetition_penalty, top_p, top_k
        )

        stop_token_ids = []
        stop_token_ids.append(self.model.tokenizer.eos_token_id)

        input_ids = self.model.tokenizer(chat_req.get_prompt(self.model_name)).input_ids
        input_echo_len = len(input_ids)
        output_ids = list(input_ids)
        context_len = 2048
        max_new_tokens = 512
        is_encoder_decoder = True
        if is_encoder_decoder:
            max_src_len = context_len
        else:
            max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        if is_encoder_decoder:
            encoder_output = self.model.model.encoder(
                input_ids=torch.as_tensor([input_ids], device=self.model.device)
            )[0]
            start_ids = torch.as_tensor(
                [[self.model.model.generation_config.decoder_start_token_id]],
                dtype=torch.int64,
                device=self.model.device,
            )

        past_key_values = out = None
        for i in range(max_new_tokens):
            if i == 0:
                if is_encoder_decoder:
                    out = self.model.model.decoder(
                        input_ids=start_ids,
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                    )
                    logits = self.model.model.lm_head(out[0])
                else:
                    out = self.model.model(torch.as_tensor([input_ids], device=self.model.model.device), use_cache=True)
                    logits = out.logits
                past_key_values = out.past_key_values
            else:
                if is_encoder_decoder:
                    out = self.model.model.decoder(
                        input_ids=torch.as_tensor([[token]], device=self.model.model.device),
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )

                    logits = self.model.model.lm_head(out[0])
                else:
                    out = self.model.model(
                        input_ids=torch.as_tensor([[token]], device=self.model.model.device),
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    logits = out.logits
                past_key_values = out.past_key_values

            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]

            if temperature < 1e-5 or top_p < 1e-8:  # greedy
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)
            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False
        
            stream_interval = 1
            echo = False
            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                if echo:
                    tmp_output_ids = output_ids
                    rfind_start = input_echo_len
                else:
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0

                output = self.model.tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )

                partially_stopped = False
                stop_str = "###"
                if stop_str:
                    if isinstance(stop_str, str):
                        pos = output.rfind(stop_str, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                        else:
                            partially_stopped = partial_stop(output, stop_str)
                    elif isinstance(stop_str, Iterable):
                        for each_stop in stop_str:
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
                    print(output)

            if stopped:
                break

        # finish stream event, which contains finish reason
        if i == max_new_tokens - 1:
            finish_reason = "length"
        elif stopped:
            finish_reason = "stop"
        else:
            finish_reason = None
            

        # tokens = self.model.encode(chat_req.get_prompt(self.model_name))
        # input_length = len(tokens[0])
        # outputs = self.model.generate(tokens=tokens)[0]
        # res = outputs[input_length:]
        # msg = self.model.decode(res)


        completion = CompletionResponse(
            id=self.model_name,
            object="chat",
            model=self.model_name,
            created=datetime.now(),
            choices=[
                ChatChoice(message=ChatMessage(content=output, role=Role.ASSISTANT)),
            ],
            usage=TokenUsage(
                prompt_tokens=input_echo_len,
                completion_tokens=len(output),
                total_tokens=input_echo_len + len(output),
            ),
        )
        resp.data = completion.to_json()


def partial_stop(output, stop_str):
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False

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

        tokens = self.model.encode(prompt_req.get_prompt())
        input_length = len(tokens[0])
        outputs = self.model.generate(tokens=tokens)[0]
        msg = self.model.decode(outputs)
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


def build_falcon_app(args: argparse.Namespace):
    llm = LLM(args.model, args.device)
    embeddings = Embeddings(args.emb_model, args.device)
    completion = Completions(llm)
    chat_completion = ChatCompletions(llm)

    if args.dry_run:
        return

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
