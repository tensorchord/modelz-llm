import gc
from typing import Iterable, List, Union

import torch
import transformers
from llmspec import (
    ChatCompletionRequest,
    ChatResponse,
    CompletionResponse,
    LanguageModels,
    PromptCompletionRequest,
)

from modelz_llm.log import logger
from modelz_llm.utils import (
    MIN_TEMPERATURE,
    MIN_TOP_P,
    partial_stop,
    prepare_logits_processor,
)

CONTEXT_LEN = 2048


class LLM:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.model_spec = LanguageModels.find(model_name).value
        tokenizer_cls = getattr(transformers, self.model_spec.tokenizer_cls)
        model_cls = getattr(transformers, self.model_spec.transformer_model_cls)

        logger.info(
            "loading model and embedding: %s(%s) %s(%s)",
            model_cls,
            model_name,
            tokenizer_cls,
            model_name,
        )
        self.tokenizer = tokenizer_cls.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=self.model_spec.low_cpu_mem_usage,
        )
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

        try:
            if self.device == "cpu":
                self.model = self.model.float()
            else:
                self.model = self.model.half()
        except Exception as err:
            logger.debug("failed to convert the model: %s", err)

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

    def get_prompt_from_req(
        self, req: Union[ChatCompletionRequest, PromptCompletionRequest]
    ):
        if isinstance(req, ChatCompletionRequest):
            return req.get_prompt(self.model_name)
        return req.get_prompt()

    def __call__(
        self, req: Union[ChatResponse, CompletionResponse]
    ) -> Union[ChatResponse, ChatCompletionRequest]:
        """Generate chat or completion response."""
        resp_cls = (
            ChatResponse
            if isinstance(req, ChatCompletionRequest)
            else CompletionResponse
        )
        if self.model_spec is not LanguageModels.CHAT_GLM.value:
            return list(self.step_generate(req, resp_cls=resp_cls))[0]

        tokens = self.token_encode(self.get_prompt_from_req(req))
        input_length = len(tokens[0])
        outputs = self.generate(tokens, **req.get_inference_args(self.model_name))[0]
        message = self.token_decode(outputs[input_length:])
        return resp_cls.from_message(
            message=message,
            model=self.model_name,
            finish_reason=None,
            prompt_token=input_length,
            completion_token=len(outputs) - input_length,
        )

    def step_generate(
        self,
        req: ChatCompletionRequest,
        resp_cls: Union[ChatResponse, CompletionResponse],
        echo=False,
        stream_interval=1,
    ):
        """Ref to FastChat.

        https://github.com/lm-sys/FastChat/blob/8e38141ff5dd15f3138ccfd312dd73a471e986a1/fastchat/serve/inference.py#L58
        """
        prompt = self.get_prompt_from_req(req)
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

        is_encoder_decoder = getattr(self.model, "encoder", None) and getattr(
            self.model, "decoder", None
        )
        # encoding
        if is_encoder_decoder:
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
                if is_encoder_decoder:
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
                if is_encoder_decoder:
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

            if stopped:
                break

        # finish stream event, which contains finish reason
        if i == req.max_tokens - 1:
            finish_reason = "length"
        elif stopped:
            finish_reason = "stop"
        else:
            finish_reason = None

        yield resp_cls.from_message(
            message=output,
            model=self.model_name,
            finish_reason=finish_reason,
            prompt_token=input_length,
            completion_token=i,
        )

        # clean
        del past_key_values, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
