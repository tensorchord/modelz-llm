import os
from datetime import datetime
from typing import Any, Dict, List

import torch  # type: ignore
from llmspec import (
    ChatChoice,
    ChatCompletionRequest,
    ChatMessage,
    CompletionResponse,
    Role,
    TokenUsage,
)
from mosec import Server, Worker, get_logger
from transformers import AutoModel, AutoTokenizer

logger = get_logger()
TOKENIZER = os.environ.get("MODELZ_TOKENIZER", "THUDM/chatglm-6b-int4")
MODEL = os.environ.get("MODELZ_MODEL", "THUDM/chatglm-6b-int4")


class Tokenizer(Worker):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER, trust_remote_code=True
        )

    def deserialize(self, buf: bytes) -> ChatCompletionRequest:
        return ChatCompletionRequest.from_bytes(buf)

    def forward(self, req: ChatCompletionRequest):
        prompt = req.get_prompt(MODEL)
        tokens = self.tokenizer(prompt, return_tensors="pt")
        # TODO: ignore the inference configurations for now
        return tokens.input_ids[0]


class Inference(Worker):
    def __init__(self):
        self.model = AutoModel.from_pretrained(MODEL, trust_remote_code=True)
        self.device = (
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )
        if torch.cuda.is_available():
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.float()
        self.model.eval()

    def forward(self, tokens: List[Any]) -> Dict:
        inputs = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True).to(
            self.device
        )
        outputs = self.model.generate(inputs, max_length=30).tolist()
        return list(zip(tokens, outputs))


class Decoder(Worker):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER, trust_remote_code=True
        )

    def forward(self, data: Any):
        token, output = data
        res = output[len(token) :]
        msg = self.tokenizer.decode(res, skip_special_tokens=True)
        resp = CompletionResponse(
            id=MODEL,
            object="chat",
            created=datetime.now(),
            choices=[
                ChatChoice(message=ChatMessage(content=msg, role=Role.ASSISTANT)),
            ],
            usage=TokenUsage(
                prompt_tokens=len(token),
                completion_tokens=len(res),
                total_tokens=len(token) + len(res),
            ),
        )
        return resp

    def serialize(self, data: CompletionResponse):
        return data.to_json()


if __name__ == "__main__":
    server = Server()
    server.append_worker(Tokenizer, num=1, timeout=10)
    server.append_worker(Inference, num=1, max_batch_size=4, timeout=40)
    server.append_worker(Decoder, num=1, timeout=10)
    server.run()
