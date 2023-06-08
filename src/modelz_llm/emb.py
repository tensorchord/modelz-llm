import base64
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from llmspec import EmbeddingData, EmbeddingRequest, EmbeddingResponse, TokenUsage


class Emb:
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

    def __call__(self, req: EmbeddingRequest) -> EmbeddingResponse:
        token_count, embeddings = self.get_embedding_with_token_count(req.input)
        embeddings = embeddings.detach()
        if self.device != "cpu":
            embeddings = embeddings.cpu()
        embeddings = embeddings.numpy()
        if req.encoding_format == "base64":
            embeddings = [
                base64.b64encode(emb.astype(np.float32).tobytes()).decode("utf-8")
                for emb in embeddings
            ]
        else:
            embeddings = [emb.tolist() for emb in embeddings]

        resp = EmbeddingResponse(
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
        return resp
