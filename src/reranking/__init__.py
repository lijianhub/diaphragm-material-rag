"""Second-stage rerankers: score (query, passage) pairs jointly and reorder candidates.

A first-stage retriever (BM25 + bi-encoder) is fast but encodes query and
passage separately. A cross-encoder reads them together, so it judges whether a
passage actually answers the question, but it is too slow to run over a whole
corpus, so it reorders only the top ``rerank_depth`` candidates.

``create_reranker(name)``:

    ce:<hugging-face cross-encoder>          local sentence-transformers CrossEncoder
    bedrock-rerank:<model-id>                Amazon Bedrock Rerank API (e.g. amazon.rerank-v1:0)
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", model: Any = None, batch_size: int = 32):
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - needs the optional extra
                raise ImportError("CrossEncoderReranker needs: pip install -e .[local-embeddings]") from exc
            model = CrossEncoder(model_name)
        self.model_name = model_name
        self.model = model
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return f"ce:{self.model_name}"

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, passage) for passage in passages], batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


class BedrockReranker:
    """Amazon Bedrock ``Rerank`` (bedrock-agent-runtime) with documents passed inline.

    Results come back sorted by relevance; scores are mapped back to input order.
    """

    def __init__(self, model_id: str = "amazon.rerank-v1:0", region: Optional[str] = None, client: Any = None):
        if client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - needs the optional extra
                raise ImportError("BedrockReranker needs: pip install -e .[aws]") from exc
            client = boto3.client("bedrock-agent-runtime", region_name=region)
            region = region or client.meta.region_name
        if not region:
            raise ValueError("BedrockReranker needs an AWS region to build the model ARN")
        self.model_id = model_id
        self.model_arn = f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
        self.client = client

    @property
    def name(self) -> str:
        return f"bedrock-rerank:{self.model_id}"

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        response = self.client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=[
                {"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": passage}}}
                for passage in passages
            ],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": len(passages),
                    "modelConfiguration": {"modelArn": self.model_arn},
                },
            },
        )
        scores = [float("-inf")] * len(passages)
        for result in response["results"]:
            scores[result["index"]] = float(result["relevanceScore"])
        return scores


def create_reranker(name: str, **kwargs: Any):
    if name.startswith("ce:"):
        return CrossEncoderReranker(name[3:], **kwargs)
    if name.startswith("bedrock-rerank:"):
        return BedrockReranker(name[len("bedrock-rerank:") :], **kwargs)
    raise ValueError(f"unknown reranker {name!r}; expected ce:<model> or bedrock-rerank:<model>")


__all__ = ["BedrockReranker", "CrossEncoderReranker", "create_reranker"]
