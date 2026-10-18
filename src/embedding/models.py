"""Model-based embedders. Both import their SDK lazily, so the core install stays light.

Every embedder exposes the same small interface used by ``SearchIndex``:
``name`` (identifies the vector space), ``dim``, ``encode``, ``encode_many`` and
``encode_query``. ``name`` doubles as the spec ``create_embedder`` understands,
so an index can rebuild its embedder from the manifest.
"""
from __future__ import annotations

import json
import time
from typing import Any, List, Optional, Sequence


class SentenceTransformerEmbedder:
    """Local transformer model via the ``sentence-transformers`` library.

    The model is downloaded from the Hugging Face Hub on first use and cached.
    Vectors are L2-normalised by the library. Asymmetric models (e5, bge) expect
    different prefixes for queries and passages; set them here.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model: Any = None,
        batch_size: int = 32,
        query_prefix: str = "",
        passage_prefix: str = "",
        device: Optional[str] = None,
    ):
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - needs the optional extra
                raise ImportError("SentenceTransformerEmbedder needs: pip install -e .[local-embeddings]") from exc
            model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.model = model
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        # sentence-transformers 6 renamed get_sentence_embedding_dimension; support both.
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    @property
    def name(self) -> str:
        return f"st:{self.model_name}"

    def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(x) for x in vector] for vector in vectors]

    def encode_many(self, texts: List[str]) -> List[List[float]]:
        return self._encode([self.passage_prefix + text for text in texts])

    def encode(self, text: str) -> List[float]:
        return self.encode_many([text])[0]

    def encode_query(self, text: str) -> List[float]:
        return self._encode([self.query_prefix + text])[0]


TITAN_V2_DIMENSIONS = (256, 512, 1024)
_RETRYABLE = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException", "InternalServerException"}


class BedrockEmbedder:
    """Amazon Titan Text Embeddings V2 on Amazon Bedrock.

    One ``InvokeModel`` call per text (Titan V2 has no batch endpoint), with
    exponential backoff on throttling. Credentials and region come from the
    standard AWS chain (environment, profile, or the ECS/Lambda task role).
    """

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = 1024,
        region: Optional[str] = None,
        client: Any = None,
        max_retries: int = 5,
        backoff_seconds: float = 0.5,
    ):
        if "titan-embed-text-v2" in model_id and dimensions not in TITAN_V2_DIMENSIONS:
            raise ValueError(f"Titan Text Embeddings V2 supports dimensions {TITAN_V2_DIMENSIONS}, not {dimensions}")
        if client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - needs the optional extra
                raise ImportError("BedrockEmbedder needs: pip install -e .[aws]") from exc
            client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.dim = dimensions
        self.client = client
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    @property
    def name(self) -> str:
        return f"bedrock:{self.model_id}@{self.dim}"

    def _invoke(self, text: str) -> List[float]:
        body = json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True})
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id, body=body, contentType="application/json", accept="application/json"
                )
                return [float(x) for x in json.loads(response["body"].read())["embedding"]]
            except Exception as exc:
                code = getattr(exc, "response", {}).get("Error", {}).get("Code")
                if code not in _RETRYABLE or attempt == self.max_retries:
                    raise
                time.sleep(self.backoff_seconds * (2**attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    def encode(self, text: str) -> List[float]:
        return self._invoke(text)

    def encode_many(self, texts: List[str]) -> List[List[float]]:
        return [self._invoke(text) for text in texts]

    def encode_query(self, text: str) -> List[float]:
        return self._invoke(text)
