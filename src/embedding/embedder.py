from __future__ import annotations

import math
import zlib
from typing import List

from tokenization import tokenize


class SimpleEmbedder:
    """Deterministic offline embedder based on feature hashing.

    Each token is hashed into one of ``dim`` buckets and the counts are
    L2-normalised, so the dot product of two vectors is their cosine
    similarity. It captures word overlap, not meaning: synonyms do not match.
    Swap in a model-based embedder with the same interface for semantic search.
    """

    def __init__(self, dim: int = 512):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def encode(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            # crc32 is stable across runs, unlike Python's salted hash().
            vector[zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def encode_many(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(text) for text in texts]
