from __future__ import annotations

from typing import List


class SimpleEmbedder:
    """Minimal deterministic embedding implementation for offline testing."""

    def encode(self, text: str) -> List[float]:
        tokens = text.lower().split()
        vector = [0.0] * max(1, len(tokens))
        for i, token in enumerate(tokens):
            vector[i] = float(sum(ord(ch) for ch in token) % 97 + 1)
        return vector

    def encode_many(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(text) for text in texts]
