from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class RecursiveChunk:
    text: str
    source: str
    start: int
    end: int


class RecursiveChunker:
    def __init__(self, chunk_size: int = 300, overlap: int = 50, separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators

    def chunk(self, text: str, source: str = "document") -> List[RecursiveChunk]:
        if not text.strip():
            return []

        normalized = re.sub(r"\s+", " ", text).strip()
        chunks: List[RecursiveChunk] = []
        start = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            candidate = normalized[start:end]
            if end < len(normalized):
                split_index = self._find_split(candidate)
                if split_index is not None and split_index > 0:
                    end = start + split_index
                    candidate = normalized[start:end]
            chunks.append(RecursiveChunk(text=candidate.strip(), source=source, start=start, end=end))
            if end >= len(normalized):
                break
            start = max(start + self.chunk_size - self.overlap, start + 1)
        return [chunk for chunk in chunks if chunk.text]

    def _find_split(self, text: str) -> int | None:
        for separator in self.separators:
            indexes = [match.start() for match in re.finditer(re.escape(separator), text)]
            if not indexes:
                continue
            return max(1, max(indexes))
        return None
