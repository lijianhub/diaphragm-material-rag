from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    source: str
    start: int
    end: int


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 200, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str = "document") -> List[Chunk]:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if len(text) <= self.chunk_size:
            return [Chunk(text=text, source=source, start=0, end=len(text))]

        chunks: List[Chunk] = []
        step = self.chunk_size - self.overlap
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(Chunk(text=chunk_text, source=source, start=start, end=end))
            if end >= len(text):
                break
            start += step
        return chunks
