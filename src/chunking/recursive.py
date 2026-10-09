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

        normalized = self._normalize(text)
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
            start = self._next_start(normalized, start, end)
        return [chunk for chunk in chunks if chunk.text]

    @staticmethod
    def _normalize(text: str) -> str:
        """Collapse spaces but keep line and paragraph breaks, so the newline
        separators can still match."""
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def _next_start(self, text: str, start: int, end: int) -> int:
        # Step back from the actual end, not from start + chunk_size: after a
        # separator split the chunk is shorter, and the old rule skipped text.
        candidate = end - self.overlap
        if candidate <= start:
            return end
        # Move forward to the next word boundary so the overlap never starts mid-word.
        boundary = re.search(r"\s", text[candidate:end])
        return candidate + boundary.end() if boundary else end

    def _find_split(self, text: str) -> int | None:
        for separator in self.separators:
            indexes = [match.start() for match in re.finditer(re.escape(separator), text)]
            if not indexes:
                continue
            # Split after the separator so a chunk keeps its closing punctuation.
            return max(indexes) + len(separator)
        return None
