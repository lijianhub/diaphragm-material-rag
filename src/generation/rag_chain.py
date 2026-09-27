from __future__ import annotations

from typing import Any, Dict, List


class RAGPipeline:
    def answer(self, question: str, documents: List[str]) -> Dict[str, Any]:
        relevant = [doc for doc in documents if question.lower() in doc.lower() or any(word in doc.lower() for word in question.lower().split())]
        if not relevant:
            relevant = documents[:1]

        answer = "\n\n".join(relevant)
        return {
            "answer": f"Based on the retrieved context: {answer}",
            "context": relevant,
            "sources": [f"doc_{i}" for i in range(len(relevant))],
        }
