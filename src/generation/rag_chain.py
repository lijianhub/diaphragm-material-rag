from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from tokenization import tokenize

from .prompt import Prompt, build_prompt, check_citations, citation_label


@dataclass
class Citation:
    number: int
    label: str
    source: str
    text: str


@dataclass
class Answer:
    text: str
    citations: List[Citation] = field(default_factory=list)
    sources: list = field(default_factory=list)  # the SearchResults that were offered as context
    used_llm: bool = False
    invalid_citations: List[int] = field(default_factory=list)
    fallback_reason: Optional[str] = None
    prompt: Optional[Prompt] = None
    model: str = "extractive"  # the model that wrote the answer, as served
    prompt_version: str = ""
    usage: Optional[dict] = None  # token counts when an LLM was called


_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


class ExtractiveAnswerer:
    """Offline answering without a language model: quote the sentences from the
    retrieved sources that share the most question terms, each with its citation.

    It never invents text, so it is a safe default and a baseline for LLM answers.
    """

    NOT_FOUND = "The information is not in the retrieved sources."

    def __init__(self, max_sentences: int = 2, min_overlap: float = 0.3):
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def answer(self, question: str, results: Sequence) -> str:
        terms = set(tokenize(question, stem_tokens=True))
        if not terms:
            return self.NOT_FOUND
        candidates = []
        for number, result in enumerate(results, start=1):
            for sentence in _SENTENCE.split(result.text.strip()):
                sentence = sentence.strip().lstrip("#").strip()
                overlap = len(terms & set(tokenize(sentence, stem_tokens=True))) / len(terms)
                if sentence and overlap >= self.min_overlap:
                    # Higher overlap first; among equals, the better-ranked source.
                    candidates.append((-overlap, number, sentence))
        if not candidates:
            return self.NOT_FOUND
        chosen = sorted(candidates)[: self.max_sentences]
        return " ".join(f"{sentence} [{number}]" for _, number, sentence in chosen)


class RAGPipeline:
    """Retrieve, build a grounded prompt, answer, and check the citations.

    ``complete`` is any text-in, text-out language-model function (connected in a
    later step). Without one, or if it fails, answers are extractive quotes. The
    answer's citations are parsed and mapped back to real sources; numbers that
    point to no source are reported in ``invalid_citations``.
    """

    def __init__(
        self,
        index=None,
        complete: Optional[Callable[[str], str]] = None,
        top_k: int = 3,
        max_context_tokens: int = 3000,
        retrieve_fn: Optional[Callable] = None,
    ):
        self.index = index
        self.complete = complete
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.retrieve_fn = retrieve_fn
        self.extractive = ExtractiveAnswerer()

    def _retrieve(self, question: str) -> list:
        if self.index is None:
            return []
        if self.retrieve_fn is not None:
            return list(self.retrieve_fn(self.index, question, top_k=self.top_k))
        return self.index.search(question, top_k=self.top_k)

    def answer(self, question: str, documents: Optional[Sequence[str]] = None) -> Answer:
        if documents is not None:
            # Plain passages (quick experiments, no index): treat them as ranked sources.
            from retrieval.index import SearchResult

            results = [SearchResult(text=d, source=f"passage-{i}", start=0, end=len(d), score=0.0) for i, d in enumerate(documents, 1)]
        else:
            results = self._retrieve(question)

        if not results:
            return Answer(text=ExtractiveAnswerer.NOT_FOUND, model="none")

        prompt = build_prompt(question, results, self.max_context_tokens)
        offered = list(results[: prompt.included])
        text, used_llm, reason, model, usage = None, False, None, "extractive", None
        if self.complete is not None:
            try:
                chat = getattr(self.complete, "chat", None)
                # Prefer a real system prompt; plain functions get one combined string.
                text = (chat(prompt.system, prompt.user) if chat else self.complete(prompt.as_text())).strip()
                used_llm = True
                call = getattr(self.complete, "last_call", None)
                model = call.model if call else getattr(self.complete, "name", "llm")
                if call:
                    usage = {"input_tokens": call.input_tokens, "output_tokens": call.output_tokens}
            except Exception as exc:  # degrade to an extractive answer instead of failing
                reason = f"{type(exc).__name__}: {exc}"
        if text is None:
            text = self.extractive.answer(question, offered)

        valid, invalid = check_citations(text, len(offered))
        citations = [
            Citation(number=n, label=citation_label(offered[n - 1]), source=offered[n - 1].source, text=offered[n - 1].text)
            for n in valid
        ]
        return Answer(
            text=text,
            citations=citations,
            sources=offered,
            used_llm=used_llm,
            invalid_citations=invalid,
            fallback_reason=reason,
            prompt=prompt,
            model=model,
            prompt_version=prompt.version,
            usage=usage,
        )
