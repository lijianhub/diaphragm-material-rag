"""Request and response models of the HTTP API (Pydantic v2)."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_CHARS = 2000


# Metadata fields a caller may filter on: a fixed list, so a filter cannot probe arbitrary fields.
FILTER_FIELDS = ("source", "doc_type", "title", "heading", "page", "sheet")

FilterValue = Union[str, int, List[Union[str, int]]]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS, description="The user's question")
    top_k: int = Field(3, ge=1, le=10, description="Number of sources offered to the answerer")
    filter: Optional[Dict[str, FilterValue]] = Field(
        None, description=f"Search only chunks whose metadata match, e.g. {{\"doc_type\": [\"md\", \"pdf\"]}}; fields: {', '.join(FILTER_FIELDS)}"
    )

    @field_validator("filter")
    @classmethod
    def known_bounded_fields(cls, value):
        if value is None:
            return value
        unknown = sorted(set(value) - set(FILTER_FIELDS))
        if unknown:
            raise ValueError(f"unknown filter fields {unknown}; allowed: {list(FILTER_FIELDS)}")
        for name, expected in value.items():
            options = expected if isinstance(expected, list) else [expected]
            if not 1 <= len(options) <= 20 or any(len(str(option)) > 200 for option in options):
                raise ValueError(f"filter {name!r}: 1 to 20 values of at most 200 characters")
        return value

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class CitationOut(BaseModel):
    number: int
    label: str
    source: str


class SourceOut(BaseModel):
    number: int
    label: str
    source: str
    text: str
    score: float


class AskResponse(BaseModel):
    request_id: str
    answer: str
    refused: bool = Field(description="True when every sentence says the sources lack the answer")
    citations: List[CitationOut]
    sources: List[SourceOut]
    invalid_citations: List[int]
    model: str = Field(description="Model that actually served the answer, or 'extractive'")
    prompt_version: str
    used_llm: bool
    fallback_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    timings_ms: Dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    index_chunks: int = 0
    embedder: Optional[str] = None
    answerer: str = "extractive"
    reason: Optional[str] = None


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    detail: Optional[object] = None


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., pattern=r"^[A-Za-z0-9._:-]{1,64}$", description="The request ID of the answer being rated")
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(None, max_length=500)


class AgentRunRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS, description="The task or question for the agent")

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = Field(..., pattern=r"^[A-Za-z0-9._:@-]{1,64}$", description="Who decided (set by the BFF from the admin key)")
    reason: str = Field("", max_length=500)


class FeedbackResponse(BaseModel):
    status: str
    request_id: str
