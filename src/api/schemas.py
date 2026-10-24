"""Request and response models of the HTTP API (Pydantic v2)."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_CHARS = 2000


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS, description="The user's question")
    top_k: int = Field(3, ge=1, le=10, description="Number of sources offered to the answerer")

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


class FeedbackResponse(BaseModel):
    status: str
    request_id: str
