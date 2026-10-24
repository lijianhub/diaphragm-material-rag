"""HTTP service around the RAG pipeline (FastAPI).

Endpoints:
- ``POST /ask``     answer a question with citations, model, prompt version, tokens and stage timings
- ``POST /ask/stream``  the same answer as Server-Sent Events: ``sources``, ``delta``..., ``final`` (or ``error``)
- ``POST /feedback`` record a 👍/👎 rating (and an optional comment) for an answer's request ID
- ``GET /health``   liveness: the process is up
- ``GET /ready``    readiness: the index is loaded (503 otherwise)
- ``GET /metrics``  Prometheus text format: requests, latency, stage timings, outcomes, LLM calls, tokens
- ``GET /ops/...``  read-only JSON for the ops dashboard: ``summary`` (the metrics with latency
  percentiles), ``feedback``, ``evaluations`` (tracked runs) and ``ingestion`` (last indexing run).
  Internal: the BFF puts them behind an admin key.

Every response carries ``X-Request-ID`` (the caller's, if valid, or a new one), and every
request produces one JSON log line. The question text is never logged, only its length and
a hash, because the documents and the questions may be confidential.

Run: ``python scripts/serve.py`` (or ``uvicorn api.main:create_app --factory``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from generation import RAGPipeline
from generation.prompt import citation_label

from .metrics import Metrics
from .ops import read_feedback, read_runs
from .schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ReadyResponse,
    SourceOut,
)

logger = logging.getLogger("rag.api")

# Accept the caller's request ID only if it is short and plain, so it cannot inject log lines.
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


@dataclass
class Settings:
    index_dir: str = "data/index"
    llm: str = ""  # e.g. "openai:<model>@low"; empty → extractive answers
    llm_timeout: float = 30.0  # seconds per LLM call; the SDK default (120 s) is too long for a user request
    llm_max_retries: int = 1
    feedback_path: str = "logs/feedback.jsonl"  # git-ignored; one JSON object per rating
    runs_path: str = "runs/runs.jsonl"  # tracked evaluation runs (--track jsonl), for the ops dashboard

    @classmethod
    def from_env(cls, base_dir: Optional[Path] = None) -> "Settings":
        base = Path(base_dir) if base_dir else Path.cwd()
        return cls(
            index_dir=os.environ.get("RAG_INDEX_DIR") or str(base / "data" / "index"),
            llm=os.environ.get("RAG_LLM", ""),
            llm_timeout=float(os.environ.get("RAG_LLM_TIMEOUT", "30")),
            llm_max_retries=int(os.environ.get("RAG_LLM_MAX_RETRIES", "1")),
            feedback_path=os.environ.get("RAG_FEEDBACK_PATH") or str(base / "logs" / "feedback.jsonl"),
            runs_path=os.environ.get("RAG_RUNS_PATH") or str(base / "runs" / "runs.jsonl"),
        )


def _request_id(header: Optional[str]) -> str:
    return header if header and _REQUEST_ID.match(header) else uuid.uuid4().hex


def _log(fields: dict) -> None:
    logger.info(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **fields}, ensure_ascii=False))


def _error(request_id: str, status: int, error: str, detail: Any = None) -> JSONResponse:
    body = ErrorResponse(request_id=request_id, error=error, detail=detail).model_dump()
    return JSONResponse(status_code=status, content=body, headers={"X-Request-ID": request_id})


class ClosingStreamingResponse(StreamingResponse):
    """A StreamingResponse that always closes its body iterator.

    With ASGI 2.4 servers (Uvicorn), Starlette does not watch for disconnects while
    streaming; the next ``send`` after a disconnect raises, and the exception leaves the
    ``async for`` loop without ``aclose()``, so the generator is abandoned mid-stream and its
    ``finally`` (which closes the LLM stream) never runs. The model keeps generating, and
    billing, for a client that is gone. Closing here makes cancellation reach the model.
    """

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()


async def closing_stream(events):
    """Serve a synchronous generator as an async one, and close it when the client goes.

    Starlette iterates a sync generator in a thread pool and, on client disconnect, only
    cancels its own wrapper: the generator is never closed, so its ``finally`` blocks never
    run and the LLM stream keeps generating (and billing) to the end. Here each ``next()``
    runs in a worker thread; cancellation waits for that step to return (so the generator is
    never closed mid-step), and ``finally`` closes the generator, which closes the LLM stream.
    """
    import anyio

    done = object()
    try:
        while True:
            item = await anyio.to_thread.run_sync(next, events, done)
            if item is done:
                return
            yield item
    finally:
        events.close()


def _sse(event: str, data: dict) -> str:
    """One Server-Sent Event. ``data`` is a single JSON line, so newlines in the text are safe."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _source_out(number: int, source) -> SourceOut:
    return SourceOut(number=number, label=citation_label(source), source=str(source.source), text=source.text, score=float(source.score))


def _answer_response(request_id: str, answer) -> AskResponse:
    return AskResponse(
        request_id=request_id,
        answer=answer.text,
        refused=answer.refused,
        citations=[CitationOut(number=c.number, label=c.label, source=c.source) for c in answer.citations],
        sources=[_source_out(n, s) for n, s in enumerate(answer.sources, start=1)],
        invalid_citations=answer.invalid_citations,
        model=answer.model,
        prompt_version=answer.prompt_version,
        used_llm=answer.used_llm,
        fallback_reason=answer.fallback_reason,
        usage=answer.usage,
        timings_ms=answer.timings,
    )


def _answer_log(answer) -> dict:
    usage = answer.usage or {}
    return dict(
        model=answer.model, prompt_version=answer.prompt_version, used_llm=answer.used_llm, refused=answer.refused,
        fallback_reason=answer.fallback_reason, sources=len(answer.sources), citations=len(answer.citations),
        invalid_citations=len(answer.invalid_citations), tokens_in=usage.get("input_tokens", 0),
        tokens_out=usage.get("output_tokens", 0), **answer.timings,
    )


def _question_log(body) -> dict:
    return dict(
        question_chars=len(body.question),
        question_sha256=hashlib.sha256(body.question.encode("utf-8")).hexdigest()[:12],
        top_k=body.top_k,
    )


def create_app(settings: Optional[Settings] = None, *, index=None, llm=None, metrics: Optional[Metrics] = None) -> FastAPI:
    """Build the app. Tests pass ``index`` and ``llm`` directly; otherwise the index is loaded
    from ``settings.index_dir`` and the LLM is built from ``settings.llm``.

    A configured LLM that cannot be built (missing SDK or key) fails at startup, not on the
    first user request. A missing index is not fatal: ``/ready`` reports it and ``/ask``
    returns 503 until an index exists.
    """
    settings = settings or Settings.from_env()
    metrics = metrics or Metrics()
    not_ready_reason = None

    if index is None:
        from retrieval.index import SearchIndex

        try:
            index = SearchIndex.load(settings.index_dir)
        except FileNotFoundError as exc:
            not_ready_reason = f"no index: {exc}"
    if llm is None and settings.llm:
        from generation import create_llm

        llm = create_llm(settings.llm, timeout=settings.llm_timeout, max_retries=settings.llm_max_retries)

    app = FastAPI(title="diaphragm-material-rag", version="0.16.0", description="Grounded answers with citations over engineering documents.")
    app.state.index, app.state.llm, app.state.metrics, app.state.not_ready = index, llm, metrics, not_ready_reason
    answerer = getattr(llm, "name", "llm") if llm else "extractive"

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = _request_id(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        request.state.log = {}
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # last line of defence: no stack trace reaches the client
            request.state.log["error_type"] = type(exc).__name__
            response = _error(request_id, 500, "internal_error")
        route = getattr(request.scope.get("route"), "path", "unmatched")  # template, never the raw path
        seconds = time.perf_counter() - started
        metrics.record_request(request.method, route, response.status_code, seconds)
        response.headers["X-Request-ID"] = request_id
        if route != "/metrics":  # scrapes every few seconds would drown the log
            _log({"event": "request", "request_id": request_id, "method": request.method, "route": route,
                  "status": response.status_code, "latency_ms": round(seconds * 1000, 2), **request.state.log})
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        detail = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()]
        return _error(getattr(request.state, "request_id", uuid.uuid4().hex), 422, "invalid_request", detail)

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}})
    def ready():
        if app.state.index is None:
            body = ReadyResponse(ready=False, answerer=answerer, reason=app.state.not_ready)
            return JSONResponse(status_code=503, content=body.model_dump())
        return ReadyResponse(
            ready=True, index_chunks=len(app.state.index), embedder=getattr(app.state.index.embedder, "name", None), answerer=answerer
        )

    @app.post("/ask", response_model=AskResponse, responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
    def ask(body: AskRequest, request: Request):
        request_id = request.state.request_id
        request.state.log.update(_question_log(body))
        if app.state.index is None:
            return _error(request_id, 503, "not_ready", app.state.not_ready)

        answer = RAGPipeline(app.state.index, complete=app.state.llm, top_k=body.top_k).answer(body.question)
        metrics.record_answer(answer)
        request.state.log.update(_answer_log(answer))
        return _answer_response(request_id, answer)

    @app.post("/ask/stream", responses={200: {"content": {"text/event-stream": {}}}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    def ask_stream(body: AskRequest, request: Request):
        """Server-Sent Events: ``sources`` once, ``delta`` per text piece, then ``final`` with the
        checked answer (same body as ``/ask``), which replaces the streamed text. An unexpected
        failure ends the stream with ``error``. If the client disconnects, generation stops."""
        request_id = request.state.request_id
        request.state.log.update(stream=True, **_question_log(body))
        if app.state.index is None:
            return _error(request_id, 503, "not_ready", app.state.not_ready)
        pipeline = RAGPipeline(app.state.index, complete=app.state.llm, top_k=body.top_k)
        question_fields = _question_log(body)

        def events():
            started = time.perf_counter()
            answer, completed, error_type, deltas = None, False, None, 0
            try:
                for event in pipeline.stream(body.question):
                    if event["type"] == "sources":
                        sources = [_source_out(n, s).model_dump() for n, s in enumerate(event["sources"], start=1)]
                        yield _sse("sources", {"request_id": request_id, "sources": sources})
                    elif event["type"] == "delta":
                        deltas += 1
                        yield _sse("delta", {"text": event["text"]})
                    else:
                        answer = event["answer"]
                        metrics.record_answer(answer)
                        yield _sse("final", _answer_response(request_id, answer).model_dump())
                        completed = True
            except Exception as exc:  # the status line is already sent: report in-band, without internals
                error_type = type(exc).__name__
                yield _sse("error", {"request_id": request_id, "error": "internal_error"})
            finally:
                # Runs on completion, on error, and when the client disconnects (GeneratorExit).
                fields = {"event": "stream", "request_id": request_id, "route": "/ask/stream", "completed": completed,
                          "disconnected": not completed and error_type is None, "deltas": deltas,
                          "stream_ms": round((time.perf_counter() - started) * 1000, 2), **question_fields}
                if error_type:
                    fields["error_type"] = error_type
                if answer is not None:
                    fields.update(_answer_log(answer))
                _log(fields)

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}  # no proxy buffering of the stream
        # closing_stream: a client disconnect must close the generator, and with it the LLM stream.
        return ClosingStreamingResponse(closing_stream(events()), media_type="text/event-stream", headers=headers)

    feedback_lock = threading.Lock()

    @app.post("/feedback", status_code=201, response_model=FeedbackResponse, responses={422: {"model": ErrorResponse}})
    def feedback(body: FeedbackRequest, request: Request):
        """Store a rating for the answer with ``body.request_id``. The comment is kept in the
        feedback file (the store), not in the request log."""
        request_id = request.state.request_id
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "answer_request_id": body.request_id,
            "rating": body.rating,
            "comment": body.comment,
            "feedback_request_id": request_id,
        }
        path = Path(settings.feedback_path)
        with feedback_lock:  # concurrent requests run in a thread pool: one writer at a time
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        metrics.record_feedback(body.rating)
        request.state.log.update(answer_request_id=body.request_id, rating=body.rating, has_comment=bool(body.comment))
        return FeedbackResponse(status="recorded", request_id=request_id)

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics():
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    # ---- ops dashboard (read-only) ----------------------------------------------------

    @app.get("/ops/summary")
    def ops_summary():
        """Service health since start-up: traffic, errors, latency percentiles per route and
        stage, answer outcomes, LLM calls and tokens per model, feedback."""
        index = app.state.index
        return {
            **metrics.snapshot(),
            "ready": index is not None,
            "not_ready_reason": app.state.not_ready,
            "answerer": answerer,
            "index_chunks": len(index) if index is not None else 0,
        }

    @app.get("/ops/feedback")
    def ops_feedback(limit: int = Query(50, ge=1, le=200)):
        return read_feedback(settings.feedback_path, limit)

    @app.get("/ops/evaluations")
    def ops_evaluations(limit: int = Query(100, ge=1, le=500)):
        return read_runs(settings.runs_path, limit)

    @app.get("/ops/ingestion")
    def ops_ingestion():
        from ingestion.report import read_report

        try:
            report = read_report(settings.index_dir)
        except (OSError, ValueError):
            return {"report": None, "reason": "the ingestion report could not be read"}
        return {"report": report, "reason": None if report else "no ingestion report: run scripts/index.py"}

    return app
