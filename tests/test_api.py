import json
import logging

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api import Metrics, Settings, create_app  # noqa: E402
from ingestion.loader import Document  # noqa: E402
from retrieval.index import SearchIndex  # noqa: E402
from tests.fakes import FakeLLM, FakeStreamingLLM  # noqa: E402

DOCS = [
    Document(source="forming.txt", content="Springback is compensated by over-forming the profile by a small angle."),
    Document(source="lots.txt", content="Lot L-302 was rejected for tearing at the rim."),
]


def _index():
    index = SearchIndex()
    index.sync(DOCS)
    return index


def _client(llm=None, index="build", **kwargs):
    app = create_app(Settings(index_dir="does/not/exist"), index=_index() if index == "build" else index, llm=llm, **kwargs)
    return TestClient(app, raise_server_exceptions=False)


def test_health_is_always_ok_and_ready_needs_an_index():
    assert _client().get("/health").json() == {"status": "ok"}

    ready = _client().get("/ready")
    not_ready = _client(index=None).get("/ready")

    assert ready.status_code == 200 and ready.json()["ready"] and ready.json()["index_chunks"] == 2
    assert not_ready.status_code == 503 and not not_ready.json()["ready"] and "no index" in not_ready.json()["reason"]


def test_ask_returns_a_cited_answer_with_versions_and_timings():
    llm = FakeLLM("Dies are over-formed by a small angle [1].")

    response = _client(llm=llm).post("/ask", json={"question": "How is springback compensated?", "top_k": 2})
    body = response.json()

    assert response.status_code == 200
    assert body["answer"] == "Dies are over-formed by a small angle [1]."
    assert body["citations"][0] == {"number": 1, "label": "forming.txt", "source": "forming.txt"}
    assert [s["number"] for s in body["sources"]] == [1, 2] and body["sources"][0]["text"].startswith("Springback")
    assert body["used_llm"] and body["prompt_version"].startswith("grounded-v2+")
    assert set(body["timings_ms"]) == {"retrieval_ms", "generation_ms"}
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_ask_without_an_llm_answers_extractively():
    body = _client().post("/ask", json={"question": "Which lot was rejected for tearing?"}).json()

    assert body["model"] == "extractive" and not body["used_llm"]
    assert "L-302" in body["answer"]


def test_a_refusal_comes_back_without_citations():
    body = _client(llm=FakeLLM("The price is not in the sources. [1][2]")).post("/ask", json={"question": "Price of titanium?"}).json()

    assert body["refused"] and body["citations"] == [] and body["answer"] == "The price is not in the sources."


def test_an_llm_failure_degrades_to_an_extractive_answer_and_is_counted():
    client = _client(llm=FakeLLM(error=TimeoutError("slow")))

    body = client.post("/ask", json={"question": "Which lot was rejected for tearing?"}).json()
    metrics = client.get("/metrics").text

    assert body["fallback_reason"] == "TimeoutError: slow" and "L-302" in body["answer"]
    assert 'rag_answers_total{outcome="fallback"} 1' in metrics


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "   "}, {"question": "x" * 2001}, {"question": "ok", "top_k": 0}, {}])
def test_invalid_requests_get_422_with_a_request_id(payload):
    response = _client().post("/ask", json=payload, headers={"X-Request-ID": "client-req-1"})

    assert response.status_code == 422
    assert response.json()["request_id"] == "client-req-1" and response.json()["error"] == "invalid_request"


def test_ask_returns_503_until_an_index_exists():
    response = _client(index=None).post("/ask", json={"question": "anything?"})

    assert response.status_code == 503 and response.json()["error"] == "not_ready"


def test_an_unexpected_error_returns_500_without_internals():
    class BrokenIndex:
        embedder = None

        def __len__(self):
            return 1

        def search(self, *args, **kwargs):
            raise RuntimeError("database password is hunter2")

    response = _client(index=BrokenIndex()).post("/ask", json={"question": "anything?"})

    assert response.status_code == 500
    assert response.json()["error"] == "internal_error" and "hunter2" not in response.text
    assert response.headers["X-Request-ID"] == response.json()["request_id"]


def test_request_ids_are_reused_when_safe_and_replaced_when_not():
    client = _client()

    kept = client.get("/health", headers={"X-Request-ID": "bff-7f3a.1"}).headers["X-Request-ID"]
    replaced = client.get("/health", headers={"X-Request-ID": 'evil"\n{"injected": true}'}).headers["X-Request-ID"]
    generated = client.get("/health").headers["X-Request-ID"]

    assert kept == "bff-7f3a.1"
    assert replaced != kept and len(replaced) == 32 and replaced.isalnum()
    assert len(generated) == 32 and generated != replaced


def test_one_json_log_line_per_request_without_the_question_text(caplog):
    client = _client(llm=FakeLLM("Dies are over-formed by a small angle [1].", ))
    question = "How is springback compensated on the confidential part 4711?"

    with caplog.at_level(logging.INFO, logger="rag.api"):
        client.post("/ask", json={"question": question}, headers={"X-Request-ID": "trace-1"})
    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "rag.api"]

    assert len(lines) == 1
    log = lines[0]
    assert log["request_id"] == "trace-1" and log["route"] == "/ask" and log["status"] == 200
    assert log["question_chars"] == len(question) and len(log["question_sha256"]) == 12
    assert {"latency_ms", "retrieval_ms", "generation_ms", "model", "prompt_version", "tokens_in"} <= set(log)
    assert "4711" not in json.dumps(log) and "springback" not in json.dumps(log).lower()


def test_metrics_use_route_templates_and_count_requests_and_tokens():
    class TokenLLM(FakeLLM):
        name = "fake-model"

        def __call__(self, prompt):
            from generation.llm import LLMCall

            self.last_call = LLMCall(model="fake-model", stop_reason="end", input_tokens=120, output_tokens=8)
            return super().__call__(prompt)

    client = _client(llm=TokenLLM("Over-formed [1]."))
    client.post("/ask", json={"question": "How is springback compensated?"})
    client.get("/no/such/path/12345")
    text = client.get("/metrics").text

    assert 'http_requests_total{method="POST",route="/ask",status="200"} 1' in text
    assert 'route="unmatched",status="404"' in text and "12345" not in text  # no raw paths, bounded label values
    assert 'rag_llm_tokens_total{direction="input",model="fake-model"} 120' in text
    assert 'rag_stage_duration_seconds_count{stage="retrieval"} 1' in text
    assert 'http_request_duration_seconds_bucket{le="+Inf",route="/ask"} 1' in text


def test_metrics_render_histogram_buckets_cumulatively():
    metrics = Metrics()
    for seconds in (0.003, 0.04, 0.04, 12.0):
        metrics.request_seconds.observe(seconds, route="/ask")

    text = metrics.render()

    assert 'http_request_duration_seconds_bucket{le="0.005",route="/ask"} 1' in text
    assert 'http_request_duration_seconds_bucket{le="0.05",route="/ask"} 3' in text
    assert 'http_request_duration_seconds_bucket{le="30",route="/ask"} 4' in text
    assert 'http_request_duration_seconds_count{route="/ask"} 4' in text


def test_a_configured_llm_that_cannot_be_built_fails_at_startup():
    with pytest.raises(ValueError):
        create_app(Settings(index_dir="does/not/exist", llm="unknown:model"), index=_index())


def test_settings_come_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_LLM", "openai:some-model@low")
    monkeypatch.setenv("RAG_LLM_TIMEOUT", "12.5")
    monkeypatch.delenv("RAG_INDEX_DIR", raising=False)

    settings = Settings.from_env(tmp_path)

    assert settings.llm == "openai:some-model@low" and settings.llm_timeout == 12.5
    assert settings.index_dir == str(tmp_path / "data" / "index")


# ---- streaming ----------------------------------------------------------------


def _sse_events(text):
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_stream_sends_sources_deltas_and_a_final_equal_to_ask():
    pieces = ("Dies are over-formed ", "by a small angle [1].")
    client = _client(llm=FakeStreamingLLM(pieces))

    response = client.post("/ask/stream", json={"question": "How is springback compensated?", "top_k": 2}, headers={"X-Request-ID": "s-1"})
    events = _sse_events(response.text)
    whole = _client(llm=FakeLLM("".join(pieces))).post("/ask", json={"question": "How is springback compensated?", "top_k": 2}).json()

    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["X-Request-ID"] == "s-1" and response.headers["cache-control"] == "no-cache"
    assert [e for e, _ in events] == ["sources", "delta", "delta", "final"]
    assert events[0][1]["request_id"] == "s-1" and len(events[0][1]["sources"]) == 2
    final = events[-1][1]
    for key in ("answer", "refused", "citations", "sources", "invalid_citations", "model", "prompt_version"):
        assert final[key] == whole[key], key
    assert "first_token_ms" in final["timings_ms"]


def test_streamed_refusal_is_cleaned_in_the_final_event():
    client = _client(llm=FakeStreamingLLM(("The price is not in the sources. ", "[1][2]")))

    events = _sse_events(client.post("/ask/stream", json={"question": "Price of titanium?"}).text)

    assert events[-1][1]["answer"] == "The price is not in the sources." and events[-1][1]["citations"] == []


def test_stream_rejects_invalid_requests_and_waits_for_an_index_like_ask():
    assert _client().post("/ask/stream", json={"question": " "}).status_code == 422
    not_ready = _client(index=None).post("/ask/stream", json={"question": "anything?"})
    assert not_ready.status_code == 503 and not_ready.json()["error"] == "not_ready"


def test_an_error_inside_the_stream_ends_with_an_error_event_without_internals():
    class BrokenIndex:
        embedder = None

        def __len__(self):
            return 1

        def search(self, *args, **kwargs):
            raise RuntimeError("database password is hunter2")

    response = _client(index=BrokenIndex()).post("/ask/stream", json={"question": "anything?"}, headers={"X-Request-ID": "s-err"})

    assert response.status_code == 200  # the status line went out before the error happened
    assert _sse_events(response.text) == [("error", {"request_id": "s-err", "error": "internal_error"})]
    assert "hunter2" not in response.text


def test_a_finished_stream_logs_one_completion_line_without_the_question(caplog):
    client = _client(llm=FakeStreamingLLM(("Over-formed ", "[1].")))

    with caplog.at_level(logging.INFO, logger="rag.api"):
        client.post("/ask/stream", json={"question": "springback on part 4711?"}, headers={"X-Request-ID": "s-2"})
    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "rag.api"]
    stream = [line for line in lines if line["event"] == "stream"]

    assert len(stream) == 1
    assert stream[0]["request_id"] == "s-2" and stream[0]["completed"] and not stream[0]["disconnected"]
    assert stream[0]["deltas"] == 2 and "first_token_ms" in stream[0]
    assert "4711" not in json.dumps(lines)


def test_stream_records_time_to_first_token_in_metrics():
    client = _client(llm=FakeStreamingLLM())
    client.post("/ask/stream", json={"question": "How is springback compensated?"})

    assert 'rag_stage_duration_seconds_count{stage="first_token"} 1' in client.get("/metrics").text


def test_closing_the_async_stream_closes_the_sync_generator():
    # Starlette never closes a sync generator when the client disconnects; closing_stream does.
    import asyncio

    from api.main import closing_stream

    state = {"closed": False}

    def events():
        try:
            yield "first"
            yield "second"
        finally:
            state["closed"] = True  # where the LLM stream gets closed

    async def consume_one_and_disconnect():
        stream = closing_stream(events())
        assert await stream.__anext__() == "first"
        await stream.aclose()  # what a client disconnect does to the response body iterator

    asyncio.run(consume_one_and_disconnect())

    assert state["closed"]


def test_a_client_disconnect_closes_the_llm_stream_on_a_real_server():
    # The unit tests above close generators directly; only a real server showed that a
    # disconnect left the LLM stream open (it kept generating for a client that was gone).
    import threading
    import time

    import httpx
    import uvicorn

    class SlowLLM(FakeStreamingLLM):
        def stream_chat(self, system, user):
            for piece in super().stream_chat(system, user):
                time.sleep(0.05)
                yield piece

    llm = SlowLLM(pieces=[f"word{i} " for i in range(60)])  # 3 s if nobody stops it
    server = uvicorn.Server(uvicorn.Config(create_app(Settings(index_dir="none"), index=_index(), llm=llm), host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]

        with httpx.Client(timeout=10) as client:
            with client.stream("POST", f"http://127.0.0.1:{port}/ask/stream", json={"question": "springback?"}) as response:
                for line in response.iter_lines():
                    if line.startswith("event: delta"):
                        break  # the user closes the page after the first words
        deadline = time.time() + 2
        while not llm.closed_early and time.time() < deadline:
            time.sleep(0.05)

        assert llm.closed_early and not llm.finished
    finally:
        server.should_exit = True
        thread.join(timeout=5)


# ---- feedback -----------------------------------------------------------------


def test_feedback_is_stored_counted_and_kept_out_of_the_log(tmp_path, caplog):
    path = tmp_path / "feedback.jsonl"
    client = TestClient(create_app(Settings(index_dir="none", feedback_path=str(path)), index=_index()), raise_server_exceptions=False)

    with caplog.at_level(logging.INFO, logger="rag.api"):
        up = client.post("/feedback", json={"request_id": "ans-1", "rating": "up"}, headers={"X-Request-ID": "fb-1"})
        down = client.post("/feedback", json={"request_id": "ans-2", "rating": "down", "comment": "cited the wrong alloy, part 4711"})
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    metrics = client.get("/metrics").text
    logs = [json.loads(r.getMessage()) for r in caplog.records if r.name == "rag.api"]

    assert up.status_code == 201 and up.json() == {"status": "recorded", "request_id": "fb-1"}
    assert down.status_code == 201
    assert [(e["answer_request_id"], e["rating"]) for e in entries] == [("ans-1", "up"), ("ans-2", "down")]
    assert entries[1]["comment"] == "cited the wrong alloy, part 4711" and entries[0]["feedback_request_id"] == "fb-1"
    assert 'rag_feedback_total{rating="up"} 1' in metrics and 'rag_feedback_total{rating="down"} 1' in metrics
    assert logs[1]["rating"] == "down" and logs[1]["has_comment"] is True
    assert "4711" not in json.dumps(logs)  # the comment lives in the store, not the log


@pytest.mark.parametrize(
    "payload",
    [
        {"request_id": "ans-1", "rating": "meh"},
        {"request_id": "bad id with spaces", "rating": "up"},
        {"request_id": "ans-1", "rating": "up", "comment": "x" * 501},
        {"rating": "up"},
    ],
)
def test_invalid_feedback_is_rejected(payload, tmp_path):
    path = tmp_path / "feedback.jsonl"
    client = TestClient(create_app(Settings(index_dir="none", feedback_path=str(path)), index=_index()), raise_server_exceptions=False)

    response = client.post("/feedback", json=payload)

    assert response.status_code == 422 and response.json()["error"] == "invalid_request"
    assert not path.exists()
