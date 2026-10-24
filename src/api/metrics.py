"""A minimal metrics registry that renders the Prometheus text exposition format.

Only counters and histograms are needed, so this avoids a dependency. Each app gets its
own ``Metrics`` instance, which keeps tests isolated. Label values come from a fixed set
(route templates, outcomes), never from user input, so the number of series stays small.
"""
from __future__ import annotations

import threading
from bisect import bisect_left
from typing import Dict, Iterable, List, Sequence, Tuple

LabelKey = Tuple[Tuple[str, str], ...]

# Seconds. Retrieval takes milliseconds and LLM calls take seconds, so buckets span both.
DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


def _key(labels: Dict[str, str]) -> LabelKey:
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


def _escape(value: str) -> str:
    # Prometheus text format: backslash, double quote and line feed are escaped in label values.
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt_labels(key: Iterable[Tuple[str, str]]) -> str:
    items = list(key)
    if not items:
        return ""
    return "{" + ",".join(f'{k}="{_escape(v)}"' for k, v in items) + "}"


def _fmt_value(value: float) -> str:
    return repr(float(value)) if value != int(value) else f"{int(value)}"


class Counter:
    def __init__(self, name: str, help_text: str):
        self.name, self.help = name, help_text
        self.values: Dict[LabelKey, float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise ValueError("counters only go up")
        key = _key(labels)
        self.values[key] = self.values.get(key, 0.0) + amount

    def get(self, **labels: str) -> float:
        return self.values.get(_key(labels), 0.0)

    def render(self) -> List[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        lines += [f"{self.name}{_fmt_labels(k)} {_fmt_value(v)}" for k, v in sorted(self.values.items())]
        return lines


class Histogram:
    def __init__(self, name: str, help_text: str, buckets: Sequence[float] = DEFAULT_BUCKETS):
        self.name, self.help = name, help_text
        self.buckets = tuple(sorted(buckets))
        self.series: Dict[LabelKey, dict] = {}

    def observe(self, value: float, **labels: str) -> None:
        key = _key(labels)
        s = self.series.setdefault(key, {"counts": [0] * len(self.buckets), "sum": 0.0, "count": 0})
        index = bisect_left(self.buckets, value)
        if index < len(self.buckets):
            s["counts"][index] += 1
        s["sum"] += value
        s["count"] += 1

    def count(self, **labels: str) -> int:
        return self.series.get(_key(labels), {"count": 0})["count"]

    def render(self) -> List[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key, s in sorted(self.series.items()):
            cumulative = 0
            for bound, n in zip(self.buckets, s["counts"]):
                cumulative += n
                lines.append(f"{self.name}_bucket{_fmt_labels(sorted(key + (('le', _fmt_value(bound)),)))} {cumulative}")
            lines.append(f"{self.name}_bucket{_fmt_labels(sorted(key + (('le', '+Inf'),)))} {s['count']}")
            lines.append(f"{self.name}_sum{_fmt_labels(key)} {_fmt_value(s['sum'])}")
            lines.append(f"{self.name}_count{_fmt_labels(key)} {s['count']}")
        return lines


class Metrics:
    """All metrics of one API instance. Thread-safe: FastAPI runs sync endpoints in a pool."""

    def __init__(self):
        self._lock = threading.Lock()
        self.requests = Counter("http_requests_total", "HTTP requests by method, route and status code.")
        self.request_seconds = Histogram("http_request_duration_seconds", "HTTP request latency by route.")
        self.stage_seconds = Histogram("rag_stage_duration_seconds", "Time spent per pipeline stage (retrieval, generation).")
        self.answers = Counter("rag_answers_total", "Answers by outcome: answered, refused, fallback.")
        self.llm_calls = Counter("rag_llm_calls_total", "LLM calls by model.")
        self.tokens = Counter("rag_llm_tokens_total", "LLM tokens by model and direction (input, output).")
        self.feedback = Counter("rag_feedback_total", "User feedback on answers by rating (up, down).")

    def record_request(self, method: str, route: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests.inc(method=method, route=route, status=str(status))
            self.request_seconds.observe(seconds, route=route)

    def record_answer(self, answer) -> None:
        outcome = "fallback" if answer.fallback_reason else ("refused" if answer.refused else "answered")
        with self._lock:
            self.answers.inc(outcome=outcome)
            for stage, ms in (answer.timings or {}).items():
                self.stage_seconds.observe(ms / 1000, stage=stage.removesuffix("_ms"))
            if answer.used_llm:
                self.llm_calls.inc(model=answer.model)
                usage = answer.usage or {}
                self.tokens.inc(usage.get("input_tokens", 0), model=answer.model, direction="input")
                self.tokens.inc(usage.get("output_tokens", 0), model=answer.model, direction="output")

    def record_feedback(self, rating: str) -> None:
        with self._lock:
            self.feedback.inc(rating=rating)

    def render(self) -> str:
        with self._lock:
            parts = [self.requests, self.request_seconds, self.stage_seconds, self.answers, self.llm_calls, self.tokens, self.feedback]
            return "\n".join(line for metric in parts for line in metric.render()) + "\n"
