"""A minimal metrics registry that renders the Prometheus text exposition format.

Only counters and histograms are needed, so this avoids a dependency. Each app gets its
own ``Metrics`` instance, which keeps tests isolated. Label values come from a fixed set
(route templates, outcomes), never from user input, so the number of series stays small.
"""
from __future__ import annotations

import threading
from bisect import bisect_left
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

LabelKey = Tuple[Tuple[str, str], ...]

# Routes that carry user traffic. Health checks, scrapes and the ops dashboard's own polling
# are excluded from the traffic totals, or watching the dashboard would inflate its numbers.
TRAFFIC_ROUTES = ("/ask", "/ask/stream", "/feedback")

# Seconds. Retrieval takes milliseconds and LLM calls take seconds, so buckets span both.
# Percentiles are estimated inside a bucket, so they are only as fine as the buckets: with
# 5 ms as the first bound, every 2 ms request showed p95 = 4.8 ms. 1 and 2.5 ms resolve it.
DEFAULT_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


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

    def quantile(self, q: float, **labels: str) -> Optional[float]:
        """Estimate a quantile the way Prometheus' ``histogram_quantile`` does: find the
        bucket holding the q-th observation and interpolate linearly inside it. The result
        is only as precise as the bucket is wide. Above the last bucket, the last bound is
        returned (a lower limit). None when there are no observations."""
        s = self.series.get(_key(labels))
        if not s or s["count"] == 0:
            return None
        rank = q * s["count"]
        cumulative, lower = 0, 0.0
        for bound, n in zip(self.buckets, s["counts"]):
            if n and cumulative + n >= rank:
                return lower + (bound - lower) * (rank - cumulative) / n
            cumulative += n
            lower = bound
        return self.buckets[-1]

    def label_values(self, label: str) -> List[str]:
        return sorted({v for key in self.series for k, v in key if k == label})

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

    def __init__(self, clock=time.time):
        self._lock = threading.Lock()
        self.started_at = clock()
        self._clock = clock
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

    def snapshot(self) -> dict:
        """The registry as JSON for the ops dashboard: totals since the process started and
        latency percentiles estimated from the histograms. Counters reset on restart."""
        ms = lambda seconds: None if seconds is None else round(seconds * 1000, 1)  # noqa: E731
        with self._lock:
            routes: Dict[str, dict] = {}
            for key, n in self.requests.values.items():
                labels = dict(key)
                r = routes.setdefault(labels["route"], {"route": labels["route"], "count": 0, "client_errors": 0, "server_errors": 0})
                r["count"] += int(n)
                status = int(labels["status"])
                if 400 <= status < 500:
                    r["client_errors"] += int(n)
                elif status >= 500:
                    r["server_errors"] += int(n)
            for r in routes.values():
                r["p50_ms"] = ms(self.request_seconds.quantile(0.5, route=r["route"]))
                r["p95_ms"] = ms(self.request_seconds.quantile(0.95, route=r["route"]))
            traffic = [r for name, r in routes.items() if name in TRAFFIC_ROUTES]
            models: Dict[str, dict] = {}
            for key, n in self.llm_calls.values.items():
                models.setdefault(dict(key)["model"], {"calls": 0, "input_tokens": 0, "output_tokens": 0})["calls"] = int(n)
            for key, n in self.tokens.values.items():
                labels = dict(key)
                models.setdefault(labels["model"], {"calls": 0, "input_tokens": 0, "output_tokens": 0})[f"{labels['direction']}_tokens"] = int(n)
            return {
                "started_at": self.started_at,
                "uptime_s": round(self._clock() - self.started_at, 1),
                "traffic": {
                    "requests": sum(r["count"] for r in traffic),
                    "client_errors": sum(r["client_errors"] for r in traffic),
                    "server_errors": sum(r["server_errors"] for r in traffic),
                },
                "routes": sorted(routes.values(), key=lambda r: r["route"]),
                "stages": [
                    {"stage": stage, "count": self.stage_seconds.count(stage=stage),
                     "p50_ms": ms(self.stage_seconds.quantile(0.5, stage=stage)), "p95_ms": ms(self.stage_seconds.quantile(0.95, stage=stage))}
                    for stage in self.stage_seconds.label_values("stage")
                ],
                "answers": {o: int(self.answers.get(outcome=o)) for o in ("answered", "refused", "fallback")},
                "models": [{"model": m, **v} for m, v in sorted(models.items())],
                "feedback": {r: int(self.feedback.get(rating=r)) for r in ("up", "down")},
            }

    def render(self) -> str:
        with self._lock:
            parts = [self.requests, self.request_seconds, self.stage_seconds, self.answers, self.llm_calls, self.tokens, self.feedback]
            return "\n".join(line for metric in parts for line in metric.render()) + "\n"
