"""Test doubles for model SDKs, so tests never download a model or call AWS."""
import json


class FakeSentenceModel:
    """Stands in for sentence_transformers.SentenceTransformer: no download, no torch."""

    def __init__(self):
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        self.calls.append({"texts": list(texts), "batch_size": batch_size, "normalize": normalize_embeddings})
        return [[float(len(t)), 1.0, 0.0] for t in texts]


class FakeBedrockClient:
    """Records invoke_model calls; can fail a few times with throttling first."""

    def __init__(self, throttle_times=0, dim=256):
        self.requests = []
        self.throttle_times = throttle_times
        self.dim = dim

    def invoke_model(self, modelId, body, contentType, accept):
        self.requests.append({"modelId": modelId, "body": json.loads(body)})
        if self.throttle_times:
            self.throttle_times -= 1
            error = Exception("Rate exceeded")
            error.response = {"Error": {"Code": "ThrottlingException"}}
            raise error
        payload = json.dumps({"embedding": [0.5] * self.dim}).encode("utf-8")

        class Body:
            def read(self_inner):
                return payload

        return {"body": Body()}


class FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder: scores a (query, passage) pair by
    how many query words the passage contains, and records every batch it is given."""

    def __init__(self, preferred=None):
        self.calls = []
        self.preferred = preferred  # a phrase this "model" judges highly relevant

    def predict(self, pairs, batch_size, show_progress_bar):
        self.calls.append(list(pairs))
        scores = []
        for query, passage in pairs:
            score = float(sum(word in passage.lower() for word in query.lower().split()))
            if self.preferred and self.preferred in passage.lower():
                score += 100.0
            scores.append(score)
        return scores


class FakeBedrockAgentClient:
    """Records rerank requests and returns results out of order, as the real API may."""

    def __init__(self, scores):
        self.scores = scores
        self.requests = []

    def rerank(self, queries, sources, rerankingConfiguration):
        self.requests.append({"queries": queries, "sources": sources, "config": rerankingConfiguration})
        ranked = sorted(range(len(sources)), key=lambda i: self.scores[i], reverse=True)
        return {"results": [{"index": i, "relevanceScore": self.scores[i]} for i in ranked]}


class FakeLLM:
    """A text-in, text-out stand-in for a language model: returns a scripted reply and
    records every prompt it was given."""

    def __init__(self, reply="", error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply


class FakeAnthropicClient:
    """Stands in for anthropic.Anthropic(): records messages.create / beta.messages.create
    calls and returns a response shaped like the SDK's (thinking + text blocks, usage)."""

    def __init__(self, text="Answer [1].", stop_reason="end_turn", model="claude-opus-5-5", refusal_category=None, error=None):
        from types import SimpleNamespace

        self.calls = []
        self._ns = SimpleNamespace
        self.text, self.stop_reason, self.model = text, stop_reason, model
        self.refusal_category, self.error = refusal_category, error
        self.messages = SimpleNamespace(create=lambda **kw: self._create("messages", kw))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: self._create("beta", kw)))

    def _create(self, path, kwargs):
        self.calls.append({"path": path, **kwargs})
        if self.error:
            raise self.error
        ns = self._ns
        return ns(
            content=[ns(type="thinking", thinking=""), ns(type="text", text=self.text)],
            stop_reason=self.stop_reason,
            stop_details=ns(type="refusal", category=self.refusal_category, explanation="declined") if self.stop_reason == "refusal" else None,
            model=self.model,
            usage=ns(input_tokens=1200, output_tokens=45),
        )


class FakeOpenAIClient:
    """Stands in for openai.OpenAI(): records responses.create calls and returns a
    Responses-API-shaped object (reasoning item + message with text or refusal)."""

    def __init__(self, text="Answer [1].", refusal=None, status="completed", incomplete_reason=None, model="model-under-test"):
        from types import SimpleNamespace as ns

        self.calls = []
        content = [ns(type="refusal", refusal=refusal)] if refusal else [ns(type="output_text", text=text)]
        self._response = ns(
            output=[ns(type="reasoning"), ns(type="message", role="assistant", content=content)],
            status=status,
            incomplete_details=ns(reason=incomplete_reason) if incomplete_reason else None,
            usage=ns(input_tokens=900, output_tokens=60),
            model=model,
        )
        self.responses = ns(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class FakeMlflow:
    """Stands in for the ``mlflow`` module: records experiments, runs, params, metrics,
    tags and artifacts, so MlflowTracker is tested without MLflow installed."""

    class _Run:
        def __init__(self, run_id):
            self.info = type("Info", (), {"run_id": run_id})()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def __init__(self, existing_experiments=()):
        self.tracking_uri = None
        self.experiments = {name: {"artifact_location": None} for name in existing_experiments}
        self.active_experiment = None
        self.runs = []

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def get_experiment_by_name(self, name):
        return self.experiments.get(name)

    def create_experiment(self, name, artifact_location=None):
        self.experiments[name] = {"artifact_location": artifact_location}

    def set_experiment(self, name):
        self.experiments.setdefault(name, {"artifact_location": None})
        self.active_experiment = name

    def start_run(self, run_name=None):
        self.runs.append({"name": run_name, "experiment": self.active_experiment, "params": {}, "metrics": {}, "tags": {}, "artifacts": []})
        return self._Run(f"run-{len(self.runs)}")

    def log_params(self, params):
        self.runs[-1]["params"].update(params)

    def log_metrics(self, metrics):
        self.runs[-1]["metrics"].update(metrics)

    def set_tags(self, tags):
        self.runs[-1]["tags"].update(tags)

    def log_artifact(self, path):
        self.runs[-1]["artifacts"].append(path)
