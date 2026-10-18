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
