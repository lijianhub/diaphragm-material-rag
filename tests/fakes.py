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
