"""Embedders turn text into vectors. ``create_embedder(name)`` builds one from its name:

    simple-hash[-<dim>]                         offline feature hashing (default, tests)
    st:<hugging-face-model-id>                  local sentence-transformers model
    bedrock:<model-id>[@<dimensions>]           Amazon Bedrock (Titan Text Embeddings V2)

An embedder's ``name`` is a valid spec for its own kind, which is how a saved
index rebuilds the exact embedder that created it.
"""
from __future__ import annotations

from typing import Any

from .embedder import SimpleEmbedder
from .models import BedrockEmbedder, SentenceTransformerEmbedder

DEFAULT_LOCAL_MODEL = "st:sentence-transformers/all-MiniLM-L6-v2"


def create_embedder(name: str, **kwargs: Any):
    if name == "simple-hash":
        return SimpleEmbedder(**kwargs)
    if name.startswith("simple-hash-"):
        return SimpleEmbedder(dim=int(name.rsplit("-", 1)[1]), **kwargs)
    if name.startswith("st:"):
        return SentenceTransformerEmbedder(name[3:], **kwargs)
    if name.startswith("bedrock:"):
        model_id, _, dims = name[len("bedrock:") :].partition("@")
        return BedrockEmbedder(model_id, dimensions=int(dims) if dims else 1024, **kwargs)
    raise ValueError(f"unknown embedder {name!r}; expected simple-hash[-dim], st:<model> or bedrock:<model>[@dims]")


def query_vector(embedder, text: str):
    """Embed a query, using the embedder's query-side encoding when it has one."""
    encode_query = getattr(embedder, "encode_query", None)
    return encode_query(text) if encode_query else embedder.encode(text)


__all__ = [
    "DEFAULT_LOCAL_MODEL",
    "BedrockEmbedder",
    "SentenceTransformerEmbedder",
    "SimpleEmbedder",
    "create_embedder",
    "query_vector",
]
