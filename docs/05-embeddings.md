# Embeddings

An embedder turns text into a vector so that similar meanings end up close together. Retrieval quality on paraphrased questions depends almost entirely on it.

## Available embedders

| Name (spec) | Class | Runs | Dimensions | Install |
|---|---|---|---|---|
| `simple-hash` / `simple-hash-<dim>` | `SimpleEmbedder` | offline, deterministic | 512 (default) | core |
| `st:<hugging-face-model-id>` | `SentenceTransformerEmbedder` | local CPU/GPU; model downloaded once | model-defined (MiniLM: 384) | `pip install -e .[local-embeddings]` |
| `bedrock:<model-id>[@dims]` | `BedrockEmbedder` | Amazon Bedrock API | Titan V2: 256 / 512 / 1024 | `pip install -e .[aws]` + AWS credentials |

```bash
python scripts/evaluate.py --embedder st:sentence-transformers/all-MiniLM-L6-v2
python scripts/index.py --rebuild --embedder st:sentence-transformers/all-MiniLM-L6-v2
python scripts/index.py --rebuild --embedder bedrock:amazon.titan-embed-text-v2:0@1024
```

`create_embedder(name)` builds any of them from its name. Every embedder's `name` is also a valid spec, so a saved index rebuilds the exact embedder that created it from `manifest.json`. `SearchIndex.load` refuses an embedder with a different name, because vectors from different models are not comparable.

## Interface

```python
embedder.name                     # identifies the vector space, e.g. "st:sentence-transformers/all-MiniLM-L6-v2"
embedder.dim                      # vector length
embedder.encode_many(passages)    # documents / chunks
embedder.encode_query(question)   # queries (falls back to encode() if absent)
```

### Query vs passage encoding

Asymmetric models (e5, bge) are trained with different prefixes for queries and passages, for example `query: ` and `passage: `. `SentenceTransformerEmbedder(query_prefix=..., passage_prefix=...)` applies them, and `SearchIndex` always embeds questions through `encode_query`. Symmetric models such as all-MiniLM-L6-v2 need no prefixes.

## Bedrock details

- Titan Text Embeddings V2 has no batch endpoint, so there is one `InvokeModel` call per chunk. The index only embeds new or changed documents (content-hash sync), which keeps the cost proportional to what changed.
- Throttling and transient errors (`ThrottlingException`, `ServiceUnavailableException`, …) are retried with exponential backoff; other errors fail immediately.
- Credentials and region come from the standard AWS chain: environment variables, a profile, or the ECS/Lambda task role in production.
- `normalize: true` is requested, so vectors are unit length, as the vector store expects.

## Testing

Unit tests never download a model or call AWS: `tests/fakes.py` provides a fake sentence-transformers model and a fake Bedrock client that records requests and can simulate throttling. One opt-in test uses the real local model:

```bash
RUN_MODEL_TESTS=1 pytest tests/test_embedding.py -q
```

## Results

On the frozen 30-question benchmark (details in `docs/11-rag-evaluation.md`), replacing the hashing embedder with `all-MiniLM-L6-v2` raises dense-only hit@1 from 0.800 to 0.933 and paraphrase questions from 0.67 to 1.00. Combined with BM25 (weighted, alpha 0.5), it reaches hit@1 = 1.000. Dense retrieval alone still misses exact-term and distractor questions that BM25 handles, which is why hybrid search remains the default.
