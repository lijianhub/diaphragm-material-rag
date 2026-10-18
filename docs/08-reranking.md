# Reranking

Reranking adds a second, more precise stage to retrieval. The first stage (BM25 + embeddings) is fast because it scores the query and each passage independently. A **cross-encoder** reads the query and a passage *together* and judges whether the passage answers the question. It is too slow to run over a whole corpus, so it reorders only the first stage's top candidates.

```
question ─► first stage: hybrid retrieval over all chunks ─► top rerank_depth (20)
         ─► optional: collapse duplicate overlapping chunks
         ─► second stage: reranker scores each (question, passage) pair ─► top-k
```

## Usage

```bash
pip install -e .[local-embeddings]          # sentence-transformers provides the CrossEncoder
python scripts/evaluate.py --k 1 --reranker ce:cross-encoder/ms-marco-MiniLM-L-6-v2
python scripts/query.py "Why don't these alloys rust in the sea?" --reranker ce:cross-encoder/ms-marco-MiniLM-L-6-v2
python scripts/index.py --reranker ce:cross-encoder/ms-marco-MiniLM-L-6-v2   # record it in the index manifest
```

| Name (spec) | Class | Runs |
|---|---|---|
| `ce:<hugging-face cross-encoder>` | `CrossEncoderReranker` | Local CPU/GPU (`[local-embeddings]` extra) |
| `bedrock-rerank:<model-id>` | `BedrockReranker` | Amazon Bedrock Rerank API, e.g. `amazon.rerank-v1:0` (`[aws]` extra) |

In code: `SearchIndex(reranker=create_reranker(name), rerank_depth=20)`. Results then carry the reranker's score in `score`, and the first-stage score in `metadata["retrieval_score"]`.

The Bedrock reranker sends documents inline through the `bedrock-agent-runtime` `Rerank` API and maps the relevance scores back to input order. It is covered by tests with a fake client but has not yet been run against a live AWS account.

## Duplicate overlapping chunks

Chunk overlap creates **fragments**: a short tail that lies almost entirely inside a neighbouring full chunk. Fragments often outrank the full chunk (shorter text scores higher on BM25 and cosine) while carrying less context. `dedupe_overlaps=True` collapses duplicates before reranking:

- Two chunks of the same section are duplicates when their overlap covers **at least half of the shorter one**.
- Each group keeps the best rank position but the **longest** span.
- Neighbouring chunks that share only the configured overlap border (about 50 of 300 characters) are **not** duplicates.

That last rule matters. An earlier version treated *any* overlap as a duplicate, which merged every pair of adjacent chunks and dropped benchmark hit@1 from 0.889 to 0.422.

## Results

Frozen 45-question benchmark over the tracked txt, md, html and csv samples (40 chunks). Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, depth 20, CPU.

hit@1 (MRR@3):

| First stage | Without reranking | With reranking |
|---|---|---|
| Hashing embedder, dense only | 0.689 (0.796) | **0.911** (0.930) |
| Hashing embedder, weighted hybrid (default) | 0.889 (0.919) | **0.933** (0.952) |
| Hashing embedder, RRF | 0.822 (0.885) | **0.933** (0.952) |
| MiniLM, dense only | 0.844 (0.904) | **0.933** (0.952) |
| MiniLM, weighted hybrid | 0.933 (0.956) | 0.933 (0.952) |
| MiniLM, RRF | 0.933 (0.956) | 0.933 (0.952) |
| BM25 only | 0.933 (0.952) | 0.933 (0.952) |

Latency per query: about 2–10 ms without reranking, **75–110 ms** with reranking 20 candidates on a laptop CPU.

**Findings**

- **Reranking lifts weak first stages to the reranker's ceiling.** With the offline hashing embedder, it adds 4–22 points of hit@1. With any first stage, the reranked result is the same 0.933 with the same three misses, because the cross-encoder decides the order once the right chunk is among the 20 candidates.
- **It does not beat a good hybrid here.** MiniLM + BM25 already reaches 0.933, and reranking changes which questions are missed (q31, q33, q44 instead of q33, q37, q41) but not how many, while adding about 70 ms per query.
- **The reranker is a general-domain model.** `ms-marco` was trained on web search. On q33 ("Can the parts be taken out into the air straight after aging?") it prefers a heat-treatment paragraph that mentions aging over the procedure step "cool the parts in the furnace atmosphere before exposing them to air". A domain-tuned or larger reranker could do better; this one is a baseline.
- **Duplicate collapsing changed nothing on this benchmark.** The corpus has one overlap fragment in 40 chunks. It stays off by default, available for corpora with longer paragraphs or more overlap.

**Decision.** Reranking stays **off by default**: tests and the default demo must run offline, and with a good hybrid first stage it costs latency without a measured gain here. Turn it on when the first stage is cheap (hashing or BM25 only), when answers must be at rank 1, or with a stronger or domain-tuned reranker, and measure on the benchmark.
