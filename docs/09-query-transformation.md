# Query Transformation

Users rarely phrase questions the way documents are written. Query transformation rewrites or multiplies the question before retrieval, so that at least one query matches the document's wording.

```python
from query import decompose, retrieve, LLMQueryRewriter

retrieve(index, question, top_k=3)                                          # plain search
retrieve(index, question, transforms=[decompose], fusion="interleave")      # compound questions
retrieve(index, question, prf_terms=3)                                      # pseudo-relevance feedback
retrieve(index, question, transforms=[LLMQueryRewriter(llm)], fusion="rrf") # LLM paraphrases
retrieve(index, question, transforms=[LLMQueryRewriter(llm, mode="hyde")])  # hypothetical answer
```

```bash
python scripts/evaluate.py --k 3 --decompose              # interleaved sub-query results (default)
python scripts/evaluate.py --k 1 --prf 3                  # add 3 feedback terms
```

## Methods

| Method | What it does | Needs |
|---|---|---|
| `decompose` | Splits compound questions at ", and what …", " and how …" (a conjunction followed by a question word). "Strength and ductility" is not split. The original question is kept. | Nothing (rules) |
| Pseudo-relevance feedback (`prf_terms`) | Searches once, takes the top `prf_docs` passages, and adds the `prf_terms` terms that are frequent there but rare in the corpus (IDF), excluding words already in the question. Terms are matched by stem but added as real words. | Nothing |
| `LLMQueryRewriter(mode="multi-query")` | Asks a language model for `n_variants` rephrasings in document language | A text-in, text-out LLM function |
| `LLMQueryRewriter(mode="hyde")` | Asks a language model to write a short passage that would answer the question, and searches for it alongside the question (Hypothetical Document Embeddings) | A text-in, text-out LLM function |

The LLM rewriter accepts any `complete(prompt) -> str` function, so a local model, the Anthropic API or Amazon Bedrock can be plugged in. If the call fails, it falls back to the original question.

## How multiple queries are combined

Each query is searched with the first stage only (`search(..., rerank=False)`). The rankings are then fused, and if the index has a reranker, the fused candidates are reranked **against the original question**, never against a variant, so a drifting rewrite cannot change what counts as relevant.

The fusion method must match what the queries are:

| Queries are… | Want | Fusion |
|---|---|---|
| Rephrasings of **one** question (multi-query, HyDE) | Consensus: chunks that several phrasings agree on | `rrf` |
| **Different** sub-questions (decomposition) | Coverage: each part's best evidence | `interleave` (round-robin: every list's #1, then every #2, …) |

With decomposition, RRF fails in a specific way. A generic chunk that ranks 5th–10th for *every* sub-query sums several small contributions (≈ 1/62 + 1/66 + 1/67 = 0.046) and beats the chunk that is **first** for two of them (1/61 + 1/61 = 0.033). On the benchmark, RRF fusion dropped multi-document recall@3 from 0.833 to 0.633; interleaving restored it.

## Results

Frozen 45-question benchmark (v3). hit@1 / MRR@3 / recall@3:

| | Hashing embedder | MiniLM |
|---|---|---|
| None (baseline) | 0.889 / 0.919 / 0.937 | 0.933 / 0.956 / 0.952 |
| Decompose, interleave | 0.889 / 0.919 / 0.937 (unchanged) | unchanged |
| Decompose, RRF | 0.867 / 0.896 / 0.915 (multi recall 0.833 → 0.633) | unchanged |
| PRF, 3 terms | **0.600** / 0.759 / 0.915 | **0.689** / 0.822 / 0.941 |
| PRF, 5 terms | **0.533** / 0.711 / 0.915 | **0.578** / 0.763 / 0.952 |
| Any of the above + reranker | 0.933 / 0.952 / 0.952 | 0.933 / 0.952 / 0.952 |

**Findings**

- **Pseudo-relevance feedback drifts.** It fixed a few paraphrase questions (q22, q35, q37 with hashing; q37, q41 with MiniLM) but broke 13–19 others at rank 1. When the first pass is partly wrong or the top passages cover several topics, the added terms pull the query toward them. Recall@3 barely moves; rank 1 suffers. With 11 short documents there is little to feed back from, and PRF typically does better on large corpora with long documents.
- **The reranker neutralizes query changes** because it scores the original question: every transformation ends at the same 0.933. That is the safety net working as designed, and it also means the transformations add nothing once a reranker is on.
- **Decomposition is correct but has no room here.** Only two benchmark questions are compound (q10, q44), and plain search already retrieves both parts within the top 3.
- **q33 is still unsolved** ("Can the parts be taken out into the air straight after aging?" vs "cool the parts in the furnace atmosphere before exposing them to air"). Bridging that wording gap is what multi-query rewriting and HyDE are for. They are implemented and unit-tested with a fake model, and will be measured once a real LLM backend is connected.

**Decision.** No transformation is enabled by default. Decomposition (with interleave) is safe to turn on for corpora with many compound questions; PRF should be measured before use; LLM rewriting is the next experiment.
