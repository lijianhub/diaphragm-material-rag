# RAG Evaluation

Retrieval is evaluated before generation: if the right passage is not retrieved, no prompt can produce a grounded answer.

## Run it

```bash
python scripts/evaluate.py                    # k=3, weighted fusion, alpha=0.5
python scripts/evaluate.py --k 1 --alpha 0     # lexical only, strict top-1
python scripts/evaluate.py --k 1 --fusion rrf  # Reciprocal Rank Fusion
```

The script also prints metrics per question category, so a weakness in one kind of question is not averaged away.

The script indexes the tracked `data/raw/*.txt` samples, runs every question in `evaluation/questions.jsonl`, prints the aggregate metrics and the misses, and writes per-question results to `evaluation/results.jsonl`.

## Ground truth: evidence phrases, not chunk ids

Each entry in `evaluation/ground_truth.jsonl` lists one or more **evidence phrases** that appear verbatim in the corpus:

```json
{"id": "q09", "answer": "Dies are compensated by over-forming the profile by a small angle.",
 "evidence": ["over-forming the profile by a small angle"]}
```

A retrieved chunk is relevant if it contains an evidence phrase, ignoring case and whitespace. Chunk ids change whenever chunk size, overlap or the chunking algorithm changes; evidence phrases do not, so one dataset can compare chunking strategies. The tradeoff is that a phrase split across a chunk boundary counts as a miss, which is a real retrieval failure, because the chunk does not hold the full evidence.

Questions are paraphrased rather than copied from the text, so pure keyword matching cannot solve them trivially. `tests/test_evaluation.py` checks that every evidence phrase still exists in the corpus.

## Metrics

| Metric | Meaning |
|---|---|
| hit@k | Share of questions with at least one relevant chunk in the top k |
| recall@k | Share of a question's evidence phrases covered by the top k, averaged over questions |
| MRR | Mean of 1 / rank of the first relevant chunk (0 if none is retrieved) |

## Question categories

| Category | What it tests | Example |
|---|---|---|
| `keyword` | The question shares distinctive terms with the evidence | "What is galling?" |
| `paraphrase` | Different words from the source (synonyms, informal wording) | "too much forming **oil**" vs "too much **lubricant**" |
| `distractor` | Several chunks share the question's terms; only one holds the answer | "Why must parts be cleaned before aging?" (aging also appears in the heat-treatment notes) |
| `multi` | Evidence spread over two or more documents; measured by recall@k | "How do cobalt alloys behave differently from ordinary steels?" |

## Results log

Entries are in chronological order. Earlier entries used the first benchmark of 14 questions over 4 documents; from "Expanded benchmark and RRF" onwards it has 30 questions over 8 documents. All runs use `RecursiveChunker(chunk_size=300, overlap=50)`.

### Baseline (initial scaffold)

| | hit | recall | MRR |
|---|---|---|---|
| k=1 | 0.929 | 0.893 | 0.929 |
| k=3 | 0.929 | 0.893 | 0.929 |

The k=3 scores equal the k=1 scores because one question (q13) could never be retrieved. Diagnosis:

1. **Chunking dropped text.** After a separator split, the next chunk started at `start + chunk_size - overlap` instead of `end - overlap`, which skipped 269 characters across the corpus. The answer to q13 was in one of those gaps.
2. **The dense term ignored the query.** The hybrid score added the mean of the *document* vector, a constant per chunk, so only lexical overlap affected the ranking. The embedder's vectors also had variable length, so cosine similarity was impossible.
3. **Lexical matching used substrings**, so `"form"` matched `"information"`.

### After the fixes

Changes: no-gap chunking, paragraph-aware separators, overlap starting on word boundaries, a fixed-dimension feature-hashing embedder with cosine similarity, and whole-word lexical coverage with a shared tokenizer.

| | hit | recall | MRR |
|---|---|---|---|
| k=1 | 0.857 | 0.857 | 0.857 |
| k=3 | **1.000** | **1.000** | 0.917 |

Every question now retrieves its evidence within the top 3. Top-1 accuracy dropped, and the alpha sweep (k=1) shows why:

| alpha (dense weight) | 0.0 | 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|
| hit@1 | 0.929 | 0.857 | 0.857 | 0.857 | 0.786 |

The hashing embedder is a bag of words with collisions, so it adds noise rather than meaning. Lexical-only scoring wins on this corpus. The default stays at 0.5: tuning a weight on 14 questions would overfit the test set, and the dense slot is where a real embedding model will go.

The remaining rank-1 misses show what to fix next:

- **q11** ("…how is fatigue performance *verified*?"): the source says "verify", and with no stemming the two do not match. The frequent phrase "fatigue performance" pulls in a chunk from another document. BM25's IDF weighting and light stemming address both.
- **q14**: the top hit is an overlap fragment that repeats the question's wording but not the answer; the full chunk ranks second. Overlap improves recall but creates near-duplicates. Reranking, or merging adjacent chunks, addresses this.

### BM25 and stemming

Changes: `retrieval/bm25.py` (Okapi BM25 with Lucene IDF, k1=1.5, b=0.75), a light suffix-stripping stemmer in `tokenization.py`, and two `SearchIndex` options, `lexical` and `stem_tokens`. BM25 scores are unbounded, so they are max-normalized per query before being blended with cosine.

Ablation, `chunk_size=300`, k=1:

| lexical | stemming | alpha 0.0 | alpha 0.5 |
|---|---|---|---|
| coverage | no | 0.929 | 0.857 |
| coverage | yes | **1.000** | **1.000** |
| bm25 | no | 0.929 | 0.929 |
| bm25 | yes | **1.000** | **1.000** |

With the new defaults (`bm25`, stemming, alpha 0.5), hit@1, recall@3 and MRR are all 1.000; MRR was 0.917 before.

- **Stemming is the change that matters.** It fixes q11 ("verified" vs "verify") and q14 ("measures" vs "measuring"), which is vocabulary mismatch that no weighting scheme can solve.
- **This benchmark cannot separate BM25 from coverage.** Both reach 1.000 with stemming. A stress test with smaller chunks (k=5, chunk sizes 150 and 200) gives MRR differences of 0.02–0.04 in either direction. With 14 questions, a single rank change moves MRR by about 0.036, so these differences are noise. With 4 documents and a small vocabulary, IDF has almost no signal to work with.
- **Why BM25 is still the default:** IDF, term-frequency saturation and length normalization are what keep lexical retrieval stable on a larger corpus with many common domain terms ("diaphragm", "alloy", "forming"), and BM25 did not regress here. The choice rests on that reasoning, not on a score difference this benchmark cannot measure.
- **Next:** the benchmark is saturated. Before comparing fusion methods (RRF), expand it with more documents, distractor passages and harder questions; otherwise every method will score 1.000.

### Expanded benchmark and RRF

**Benchmark first.** The 14-question set had saturated, with every method at 1.000. It was expanded to 30 questions over 8 documents and **frozen before RRF was implemented**, so the new questions could not be tailored to any method. The four new documents (springs, lubrication and tooling, corrosion, quality records) reuse the existing vocabulary on purpose (fatigue, aging, inspection, lot, surface) to act as distractors.

Method comparison (30 questions):

| Method | hit@1 | hit@3 | recall@3 | MRR@3 |
|---|---|---|---|---|
| BM25 only (alpha 0) | **0.967** | 1.000 | 0.989 | **0.983** |
| Dense only (alpha 1) | 0.800 | 0.967 | 0.922 | 0.872 |
| Weighted sum, alpha 0.5 (default) | 0.933 | 1.000 | 0.989 | 0.967 |
| RRF, k=60 | 0.900 | 0.967 | 0.956 | 0.933 |

By category (k=1, default): keyword 1.000 · distractor 1.000 · multi recall 0.444 (one chunk cannot hold evidence from two documents) · **paraphrase 0.667**.

**Findings**

- **Fusion with a weak retriever hurts.** The hashing embedder is a bag of words with collisions (dense only: 0.800). Blending it in, by either method, scores below BM25 alone.
- **RRF did worse than the weighted sum here.** RRF is robust to *score scale*, not to *retriever quality*: it gives the weak retriever an equal vote and discards how confident each retriever was. Worked example, q11:

  | Chunk | dense rank | BM25 rank | RRF = 1/(60+r₁) + 1/(60+r₂) |
  |---|---|---|---|
  | correct (pressure cycling tests) | 3 (cos 0.197) | **1** (BM25 7.39) | 0.03227 |
  | distractor (fatigue performance) | **1** (cos 0.281) | 3 (BM25 4.74) | 0.03227 |

  The scores tie exactly, and the tie is broken by chunk order. BM25's large margin (7.39 vs 4.74) is invisible to RRF.
- **Sensitivity (analysis only, not used to pick defaults):** RRF gets worse as the dense weight rises (alpha 0.2: 0.933 → 0.7: 0.833), and `rrf_k` between 10 and 100 makes no difference.
- **Paraphrase is the real gap** (q22 "rust in the sea" vs "oxide film … seawater", q24 "passivation" vs "passivated"). No lexical method and no bag-of-words embedder can bridge synonyms. That is the job of a real embedding model, query rewriting (step 7) or reranking (step 6).

**Decision.** The defaults stay as they are (weighted, alpha 0.5). Switching to BM25 only, or to alpha 0.3, would be tuning on the test set, and the dense slot is reserved for a real embedding model. RRF stays available as `--fusion rrf`; it should be re-evaluated once the dense retriever is comparable in quality to BM25, which is the setting where RRF is usually recommended.
