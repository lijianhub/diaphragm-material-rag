# Vector Database

Chunks and their embeddings are stored in a **vector store**, so documents are embedded once at indexing time instead of on every query.

## Build, update and query an index

```bash
python scripts/index.py                    # first run: chunk, embed and store data/raw -> data/index
python scripts/index.py                    # later runs: only new or changed files are re-embedded
python scripts/index.py --store faiss      # FAISS backend (pip install -e .[faiss])
python scripts/index.py --rebuild          # start from scratch
python scripts/query.py "How is springback compensated?"   # loads data/index
```

`data/index/` is git-ignored; every environment builds its own.

## Design

```
SearchIndex ── sync / save / load / search
   │
   ├── VectorStore  (abstract: add, delete_document, search, save, load)
   │     ├── InMemoryVectorStore   pure Python, exact; reference implementation
   │     └── FaissVectorStore      IndexIDMap2(IndexFlatIP), exact cosine; optional dependency
   │
   └── BM25         rebuilt in memory from the stored chunk texts
```

- **One interface, swappable backends.** `SearchIndex` only calls the `VectorStore` methods, so a managed service (OpenSearch k-NN, pgvector) can be added without touching retrieval code.
- **Cosine everywhere.** Stores L2-normalize vectors on insert and query, so the inner product equals cosine similarity even if an embedder does not return unit vectors.
- **Chunk records carry metadata.** Each record holds `id` (`<source>#<n>`), `doc_id`, text, offsets and a `metadata` dict (currently `doc_type`), so results can be cited and later filtered.
- **Dense candidates.** A query asks the store for its `dense_candidates` nearest chunks (default 100), as it would with an approximate index; chunks outside that set get a dense score of 0 before fusion with BM25.

## Persistence layout

```
data/index/
├── manifest.json        format version, embedder name and dim, chunker settings,
│                        search options, and a SHA-256 hash per document
└── store/
    ├── store.json       backend kind, dimension, count
    ├── records.jsonl    one chunk record per line
    └── vectors.json     (memory)   or   index.faiss + faiss_ids.json   (FAISS)
```

## Incremental sync

`SearchIndex.sync(documents)` compares each document's SHA-256 with the manifest:

| Situation | Action |
|---|---|
| New source | chunk, embed, add |
| Same source, different hash | delete its old chunks, then re-add |
| Same source, same hash | skip (no embedding cost) |
| In the index but not in the input | delete its chunks |

Sources are stored relative to the repository with forward slashes, so an index built on one machine or container matches the files on another.

## Embedding-model safety

The manifest records the embedder's `name` (for example `simple-hash-512`). `SearchIndex.load` refuses to open an index with a different embedder, because vectors from different models live in different spaces and their similarities are meaningless. A model change requires `--rebuild`.

## Exact vs approximate search

Both backends here are **exact** and return identical rankings. On the benchmark, `--store memory` and `--store faiss` give the same scores. Scale is the difference (512-dim vectors, top-10, average of 5 queries, on a laptop):

| Chunks | In-memory query | FAISS flat query | Same top-10 |
|---|---|---|---|
| 1,000 | 27 ms | 2.4 ms | yes |
| 10,000 | 278 ms | 4.7 ms | yes |
| 50,000 | 1,817 ms | 4.5 ms | yes |

Beyond a few million vectors, exact search becomes too slow or too memory-hungry, and an **approximate** (ANN) index is used instead:

| Index | Idea | Trade-off |
|---|---|---|
| HNSW | Layered proximity graph; greedy search from coarse to fine layers | Very fast, high recall; more memory; deletes are hard |
| IVF | Cluster vectors (k-means); search only the closest `nprobe` clusters | Less memory; needs training; recall depends on `nprobe` |
| PQ | Compress vectors into short codes | Huge memory savings; lower precision; often combined with IVF |

Switching to an ANN index must be validated with `scripts/evaluate.py`: approximate search can miss true neighbours, and that shows up as lower recall.
