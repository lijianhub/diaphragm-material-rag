# Experiment Tracking and Versioning

A score is only useful if you can say exactly what produced it. "hit@1 = 0.933" means nothing without the embedder, fusion settings, prompt, model, code commit and benchmark version behind it. Every evaluation run can therefore be recorded with its configuration, its versions and its results.

## Run it

```bash
pip install -e .[tracking]                                   # MLflow (optional; the core needs nothing)

python scripts/evaluate.py --k 3 --track mlflow
python scripts/evaluate_generation.py --llm openai:<model>@low --track mlflow
python scripts/compare_runs.py                               # side-by-side table in the terminal
mlflow ui --backend-store-uri sqlite:///mlflow.db            # browse and plot runs at http://127.0.0.1:5000
```

`--track jsonl` writes the same record to `runs/runs.jsonl` with no dependencies; `--track none` (the default) records nothing.

## What a run records

| Part | Content | Example |
|---|---|---|
| **params** (the configuration) | retrieval: `k`, `alpha`, `lexical`, `stem`, `fusion`, `rrf_k`, `embedder`, `reranker`, `rewrite`, `chunk_size`, `overlap`, counts of questions, documents and chunks · generation: `answerer`, `served_models`, `prompt_version`, `embedder`, `top_k`, `cases` | `embedder = st:sentence-transformers/all-MiniLM-L6-v2` |
| **metrics** (the scores) | retrieval: `hit_at_k`, `recall_at_k`, `mrr_at_k`, the same per category, LLM calls · generation: every generation metric, correct share per case type, one `gate.<name>` (1 = pass) per gate, `gates_passed`, tokens | `paraphrase.hit_at_3 = 0.917`, `gates_passed = 6` |
| **tags** (the versions) | `git_commit`, `git_dirty`, `data.<file>.sha256` for each evaluation file, `data.corpus.fingerprint`, `data.corpus.files`, generation `gates = pass/fail` | `git_dirty = false`, `data.questions.sha256 = 4b84df2e…` |
| **artifact** | the per-question or per-case results file | `r_minilm_k3.jsonl` |

Metric names are made MLflow-safe (`hit@3` → `hit_at_3`), and long parameter values are cut at 500 characters rather than failing a run.

## Versioning strategy

| What changes | How it is versioned | Recorded as |
|---|---|---|
| Code | git commit, plus a flag when the working tree has uncommitted changes | tags `git_commit`, `git_dirty` |
| Prompt | `PROMPT_LABEL` (bumped on purpose) + SHA-256 fingerprint of the system prompt | param `prompt_version`, e.g. `grounded-v2+5647c36b` |
| LLM | the requested spec **and** the model that actually served the answers | params `answerer`, `served_models` |
| Embedder | its name, which is also a spec that rebuilds it (`simple-hash-512`, `st:<model>`) | param `embedder` |
| Retrieval configuration | every command-line option of `evaluate.py` | params |
| Evaluation data | SHA-256 of `questions.jsonl`, `ground_truth.jsonl`, `generation_cases.jsonl` | tags `data.*.sha256` |
| Corpus | one fingerprint over the benchmark files' names and contents | tags `data.corpus.fingerprint`, `data.corpus.files` |
| Results | the per-question file of that run | artifact |

**Rules that follow from it:**

1. **Commit before a tracked run.** A run on a dirty tree is recorded but flagged (`git_dirty = true`, and the script says so), because its code may not exist in any commit.
2. **Compare only runs with the same data hashes.** When a benchmark is expanded, its hash changes, and older runs belong to a different comparison group. `compare_runs.py` prints the questions hash on every row for this reason.
3. **Write tracked results outside the repository** (`--output runs/<name>.jsonl`, git-ignored), so the first run of a series does not dirty the tree for the next one.
4. **Bump `PROMPT_LABEL` on intentional prompt edits;** the fingerprint catches the ones nobody labelled.

## Where runs are stored

| Store | Location | When |
|---|---|---|
| MLflow, local | `mlflow.db` (SQLite) and `mlartifacts/` at the **repository root**, git-ignored | default for `--track mlflow` |
| MLflow, shared | whatever `MLFLOW_TRACKING_URI` points to (e.g. a tracking server) | team or CI use |
| JSONL | `runs/runs.jsonl`, git-ignored | no MLflow installed |

MLflow 3 otherwise creates its store in the *current* directory, so runs would scatter depending on where a script is launched. The tracker pins the store and the artifact folder to the repository root.

## Recorded runs (commit `2677ae9`, clean tree, same benchmark hashes)

Produced by `python scripts/compare_runs.py`.

| Run | hit@1 | hit@3 | recall@3 | MRR@3 | paraphrase hit@3 |
|---|---|---|---|---|---|
| BM25 only | 0.933 | 0.978 | 0.959 | 0.952 | 0.917 |
| hashing hybrid (offline default) | 0.889 | 0.956 | 0.937 | 0.919 | 0.833 |
| MiniLM hybrid | 0.933 | 0.978 | 0.952 | 0.956 | 0.917 |
| MiniLM hybrid + multi-query rewriting | 0.933 | **1.000** | 0.963 | 0.967 | **1.000** |

| Run | answer accuracy | refusal accuracy | injection resistance | ambiguity | gates passed |
|---|---|---|---|---|---|
| extractive (offline) | 0.756 | 0.000 | 1.000 | 0.500 | 4/6 |
| `gpt-5.4-mini@low`, `grounded-v2` | 0.933 | 1.000 | 1.000 | 0.333 | 6/6 |

These reproduce the numbers in [11-rag-evaluation.md](11-rag-evaluation.md) exactly; the LLM rewrites and answers came from the response cache, so recording them cost no tokens.

## Design decisions

| Decision | Alternatives | Why |
|---|---|---|
| A `Tracker` interface with null, JSONL and MLflow backends | Call MLflow directly from the scripts | The core and the tests stay free of MLflow; a different backend (a team server, another tool) is one class |
| Tracking failures print a warning and the evaluation completes | Let the exception propagate | Tracking is observability. A tracking server being down must not cost an evaluation, which may have spent tokens |
| Versions as tags, configuration as params | Everything as params | Tags describe *what the run was measured on*; params describe *what was tried*. The split keeps comparisons readable in the UI |
| Content hashes for data | File modification times, file names | Hashes change exactly when content changes, on every machine and after a clone |
| SQLite store | MLflow's file store, a tracking server | A single file, no server process, works offline; a server is one environment variable away |
| One run per evaluation, with aggregate metrics | One run per question, or tracing every LLM call | Runs answer "which configuration is better"; per-request tracing belongs to the service layer (request IDs, structured logs, step 16) |

## Limitations

- **The environment is not recorded:** Python and package versions (e.g. sentence-transformers, the OpenAI SDK) are not yet tags, so a dependency upgrade that changes scores would not be visible in the run.
- **The index is not versioned as an artifact:** each evaluation rebuilds the index from the corpus, which the corpus fingerprint covers, but a saved index (`data/index`) has no link to runs.
- **No model registry:** prompts and configurations are versioned by label and hash, not promoted through stages.
- **Local only so far:** no shared tracking server; CI will log to one later.
