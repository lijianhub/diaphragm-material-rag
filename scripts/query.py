"""Ask a question against the indexed documents.

Uses the persistent index in data/index (build it with scripts/index.py). If no
index exists yet, it indexes --source in memory for this run only.

Usage:
    python scripts/query.py "How is springback compensated?"
    python scripts/query.py "Why age after forming?" --top-k 2
    python scripts/query.py "..." --source data/raw --no-index   # ignore the saved index
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generation.rag_chain import RAGPipeline
from ingestion.loader import ingest
from reranking import create_reranker
from retrieval.index import MANIFEST_FILE, SearchIndex


def citation(result) -> str:
    """"manual.pdf, page 3" / "lots.xlsx, sheet Q3" / "guide.md, section Cooling"."""
    parts = [Path(result.source).name]
    meta = result.metadata
    if "page" in meta:
        parts.append(f"page {meta['page']}")
    if "sheet" in meta:
        parts.append(f"sheet {meta['sheet']}")
    if "heading" in meta:
        parts.append(f"section \"{meta['heading']}\"")
    return ", ".join(parts) + f" [{result.start}:{result.end}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--index", type=Path, default=REPO_ROOT / "data" / "index", help="saved index directory")
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "data" / "raw", help="documents to use when there is no saved index")
    parser.add_argument("--no-index", action="store_true", help="always index --source in memory")
    parser.add_argument("--reranker", default=None, help="override the index's reranker, e.g. ce:cross-encoder/ms-marco-MiniLM-L-6-v2")
    args = parser.parse_args()

    if not args.no_index and (args.index / MANIFEST_FILE).exists():
        index = SearchIndex.load(args.index)
    else:
        print(f"(no saved index at {args.index}; indexing {args.source} in memory. Run scripts/index.py to persist it.)\n")
        index = SearchIndex()
        index.sync(ingest(args.source).documents)

    if args.reranker:
        index.reranker = create_reranker(args.reranker)
    if len(index) == 0:
        sys.exit("The index is empty.")

    # No language model is connected yet, so answers are extractive quotes with citations.
    answer = RAGPipeline(index, top_k=args.top_k).answer(args.question)

    print("Question:", args.question)
    print("\nAnswer:\n" + answer.text)
    if answer.invalid_citations:
        print(f"\n(warning: the answer cites sources that do not exist: {answer.invalid_citations})")
    print("\nSources (* = cited):")
    cited = {c.number for c in answer.citations}
    for number, result in enumerate(answer.sources, start=1):
        marker = "*" if number in cited else " "
        print(f"{marker}[{number}] {citation(result)}  score={result.score:.3f}")


if __name__ == "__main__":
    main()
