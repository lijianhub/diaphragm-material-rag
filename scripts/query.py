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
    args = parser.parse_args()

    if not args.no_index and (args.index / MANIFEST_FILE).exists():
        index = SearchIndex.load(args.index)
    else:
        print(f"(no saved index at {args.index}; indexing {args.source} in memory. Run scripts/index.py to persist it.)\n")
        index = SearchIndex()
        index.sync(ingest(args.source).documents)

    results = index.search(args.question, top_k=args.top_k)
    if not results:
        sys.exit("The index is empty.")

    answer = RAGPipeline().answer(args.question, [result.text for result in results])

    print("Question:", args.question)
    print("\nAnswer:\n" + answer["answer"])
    print("\nSources:")
    for result in results:
        print(f"- {citation(result)}  score={result.score:.3f}")


if __name__ == "__main__":
    main()
