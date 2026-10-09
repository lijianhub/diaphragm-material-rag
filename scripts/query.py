"""Ask a question against the documents in data/raw.

Usage:
    python scripts/query.py "How is springback compensated?"
    python scripts/query.py "Why age after forming?" --top-k 2 --source data/raw
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
from ingestion.loader import load_documents
from retrieval.index import SearchIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "data" / "raw", help="file or directory to index")
    args = parser.parse_args()

    index = SearchIndex()
    index.add_documents(load_documents(args.source))
    results = index.search(args.question, top_k=args.top_k)
    if not results:
        sys.exit(f"No documents found in {args.source}")

    answer = RAGPipeline().answer(args.question, [result.text for result in results])

    print("Question:", args.question)
    print("\nAnswer:\n" + answer["answer"])
    print("\nSources:")
    for result in results:
        print(f"- {Path(result.source).name} [{result.start}:{result.end}] score={result.score:.3f}")


if __name__ == "__main__":
    main()
