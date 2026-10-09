"""Build or update the persistent search index.

The first run chunks, embeds and stores every document. Later runs are
incremental: only new or changed files are re-embedded, and files that were
deleted from the source folder are removed from the index.

Usage:
    python scripts/index.py                       # sync data/raw -> data/index
    python scripts/index.py --store faiss         # FAISS-backed store (pip install faiss-cpu)
    python scripts/index.py --rebuild             # discard the existing index first
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingestion.loader import Document, ingest
from retrieval.index import MANIFEST_FILE, SearchIndex
from vectorstore import STORE_KINDS


def _portable(document: Document, base: Path) -> Document:
    """Store sources relative to the repo with forward slashes, so an index built
    on one machine (or in a container) still matches the files on another."""
    path = Path(document.source).resolve()
    source = path.relative_to(base).as_posix() if path.is_relative_to(base) else path.as_posix()
    return dataclasses.replace(document, source=source)  # keep metadata and sections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "data" / "raw", help="file or directory to index")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "index", help="index directory")
    parser.add_argument("--store", choices=STORE_KINDS, default="memory", help="vector store for a new index")
    parser.add_argument("--rebuild", action="store_true", help="delete the existing index and build from scratch")
    args = parser.parse_args()

    if args.rebuild and (args.out / MANIFEST_FILE).exists():
        shutil.rmtree(args.out)
        args.out.mkdir(parents=True, exist_ok=True)

    if (args.out / MANIFEST_FILE).exists():
        index = SearchIndex.load(args.out)
        print(f"loaded index from {args.out} ({len(index)} chunks, store={index.store.kind})")
    else:
        index = SearchIndex(store=args.store)
        print(f"creating new index (store={args.store})")

    started = time.perf_counter()
    result = ingest(args.source)
    print(f"ingest: {result.summary()}")
    for failure in result.failures:
        print(f"  FAILED  {failure.source}: {failure.reason}")
    for duplicate, original in result.duplicates:
        print(f"  duplicate  {duplicate} has the same content as {original}")
    if result.failures:
        # Never delete a failed file's previous chunks just because it could not be read this time.
        print("  (files that failed keep their previously indexed version)")

    documents = [_portable(document, REPO_ROOT) for document in result.documents]
    failed = {_portable(Document(source=f.source, content=""), REPO_ROOT).source for f in result.failures}
    keep = [doc_id for doc_id in index.document_ids() if doc_id in failed]
    stats = index.sync(documents, keep=keep)
    index.save(args.out)

    print(f"documents={len(documents)} {stats}")
    print(f"index now has {len(index.document_ids())} documents, {len(index)} chunks; saved to {args.out}")
    print(f"took {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
