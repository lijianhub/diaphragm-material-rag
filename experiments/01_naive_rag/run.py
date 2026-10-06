from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generation.rag_chain import RAGPipeline
from ingestion.loader import load_documents


def main() -> None:
    source_file = REPO_ROOT / "data" / "raw" / "diaphragm_material_notes.txt"

    documents = load_documents(source_file)
    content = [doc.content for doc in documents]

    question = "What material properties matter for diaphragm forming?"
    result = RAGPipeline().answer(question, content)

    print("Question:", question)
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print("-", source)


if __name__ == "__main__":
    main()
