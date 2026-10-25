from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
# Always first, also after `pip install -e .` (which appends src/ at the end): otherwise this
# script's own folder wins and `import query` / `import agent` find the scripts of that name.
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
    print(result.text)
    print("\nCited passages:")
    for citation in result.citations:
        print(f"- [{citation.number}] {citation.text[:80]}...")


if __name__ == "__main__":
    main()
