from generation.rag_chain import RAGPipeline


def test_rag_pipeline_answer_contains_context_and_sources():
    docs = [
        "Diaphragm forming involves controlled pressure and metal flow.",
        "RAG systems retrieve relevant passages before generating answers.",
    ]

    pipeline = RAGPipeline()
    result = pipeline.answer("What is diaphragm forming?", docs)

    assert "diaphragm" in result["answer"].lower()
    assert result["sources"]
    assert "context" in result
