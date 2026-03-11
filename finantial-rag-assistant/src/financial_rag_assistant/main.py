from financial_rag_assistant.pipeline.rag_pipeline import RAGSystemBuilder


def main() -> None:
    pdf_path = "data/infineon_report.pdf"

    rag_builder = RAGSystemBuilder(pdf_path=pdf_path)
    rag_pipeline = rag_builder.build_pipeline()

    question = "What is the total revenue?"
    response = rag_pipeline.ask(question, k=5)

    print("\nQuestion:")
    print(response.question)

    print("\nAnswer:")
    print(response.answer)

    print("\nSources:")
    for source in response.sources:
        print(
            f"- {source.chunk_id} | page {source.page_number} | score {source.score:.4f}"
        )


if __name__ == "__main__":
    main()
