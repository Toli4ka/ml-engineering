from pathlib import Path

import streamlit as st

from financial_rag_assistant.pipeline.rag_pipeline import RAGSystemBuilder


PDF_PATH = Path("data/infineon_report.pdf")
INDEX_DIR = Path("storage/my_pdf_index")

def main() -> None:
    st.set_page_config(
        page_title="PDF RAG Assistant",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 PDF RAG Assistant")
    st.write("Ask questions about your PDF document using retrieval-augmented generation.")

    with st.sidebar:
        st.header("Settings")
        st.write(f"**Document:** `{PDF_PATH.name}`")
        st.write(f"**Index folder:** `{INDEX_DIR}`")

    try:
        rag_builder = RAGSystemBuilder(PDF_PATH)
        rag_pipeline = rag_builder.build_pipeline()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return

    question = st.text_input(
        "Enter your question:",
        placeholder="What risks does the company mention?"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_clicked = st.button("Ask")
    with col2:
        clear_clicked = st.button("Clear")

    if clear_clicked:
        st.rerun()

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Searching document and generating answer..."):
            try:
                response = rag_pipeline.ask(question, k=5)
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
                return

        st.subheader("Answer")
        st.write(response.answer)

        st.subheader("Sources")
        for i, source in enumerate(response.sources, start=1):
            with st.expander(
                f"Source {i} — page {source.page_number} — score {source.score:.4f}"
            ):
                st.markdown(f"**Chunk ID:** `{source.chunk_id}`")
                st.write(source.text)


if __name__ == "__main__":
    main()
