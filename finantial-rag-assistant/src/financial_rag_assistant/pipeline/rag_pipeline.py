from dataclasses import dataclass
from typing import List
from pathlib import Path

from openai import OpenAI

from financial_rag_assistant.llm.generator import GeneratedAnswer, generate_answer
from financial_rag_assistant.retrieval.vector_store import SearchResult, VectorStore
from financial_rag_assistant.ingestion.pdf_loader import load_pdf
from financial_rag_assistant.ingestion.text_splitter import split_text
from financial_rag_assistant.config import get_openai_client
from financial_rag_assistant.ingestion.embedder import embed_chunks, embed_query


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: List[SearchResult]

class RAGSystemBuilder:
    _cache: dict[str, tuple[VectorStore, OpenAI]] = {}

    def __init__(self, pdf_path: Path) -> None:
        self.pdf_path = Path(pdf_path)
        self.client: OpenAI | None = None
        self.vector_store: VectorStore | None = None

    def build(self) -> tuple[VectorStore, OpenAI]:
        store_dir = Path("storage/my_pdf_index")

        try:
            vector_store = VectorStore.load(store_dir)
            client = get_openai_client()
            print("Loaded existing FAISS index from disk.")
        except FileNotFoundError:
            pages = load_pdf("data/infineon_report.pdf")
            chunks = split_text(pages, chunk_size=800, chunk_overlap=150)
            client = get_openai_client()
            embedded_chunks = embed_chunks(chunks, client=client, batch_size=32)

            vector_store = VectorStore()
            vector_store.build_index(embedded_chunks)
            vector_store.save(store_dir)
            print("Built and saved FAISS index.")

        self.vector_store = vector_store
        self.client = client

        return vector_store, client

    def build_pipeline(self) -> "RAGPipeline":
        vector_store, client = self.build()
        return RAGPipeline(client=client, vector_store=vector_store)


class RAGPipeline:
    def __init__(
        self,
        client: OpenAI,
        vector_store: VectorStore,
        embedding_model: str = "text-embedding-3-small",
        generation_model: str = "gpt-4.1-mini",
    ) -> None:
        self.client = client
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.generation_model = generation_model

    def ask(self, question: str, k: int = 5) -> RAGResponse:
        """
        Answer a user question using retrieval-augmented generation.

        Args:
            question: User question.
            k: Number of chunks to retrieve.

        Returns:
            RAGResponse containing the answer and used sources.
        """
        if not question.strip():
            raise ValueError("Question must not be empty.")

        query_embedding = embed_query(
            query_text=question,
            client=self.client,
            model=self.embedding_model,
        )

        search_results = self.vector_store.search(query_embedding=query_embedding, k=k)

        generated: GeneratedAnswer = generate_answer(
            question=question,
            search_results=search_results,
            client=self.client,
            model=self.generation_model,
        )

        return RAGResponse(
            question=question,
            answer=generated.answer,
            sources=generated.sources,
        )
