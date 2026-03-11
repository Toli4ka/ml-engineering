from dataclasses import dataclass
from typing import List

from openai import OpenAI

from financial_rag_assistant.ingestion.text_splitter import TextChunk


@dataclass
class EmbeddedChunk:
    chunk_id: str
    page_number: int
    text: str
    embedding: List[float]


def embed_chunks(
    chunks: List[TextChunk],
    client: OpenAI,
    model: str = "text-embedding-3-small",
    batch_size: int = 32
) -> List[EmbeddedChunk]:
    """
    Generate embeddings for a list of text chunks in batches.
    """
    if not chunks:
        raise ValueError("No chunks provided for embedding.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    embedded_chunks: List[EmbeddedChunk] = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [chunk.text for chunk in batch]

        response = client.embeddings.create(
            model=model,
            input=texts
        )

        for chunk, item in zip(batch, response.data):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    embedding=item.embedding
                )
            )

    return embedded_chunks

def embed_query(
    query_text: str,
    client: OpenAI,
    model: str = "text-embedding-3-small"
) -> List[float]:
    """
    Generate an embedding for a single query string.
    """
    if not query_text.strip():
        raise ValueError("Query text must not be empty.")

    response = client.embeddings.create(
        model=model,
        input=[query_text]
    )

    return response.data[0].embedding
