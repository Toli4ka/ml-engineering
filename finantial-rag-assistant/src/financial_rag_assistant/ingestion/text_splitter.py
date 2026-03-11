from dataclasses import dataclass
from typing import List

from financial_rag_assistant.ingestion.pdf_loader import PDFPage


@dataclass
class TextChunk:
    chunk_id: str
    page_number: int
    text: str


def split_text(
    pages: List[PDFPage],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[TextChunk]:
    """
    Split PDF pages into overlapping text chunks, preferring whitespace boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[TextChunk] = []
    step = chunk_size - chunk_overlap

    for page in pages:
        text = page.text.strip()

        if not text:
            continue

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # avoid cutting words
            if end < len(text):
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"page_{page.page_number}_chunk_{chunk_index}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        page_number=page.page_number,
                        text=chunk_text
                    )
                )

            start += step
            chunk_index += 1

    return chunks
