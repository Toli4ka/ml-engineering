import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np

from financial_rag_assistant.ingestion.embedder import EmbeddedChunk


@dataclass
class SearchResult:
    chunk_id: str
    page_number: int
    text: str
    score: float


class VectorStore:
    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunks: List[EmbeddedChunk] = []
        self.embedding_dimension: int | None = None

    def build_index(self, embedded_chunks: List[EmbeddedChunk]) -> None:
        """
        Build a FAISS index from embedded chunks.

        Args:
            embedded_chunks: List of chunks with embeddings.

        Raises:
            ValueError: If no chunks are provided or embedding sizes are inconsistent.
        """
        if not embedded_chunks:
            raise ValueError("No embedded chunks provided.")

        self.chunks = embedded_chunks

        embeddings = np.array(
            [chunk.embedding for chunk in embedded_chunks],
            dtype=np.float32
        )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must form a 2D array.")

        self.embedding_dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.index.add(embeddings)

    def search(self, query_embedding: List[float], k: int = 5) -> List[SearchResult]:
        """
        Search the FAISS index using a query embedding.

        Args:
            query_embedding: Embedding vector of the user query.
            k: Number of nearest chunks to return.

        Returns:
            A list of SearchResult objects.

        Raises:
            ValueError: If the index is not built or dimensions do not match.
        """
        if self.index is None:
            raise ValueError("The FAISS index has not been built yet.")

        if self.embedding_dimension is None:
            raise ValueError("Embedding dimension is unknown.")

        if k <= 0:
            raise ValueError("k must be greater than 0.")
        k = min(k, len(self.chunks))

        query = np.array([query_embedding], dtype=np.float32)

        if query.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension {query.shape[1]} does not match "
                f"index dimension {self.embedding_dimension}."
            )

        distances, indices = self.index.search(query, k) # returns vector with one list element

        results: List[SearchResult] = []

        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=float(distance)
                )
            )

        return results

    def save(self, directory: str | Path) -> None:
        """
        Save the FAISS index and chunk metadata to disk.
        """
        if self.index is None:
            raise ValueError("The FAISS index has not been built yet.")

        if self.embedding_dimension is None:
            raise ValueError("Embedding dimension is unknown.")

        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        index_path = save_dir / "faiss.index"
        metadata_path = save_dir / "chunks.json"

        faiss.write_index(self.index, str(index_path))

        metadata = {
            "embedding_dimension": self.embedding_dimension,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """
        Load the FAISS index and chunk metadata from disk.
        """
        load_dir = Path(directory)
        index_path = load_dir / "faiss.index"
        metadata_path = load_dir / "chunks.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Chunk metadata file not found: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        vector_store = cls()
        vector_store.index = faiss.read_index(str(index_path))
        vector_store.embedding_dimension = metadata["embedding_dimension"]
        vector_store.chunks = [EmbeddedChunk(**chunk) for chunk in metadata["chunks"]]

        return vector_store
