from pathlib import Path
from typing import Any

import chromadb

from app.core.config import settings


class VectorStoreService:
    """Chroma-backed vector store for RAG retrieval."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_chunks(self, chunks: list[dict[str, Any]], batch_size: int = 100) -> int:
        if not chunks:
            return 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[chunk["id"] for chunk in batch],
                documents=[chunk["text"] for chunk in batch],
                metadatas=[self._sanitize_metadata(chunk["metadata"]) for chunk in batch],
            )

        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []

        limit = top_k or settings.chroma_top_k
        results = self._collection.query(
            query_texts=[query],
            n_results=min(limit, self.count()),
        )

        return self._format_results(results)

    def _format_results(self, results: dict[str, Any]) -> list[dict[str, Any]]:
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        formatted: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            formatted.append(
                {
                    "id": ids[index],
                    "text": document,
                    "metadata": metadatas[index] or {},
                    "distance": distances[index] if index < len(distances) else None,
                }
            )
        return formatted

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        clean: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean
