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

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        price_max: float | None = None,
        price_min: float | None = None,
        category: str | None = None,
        rating_min: int | None = None,
        sort_by_price: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []

        where = self._build_where_clause(
            price_max=price_max,
            price_min=price_min,
            category=category,
            rating_min=rating_min,
        )
        limit = top_k or settings.chroma_top_k

        results = self._collection.query(
            query_texts=[query],
            n_results=min(limit, self.count()),
            where=where,
        )
        formatted = self._dedupe_by_product_url(self._format_results(results))

        if sort_by_price == "asc":
            formatted.sort(key=lambda item: self._price_value(item) or float("inf"))
        elif sort_by_price == "desc":
            formatted.sort(key=lambda item: self._price_value(item) or 0.0, reverse=True)

        return formatted[:limit]

    def search_filtered(
        self,
        query: str,
        *,
        price_max: float | None = None,
        price_min: float | None = None,
        category: str | None = None,
        rating_min: int | None = None,
        top_k: int | None = None,
        sort_by_price: str | None = None,
        fallback_semantic: bool = True,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Search with metadata filters. Returns (results, used_fallback)."""
        where = self._build_where_clause(
            price_max=price_max,
            price_min=price_min,
            category=category,
            rating_min=rating_min,
        )
        limit = top_k or settings.chroma_filter_top_k

        if where is None:
            return self.search(query, top_k=limit), False

        results = self._collection.query(
            query_texts=[query],
            n_results=min(limit, self.count()),
            where=where,
        )
        formatted = self._dedupe_by_product_url(self._format_results(results))

        if sort_by_price == "asc":
            formatted.sort(key=lambda item: self._price_value(item) or float("inf"))
        elif sort_by_price == "desc":
            formatted.sort(key=lambda item: self._price_value(item) or 0.0, reverse=True)

        formatted = formatted[:limit]

        if formatted or not fallback_semantic:
            return formatted, False

        return self.search(query, top_k=limit), True

    def count_filtered(
        self,
        *,
        price_max: float | None = None,
        price_min: float | None = None,
        category: str | None = None,
        rating_min: int | None = None,
    ) -> int:
        where = self._build_where_clause(
            price_max=price_max,
            price_min=price_min,
            category=category,
            rating_min=rating_min,
        )
        if where is None:
            return self.count()

        results = self._collection.get(where=where, include=["metadatas"])
        metadatas = results.get("metadatas") or []
        product_urls = {
            metadata.get("product_url")
            for metadata in metadatas
            if metadata and metadata.get("product_url")
        }
        if product_urls:
            return len(product_urls)
        return len(results.get("ids") or [])

    def _build_where_clause(
        self,
        *,
        price_max: float | None = None,
        price_min: float | None = None,
        category: str | None = None,
        rating_min: int | None = None,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []

        if price_max is not None:
            clauses.append({"price_value": {"$lt": price_max}})
        if price_min is not None:
            clauses.append({"price_value": {"$gt": price_min}})
        if category:
            clauses.append({"category": category})
        if rating_min is not None:
            clauses.append({"rating_value": {"$gte": rating_min}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

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

    def _dedupe_by_product_url(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        for item in results:
            metadata = item.get("metadata") or {}
            key = metadata.get("product_url") or item.get("id", "")
            if key not in seen:
                seen[key] = item
                order.append(key)
                continue

            existing = seen[key]
            if self._chunk_priority(item) > self._chunk_priority(existing):
                seen[key] = item

        return [seen[key] for key in order]

    def _chunk_priority(self, item: dict[str, Any]) -> tuple[int, int]:
        metadata = item.get("metadata") or {}
        page_type = metadata.get("page_type", "")
        has_description = 1 if metadata.get("has_description") else 0
        type_score = 2 if page_type == "product-detail" else 1 if page_type else 0
        return (type_score, has_description)

    def _price_value(self, item: dict[str, Any]) -> float | None:
        metadata = item.get("metadata") or {}
        value = metadata.get("price_value")
        if isinstance(value, (int, float)):
            return float(value)
        return None

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
