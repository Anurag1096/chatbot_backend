import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

GENERIC_CATEGORIES = {"All products", "Books"}


@dataclass
class ChunkResult:
    site_name: str
    source_dir: Path
    output_dir: Path
    chunk_count: int
    deduplicated_count: int
    output_file: Path


class DocumentChunker:
    """Chunk scraped page files for RAG using LangChain text splitters."""

    FIELD_PATTERN = re.compile(r"^([A-Za-z-]+):\s*(.+)$", re.MULTILINE)
    PRODUCT_HEADER_PATTERN = re.compile(r"^--- Product \d+ ---\s*$", re.MULTILINE)
    AVAILABILITY_PATTERN = re.compile(r"^Availability:.*$", re.MULTILINE)

    def __init__(
        self,
        scraped_dir: str | Path,
        output_dir: str | Path | None = None,
        separator: str | None = None,
    ) -> None:
        self.scraped_dir = Path(scraped_dir)
        self.site_name = self.scraped_dir.name
        self.output_dir = Path(output_dir or settings.chunks_output_dir) / self.site_name
        self.separator = separator or settings.product_chunk_separator

        self._detail_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
        )

    def chunk_site(self) -> ChunkResult:
        manifest_path = self.scraped_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_chunks: list[dict] = []

        for page in manifest.get("pages", []):
            page_file = self.scraped_dir / page["file"]
            if not page_file.exists():
                continue

            text = page_file.read_text(encoding="utf-8")
            page_type = page.get("page_type", "generic")

            if page_type == "category-listing":
                page_chunks = self._chunk_listing_page(text, page)
            elif page_type == "product-detail":
                page_chunks = self._chunk_product_detail_page(text, page)
            else:
                page_chunks = self._chunk_single_page(text, page)

            raw_chunks.extend(page_chunks)

        chunks = self._deduplicate_chunks(raw_chunks)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "chunks.json"
        payload = {
            "site_name": self.site_name,
            "source_dir": str(self.scraped_dir),
            "raw_chunk_count": len(raw_chunks),
            "chunk_count": len(chunks),
            "separator": self.separator,
            "chunks": chunks,
        }
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return ChunkResult(
            site_name=self.site_name,
            source_dir=self.scraped_dir,
            output_dir=self.output_dir,
            chunk_count=len(chunks),
            deduplicated_count=len(raw_chunks) - len(chunks),
            output_file=output_file,
        )

    def _chunk_listing_page(self, text: str, page: dict) -> list[dict]:
        header_fields = self._extract_fields(text.split(self.separator)[0])
        category = header_fields.get("Category", "")
        page_url = header_fields.get("URL", page.get("url", ""))

        results: list[dict] = []
        for index, chunk in enumerate(self._split_product_blocks(text), start=1):
            fields = self._extract_fields(chunk)
            book_title = fields.get("Book", "")
            if not book_title:
                continue

            cleaned = self._build_chunk_text(
                fields=fields,
                category=category,
            )
            chunk_id = self._make_chunk_id(page["id"], index, book_title)

            results.append(
                self._make_chunk_record(
                    chunk_id=chunk_id,
                    text=cleaned,
                    page=page,
                    page_url=page_url,
                    fields=fields,
                    category=category,
                    chunk_index=index,
                    page_type="category-listing",
                )
            )

        return results

    def _chunk_product_detail_page(self, text: str, page: dict) -> list[dict]:
        fields = self._extract_fields(text)
        book_title = fields.get("Book", page.get("title", ""))
        page_url = fields.get("URL", page.get("url", ""))
        category = fields.get("Category", "")

        description = self._extract_description(text)
        if description:
            fields["Description"] = description

        cleaned = self._build_chunk_text(fields=fields, category=category)
        chunk_id = self._make_chunk_id(page["id"], 1, book_title)

        return [
            self._make_chunk_record(
                chunk_id=chunk_id,
                text=cleaned,
                page=page,
                page_url=page_url,
                fields=fields,
                category=category,
                chunk_index=1,
                page_type="product-detail",
            )
        ]

    def _chunk_single_page(self, text: str, page: dict) -> list[dict]:
        fields = self._extract_fields(text)
        book_title = fields.get("Book", page.get("title", ""))
        page_url = fields.get("URL", page.get("url", ""))
        category = fields.get("Category", "")

        documents = self._detail_splitter.split_documents(
            [
                Document(
                    page_content=self._clean_chunk_text(text.strip()),
                    metadata={"source_file": page["file"]},
                )
            ]
        )

        results: list[dict] = []
        for index, doc in enumerate(documents, start=1):
            chunk_fields = self._extract_fields(doc.page_content)
            chunk_id = self._make_chunk_id(page["id"], index, book_title)

            results.append(
                self._make_chunk_record(
                    chunk_id=chunk_id,
                    text=doc.page_content.strip(),
                    page=page,
                    page_url=page_url,
                    fields=chunk_fields or fields,
                    category=category,
                    chunk_index=index,
                    page_type=page.get("page_type", "generic"),
                )
            )

        return results

    def _make_chunk_record(
        self,
        chunk_id: str,
        text: str,
        page: dict,
        page_url: str,
        fields: dict[str, str],
        category: str,
        chunk_index: int,
        page_type: str,
    ) -> dict:
        product_url = fields.get("Product-URL", page_url if page_type == "product-detail" else "")

        return {
            "id": chunk_id,
            "text": text,
            "metadata": {
                "site_name": self.site_name,
                "source_file": page["file"],
                "page_id": page["id"],
                "page_url": page_url,
                "page_type": page_type,
                "category": category or fields.get("Category", ""),
                "book_title": fields.get("Book", ""),
                "product_url": product_url,
                "price": fields.get("Price", ""),
                "rating": fields.get("Rating", ""),
                "has_description": bool(fields.get("Description")),
                "chunk_index": chunk_index,
            },
        }

    def _build_chunk_text(self, fields: dict[str, str], category: str) -> str:
        lines: list[str] = []

        book = fields.get("Book")
        if book:
            lines.append(f"Book: {book}")

        effective_category = fields.get("Category") or category
        if effective_category:
            lines.append(f"Category: {effective_category}")

        for key in ("Price", "Rating", "Description", "Product-URL"):
            value = fields.get(key)
            if value:
                lines.append(f"{key}: {value}")

        return "\n".join(lines)

    def _extract_description(self, text: str) -> str:
        match = re.search(r"Description:\s*(.+?)(?:\n[A-Za-z-]+:|\Z)", text, re.DOTALL)
        if not match:
            return ""
        return re.sub(r"\s+", " ", match.group(1)).strip()

    def _clean_chunk_text(self, text: str) -> str:
        cleaned = self.PRODUCT_HEADER_PATTERN.sub("", text)
        cleaned = self.AVAILABILITY_PATTERN.sub("", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _deduplicate_chunks(self, chunks: list[dict]) -> list[dict]:
        best_by_url: dict[str, dict] = {}

        for chunk in chunks:
            product_url = chunk["metadata"].get("product_url", "").strip()
            if not product_url:
                best_by_url[chunk["id"]] = chunk
                continue

            current = best_by_url.get(product_url)
            if current is None or self._chunk_priority(chunk) > self._chunk_priority(current):
                best_by_url[product_url] = chunk

        return list(best_by_url.values())

    def _chunk_priority(self, chunk: dict) -> tuple[int, int, int]:
        metadata = chunk["metadata"]
        page_type = metadata.get("page_type", "")
        category = metadata.get("category", "")
        has_description = metadata.get("has_description", False)

        type_score = 3 if page_type == "product-detail" else 1
        description_score = 1 if has_description else 0
        category_score = 0 if category in GENERIC_CATEGORIES else 1

        return (type_score, description_score, category_score)

    def _split_product_blocks(self, text: str) -> list[str]:
        parts = text.split(self.separator)
        if len(parts) <= 1:
            return [text]

        return [f"{self.separator}{part}" for part in parts[1:]]

    def chunk_to_langchain_documents(self, chunks: list[dict]) -> list[Document]:
        return [
            Document(page_content=chunk["text"], metadata=chunk["metadata"])
            for chunk in chunks
        ]

    def _extract_fields(self, text: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        for match in self.FIELD_PATTERN.finditer(text):
            key = match.group(1).strip()
            value = match.group(2).strip()
            fields[key] = value
        return fields

    def _make_chunk_id(self, page_id: str, index: int, book_title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", book_title.lower()).strip("_")
        slug = slug[:60] if slug else f"item_{index}"
        return f"{page_id}__{index}__{slug}"
