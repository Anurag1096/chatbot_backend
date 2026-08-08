import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from app.core.config import settings


@dataclass
class PageRecord:
    page_id: str
    url: str
    title: str
    file: str
    word_count: int


@dataclass
class DownloadResult:
    site_name: str
    base_url: str
    output_dir: Path
    pages: list[PageRecord] = field(default_factory=list)

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


class SiteDownloader:
    """Download a small ecommerce site into RAG-friendly text files."""

    def __init__(
        self,
        base_url: str | None = None,
        output_dir: str | Path | None = None,
        max_pages: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.scrape_default_url).rstrip("/")
        self.output_root = Path(output_dir or settings.scrape_output_dir)
        self.max_pages = max_pages or settings.scrape_max_pages
        self.timeout = timeout_seconds or settings.scrape_request_timeout_seconds

        parsed = urlparse(self.base_url)
        self.site_name = re.sub(r"[^a-z0-9]+", "_", parsed.netloc.lower()).strip("_")
        self.output_dir = self.output_root / self.site_name
        self.pages_dir = self.output_dir / "pages"

    async def download(self) -> DownloadResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)

        visited: set[str] = set()
        queue: list[str] = [self.base_url + "/"]
        records: list[PageRecord] = []

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": "ChatbotRAG-SiteDownloader/1.0"},
        ) as client:
            while queue and len(records) < self.max_pages:
                url = queue.pop(0)
                normalized = self._normalize_url(url)
                if normalized in visited:
                    continue
                visited.add(normalized)

                html = await self._fetch_html(client, normalized)
                if html is None:
                    continue

                title, text, links = self._parse_page(html, normalized)
                page_id = self._page_id(normalized)
                file_name = f"{page_id}.txt"
                file_path = self.pages_dir / file_name

                file_path.write_text(
                    self._format_page_document(normalized, title, text),
                    encoding="utf-8",
                )

                word_count = len(text.split())
                records.append(
                    PageRecord(
                        page_id=page_id,
                        url=normalized,
                        title=title,
                        file=f"pages/{file_name}",
                        word_count=word_count,
                    )
                )

                for link in links:
                    if link not in visited and link not in queue:
                        queue.append(link)

                await asyncio.sleep(0.2)

        result = DownloadResult(
            site_name=self.site_name,
            base_url=self.base_url,
            output_dir=self.output_dir,
            pages=records,
        )
        self._write_manifest(result)
        return result

    async def _fetch_html(self, client: httpx.AsyncClient, url: str) -> str | None:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPError:
            return None

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return None

        return response.text

    def _parse_page(
        self,
        html: str,
        page_url: str,
    ) -> tuple[str, str, list[str]]:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else "Untitled"
        main = soup.find("main") or soup.find("article") or soup.body
        text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if not href or href.startswith("#") or href.lower().startswith("mailto:"):
                continue

            absolute = urljoin(page_url, href)
            normalized = self._normalize_url(absolute)
            if self._is_internal(normalized):
                links.append(normalized)

        return title, text, links

    def _format_page_document(self, url: str, title: str, text: str) -> str:
        scraped_at = datetime.now(UTC).isoformat()
        return (
            f"URL: {url}\n"
            f"Title: {title}\n"
            f"Scraped-At: {scraped_at}\n"
            f"\n"
            f"{text}\n"
        )

    def _write_manifest(self, result: DownloadResult) -> None:
        manifest = {
            "site_name": result.site_name,
            "base_url": result.base_url,
            "scraped_at": datetime.now(UTC).isoformat(),
            "page_count": len(result.pages),
            "pages": [
                {
                    "id": page.page_id,
                    "url": page.url,
                    "title": page.title,
                    "file": page.file,
                    "word_count": page.word_count,
                }
                for page in result.pages
            ],
        }
        result.manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        normalized = urlunparse(
            (
                parsed.scheme,
                parsed.netloc.lower(),
                path,
                parsed.params,
                parsed.query,
                "",
            )
        )
        return normalized

    def _is_internal(self, url: str) -> bool:
        parsed = urlparse(url)
        base = urlparse(self.base_url)
        return parsed.netloc.lower() == base.netloc.lower()

    def _page_id(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.strip("/") or "index"
        page_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", path)
        if parsed.query:
            query_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", parsed.query)
            page_id = f"{page_id}__{query_id}"
        return page_id[:120]
