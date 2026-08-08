import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup, Tag

from app.core.config import settings

STAR_RATINGS = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


@dataclass
class PageRecord:
    page_id: str
    url: str
    title: str
    file: str
    word_count: int
    page_type: str = "generic"
    item_count: int = 0


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
        self._uses_book_parser = "books.toscrape.com" in parsed.netloc.lower()
        self._skip_paths = set(settings.scrape_skip_paths)
        self._max_product_details = settings.scrape_max_product_details
        self._product_details_saved = 0

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
                if normalized in visited or self._should_skip_url(normalized):
                    continue
                visited.add(normalized)

                html = await self._fetch_html(client, normalized)
                if html is None:
                    continue

                parsed = self._parse_page(html, normalized)
                page_type = parsed["page_type"]

                if (
                    page_type == "product-detail"
                    and self._product_details_saved >= self._max_product_details
                ):
                    continue

                page_id = self._page_id(normalized)
                file_name = f"{page_id}.txt"
                file_path = self.pages_dir / file_name

                file_path.write_text(
                    self._format_page_document(normalized, parsed),
                    encoding="utf-8",
                )

                word_count = len(parsed["text"].split())
                records.append(
                    PageRecord(
                        page_id=page_id,
                        url=normalized,
                        title=parsed["title"],
                        file=f"pages/{file_name}",
                        word_count=word_count,
                        page_type=page_type,
                        item_count=parsed["item_count"],
                    )
                )

                if page_type == "product-detail":
                    self._product_details_saved += 1

                for product_url in parsed.get("product_urls", []):
                    if (
                        product_url not in visited
                        and product_url not in queue
                        and self._product_details_saved < self._max_product_details
                    ):
                        queue.insert(0, product_url)

                for link in parsed["links"]:
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

    def _parse_page(self, html: str, page_url: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        links = self._extract_links(soup, page_url)

        if self._uses_book_parser:
            if soup.select_one("div.product_main") or self._is_product_detail_url(page_url):
                return self._parse_book_detail_page(soup, page_url, links)
            if soup.select("article.product_pod"):
                return self._parse_book_listing_page(soup, page_url, links)

        return self._parse_generic_page(soup, page_url, links)

    def _parse_book_listing_page(
        self,
        soup: BeautifulSoup,
        page_url: str,
        links: list[str],
    ) -> dict:
        title = soup.title.get_text(strip=True) if soup.title else "Untitled"
        category = self._extract_category(soup)
        products: list[str] = []
        product_urls: list[str] = []

        for index, pod in enumerate(soup.select("article.product_pod"), start=1):
            product = self._extract_listing_product(pod, page_url)
            if product:
                products.append(self._format_product_block(index, product))
                product_urls.append(product["Product-URL"])

        header = [
            f"Category: {category or 'All products'}",
            f"Product-Count: {len(products)}",
            "",
        ]
        text = "\n".join(header + products)

        return {
            "title": title,
            "text": text,
            "links": links,
            "product_urls": product_urls,
            "page_type": "category-listing",
            "item_count": len(products),
            "category": category,
        }

    def _parse_book_detail_page(
        self,
        soup: BeautifulSoup,
        page_url: str,
        links: list[str],
    ) -> dict:
        title_tag = soup.select_one("div.product_main h1") or soup.select_one("h1")
        book_title = title_tag.get_text(strip=True) if title_tag else "Untitled"

        lines = [
            f"Book: {book_title}",
            f"Product-URL: {page_url}",
        ]

        price = self._extract_price(soup)
        if price:
            lines.append(f"Price: {price}")

        rating = self._extract_rating(soup)
        if rating:
            lines.append(f"Rating: {rating}")

        category = self._extract_category(soup)
        if category:
            lines.append(f"Category: {category}")

        description = self._extract_product_description(soup)
        if description:
            lines.extend(["", "Description:", description])

        attributes = self._extract_product_attributes(soup)
        if attributes:
            lines.extend(["", "Product-Information:"])
            lines.extend(f"{key}: {value}" for key, value in attributes.items())

        text = "\n".join(lines)

        return {
            "title": book_title,
            "text": text,
            "links": links,
            "product_urls": [],
            "page_type": "product-detail",
            "item_count": 1,
            "category": category,
        }

    def _parse_generic_page(
        self,
        soup: BeautifulSoup,
        page_url: str,
        links: list[str],
    ) -> dict:
        title = soup.title.get_text(strip=True) if soup.title else "Untitled"
        main = soup.find("main") or soup.find("article") or soup.body
        text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return {
            "title": title,
            "text": text,
            "links": links,
            "product_urls": [],
            "page_type": "generic",
            "item_count": 0,
            "category": None,
        }

    def _extract_listing_product(self, pod: Tag, page_url: str) -> dict[str, str] | None:
        title_link = pod.select_one("h3 a")
        if title_link is None:
            return None

        book_title = title_link.get("title") or title_link.get_text(strip=True)
        product_url = urljoin(page_url, title_link.get("href", ""))

        product: dict[str, str] = {
            "Book": book_title,
            "Product-URL": self._normalize_url(product_url),
        }

        price = self._extract_price(pod)
        if price:
            product["Price"] = price

        rating = self._extract_rating(pod)
        if rating:
            product["Rating"] = rating

        return product

    def _format_product_block(self, index: int, product: dict[str, str]) -> str:
        lines = [f"--- Product {index} ---"]
        field_order = ["Book", "Price", "Rating", "Product-URL"]
        for field in field_order:
            if field in product:
                lines.append(f"{field}: {product[field]}")
        lines.append("")
        return "\n".join(lines)

    def _extract_price(self, node: Tag | BeautifulSoup) -> str | None:
        price_tag = node.select_one("p.price_color")
        if price_tag is None:
            return None
        return price_tag.get_text(strip=True)

    def _extract_rating(self, node: Tag | BeautifulSoup) -> str | None:
        rating_tag = node.select_one("p.star-rating")
        if rating_tag is None:
            return None

        classes = rating_tag.get("class", [])
        for class_name, stars in STAR_RATINGS.items():
            if class_name in classes:
                return f"{stars} out of 5"
        return None

    def _extract_availability(self, node: Tag | BeautifulSoup) -> str | None:
        availability_tag = node.select_one("p.instock.availability")
        if availability_tag is None:
            return None

        text = availability_tag.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text or None

    def _extract_category(self, soup: BeautifulSoup) -> str | None:
        page_heading = soup.select_one(".page-header.action h1")
        if page_heading:
            heading = page_heading.get_text(strip=True)
            if heading:
                return heading

        crumbs = [
            anchor.get_text(strip=True)
            for anchor in soup.select(".breadcrumb li a")
            if anchor.get_text(strip=True).lower() != "home"
        ]
        if not crumbs:
            return None
        return crumbs[-1]

    def _extract_product_description(self, soup: BeautifulSoup) -> str | None:
        header = soup.select_one("#product_description")
        if header is None:
            return None

        description_tag = header.find_next_sibling("p")
        if description_tag is None:
            return None

        text = description_tag.get_text(" ", strip=True)
        return text or None

    def _extract_product_attributes(self, soup: BeautifulSoup) -> dict[str, str]:
        attributes: dict[str, str] = {}
        for row in soup.select("table.table.table-striped tr"):
            header = row.find("th")
            value = row.find("td")
            if header is None or value is None:
                continue
            key = header.get_text(strip=True)
            val = value.get_text(" ", strip=True)
            if key and val:
                attributes[key] = val
        return attributes

    def _extract_links(self, soup: BeautifulSoup, page_url: str) -> list[str]:
        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if not href or href.startswith("#") or href.lower().startswith("mailto:"):
                continue

            absolute = urljoin(page_url, href)
            normalized = self._normalize_url(absolute)
            if self._is_internal(normalized):
                links.append(normalized)
        return links

    def _format_page_document(self, url: str, parsed: dict) -> str:
        scraped_at = datetime.now(UTC).isoformat()
        header = [
            f"URL: {url}",
            f"Title: {parsed['title']}",
            f"Page-Type: {parsed['page_type']}",
            f"Scraped-At: {scraped_at}",
            "",
        ]
        return "\n".join(header) + parsed["text"] + "\n"

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
                    "page_type": page.page_type,
                    "item_count": page.item_count,
                }
                for page in result.pages
            ],
        }
        result.manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def _should_skip_url(self, url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        return path in self._skip_paths

    def _is_product_detail_url(self, url: str) -> bool:
        path = urlparse(url).path
        return bool(re.search(r"/catalogue/[^/]+_\d+/index\.html$", path))

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path or "/"
        if path == "/index.html":
            path = "/"
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
