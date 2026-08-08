#!/usr/bin/env python3
"""Download a small ecommerce site into RAG-friendly text files."""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.site_downloader import SiteDownloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a small ecommerce website for RAG ingestion.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Site base URL (default: books.toscrape.com from config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: data/scraped)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to download (default: 25)",
    )
    return parser.parse_args()


async def run() -> int:
    args = parse_args()
    downloader = SiteDownloader(
        base_url=args.url,
        output_dir=args.output,
        max_pages=args.max_pages,
    )

    print(f"Downloading: {downloader.base_url}")
    print(f"Output dir:  {downloader.output_dir}")
    print(f"Max pages:   {downloader.max_pages}")
    print()

    result = await downloader.download()

    print(f"Saved {len(result.pages)} pages")
    print(f"Manifest:    {result.manifest_path}")
    print(f"Pages dir:   {downloader.pages_dir}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
