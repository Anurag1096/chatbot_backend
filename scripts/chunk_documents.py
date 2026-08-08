#!/usr/bin/env python3
"""Chunk scraped site files for RAG ingestion."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.services.document_chunker import DocumentChunker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk scraped pages by product blocks using LangChain.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to scraped site dir containing manifest.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for chunks (default: data/chunks)",
    )
    return parser.parse_args()


def default_scraped_dir() -> Path:
    scraped_root = Path(settings.scrape_output_dir)
    candidates = sorted(scraped_root.glob("*/manifest.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No scraped manifest found under {scraped_root}. Run download_site.py first.",
        )
    return candidates[0].parent


def main() -> None:
    args = parse_args()
    scraped_dir = Path(args.input) if args.input else default_scraped_dir()

    chunker = DocumentChunker(
        scraped_dir=scraped_dir,
        output_dir=args.output,
    )

    print(f"Input:     {chunker.scraped_dir}")
    print(f"Separator: {chunker.separator!r}")
    print(f"Output:    {chunker.output_dir}")
    print()

    result = chunker.chunk_site()

    print(f"Created {result.chunk_count} chunks")
    if result.deduplicated_count:
        print(f"Removed {result.deduplicated_count} duplicate chunks")
    print(f"Saved to: {result.output_file}")


if __name__ == "__main__":
    main()
