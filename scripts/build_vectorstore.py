#!/usr/bin/env python3
"""Load chunked documents into a Chroma vector store."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.services.vector_store import VectorStoreService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma vector store from chunks.json.")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to chunks.json (default: latest under data/chunks)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before ingesting",
    )
    return parser.parse_args()


def default_chunks_file() -> Path:
    chunks_root = Path(settings.chunks_output_dir)
    candidates = sorted(chunks_root.glob("*/chunks.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No chunks.json found under {chunks_root}. Run chunk_documents.py first.",
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    chunks_file = Path(args.input) if args.input else default_chunks_file()
    payload = json.loads(chunks_file.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])

    store = VectorStoreService()

    if args.reset and store.count() > 0:
        print("Resetting existing collection...")
        store.reset()

    print(f"Chunks file: {chunks_file}")
    print(f"Chroma path: {settings.chroma_persist_dir}")
    print(f"Collection:  {settings.chroma_collection_name}")
    print(f"Ingesting {len(chunks)} chunks...")

    ingested = store.ingest_chunks(chunks)

    print(f"Ingested {ingested} chunks")
    print(f"Collection size: {store.count()}")

    sample = store.search("cheapest poetry book", top_k=2)
    if sample:
        print("\nSample query: 'cheapest poetry book'")
        for item in sample:
            title = item["metadata"].get("book_title", "Unknown")
            price = item["metadata"].get("price", "N/A")
            print(f"  - {title} ({price})")


if __name__ == "__main__":
    main()
