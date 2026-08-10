#!/usr/bin/env python3
"""Run the full RAG ingestion pipeline: scrape → chunk → vector store."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scrape, chunk, and Chroma ingest in order.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download_site.py (reuse existing scraped data)",
    )
    parser.add_argument(
        "--skip-chunk",
        action="store_true",
        help="Skip chunk_documents.py",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not pass --reset to build_vectorstore.py",
    )
    return parser.parse_args()


def run_step(script: str, extra_args: list[str] | None = None) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *(extra_args or [])]
    print(f"\n>>> {' '.join(command)}\n")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()

    if not args.skip_download:
        run_step("download_site.py")
    else:
        print("Skipping download (using existing scraped data).")

    if not args.skip_chunk:
        run_step("chunk_documents.py")
    else:
        print("Skipping chunk step.")

    build_args = [] if args.no_reset else ["--reset"]
    run_step("build_vectorstore.py", build_args)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
