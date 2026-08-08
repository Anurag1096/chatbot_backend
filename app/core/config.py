class Settings:
    app_title: str = "Chatbot RAG API"
    cors_origins: list[str] = ["*"]
    stream_token_delay_seconds: float = 0.04

    # Site downloader (RAG ingestion)
    scrape_output_dir: str = "data/scraped"
    scrape_default_url: str = "https://books.toscrape.com"
    scrape_max_pages: int = 200
    scrape_max_product_details: int = 300
    scrape_request_timeout_seconds: float = 15.0
    scrape_skip_paths: tuple[str, ...] = (
        "/index.html",
        "/catalogue/category/books_1/index.html",
    )

    # Document chunking (RAG ingestion)
    chunks_output_dir: str = "data/chunks"
    product_chunk_separator: str = "--- Product"

    # Chroma vector store
    chroma_persist_dir: str = "data/chroma"
    chroma_collection_name: str = "books"
    chroma_top_k: int = 4


settings = Settings()
