class Settings:
    app_title: str = "Chatbot RAG API"
    cors_origins: list[str] = ["*"]
    stream_token_delay_seconds: float = 0.04

    # Site downloader (RAG ingestion)
    scrape_output_dir: str = "data/scraped"
    scrape_default_url: str = "https://books.toscrape.com"
    scrape_max_pages: int = 25
    scrape_request_timeout_seconds: float = 15.0


settings = Settings()
