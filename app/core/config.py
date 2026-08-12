import os

from dotenv import load_dotenv

load_dotenv()


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if not raw:
        return [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class Settings:
    app_title: str = "Chatbot RAG API"
    cors_origins: list[str] = _parse_cors_origins()
    cors_allow_credentials: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
    cors_allow_methods: list[str] = ["GET", "POST", "OPTIONS"]
    cors_allow_headers: list[str] = ["Content-Type", "Accept", "Authorization"]
    stream_token_delay_seconds: float = 0.04

    # Site downloader (RAG ingestion)
    scrape_output_dir: str = "data/scraped"
    scrape_default_url: str = "https://books.toscrape.com"
    scrape_max_pages: int = 400
    scrape_max_product_details: int = 350
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
    chroma_filter_top_k: int = 10
    chroma_cheapest_top_k: int = 5

    # Retrieval context
    retrieval_history_turns: int = 2

    # LLM generation (Google Gemini)
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_max_tokens: int = int(os.getenv("GEMINI_MAX_TOKENS", "1024"))
    gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
    llm_max_history_turns: int = int(os.getenv("LLM_MAX_HISTORY_TURNS", "6"))
    llm_max_chunk_chars: int = int(os.getenv("LLM_MAX_CHUNK_CHARS", "600"))

    # LLM rate limiting (per client IP)
    llm_rate_limit_enabled: bool = os.getenv("LLM_RATE_LIMIT_ENABLED", "true").lower() == "true"
    llm_rate_limit_per_minute: int = int(os.getenv("LLM_RATE_LIMIT_PER_MINUTE", "10"))
    llm_rate_limit_message: str = os.getenv(
        "LLM_RATE_LIMIT_MESSAGE",
        "AI rate limit reached for now. Showing catalog results instead.",
    )
    llm_error_fallback_message: str = os.getenv(
        "LLM_ERROR_FALLBACK_MESSAGE",
        "AI is temporarily unavailable. Showing catalog results instead.",
    )

    # Chat request limits
    chat_max_message_length: int = int(os.getenv("CHAT_MAX_MESSAGE_LENGTH", "2000"))
    chat_max_history_turns: int = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "20"))
    chat_max_history_content_length: int = int(os.getenv("CHAT_MAX_HISTORY_CONTENT_LENGTH", "4000"))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()
