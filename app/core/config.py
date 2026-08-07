class Settings:
    app_title: str = "Chatbot RAG API"
    cors_origins: list[str] = ["*"]
    stream_token_delay_seconds: float = 0.04


settings = Settings()
