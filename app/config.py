from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Chatbot API"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/rag_db"
    policy_file_path: str = "data/company_policy.txt"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4

    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    min_similarity: float = 0.50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
