"""Central application configuration loaded from the project .env file."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Shared settings for the API, RAG workflow, and agent workflow."""

    app_name: str = "RAG Chatbot API"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/rag_db"
    policy_file_path: str = "data/company_policy.txt"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 6
    log_level: str = "INFO"
    cors_origins: str = "*"

    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGENT_LLM_API_KEY", "LLM_API_KEY"),
    )
    llm_model: str = Field(
        default="openai/gpt-oss-20b",
        validation_alias=AliasChoices("AGENT_LLM_MODEL", "LLM_MODEL"),
    )
    llm_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGENT_LLM_BASE_URL", "LLM_BASE_URL"),
    )

    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    min_similarity: float = 0.50

    project_api_base_url: str = "http://localhost:9000"
    project_api_path_prefix: str = "/projects"
    project_api_timeout_seconds: float = 5.0
    memory_max_messages: int = 12

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = AppSettings()

# Feature modules can use this descriptive alias while all values still come
# from the single shared settings object above.
AgentSettings = AppSettings
