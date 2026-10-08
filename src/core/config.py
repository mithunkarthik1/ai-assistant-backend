"""Central application configuration loaded from the project .env file."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Shared settings for the API, RAG workflow, and agent workflow."""

    app_name: str = Field(
        default="WorkPilot AI Assistant",
        validation_alias=AliasChoices("APP_NAME"),
    )
    database_url: str = Field(
        default="postgresql+psycopg://postgres:12345678@localhost:5432/chatbot_db",
        validation_alias=AliasChoices("DATABASE_URL"),
    )
    policy_file_path: str = Field(
        default="data/WorkPilot_Company_Policy.pdf",
        validation_alias=AliasChoices("POLICY_FILE_PATH"),
    )
    policy_doc_id: str = Field(
        default="00000000-0000-0000-0000-000000000002",
        validation_alias=AliasChoices("POLICY_DOC_ID"),
    )
    chunk_size: int = Field(
        default=1000,
        validation_alias=AliasChoices("CHUNK_SIZE"),
    )
    chunk_overlap: int = Field(
        default=200,
        validation_alias=AliasChoices("CHUNK_OVERLAP"),
    )
    top_k: int = Field(
        default=6,
        validation_alias=AliasChoices("TOP_K"),
    )
    min_similarity: float = Field(
        default=0.60,
        validation_alias=AliasChoices("MIN_SIMILARITY"),
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL"),
    )
    cors_origins: str = Field(
        default="*",
        validation_alias=AliasChoices("CORS_ORIGINS"),
    )

    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGENT_LLM_API_KEY", "LLM_API_KEY"),
    )
    llm_model: str = Field(
        default="gemini-3.5-flash-lite",
        validation_alias=AliasChoices("AGENT_LLM_MODEL", "LLM_MODEL"),
    )
    llm_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGENT_LLM_BASE_URL", "LLM_BASE_URL"),
    )

    embedding_provider: str = Field(
        default="fastembed",
        validation_alias=AliasChoices("EMBEDDING_PROVIDER"),
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        validation_alias=AliasChoices("EMBEDDING_MODEL"),
    )
    embedding_dimension: int = Field(
        default=384,
        validation_alias=AliasChoices("EMBEDDING_DIMENSION"),
    )

    qdrant_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("QDRANT_URL"),
    )
    qdrant_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("QDRANT_API_KEY"),
    )
    qdrant_collection: str = Field(
        default="document_chunks",
        validation_alias=AliasChoices("QDRANT_COLLECTION"),
    )

    project_api_base_url: str = Field(
        default="http://mock_project_api:9000",
        validation_alias=AliasChoices("PROJECT_API_BASE_URL"),
    )
    project_api_path_prefix: str = Field(
        default="/api/v1/projects",
        validation_alias=AliasChoices("PROJECT_API_PATH_PREFIX"),
    )
    project_api_timeout_seconds: float = Field(
        default=5.0,
        validation_alias=AliasChoices("PROJECT_API_TIMEOUT_SECONDS"),
    )
    memory_max_messages: int = Field(
        default=12,
        validation_alias=AliasChoices("AGENT_MEMORY_MAX_MESSAGES", "MEMORY_MAX_MESSAGES"),
    )

    jwt_secret_key: str = Field(
        default="your-secret-key",
        validation_alias=AliasChoices("JWT_SECRET_KEY"),
    )

    jwt_algorithm: str = Field(
        default="HS256",
        validation_alias=AliasChoices("JWT_ALGORITHM"),
    )

    jwt_access_token_expire_minutes: int = Field(
        
        default=15,
        validation_alias=AliasChoices("JWT_ACCESS_TOKEN_EXPIRE_MINUTES"),
    )

    jwt_refresh_token_expire_days: int = Field(
        default=7,
        validation_alias=AliasChoices("JWT_REFRESH_TOKEN_EXPIRE_DAYS"),
    )

    

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = AppSettings()


def get_settings() -> AppSettings:
    """Return the shared application settings instance."""
    return settings


# Feature modules can use this descriptive alias while all values still come
# from the single shared settings object above.
AgentSettings = AppSettings
