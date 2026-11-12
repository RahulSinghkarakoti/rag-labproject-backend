from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "RAG API"
    DEBUG: bool = False

    DATABASE_URL: str
    NEXTAUTH_INTERNAL_SECRET:str
    JWT_SECRET_KEY:str
    JWT_ALGORITHM:str
    ACCESS_TOKEN_EXPIRE_MINUTES:int
    REFRESH_TOKEN_EXPIRE_DAYS:int

    REDIS_HOST:str
    REDIS_PORT:int

    HF_TOKEN:str

    # Model Preloading Configuration
    PRELOAD_ALL_MODELS: bool = False
    PRELOAD_EMBEDDING_MODELS: str = "none"
    PRELOAD_RERANKING_MODELS: str = "none"

    # Thread Summary & Memory Compaction Configuration
    THREAD_SUMMARY_RECENT_TURNS: int = 8
    THREAD_SUMMARY_TRIGGER_THRESHOLD: int = 8
    THREAD_SUMMARY_MAX_TOKENS: int = 300
    THREAD_SUMMARY_PROVIDER: str = ""
    THREAD_SUMMARY_MODEL: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
