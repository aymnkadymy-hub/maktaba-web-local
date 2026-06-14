from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM providers
    GROQ_API_KEY:     str   = ""
    OPENAI_API_KEY:   str   = ""
    ANTHROPIC_API_KEY: str  = ""

    # LLM behaviour
    LLM_PROVIDER:   str   = "auto"
    OLLAMA_MODEL:   str   = "qwen2.5:3b"
    TEMPERATURE:    float = 0.35
    NUM_CTX:        int   = 4096
    NUM_PREDICT:    int   = -1

    # RAG
    RAG_K:                  int   = 6
    RAG_RELEVANCE_THRESHOLD: float = 0.30
    ENABLE_HYDE:            bool  = True
    ENABLE_RERANKER:        bool  = True
    ENABLE_MULTI_QUERY:     bool  = True

    # Rate limits
    CHAT_MAX_PER_MINUTE:  int = 30
    REGISTER_MAX_PER_HOUR: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
