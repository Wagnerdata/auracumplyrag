import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    auracomply_url: str = "http://localhost:8000"
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "eu_ai_act"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    eu_ai_act_pdf: str = "./data/eu_ai_act.pdf"
    tenant_id: str = "rag-demo"


settings = Settings()

Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(settings.eu_ai_act_pdf) or ".").mkdir(parents=True, exist_ok=True)
