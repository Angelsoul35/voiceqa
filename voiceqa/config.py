"""Application configuration, loaded from environment variables (prefix ``VQ_``)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VQ_", env_file=".env", extra="ignore")

    environment: str = "local"

    # Retrieval: how many knowledge-base entries to ground an answer on.
    top_k: int = 3
    # An answer graded below this groundedness is marked low-confidence and the
    # pipeline recommends escalating to a human.
    min_groundedness: float = 0.5

    # LLM provider cascade, tried in order. Each reads its own key from the env;
    # when all fail, the pipeline falls back to the best retrieved answer.
    llm_spec: str = "anthropic,openai,gemini,deepseek,ollama"
    llm_max_tokens: int = 400

    # Speech-to-text (used only by the HTTP transcriber).
    asr_model: str = "whisper-1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
