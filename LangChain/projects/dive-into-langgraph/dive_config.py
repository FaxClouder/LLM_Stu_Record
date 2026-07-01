"""
Shared local configuration helpers for the dive-into-langgraph examples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    embedding_model: str = ""


def find_workspace_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        if (directory / "pyproject.toml").exists() and (directory / "LangChain").is_dir():
            return directory

    return Path(__file__).resolve().parents[3]


def load_project_env(start: Path | None = None) -> Path | None:
    root = find_workspace_root(start)
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        return env_path
    return None


def _first_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def get_model_config(provider: str | None = None) -> ModelConfig:
    load_project_env()
    selected = (
        provider
        or os.getenv("DIVE_PROVIDER")
        or os.getenv("LANGCHAIN_PROVIDER")
        or "dashscope"
    ).strip().lower()

    if selected in {"dashscope", "bailian"}:
        return ModelConfig(
            provider="dashscope",
            model=_first_value(
                "DIVE_CHAT_MODEL",
                "LANGCHAIN_CHAT_MODEL",
                "DASHSCOPE_MODEL",
                default="qwen-plus",
            ),
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            base_url=_first_value(
                "DASHSCOPE_BASE_URL",
                default="https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            embedding_model=_first_value(
                "DIVE_EMBEDDING_MODEL",
                "LANGCHAIN_EMBEDDING_MODEL",
                "DASHSCOPE_EMBEDDING_MODEL",
                default="text-embedding-v4",
            ),
        )

    if selected == "deepseek":
        return ModelConfig(
            provider="deepseek",
            model=_first_value(
                "DIVE_CHAT_MODEL",
                "LANGCHAIN_CHAT_MODEL",
                "DEEPSEEK_MODEL",
                default="deepseek-chat",
            ),
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=_first_value("DEEPSEEK_BASE_URL", default="https://api.deepseek.com"),
            embedding_model=_first_value(
                "DIVE_EMBEDDING_MODEL",
                "LANGCHAIN_EMBEDDING_MODEL",
                "OPENAI_EMBEDDING_MODEL",
                default="text-embedding-3-small",
            ),
        )

    if selected == "openai":
        return ModelConfig(
            provider="openai",
            model=_first_value(
                "DIVE_CHAT_MODEL",
                "LANGCHAIN_CHAT_MODEL",
                "OPENAI_CHAT_MODEL",
                default="gpt-4o-mini",
            ),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=_first_value("OPENAI_BASE_URL", default="https://api.openai.com/v1"),
            embedding_model=_first_value(
                "DIVE_EMBEDDING_MODEL",
                "LANGCHAIN_EMBEDDING_MODEL",
                "OPENAI_EMBEDDING_MODEL",
                default="text-embedding-3-small",
            ),
        )

    if selected == "ark":
        return ModelConfig(
            provider="ark",
            model=_first_value(
                "DIVE_CHAT_MODEL",
                "LANGCHAIN_CHAT_MODEL",
                "ARK_MODEL",
                default="deepseek-v3-2-251201",
            ),
            api_key=os.getenv("ARK_API_KEY", ""),
            base_url=_first_value("ARK_BASE_URL", default="https://ark.cn-beijing.volces.com/api/v3"),
        )

    if selected == "ollama":
        return ModelConfig(
            provider="ollama",
            model=_first_value(
                "DIVE_CHAT_MODEL",
                "LANGCHAIN_CHAT_MODEL",
                "OLLAMA_MODEL",
                default="qwen3:4b",
            ),
            api_key=os.getenv("OLLAMA_API_KEY", "-"),
            base_url=_first_value("OLLAMA_BASE_URL", default="http://127.0.0.1:11435/v1"),
        )

    raise ValueError(
        f"Unsupported provider {selected!r}. "
        "Expected one of: dashscope, deepseek, openai, ark, ollama."
    )


def create_chat_model(
    provider: str | None = None,
    *,
    temperature: float | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    config = get_model_config(provider)
    model_kwargs: dict[str, Any] = {
        "model": config.model,
        "api_key": config.api_key,
        "base_url": config.base_url,
    }
    if temperature is not None:
        model_kwargs["temperature"] = temperature
    model_kwargs.update(kwargs)
    return ChatOpenAI(**model_kwargs)


def create_openai_client(provider: str | None = None) -> OpenAI:
    config = get_model_config(provider)
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


class CompatibleEmbeddings(Embeddings):
    """Small OpenAI-compatible embeddings wrapper for the tutorial examples."""

    def __init__(
        self,
        provider: str | None = None,
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self.config = get_model_config(provider)
        self.client = create_openai_client(provider)
        self.model = model or self.config.embedding_model
        self.dimensions = dimensions

    def _kwargs(self, texts: list[str]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        return kwargs

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 10):
            response = self.client.embeddings.create(**self._kwargs(texts[i : i + 10]))
            out.extend([item.embedding for item in response.data])
        return out

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(**self._kwargs([text]))
        return response.data[0].embedding


def create_embeddings(
    provider: str | None = None,
    *,
    model: str | None = None,
    dimensions: int | None = None,
) -> CompatibleEmbeddings:
    return CompatibleEmbeddings(provider, model=model, dimensions=dimensions)
