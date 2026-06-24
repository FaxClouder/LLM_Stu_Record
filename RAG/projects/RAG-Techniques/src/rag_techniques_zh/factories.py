"""根据集中配置创建模型客户端，避免案例内重复硬编码。"""

from __future__ import annotations

from .config import Settings, apply_runtime_environment, get_settings


def create_chat_model(
    settings: Settings | None = None,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    from langchain_openai import ChatOpenAI

    settings = apply_runtime_environment(settings or get_settings())
    settings.require("OPENAI_API_KEY")
    return ChatOpenAI(
        model=model or settings.openai.chat_model,
        api_key=settings.openai.api_key,
        base_url=settings.openai.base_url,
        temperature=settings.openai.temperature if temperature is None else temperature,
        max_tokens=max_tokens or settings.openai.max_tokens,
    )


def create_embeddings(settings: Settings | None = None, *, model: str | None = None):
    from langchain_openai import OpenAIEmbeddings

    settings = apply_runtime_environment(settings or get_settings())
    settings.require("OPENAI_API_KEY")
    return OpenAIEmbeddings(
        model=model or settings.openai.embedding_model,
        api_key=settings.openai.api_key,
        base_url=settings.openai.base_url,
    )


def create_openai_client(settings: Settings | None = None, *, asynchronous: bool = False):
    from openai import AsyncOpenAI, OpenAI

    settings = apply_runtime_environment(settings or get_settings())
    settings.require("OPENAI_API_KEY")
    client_type = AsyncOpenAI if asynchronous else OpenAI
    return client_type(api_key=settings.openai.api_key, base_url=settings.openai.base_url)

