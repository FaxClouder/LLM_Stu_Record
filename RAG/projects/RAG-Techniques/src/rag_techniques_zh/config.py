"""集中读取模型、API 地址和访问凭证。

非敏感配置保存在 ``config/models.toml``，密钥保存在项目根目录的
``.env``。环境变量具有最高优先级，便于 CI 和部署环境覆盖。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def find_project_root(start: Path | None = None) -> Path:
    """从当前路径向上查找包含 ``pyproject.toml`` 的项目根目录。"""
    starts = [start] if start else [Path.cwd(), Path(__file__)]
    for location in starts:
        current = location.resolve()
        if current.is_file():
            current = current.parent
        for candidate in (current, *current.parents):
            if (candidate / "pyproject.toml").exists():
                return candidate
    raise FileNotFoundError("未找到项目根目录，请从 RAG-Techniques 目录或其子目录运行。")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    base_url: str
    chat_model: str
    embedding_model: str
    evaluation_model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class AzureOpenAISettings:
    api_key: str
    endpoint: str
    api_version: str
    chat_deployment: str
    embedding_deployment: str


@dataclass(frozen=True)
class CohereSettings:
    api_key: str
    chat_model: str
    embedding_model: str
    rerank_model: str


@dataclass(frozen=True)
class GoogleSettings:
    api_key: str
    chat_model: str


@dataclass(frozen=True)
class GroqSettings:
    api_key: str
    fast_model: str
    large_model: str


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str
    embedding_model: str


@dataclass(frozen=True)
class ContextualSettings:
    api_key: str
    base_url: str
    rerank_model: str


@dataclass(frozen=True)
class MilvusSettings:
    uri: str
    token: str


@dataclass(frozen=True)
class LocalModelSettings:
    sentence_transformer_model: str
    cross_encoder_model: str


@dataclass(frozen=True)
class Settings:
    root: Path
    openai: OpenAISettings
    azure_openai: AzureOpenAISettings
    cohere: CohereSettings
    google: GoogleSettings
    groq: GroqSettings
    ollama: OllamaSettings
    contextual: ContextualSettings
    milvus: MilvusSettings
    local_models: LocalModelSettings
    huggingface_token: str

    def require(self, *names: str) -> None:
        """校验运行当前案例所需的环境变量。"""
        missing = [name for name in names if not os.getenv(name)]
        if missing:
            joined = "、".join(missing)
            raise RuntimeError(f"缺少配置：{joined}。请复制 .env.example 为 .env 后填写。")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = find_project_root()
    load_dotenv(root / ".env")
    config_path = root / "config" / "models.toml"
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    openai = raw["openai"]
    azure = raw["azure_openai"]
    cohere = raw["cohere"]
    google = raw["google"]
    groq = raw["groq"]
    ollama = raw["ollama"]
    contextual = raw["contextual"]
    milvus = raw["milvus"]
    local_models = raw["local_models"]

    return Settings(
        root=root,
        openai=OpenAISettings(
            api_key=_env("OPENAI_API_KEY"),
            base_url=_env("OPENAI_BASE_URL", openai["base_url"]),
            chat_model=_env("RAG_CHAT_MODEL", openai["chat_model"]),
            embedding_model=_env("RAG_EMBEDDING_MODEL", openai["embedding_model"]),
            evaluation_model=_env("RAG_EVALUATION_MODEL", openai["evaluation_model"]),
            temperature=_float_env("RAG_TEMPERATURE", openai["temperature"]),
            max_tokens=_int_env("RAG_MAX_TOKENS", openai["max_tokens"]),
        ),
        azure_openai=AzureOpenAISettings(
            api_key=_env("AZURE_OPENAI_API_KEY"),
            endpoint=_env("AZURE_OPENAI_ENDPOINT", azure["endpoint"]),
            api_version=_env("AZURE_OPENAI_API_VERSION", azure["api_version"]),
            chat_deployment=_env("AZURE_OPENAI_CHAT_DEPLOYMENT", azure["chat_deployment"]),
            embedding_deployment=_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", azure["embedding_deployment"]),
        ),
        cohere=CohereSettings(
            api_key=_env("COHERE_API_KEY"),
            chat_model=_env("COHERE_CHAT_MODEL", cohere["chat_model"]),
            embedding_model=_env("COHERE_EMBEDDING_MODEL", cohere["embedding_model"]),
            rerank_model=_env("COHERE_RERANK_MODEL", cohere["rerank_model"]),
        ),
        google=GoogleSettings(
            api_key=_env("GOOGLE_API_KEY"),
            chat_model=_env("GOOGLE_CHAT_MODEL", google["chat_model"]),
        ),
        groq=GroqSettings(
            api_key=_env("GROQ_API_KEY"),
            fast_model=_env("GROQ_FAST_MODEL", groq["fast_model"]),
            large_model=_env("GROQ_LARGE_MODEL", groq["large_model"]),
        ),
        ollama=OllamaSettings(
            base_url=_env("OLLAMA_BASE_URL", ollama["base_url"]),
            embedding_model=_env("OLLAMA_EMBEDDING_MODEL", ollama["embedding_model"]),
        ),
        contextual=ContextualSettings(
            api_key=_env("CONTEXTUAL_API_KEY"),
            base_url=_env("CONTEXTUAL_BASE_URL", contextual["base_url"]),
            rerank_model=_env("CONTEXTUAL_RERANK_MODEL", contextual["rerank_model"]),
        ),
        milvus=MilvusSettings(
            uri=_env("MILVUS_URI", milvus["uri"]),
            token=_env("MILVUS_TOKEN"),
        ),
        local_models=LocalModelSettings(
            sentence_transformer_model=_env(
                "SENTENCE_TRANSFORMER_MODEL", local_models["sentence_transformer_model"]
            ),
            cross_encoder_model=_env("CROSS_ENCODER_MODEL", local_models["cross_encoder_model"]),
        ),
        huggingface_token=_env("HF_TOKEN"),
    )


def apply_runtime_environment(settings: Settings | None = None) -> Settings:
    """将集中配置同步到依赖库约定的标准环境变量。"""
    settings = settings or get_settings()
    values = {
        "OPENAI_API_KEY": settings.openai.api_key,
        "OPENAI_BASE_URL": settings.openai.base_url,
        "AZURE_OPENAI_API_KEY": settings.azure_openai.api_key,
        "AZURE_OPENAI_ENDPOINT": settings.azure_openai.endpoint,
        "AZURE_OPENAI_API_VERSION": settings.azure_openai.api_version,
        "COHERE_API_KEY": settings.cohere.api_key,
        "GOOGLE_API_KEY": settings.google.api_key,
        "GROQ_API_KEY": settings.groq.api_key,
        "HF_TOKEN": settings.huggingface_token,
        "HF_token": settings.huggingface_token,
        "OLLAMA_BASE_URL": settings.ollama.base_url,
        "CONTEXTUAL_API_KEY": settings.contextual.api_key,
        "CONTEXTUAL_BASE_URL": settings.contextual.base_url,
        "MILVUS_URI": settings.milvus.uri,
        "MILVUS_TOKEN": settings.milvus.token,
    }
    for name, value in values.items():
        if value:
            os.environ[name] = value
    return settings
