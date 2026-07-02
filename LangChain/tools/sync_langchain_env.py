from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys

from dotenv import dotenv_values


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    api_key_name: str
    has_api_key: bool
    base_url: str
    chat_model: str
    embedding_model: str
    compare_models: tuple[str, ...]


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "LangChain").is_dir():
            return candidate
    raise FileNotFoundError("未找到仓库根目录。")


def normalize(values: dict[str, str | None]) -> dict[str, str]:
    return {key: (value or "").strip() for key, value in values.items()}


def load_root_env(root: Path) -> dict[str, str]:
    env_path = root / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"未找到 {env_path}。请先从根目录 .env.example 创建 .env。")
    return normalize(dotenv_values(env_path))


def _first(values: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = values.get(name, "")
        if value:
            return value
    return default


def resolve_provider(values: dict[str, str], provider: str | None) -> ProviderConfig:
    selected = (provider or values.get("LANGCHAIN_PROVIDER") or "").strip().lower()
    if not selected:
        if values.get("DEEPSEEK_API_KEY"):
            selected = "deepseek"
        elif values.get("DASHSCOPE_API_KEY"):
            selected = "dashscope"
        elif values.get("OPENAI_API_KEY"):
            selected = "openai"
        else:
            selected = "deepseek"

    shared_chat_model = values.get("LANGCHAIN_CHAT_MODEL") or ""
    shared_embedding_model = values.get("LANGCHAIN_EMBEDDING_MODEL") or ""

    if selected == "openai":
        return ProviderConfig(
            provider=selected,
            api_key_name="OPENAI_API_KEY",
            has_api_key=bool(values.get("OPENAI_API_KEY")),
            base_url=values.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            chat_model=_first(values, "OPENAI_MODEL", "OPENAI_CHAT_MODEL", default=shared_chat_model or "gpt-4o-mini"),
            embedding_model=_first(
                values,
                "OPENAI_EMBEDDING_MODEL",
                default=shared_embedding_model or "text-embedding-3-small",
            ),
            compare_models=("gpt-4o", "gpt-4o-mini"),
        )

    if selected == "deepseek":
        return ProviderConfig(
            provider=selected,
            api_key_name="DEEPSEEK_API_KEY",
            has_api_key=bool(values.get("DEEPSEEK_API_KEY")),
            base_url=values.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            chat_model=_first(values, "DEEPSEEK_MODEL", default=shared_chat_model or "deepseek-chat"),
            embedding_model=_first(
                values,
                "DEEPSEEK_EMBEDDING_MODEL",
                default=shared_embedding_model or "text-embedding-3-small",
            ),
            compare_models=("deepseek-v4-flash", "deepseek-chat"),
        )

    if selected in {"dashscope", "bailian"}:
        return ProviderConfig(
            provider="dashscope",
            api_key_name="DASHSCOPE_API_KEY",
            has_api_key=bool(values.get("DASHSCOPE_API_KEY")),
            base_url=values.get("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            chat_model=_first(values, "DASHSCOPE_MODEL", default=shared_chat_model or "qwen-plus"),
            embedding_model=_first(
                values,
                "DASHSCOPE_EMBEDDING_MODEL",
                default=shared_embedding_model or "text-embedding-v4",
            ),
            compare_models=("qwen-plus", "qwen-turbo"),
        )

    raise RuntimeError(
        f"不支持的 LANGCHAIN_PROVIDER={selected!r}。可选值：openai, deepseek, dashscope"
    )


def resolve_compare_models(values: dict[str, str], provider: ProviderConfig) -> tuple[str, ...]:
    raw = values.get("LANGCHAIN_COMPARE_MODELS", "")
    if raw:
        models = tuple(item.strip() for item in raw.split(",") if item.strip())
        if models:
            return models
    return provider.compare_models


def expected_aliases(provider: ProviderConfig, values: dict[str, str]) -> dict[str, str]:
    compare_models = ",".join(resolve_compare_models(values, provider))
    return {
        "AI_API_KEY": f"same value as {provider.api_key_name}",
        "AI_ENDPOINT": provider.base_url,
        "AI_MODEL": provider.chat_model,
        "AI_EMBEDDING_MODEL": provider.embedding_model,
        "AI_COMPARE_MODELS": compare_models,
        "OPENAI_BASE_URL": provider.base_url,
        "OPENAI_MODEL": provider.chat_model,
        "OPENAI_EMBEDDING_MODEL": provider.embedding_model,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="检查根目录 .env 是否已包含各 LangChain 学习项目所需配置。"
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "deepseek", "dashscope"),
        help="覆盖根 .env 中的 LANGCHAIN_PROVIDER。",
    )
    args = parser.parse_args()

    root = find_repo_root()
    values = load_root_env(root)
    provider = resolve_provider(values, args.provider)
    aliases = expected_aliases(provider, values)

    print(f"Provider: {provider.provider}")
    print(f"Chat model: {provider.chat_model}")
    print(f"Embedding model: {provider.embedding_model}")
    print(f"Compare models: {aliases['AI_COMPARE_MODELS']}")
    print("Project-local .env generation is disabled; all projects should read repository-root .env.")

    missing: list[str] = []
    if not provider.has_api_key:
        missing.append(provider.api_key_name)
    for name in aliases:
        if not values.get(name):
            missing.append(name)

    if missing:
        print("Missing or empty root .env entries:")
        for name in missing:
            hint = aliases.get(name, "")
            suffix = f" ({hint})" if hint else ""
            print(f"  - {name}{suffix}")
        print("Please fill these in the repository-root .env instead of creating child .env files.")
        return 1

    print("Root .env contains the compatibility entries used by LangChain learning projects.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
