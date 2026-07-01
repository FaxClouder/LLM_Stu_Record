from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys

from dotenv import dotenv_values


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    api_key: str
    base_url: str
    chat_model: str
    embedding_model: str
    compare_models: tuple[str, ...]


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("未找到仓库根目录。")


def normalize(values: dict[str, str | None]) -> dict[str, str]:
    return {key: (value or "").strip() for key, value in values.items()}


def require(values: dict[str, str], *names: str) -> None:
    missing = [name for name in names if not values.get(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"缺少必填配置：{joined}")


def load_root_env(root: Path) -> dict[str, str]:
    env_path = root / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"未找到 {env_path}")
    return normalize(dotenv_values(env_path))


def resolve_provider(values: dict[str, str], provider: str | None) -> ProviderConfig:
    selected = (provider or values.get("LANGCHAIN_PROVIDER") or "").strip().lower()
    if not selected:
        if values.get("DEEPSEEK_API_KEY"):
            selected = "deepseek"
        elif values.get("OPENAI_API_KEY"):
            selected = "openai"
        elif values.get("DASHSCOPE_API_KEY"):
            selected = "dashscope"
        else:
            selected = "deepseek"
    shared_chat_model = values.get("LANGCHAIN_CHAT_MODEL") or ""
    shared_embedding_model = values.get("LANGCHAIN_EMBEDDING_MODEL") or ""

    if selected == "openai":
        require(values, "OPENAI_API_KEY")
        return ProviderConfig(
            provider=selected,
            api_key=values["OPENAI_API_KEY"],
            base_url=values.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            chat_model=values.get("OPENAI_CHAT_MODEL") or shared_chat_model or "gpt-4o-mini",
            embedding_model=values.get("OPENAI_EMBEDDING_MODEL")
            or shared_embedding_model
            or "text-embedding-3-small",
            compare_models=("gpt-4o", "gpt-4o-mini"),
        )

    if selected == "deepseek":
        require(values, "DEEPSEEK_API_KEY")
        return ProviderConfig(
            provider=selected,
            api_key=values["DEEPSEEK_API_KEY"],
            base_url=values.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            chat_model=values.get("DEEPSEEK_MODEL") or shared_chat_model or "deepseek-chat",
            embedding_model=values.get("DEEPSEEK_EMBEDDING_MODEL")
            or shared_embedding_model
            or "text-embedding-3-small",
            compare_models=("deepseek-v4-flash", "deepseek-chat"),
        )

    if selected in {"dashscope", "bailian"}:
        require(values, "DASHSCOPE_API_KEY")
        return ProviderConfig(
            provider="dashscope",
            api_key=values["DASHSCOPE_API_KEY"],
            base_url=values.get("DASHSCOPE_BASE_URL") or "",
            chat_model=values.get("DASHSCOPE_MODEL") or shared_chat_model or "qwen-plus",
            embedding_model=values.get("DASHSCOPE_EMBEDDING_MODEL")
            or shared_embedding_model
            or "text-embedding-v4",
            compare_models=("qwen-plus", "qwen-turbo"),
        )

    raise RuntimeError(
        f"不支持的 LANGCHAIN_PROVIDER={selected!r}。可选值：openai, deepseek, dashscope"
    )


def write_env(path: Path, items: list[tuple[str, str]], *, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [header, ""]
    for key, value in items:
        body.append(f"{key}={value}")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def resolve_compare_models(values: dict[str, str], provider: ProviderConfig) -> tuple[str, ...]:
    raw = values.get("LANGCHAIN_COMPARE_MODELS", "")
    if raw:
        models = tuple(item.strip() for item in raw.split(",") if item.strip())
        if models:
            return models
    return provider.compare_models


def sync_langchain_for_beginners(root: Path, provider: ProviderConfig) -> Path:
    project_root = root / "LangChain" / "projects" / "langchain-for-beginners"
    env_path = project_root / ".env"
    values = load_root_env(root)
    compare_models = ",".join(resolve_compare_models(values, provider))
    write_env(
        env_path,
        [
            ("AI_API_KEY", provider.api_key),
            ("AI_ENDPOINT", provider.base_url),
            ("AI_MODEL", provider.chat_model),
            ("AI_EMBEDDING_MODEL", provider.embedding_model),
            ("AI_COMPARE_MODELS", compare_models),
        ],
        header="# Generated from ../../.env by LangChain/tools/sync_langchain_env.py",
    )
    return env_path


def sync_langchain_academy(root: Path, provider: ProviderConfig, values: dict[str, str]) -> list[Path]:
    project_root = root / "LangChain" / "projects" / "langchain-academy"
    root_env_path = project_root / ".env"
    common_items = [
        ("OPENAI_API_KEY", provider.api_key),
        ("OPENAI_BASE_URL", provider.base_url),
        ("OPENAI_MODEL", provider.chat_model),
        ("OPENAI_EMBEDDING_MODEL", provider.embedding_model),
        ("LANGSMITH_API_KEY", values.get("LANGSMITH_API_KEY", "")),
        ("LANGSMITH_TRACING", values.get("LANGSMITH_TRACING", "true") or "true"),
        ("LANGSMITH_TRACING_V2", values.get("LANGSMITH_TRACING", "true") or "true"),
        ("LANGSMITH_PROJECT", "langchain-academy"),
        ("LANGSMITH_ENDPOINT", values.get("LANGSMITH_ENDPOINT", "")),
        ("TAVILY_API_KEY", values.get("TAVILY_API_KEY", "")),
    ]
    write_env(
        root_env_path,
        common_items,
        header="# Generated from ../../.env by LangChain/tools/sync_langchain_env.py",
    )

    studio_paths: list[Path] = [root_env_path]
    for module in ("module-1", "module-2", "module-3", "module-4", "module-5"):
        env_path = project_root / module / "studio" / ".env"
        items = [
            ("OPENAI_API_KEY", provider.api_key),
            ("OPENAI_BASE_URL", provider.base_url),
            ("OPENAI_MODEL", provider.chat_model),
            ("OPENAI_EMBEDDING_MODEL", provider.embedding_model),
        ]
        if values.get("TAVILY_API_KEY") and module == "module-4":
            items.append(("TAVILY_API_KEY", values["TAVILY_API_KEY"]))
        write_env(
            env_path,
            items,
            header="# Generated from ../../../.env by LangChain/tools/sync_langchain_env.py",
        )
        studio_paths.append(env_path)
    return studio_paths


def sync_dive_into_langgraph(root: Path, values: dict[str, str]) -> list[Path]:
    _ = (root, values)
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="同步 LangChain 学习项目所需的 .env 配置。")
    parser.add_argument(
        "--provider",
        choices=("openai", "deepseek", "dashscope"),
        help="覆盖根 .env 中的 LANGCHAIN_PROVIDER。",
    )
    args = parser.parse_args()

    root = find_repo_root()
    values = load_root_env(root)
    provider = resolve_provider(values, args.provider)

    generated_paths = [
        sync_langchain_for_beginners(root, provider),
        *sync_langchain_academy(root, provider, values),
        *sync_dive_into_langgraph(root, values),
    ]

    print(f"Provider: {provider.provider}")
    print(f"Chat model: {provider.chat_model}")
    print(f"Embedding model: {provider.embedding_model}")
    print(f"Compare models: {', '.join(resolve_compare_models(values, provider))}")
    print("Generated files:")
    for path in generated_paths:
        print(f"  - {path.relative_to(root)}")
    print(
        "dive-into-langgraph: uses the repository-root .env directly; "
        "no project-local .env files are generated."
    )

    if provider.provider != "openai":
        print(
            "Note: langchain-academy 的可执行代码已改为读取 OPENAI_MODEL；"
            "少量 Notebook 说明文字和历史输出仍会提到 gpt-4o，这不影响当前运行。"
        )

    if not values.get("DASHSCOPE_API_KEY"):
        print(
            "Note: dive-into-langgraph 的 LLM 初始化已统一读取仓库根目录 .env；"
            "第 11、12 章的 DashScope 搜索工具仍需要 DASHSCOPE_API_KEY。"
        )
    elif provider.provider != "dashscope":
        print(
            "Note: dive-into-langgraph 可通过 DIVE_PROVIDER / DIVE_CHAT_MODEL 覆盖 provider 和模型；"
            "第 11、12 章的 DashScope 搜索工具会继续使用 DASHSCOPE_API_KEY。"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
