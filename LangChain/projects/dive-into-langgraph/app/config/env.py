"""
Shared environment loading for the dive-into-langgraph app.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def _candidate_dirs(start: Path) -> list[Path]:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    return [current, *current.parents]


def find_workspace_env(start: Path | None = None) -> Path | None:
    """Find the repository-root .env, falling back to the nearest local .env."""
    search_start = start or Path(__file__)
    candidates = _candidate_dirs(search_start)

    for directory in candidates:
        if (directory / "pyproject.toml").exists() and (directory / "LangChain").is_dir():
            env_path = directory / ".env"
            if env_path.exists():
                return env_path

    for directory in candidates:
        env_path = directory / ".env"
        if env_path.exists():
            return env_path

    return None


def load_workspace_env(start: Path | None = None) -> Path | None:
    """Load the shared workspace .env without overriding existing process env."""
    env_path = find_workspace_env(start)
    if env_path is not None:
        load_dotenv(env_path, override=False)
    return env_path
