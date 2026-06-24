"""Notebook 启动辅助函数。"""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_notebook() -> tuple[Path, object]:
    """定位项目根目录、加载配置，并返回根目录与 Settings。"""
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            root = candidate
            break
    else:
        raise FileNotFoundError("未找到 RAG-Techniques 项目根目录。")

    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from rag_techniques_zh.config import apply_runtime_environment

    settings = apply_runtime_environment()
    return root, settings

