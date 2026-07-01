from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dive_config import (
    create_chat_model,
    create_embeddings,
    create_openai_client,
    get_model_config,
    load_project_env,
)

__all__ = [
    "create_chat_model",
    "create_embeddings",
    "create_openai_client",
    "get_model_config",
    "load_project_env",
]
