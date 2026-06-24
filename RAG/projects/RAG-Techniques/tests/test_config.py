from pathlib import Path

from rag_techniques_zh.config import get_settings


def test_config_can_be_loaded() -> None:
    settings = get_settings()
    assert settings.root == Path(__file__).resolve().parents[1]
    assert settings.openai.chat_model
    assert settings.openai.embedding_model
    assert settings.openai.base_url.startswith(("http://", "https://"))
    assert settings.cohere.rerank_model
    assert settings.groq.fast_model
    assert settings.google.chat_model
    assert settings.ollama.base_url.startswith(("http://", "https://"))
    assert settings.local_models.cross_encoder_model
