import unittest
from unittest.mock import patch

from app import AgentService, AppConfig, LLMConfig, MCPConfig


class NamedTool:
    def __init__(self, name):
        self.name = name


class AgentServiceToolTests(unittest.TestCase):
    def _service_for_provider(self, provider):
        service = AgentService(
            AppConfig(
                llm=LLMConfig.from_env(provider),
                mcp=MCPConfig(enabled=frozenset()),
            )
        )
        service._make_search_brief_tool = lambda: NamedTool("subagent_search_brief")
        return service

    def test_dashscope_provider_registers_dashscope_search_tools(self):
        tool_names = [
            tool.name
            for tool in self._service_for_provider("dashscope")._get_local_tools()
        ]

        self.assertIn("role_play", tool_names)
        self.assertIn("dashscope_search", tool_names)
        self.assertIn("subagent_search_brief", tool_names)

    def test_non_dashscope_providers_still_register_role_play(self):
        for provider in ("deepseek", "openai", "ark", "ollama"):
            with self.subTest(provider=provider):
                tool_names = [
                    tool.name
                    for tool in self._service_for_provider(provider)._get_local_tools()
                ]

                self.assertIn("role_play", tool_names)
                self.assertNotIn("dashscope_search", tool_names)
                self.assertNotIn("subagent_search_brief", tool_names)

    def test_deepseek_provider_reads_root_env_names(self):
        with patch.dict(
            "os.environ",
            {
                "DEEPSEEK_API_KEY": "test-key",
                "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
                "DEEPSEEK_MODEL": "deepseek-test",
                "DIVE_CHAT_MODEL": "",
                "LANGCHAIN_CHAT_MODEL": "",
            },
            clear=False,
        ):
            config = LLMConfig.from_env("deepseek")

        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.model, "deepseek-test")
        self.assertEqual(config.base_url, "https://deepseek.example/v1")
        self.assertEqual(config.api_key, "test-key")
        self.assertFalse(config.enable_thinking)


if __name__ == "__main__":
    unittest.main()
