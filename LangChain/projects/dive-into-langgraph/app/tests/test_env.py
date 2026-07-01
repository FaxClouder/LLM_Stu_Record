import pathlib
import tempfile
import unittest

from config.env import find_workspace_env


class EnvLoadingTests(unittest.TestCase):
    def test_find_workspace_env_prefers_repository_root_over_project_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            nested = root / "LangChain" / "projects" / "dive-into-langgraph" / "app"
            nested.mkdir(parents=True)

            (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
            (root / ".env").write_text("ROOT_ENV=1\n", encoding="utf-8")
            (nested / ".env").write_text("LOCAL_ENV=1\n", encoding="utf-8")

            self.assertEqual(find_workspace_env(nested), (root / ".env").resolve())

    def test_find_workspace_env_falls_back_to_nearest_local_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app_dir = pathlib.Path(temp_dir) / "app"
            app_dir.mkdir()
            (app_dir / ".env").write_text("LOCAL_ENV=1\n", encoding="utf-8")

            self.assertEqual(find_workspace_env(app_dir), (app_dir / ".env").resolve())


if __name__ == "__main__":
    unittest.main()
