# Dive into LangGraph 本地接入说明

上游仓库：<https://github.com/luochang212/dive-into-langgraph>

当前目录保留了上游教程结构、Notebook、`app/` 实战示例和 `mcp_server/` 示例，并统一读取仓库根目录 `.env`。

## 上游快照

- 提交：见 [UPSTREAM_VERSION](./UPSTREAM_VERSION)
- 许可证：见 [LICENSE](./LICENSE)

## 这里的改动边界

- 保留上游 14 个章节 Notebook、`app/`、`mcp_server/` 和 `skills/` 目录结构。
- 不批量改写上游 Notebook；示例脚本统一走本地公共配置层。
- 不再维护当前项目根目录或 `app/` 子项目下的独立 `.env`。
- Notebook、[`simple_agent.py`](./simple_agent.py)、[`examples/`](./examples/)、[`langgraph.json`](./langgraph.json) 和 [`app/`](./app/) 统一使用仓库根目录 `.env`。

## 第一次使用

先在仓库根目录从 `.env.example` 复制出 `.env` 并补全配置，至少建议填写：

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `DIVE_PROVIDER`、`DIVE_CHAT_MODEL`（可选，用于覆盖当前项目的模型提供商和模型名）
- `DIVE_ADVANCED_CHAT_MODEL`、`DIVE_PARALLEL_CHAT_MODEL`、`DIVE_RAG_CHAT_MODEL`、`DIVE_WEB_SEARCH_CHAT_MODEL`、`DIVE_DEEP_AGENTS_CHAT_MODEL`（可选，用于逐章覆盖模型名）
- `TAVILY_API_KEY`（第 11 章联网搜索可选）
- `DEEPSEEK_*`、`OPENAI_*`、`ARK_*`、`OLLAMA_*`（按你的 provider 选择填写）
- `AMAP_API_KEY`、`OPENSEARCH_*`（仅 `app/` 扩展功能可选）

如果你还需要同步其他 LangChain 学习项目的配置，可以在仓库根目录执行：

```powershell
uv run python .\LangChain\tools\sync_langchain_env.py
```

这个脚本不会再为 `dive-into-langgraph` 生成子目录 `.env`。如果你之前运行过旧版本脚本，当前目录或 `app/` 下可能还残留被 `.gitignore` 忽略的 `.env` 文件；后续请以仓库根目录 `.env` 为准。

## 运行建议

主教程的依赖没有完全包含在仓库根 `pyproject.toml` 中，第一次运行前建议额外安装当前项目的依赖：

```powershell
uv pip install -r .\LangChain\projects\dive-into-langgraph\requirements.txt
```

如果你想运行教程 Notebook，建议从仓库根目录启动：

```powershell
uv run --env-file .\.env jupyter lab .\LangChain\projects\dive-into-langgraph
```

如果你想体验 `langgraph-cli` 调试页面，可在当前目录执行：

```powershell
uv run --project ..\..\.. --env-file ..\..\..\.env langgraph dev
```

如果你想运行 `app/` 里的 Gradio 示例，建议单独进入 `app/` 目录并按它自己的 `pyproject.toml` 管理依赖。应用会自动向上读取仓库根目录 `.env`。注意该子项目要求 `Python >= 3.13`。

## 兼容性说明

这个项目的可执行入口已经统一到本地公共配置层。

这意味着：

- Notebook、`simple_agent.py` 和 `examples/` 会通过 [`dive_config.py`](./dive_config.py) 读取仓库根目录 `.env`
- `simple_agent.py` 和 `examples/` 已通过 [`dive_config.py`](./dive_config.py) 支持 `dashscope`、`deepseek`、`openai`、`ark` 和 `ollama`
- `app/` 子项目支持 `dashscope`、`deepseek`、`openai`、`ark` 和 `ollama`，可以通过 `--provider` 或 `DIVE_APP_PROVIDER` 选择
- 第 11、12 章的联网搜索工具仍使用 DashScope 搜索 API；如果运行这些搜索单元，仍需配置 `DASHSCOPE_API_KEY`

## 建议阅读顺序

1. `1.quickstart.ipynb`
2. `2.stategraph.ipynb`
3. `3.middleware.ipynb`
4. `5.memory.ipynb`
5. `7.mcp_server.ipynb`
6. `8.supervisor.ipynb`
7. `13.gradio_app.ipynb`

如果你只是想快速看一个完整应用，优先阅读 [`app/README.md`](./app/README.md) 和第 13 章。
