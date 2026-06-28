# Dive into LangGraph 本地接入说明

上游仓库：<https://github.com/luochang212/dive-into-langgraph>

当前目录保留了上游教程结构、Notebook、`app/` 实战示例和 `mcp_server/` 示例，并接入了本仓库统一 `.env` 同步方式。

## 上游快照

- 提交：见 [UPSTREAM_VERSION](./UPSTREAM_VERSION)
- 许可证：见 [LICENSE](./LICENSE)

## 这里的改动边界

- 保留上游 14 个章节 Notebook、`app/`、`mcp_server/` 和 `skills/` 目录结构。
- 不批量改写上游 Notebook、示例代码或模型选择逻辑。
- 通过根目录 `.env` 统一生成当前项目根目录和 `app/` 子项目所需的 `.env`。

## 第一次使用

先在仓库根目录补全 `.env`，至少建议填写：

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `TAVILY_API_KEY`（第 11 章联网搜索可选）
- `ARK_*`、`AMAP_API_KEY`、`OPENSEARCH_*`（仅 `app/` 扩展功能可选）

然后在仓库根目录执行：

```powershell
uv run python .\LangChain\tools\sync_langchain_env.py
```

这会生成：

- 当前目录下的 `.env`
- [`app/.env`](./app/.env)

其中：

- 项目根目录 `.env` 供章节 Notebook、[`simple_agent.py`](./simple_agent.py) 和 [`langgraph.json`](./langgraph.json) 使用
- `app/.env` 供 [`app/`](./app/) 里的 Gradio 示例使用

## 运行建议

主教程的依赖没有完全包含在仓库根 `pyproject.toml` 中，第一次运行前建议额外安装当前项目的依赖：

```powershell
uv pip install -r .\LangChain\projects\dive-into-langgraph\requirements.txt
```

如果你想运行教程 Notebook，建议从当前目录启动：

```powershell
uv run --env-file .\.env jupyter lab
```

如果你想体验 `langgraph-cli` 调试页面，可在当前目录执行：

```powershell
uv run --env-file .\.env langgraph dev
```

如果你想运行 `app/` 里的 Gradio 示例，建议单独进入 `app/` 目录并按它自己的 `pyproject.toml` 管理依赖。注意该子项目要求 `Python >= 3.13`。

## 兼容性说明

这个项目的大部分 Notebook 和示例代码直接使用了 `DASHSCOPE_*` 环境变量，并且很多单元把模型名写成了 `qwen3-coder-plus`、`qwen3-max`、`qwen3.7-plus` 等 Qwen 系列。

这意味着：

- 最省事的运行方式仍然是使用 DashScope / 百炼
- 如果你想改用 DeepSeek、OpenAI 或其他 OpenAI 兼容服务，通常还需要手动调整代码里的模型名
- `app/` 子项目额外支持 `ark` 和 `ollama`，但默认配置仍然偏向 DashScope

## 建议阅读顺序

1. `1.quickstart.ipynb`
2. `2.stategraph.ipynb`
3. `3.middleware.ipynb`
4. `5.memory.ipynb`
5. `7.mcp_server.ipynb`
6. `8.supervisor.ipynb`
7. `13.gradio_app.ipynb`

如果你只是想快速看一个完整应用，优先阅读 [`app/README.md`](./app/README.md) 和第 13 章。
