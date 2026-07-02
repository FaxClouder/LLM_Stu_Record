# LangChain Academy 本地接入说明

上游仓库：<https://github.com/langchain-ai/langchain-academy>

当前目录保留了上游模块结构，并补充了本仓库统一环境变量的接入方式。

当前仓库默认推荐使用 DeepSeek 作为学习阶段的 OpenAI 兼容模型服务。

## 上游快照

- 提交：见 [UPSTREAM_VERSION](./UPSTREAM_VERSION)
- 许可证：见 [LICENSE](./LICENSE)

## 这里的改动边界

- 保留 `module-0` 到 `module-6` 的上游结构。
- `module-1` 到 `module-5` 的 `studio/langgraph.json` 直接指向仓库根目录 `.env`。
- 不批量改写上游 Notebook 内容。

## 第一次使用

在仓库根目录基于 `.env.example` 创建 `.env`，并补全课程需要的字段。常用字段包括：

```dotenv
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=
OPENAI_EMBEDDING_MODEL=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
TAVILY_API_KEY=
```

可选执行检查命令：

```powershell
uv run python .\LangChain\tools\sync_langchain_env.py
```

## Notebook 启动方式

上游 Notebook 默认依赖“环境变量已经在当前进程里存在”。因此建议从当前目录启动：

```powershell
uv run --env-file ..\..\..\.env jupyter notebook
```

这样根目录 `.env` 中的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`、`LANGSMITH_*`、`TAVILY_API_KEY` 会自动注入当前会话。

## Studio 启动方式

例如进入 `module-1/studio` 后启动：

```powershell
uv run langgraph dev
```

`langgraph.json` 已经指向仓库根目录 `.env`，所以不会再读取或生成当前目录下的 `.env`。

## 兼容性说明

本仓库已经把主要的 Python / Studio 脚本，以及常用 Notebook 里的可执行 `ChatOpenAI(...)` 代码改成读取 `OPENAI_MODEL`。

仍然可能看到两类上游残留：

- Notebook 的说明文字里还会提到 `gpt-4o`
- 历史输出单元里会保留上游作者运行时的 `gpt-4o` 响应元数据

这些内容不影响你当前用 DeepSeek 运行。
