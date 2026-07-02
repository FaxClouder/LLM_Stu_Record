# LLM Study Record

用于记录大语言模型相关的学习内容、项目实践与技术总结。

## 目录

- [Agent](./Agent/README.md)：Agent 项目、课程资料、技术说明及学习记录。
- [LangChain](./LangChain/README.md)：LangChain / LangGraph 课程与示例项目接入区。
- [RAG](./RAG/README.md)：RAG 项目、技术说明及改编学习内容。

## 根目录开发环境

仓库根目录已经使用 `uv` 配置好了一个面向 `LangChain`、`LangGraph` 的 Python 开发环境。

### 当前环境内容

- Python：`3.13`
- 包管理：`uv`
- 核心依赖：`langchain`、`langgraph`、`langchain-openai`、`langchain-text-splitters`、`langsmith`
- 开发工具：`jupyterlab`、`ipykernel`、`pytest`、`ruff`

### 常用命令

```powershell
uv sync
uv sync --group langgraph-app
uv sync --group rag
uv sync --group agent
uv sync --group all-learning
uv sync --group rag-multimodal
uv run python
uv run pytest
uv run ruff check .
uv run --env-file .\.env jupyter lab
```

`uv.lock`、`.python-version` 和 `.venv` 都统一放在仓库根目录；子项目里的
`requirements.txt` / `pyproject.toml` 仅作为上游参考或包元数据保留，不再维护独立
`uv.lock`。
`rag-multimodal` 会解析 Torch 等重依赖，只在运行多模态 RAG 案例前同步。

### 环境变量

如需调用模型或开启 LangSmith，先基于根目录的 `.env.example` 创建 `.env`，再填入自己的密钥。
仓库内各克隆项目统一读取根目录 `.env`；不要在子项目目录继续创建或提交重复的 `.env.example`。
