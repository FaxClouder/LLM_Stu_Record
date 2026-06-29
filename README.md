# LLM Study Record

用于记录大语言模型相关的学习内容、项目实践与技术总结。

## 目录

- [Agent](./Agent/README.md)：Agent 项目、课程资料、技术说明及学习记录。
- [LangChain](./LangChain/README.md)：LangChain / LangGraph 课程与示例项目接入区。
- [RAG](./RAG/README.md)：RAG 项目、技术说明及改编学习内容。

## 根目录开发环境

仓库根目录已经使用 `uv` 配置好了一个面向 `LangChain`、`LangGraph` 的 Python 开发环境。

### 当前环境内容

- Python：`3.12`
- 包管理：`uv`
- 核心依赖：`langchain`、`langgraph`、`langchain-openai`、`langchain-text-splitters`、`langsmith`
- 开发工具：`jupyterlab`、`ipykernel`、`pytest`、`ruff`

### 常用命令

```powershell
uv sync
uv run python
uv run pytest
uv run ruff check .
uv run jupyter lab
```

### 环境变量

如需调用模型或开启 LangSmith，先基于根目录的 `.env.example` 创建 `.env`，再填入自己的密钥。
