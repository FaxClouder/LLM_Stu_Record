# LangChain 学习与项目记录

本目录用于集中存放 LangChain、LangGraph 相关课程、示例工程和本地化学习配置。

## 目录结构

```text
LangChain/
├── README.md          # 本目录说明
├── projects/          # 上游课程和项目快照
├── docs/              # LangChain / LangGraph 专题文档
├── learning-notes/    # 学习笔记与项目分析
└── tools/             # 配置同步和辅助脚本
```

## 当前项目

- [langchain-for-beginners](./projects/langchain-for-beginners/LOCAL_SETUP.zh-CN.md)：Microsoft 的 LangChain 入门课程，已接入本仓库统一 `.env` 配置。
- [langchain-academy](./projects/langchain-academy/LOCAL_SETUP.zh-CN.md)：LangChain 官方 LangGraph 课程，已补充本地 `.env` 与 Studio 配置同步说明。
- [dive-into-langgraph](./projects/dive-into-langgraph/LOCAL_SETUP.zh-CN.md)：中文 LangGraph 1.0 教程与配套示例，已纳入本仓库项目导航，并补充本地接入说明。

## 统一配置方式

根目录 `.env` 作为共享配置源。执行下面的命令后，会自动生成各项目所需的 `.env` 文件：

```powershell
uv run python .\LangChain\tools\sync_langchain_env.py
```

默认建议把 `LANGCHAIN_PROVIDER` 设为 `deepseek`，并读取根目录的 `LANGCHAIN_PROVIDER`、`LANGCHAIN_CHAT_MODEL`、`LANGCHAIN_EMBEDDING_MODEL`、`LANGCHAIN_COMPARE_MODELS`、`LANGSMITH_*`、`TAVILY_API_KEY` 等变量，映射到上游项目需要的字段名。

其中 `dive-into-langgraph` 的 Notebook 和示例主要围绕 DashScope / Qwen 编写，因此除了统一配置外，仍建议在根目录 `.env` 中补全 `DASHSCOPE_API_KEY`。

## 建议顺序

1. 先在根目录补全 `.env`。
2. 运行 `sync_langchain_env.py` 生成子项目配置。
3. 进入目标项目目录，按 `LOCAL_SETUP.zh-CN.md` 的说明启动课程或示例。
