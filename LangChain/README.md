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

仓库根目录 `.env` 是唯一的本地配置源。各克隆项目目录不再维护自己的 `.env` 或 `.env.example`，需要的兼容字段统一写在根目录 `.env.example` 中。
Python 依赖也统一由仓库根目录 `pyproject.toml` / `uv.lock` 管理；子项目里的 `requirements.txt` 仅作为上游参考保留。

可以执行下面的命令检查根目录 `.env` 是否已包含 LangChain 学习项目需要的兼容字段；这个脚本只检查，不再生成子项目 `.env`：

```powershell
uv run python .\LangChain\tools\sync_langchain_env.py
```

默认建议把 `LANGCHAIN_PROVIDER` 设为 `deepseek`，并在根目录 `.env` 中补全所选 provider 的密钥、Base URL 和模型名。`langchain-for-beginners` 使用的 `AI_*` 字段、`langchain-academy` 使用的 `OPENAI_*` 字段也都放在根目录 `.env` 中。

其中 `dive-into-langgraph` 的 Notebook 和示例主要围绕 DashScope / Qwen 编写，因此除了统一配置外，仍建议在根目录 `.env` 中补全 `DASHSCOPE_API_KEY`。

## 建议顺序

1. 先在根目录补全 `.env`。
2. 运行 `sync_langchain_env.py` 检查兼容字段是否齐全。
3. 进入目标项目目录，按 `LOCAL_SETUP.zh-CN.md` 的说明启动课程或示例。
