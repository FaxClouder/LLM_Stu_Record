# Agent 学习与项目记录

本目录用于集中存放和记录 Agent 相关的项目实践、课程资料、技术说明和学习笔记。

## 目录结构

```text
Agent/
├── README.md          # 本目录说明
├── projects/          # Agent 项目、课程和上游项目快照
├── docs/              # Agent 专题文档
└── learning-notes/    # 学习笔记与项目分析
```

## 当前项目

- [hello-agents](./projects/hello-agents/README.md)：Datawhale 的系统性智能体学习教程，覆盖 Agent 基础、经典范式、低代码平台、框架实践、记忆与检索、上下文工程、协议、评估、Agentic RL 和综合案例。
- [memU](./projects/memU/README.md)：面向 AI 智能体的个人文件记忆系统（NevaMind-AI），将对话、文档、图像等多模态数据编译为可浏览的 Markdown 文件树（INDEX.md / MEMORY.md / SKILL.md），支持 memorize 写入和 retrieve 检索，具备自演化技能和可插拔存储后端。

## 内容规范

### `projects/`

每个项目使用独立子目录，建议至少记录：

- 项目来源、上游仓库和同步版本
- 项目目标与核心主题
- 本地运行方式和环境依赖
- 关键章节、代码示例和实践路径
- 学习记录、问题整理和后续计划

### `docs/`

用于存放 Agent 相关的专题说明，例如：

- Agent 架构与执行循环
- 工具调用、函数调用和 MCP/A2A 等协议
- 记忆、规划、反思、评估和多智能体协作
- 低代码 Agent 平台与代码框架对比

### `learning-notes/`

用于存放基于课程、文章、论文或项目实践整理的学习内容。记录时建议：

- 使用自己的语言重新组织内容
- 标明原始资料名称和来源链接
- 区分原文观点、个人理解和实践结论
- 避免直接复制受版权保护的大段内容
- 记录整理或更新日期
