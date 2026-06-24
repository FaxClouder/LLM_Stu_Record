# RAG 学习与项目记录

本目录用于集中记录和存储检索增强生成（Retrieval-Augmented Generation，RAG）相关的项目实践、技术说明，以及经过整理和改编的学习内容。

## 目录结构

```text
RAG/
├── README.md          # 本目录的用途和维护说明
├── projects/          # RAG 项目、实验代码和项目文档
├── docs/              # 原理说明、技术总结和专题文档
└── learning-notes/    # 经过改编、归纳和补充的学习内容
```

## 内容规范

### `projects/`

每个项目使用独立子目录，建议至少包含：

- 项目目标与使用场景
- 数据来源与处理流程
- 检索、重排及生成方案
- 环境配置与运行方法
- 实验结果、问题与后续计划

当前项目：

- [RAG Techniques 中文学习与改造版](./projects/RAG-Techniques/README.md)：按学习顺序重编排 42 个 RAG 案例，提供中文导读和统一模型/API 配置。

### `docs/`

用于存放 RAG 相关的专题说明，例如：

- 文档切分与索引构建
- Embedding 与向量数据库
- 关键词检索、向量检索与混合检索
- Reranker 和上下文构建
- RAG 评估、监控与优化

### `learning-notes/`

用于存放基于课程、文章、论文或实践材料改编的学习内容。记录时应：

- 使用自己的语言重新组织内容
- 标明原始资料名称及来源链接
- 区分原文观点、个人理解和实践结论
- 避免直接复制受版权保护的大段内容
- 记录整理或更新日期

当前学习记录：

- [NirDiamant/RAG_Techniques 项目分析](./learning-notes/nir-diamant-rag-techniques.md)

## 建议命名

- 目录和文件使用小写英文及连字符，例如 `hybrid-search.md`
- 项目目录使用能体现用途的名称，例如 `customer-service-rag/`
- 文档开头注明标题、日期、来源和当前状态
