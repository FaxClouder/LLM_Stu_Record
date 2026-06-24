# NirDiamant/RAG_Techniques 项目分析

> 分析日期：2026-06-24  
> 原始项目：[NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)  
> 分析版本：`991e16cc5e4b3aa8f49214ca1ebbd2362b03c985`  
> 内容性质：基于原项目结构和示例主题整理的中文学习说明，不是原文翻译或代码副本。

## 1. 项目定位

`RAG_Techniques` 是一个面向学习和实验的高级 RAG 技术案例库。它没有提供统一的应用框架，而是通过相互独立的 Jupyter Notebook 和部分 Python 脚本，演示不同 RAG 技术的思路与实现。

截至本次分析，仓库包含：

- 42 个 Notebook：37 个技术示例和 5 个评估示例
- 19 个可运行 Python 脚本
- 7 个用于演示的 PDF、CSV、JSON 和文本数据文件
- LangChain 与 LlamaIndex 两类实现
- OpenAI、FAISS、Milvus、知识图谱及多模态相关案例

因此，更准确的理解是：

> 这是一个 RAG 技术“实验手册”和参考实现集合，而不是可直接部署的完整产品。

## 2. 项目结构

```text
RAG_Techniques/
├── all_rag_techniques/                   # 各类技术的 Notebook
├── all_rag_techniques_runnable_scripts/  # 部分 Notebook 对应的 Python 脚本
├── evaluation/                           # RAG 评估示例
├── data/                                 # 演示数据
├── tests/                                # Notebook 和脚本的导入检查
├── helper_functions.py                   # 文档处理、检索等公共函数
└── README.md                             # 技术清单和项目介绍
```

Notebook 是项目主体。Python 脚本只覆盖其中一部分技术，不能认为每个案例都有脚本版本。

## 3. 技术体系

### 3.1 基础流水线

主要解决“先建立一个可运行基线”的问题：

- Basic RAG
- CSV RAG
- Reliable RAG
- Chunk Size Optimization
- Proposition Chunking

学习重点：

1. 文档加载与清洗
2. 文本切分
3. Embedding
4. 向量索引
5. Top-K 检索
6. 上下文组装
7. LLM 回答

基础方案的价值不是效果最好，而是建立后续实验的对照组。

### 3.2 查询增强

主要解决用户问题与知识库表达不一致的问题：

- Query Transformations
- HyDE：先生成假设答案文档，再使用该文档进行向量检索
- HyPE：围绕可能相关的提示或问题表示进行检索增强

适用情况：

- 用户查询过短或表达模糊
- 文档使用专业术语，用户使用口语
- 一个问题实际包含多个检索意图

风险是查询改写可能偏离原始意图，因此需要保留原查询，并分别评估改写前后的召回效果。

### 3.3 文档切分与上下文增强

主要解决固定长度切分导致语义破碎、上下文不足的问题：

- Semantic Chunking
- Contextual Chunk Headers
- Context Window Enhancement
- Relevant Segment Extraction
- Contextual Compression
- Document Augmentation

这些技术可以分为两个阶段：

- 索引阶段：改进 Chunk 的边界、标题、摘要和元数据
- 查询阶段：扩展相邻内容、压缩无关内容或提取关键片段

工程上应优先保证 Chunk 可追溯到原文位置，否则即使回答正确，也难以提供可靠引用。

### 3.4 高级检索与排序

主要解决单一向量相似度召回不稳定的问题：

- Fusion Retrieval
- Reranking
- Multi-faceted Filtering
- Hierarchical Indices
- Dartboard Retrieval
- Graph RAG with Milvus

典型生产方案可以抽象为：

```text
用户问题
  → 查询改写
  → 稀疏检索 + 向量检索
  → 结果融合
  → 元数据过滤
  → Reranker
  → 上下文构建
  → 回答生成
```

这里最值得优先实践的是“混合检索 + Reranker”。它通常比直接引入复杂图结构更容易验证，也更容易控制成本。

### 3.5 自适应与纠错架构

主要解决固定 RAG 流程无法处理不同问题类型的问题：

- Retrieval with Feedback Loop
- Adaptive Retrieval
- Self-RAG
- Corrective RAG（CRAG）
- Agentic RAG

这些方案会增加路由、判断、循环和外部搜索步骤。它们提高了系统处理复杂问题的能力，同时也增加：

- 延迟
- Token 和模型调用成本
- 流程不可预测性
- 调试与评估难度

在基础检索质量没有量化之前，不应优先引入 Agentic RAG。

### 3.6 层次化与图结构

主要面向跨文档总结、多跳问题和全局主题理解：

- Knowledge Graph Integration
- Microsoft GraphRAG
- RAPTOR
- Hierarchical Indices

普通向量检索擅长查找局部相似片段；图结构和层次摘要更适合回答：

- 多个实体之间有什么关系
- 一组文档的共同主题是什么
- 某个结论经过了哪些事件或关系链

代价是索引构建复杂、更新成本高，还需要单独评估实体抽取、关系抽取和摘要质量。

### 3.7 多模态与结构化数据

项目包含：

- CSV RAG
- JSON RAG
- 图片描述驱动的多模态 RAG
- ColPali 路线的文档图像检索

结构化数据不应简单地全部转换为自然语言再做向量检索。实际项目中应根据问题类型选择：

- SQL、过滤或聚合
- 关键词和字段匹配
- 向量检索
- 图像或版面检索

### 3.8 评估

评估目录涉及：

- DeepEval
- GroUSE
- End-to-End RAG Evaluation
- Open-RAG-Eval
- 自定义评估指标

RAG 评估至少要拆分为两个层级：

| 层级 | 重点指标 |
| --- | --- |
| 检索 | Recall@K、Precision@K、MRR、NDCG、命中文档与命中片段 |
| 生成 | 正确性、忠实度、相关性、完整性、引用准确性、幻觉率 |

仅评价最终答案会掩盖问题来源：答案错误可能是没有召回、排序错误、上下文组装失败，也可能是模型没有正确使用上下文。

## 4. 主要技术依赖

从脚本和 Notebook 可以观察到以下技术栈：

- 编排：LangChain、LlamaIndex
- 模型接口：OpenAI
- 向量检索：FAISS、部分 Milvus 案例
- 关键词检索：BM25
- 文档处理：PyMuPDF、Pandas
- NLP：NLTK、spaCy、sentence-transformers
- 数据处理与算法：NumPy、scikit-learn、NetworkX
- 评估：DeepEval、GroUSE、Open-RAG-Eval 等

多数可运行脚本依赖 `OPENAI_API_KEY`。部分 Notebook 还需要独立平台、模型或数据库服务，应逐个阅读其安装单元和环境变量要求。

## 5. 项目优点

- 覆盖范围广，便于建立 RAG 技术地图
- Notebook 形式适合逐步阅读和实验
- 同时包含简单基线与高级架构
- 部分案例提供独立 Python 脚本
- 将检索、上下文、架构和评估放在同一个学习集合中
- 数据样例足以支持小规模验证

## 6. 使用限制与工程风险

### 6.1 依赖不可统一复现

分析版本中没有 `requirements.txt`、`pyproject.toml` 或锁文件，但 GitHub Actions 仍尝试执行：

```text
pip install -r requirements.txt
```

这意味着当前仓库不能依靠根目录配置一次性还原全部示例环境。不同 Notebook 的依赖和版本需要分别确认。

### 6.2 测试覆盖有限

现有测试主要提取并执行 Notebook 和脚本中的 import 语句，能够发现部分缺包或导入错误，但不能验证：

- 检索结果是否正确
- 示例是否能完整运行
- 外部 API 是否兼容
- 指标是否达到预期
- 模型升级后结果是否退化

### 6.3 示例之间缺少统一接口

各技术案例偏向独立演示，没有统一的数据接口、Retriever 协议、配置系统和评估基线。横向比较时需要先进行适配。

### 6.4 外部服务和成本

部分案例依赖付费模型、托管平台或额外数据库。运行前需要检查 API 成本、数据隐私和服务可用性。

### 6.5 许可证限制

项目采用自定义非商业许可证，而不是常见的 MIT、Apache-2.0 或 BSD：

- 仅授权非商业使用、修改和分发
- 使用时必须注明作者、仓库链接以及是否修改
- 商业使用需要作者明确书面许可

因此，本仓库只记录独立分析、学习结论和重新实现思路，不直接复制原项目的大段代码或 Notebook。任何未来的商业项目都应重新确认授权。

## 7. 推荐学习路线

### 第一阶段：建立可测量基线

1. Simple RAG
2. Chunk Size Optimization
3. Reliable RAG
4. 建立 20～50 条人工标注问题
5. 记录 Recall@K、答案忠实度和响应时间

交付物：一个能够重复运行并提供来源引用的最小 RAG。

### 第二阶段：提升召回和排序

1. Query Transformations
2. BM25 与向量混合检索
3. Fusion Retrieval
4. Reranking
5. 对照实验与失败案例记录

交付物：使用同一测试集证明改进是否有效，而不是只展示单个成功案例。

### 第三阶段：改善上下文

1. Semantic Chunking
2. Contextual Chunk Headers
3. Context Window Enhancement
4. Contextual Compression

交付物：建立 Chunk、父文档和原文位置之间的追溯关系。

### 第四阶段：建立评估体系

1. 检索指标
2. LLM-as-a-Judge
3. 引用与幻觉检查
4. 成本和延迟统计
5. 回归测试

交付物：每次调整检索策略后都能输出可比较的评估报告。

### 第五阶段：按问题引入高级架构

- 多跳关系问题：Graph RAG
- 长文档和全局总结：RAPTOR 或层次索引
- 检索结果经常不可靠：CRAG
- 查询类型差异明显：Adaptive Retrieval
- 复杂任务规划：Agentic RAG
- 扫描件、图表和复杂版面：多模态 RAG

不要同时叠加多种高级技术。每次只改变一个核心变量，并与基础方案对照。

## 8. 在本仓库中的改编计划

后续不直接搬运原 Notebook，而是采用统一模板重新实现：

```text
RAG/projects/<technique-name>/
├── README.md          # 问题、原理、实验和结论
├── src/               # 独立实现
├── data/              # 可公开使用的小型数据
├── tests/             # 单元测试和回归测试
├── eval/              # 测试集与评估脚本
├── .env.example       # 环境变量名称，不存放密钥
└── pyproject.toml     # 固定依赖版本
```

每项技术的学习记录应回答：

1. 它解决了基础 RAG 的什么问题？
2. 新增了哪些计算步骤和依赖？
3. 使用什么数据与指标验证？
4. 相比基线提升多少？
5. 增加了多少延迟和成本？
6. 在什么条件下会失败？
7. 是否值得进入生产方案？

## 9. 建议的第一个实践项目

建议先实现 `baseline-hybrid-rag`，范围限定为：

- PDF 或 Markdown 文档
- 固定切分与语义切分对照
- BM25 + 向量混合检索
- Cross-Encoder 或 API Reranker
- 带原文位置的引用
- 检索和生成两级评估

该项目完成后，再选择 HyDE、CRAG 或 Graph RAG 进行单变量扩展。这样可以确保每个高级技术都建立在可测量的基线上。

## 10. 结论

`RAG_Techniques` 的最大价值是提供完整的技术视野和大量实验入口。它的不足是缺乏统一依赖、统一接口、完整执行测试和生产工程约束。

我们的使用原则是：

- 用它建立技术地图
- 用自己的语言整理原理
- 在统一基线上重新实现
- 用固定测试集量化效果
- 明确记录成本、失败条件与许可证边界

这比按顺序运行所有 Notebook 更有利于形成可维护、可验证的 RAG 工程能力。
