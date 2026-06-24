# 第一阶段：基础 RAG

先建立可运行、可评估的最小基线。

## 推荐顺序

| 顺序 | 主题 | 作用 |
| --- | --- | --- |
| 1 | [基础 RAG（LangChain）](./simple_rag.zh-CN.md) | 完成 PDF 加载、切分、向量化和 Top-K 检索。 |
| 2 | [基础 RAG（LlamaIndex）](./simple_rag_with_llamaindex.zh-CN.md) | 使用 LlamaIndex 实现同一条基础流水线。 |
| 3 | [CSV RAG（LangChain）](./simple_csv_rag.zh-CN.md) | 面向表格文本建立基础检索流程。 |
| 4 | [CSV RAG（LlamaIndex）](./simple_csv_rag_with_llamaindex.zh-CN.md) | 使用 LlamaIndex 处理 CSV 数据。 |
| 5 | [可靠 RAG](./reliable_rag.zh-CN.md) | 增加相关性判断和回答约束。 |
| 6 | [Chunk 大小优化](./choose_chunk_size.zh-CN.md) | 比较不同切分长度对检索的影响。 |
| 7 | [命题式切分](./proposition_chunking.zh-CN.md) | 把复杂段落拆分为可独立检索的原子事实。 |

完成本阶段后，请在自己的实验记录中保存：配置快照、测试问题、指标、失败案例和结论。
