# 第三阶段：上下文工程

改进切分、上下文扩展、压缩和文档增强。

## 推荐顺序

| 顺序 | 主题 | 作用 |
| --- | --- | --- |
| 1 | [上下文化 Chunk 标题](./contextual_chunk_headers.zh-CN.md) | 为片段补充文档级标题和语境。 |
| 2 | [相关片段提取](./relevant_segment_extraction.zh-CN.md) | 从较长文档中提取与问题相关的局部内容。 |
| 3 | [相邻窗口扩展（LangChain）](./context_enrichment_window_around_chunk.zh-CN.md) | 命中 Chunk 后补充前后相邻片段。 |
| 4 | [相邻窗口扩展（LlamaIndex）](./context_enrichment_window_around_chunk_with_llamaindex.zh-CN.md) | 使用 LlamaIndex 实现句子窗口检索。 |
| 5 | [语义切分](./semantic_chunking.zh-CN.md) | 按语义变化而不是固定字符数确定边界。 |
| 6 | [上下文压缩](./contextual_compression.zh-CN.md) | 在生成前删除召回结果中的无关内容。 |
| 7 | [文档增强](./document_augmentation.zh-CN.md) | 为文档生成问题、摘要或元数据以改善检索。 |

完成本阶段后，请在自己的实验记录中保存：配置快照、测试问题、指标、失败案例和结论。
