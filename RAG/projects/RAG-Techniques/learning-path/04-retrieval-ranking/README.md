# 第四阶段：检索与排序

组合多路召回、过滤和重排，提高检索稳定性。

## 推荐顺序

| 顺序 | 主题 | 作用 |
| --- | --- | --- |
| 1 | [融合检索（LangChain）](./fusion_retrieval.zh-CN.md) | 融合 BM25 与向量检索结果。 |
| 2 | [融合检索（LlamaIndex）](./fusion_retrieval_with_llamaindex.zh-CN.md) | 使用 LlamaIndex 组合多路召回。 |
| 3 | [重排序（LangChain）](./reranking.zh-CN.md) | 对初步召回结果进行更精确的二次排序。 |
| 4 | [重排序（LlamaIndex）](./reranking_with_llamaindex.zh-CN.md) | 在 LlamaIndex 检索链中加入重排。 |
| 5 | [层次索引](./hierarchical_indices.zh-CN.md) | 先检索摘要或主题，再定位细粒度片段。 |
| 6 | [Dartboard 多样性检索](./dartboard.zh-CN.md) | 在相关性之外引入结果多样性。 |
| 7 | [可解释检索](./explainable_retrieval.zh-CN.md) | 解释召回内容与问题之间的关系。 |
| 8 | [MemoRAG 记忆增强检索](./memorag.zh-CN.md) | 通过记忆表示和代理查询扩大召回范围。 |

完成本阶段后，请在自己的实验记录中保存：配置快照、测试问题、指标、失败案例和结论。
