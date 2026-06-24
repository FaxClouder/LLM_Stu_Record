# RAG Techniques 中文学习与改造版

本项目基于 [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)
的非商业学习许可进行重编排。上游快照版本记录在 `UPSTREAM_VERSION`。

本改造版完成三项工作：

1. 按由浅入深的学习顺序重新组织 42 个 Notebook。
2. 为每个案例增加中文导读、学习重点、验收问题和配置说明。
3. 将模型名、API Base URL 与访问凭证从案例代码中分离。

## 学习路径

- [第一阶段：基础 RAG](./learning-path/01-foundations/README.md)：先建立可运行、可评估的最小基线。（7 个案例）
- [第二阶段：查询增强](./learning-path/02-query-enhancement/README.md)：改善用户问题与知识库表达之间的不匹配。（3 个案例）
- [第三阶段：上下文工程](./learning-path/03-context-engineering/README.md)：改进切分、上下文扩展、压缩和文档增强。（7 个案例）
- [第四阶段：检索与排序](./learning-path/04-retrieval-ranking/README.md)：组合多路召回、过滤和重排，提高检索稳定性。（8 个案例）
- [第五阶段：自适应与智能体 RAG](./learning-path/05-adaptive-agentic/README.md)：根据问题和检索质量动态调整执行流程。（5 个案例）
- [第六阶段：图与层次化 RAG](./learning-path/06-graph-hierarchical/README.md)：处理多跳关系、跨文档主题和长文档总结。（4 个案例）
- [第七阶段：多模态与结构化数据](./learning-path/07-multimodal-structured/README.md)：处理 JSON、表格、图片和复杂文档版面。（3 个案例）
- [第八阶段：评估与回归](./learning-path/08-evaluation/README.md)：拆分检索和生成质量，建立可重复的评估体系。（5 个案例）

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[notebook,nlp,llamaindex,providers]"
Copy-Item .env.example .env
jupyter lab
```

然后：

1. 在 `.env` 中填写实际使用的 API Key。
2. 在 `config/models.toml` 中选择模型和服务地址。
3. 从 `learning-path/01-foundations/simple_rag.ipynb` 开始。

## 配置边界

- `config/models.toml`：模型名称、Base URL、温度、Token 上限等非敏感参数。
- `.env`：API Key、Token、私有 Endpoint 等敏感配置，不提交 Git。
- `src/rag_techniques_zh/config.py`：配置读取、校验和环境变量覆盖。
- `src/rag_techniques_zh/factories.py`：统一创建 Chat、Embedding 和 OpenAI 客户端。

## 目录

```text
RAG-Techniques/
├── config/             # 非敏感模型与服务配置
├── data/               # 上游演示数据
├── learning-path/      # 按学习顺序组织的 Notebook、脚本和中文导读
├── src/                # 公共配置、模型工厂和兼容辅助代码
├── tests/              # 配置与结构测试
├── tools/              # 重建学习项目的工具
├── .env.example        # 密钥模板
└── pyproject.toml      # 基础与可选依赖
```

## 运行原则

- 不在 Notebook 或脚本中填写密钥。
- 每次只调整一个核心变量，并与基础 RAG 对照。
- 检索与生成分开评估。
- 高级案例可能需要额外平台、GPU、数据库或付费 API，运行前先阅读中文导读。
- 图、多模态和评估案例按需安装 `advanced`、`multimodal`、`evaluation` 可选依赖。
- ColPali、Flash Attention 等 GPU 依赖仍应按照对应 Notebook 和本机 CUDA 版本单独安装。

## 许可证

上游项目采用自定义非商业许可证。详见 `LICENSE.UPSTREAM`：

- 仅限非商业使用。
- 使用和改编时需要注明作者、仓库链接以及修改情况。
- 商业用途必须取得原作者书面许可。

本目录中的中文导读和配置改造不改变上游内容的许可证约束。
