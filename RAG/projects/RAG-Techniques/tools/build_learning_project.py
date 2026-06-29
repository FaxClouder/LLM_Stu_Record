"""将上游 RAG_Techniques 重编排为中文学习项目。

该脚本只用于生成学习目录，不下载依赖、不调用模型，也不会写入任何密钥。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Lesson:
    stage: str
    title: str
    summary: str
    focus: str


STAGES = {
    "01-foundations": ("第一阶段：基础 RAG", "先建立可运行、可评估的最小基线。"),
    "02-query-enhancement": ("第二阶段：查询增强", "改善用户问题与知识库表达之间的不匹配。"),
    "03-context-engineering": ("第三阶段：上下文工程", "改进切分、上下文扩展、压缩和文档增强。"),
    "04-retrieval-ranking": ("第四阶段：检索与排序", "组合多路召回、过滤和重排，提高检索稳定性。"),
    "05-adaptive-agentic": ("第五阶段：自适应与智能体 RAG", "根据问题和检索质量动态调整执行流程。"),
    "06-graph-hierarchical": ("第六阶段：图与层次化 RAG", "处理多跳关系、跨文档主题和长文档总结。"),
    "07-multimodal-structured": ("第七阶段：多模态与结构化数据", "处理 JSON、表格、图片和复杂文档版面。"),
    "08-evaluation": ("第八阶段：评估与回归", "拆分检索和生成质量，建立可重复的评估体系。"),
}


LESSONS = {
    "simple_rag": Lesson("01-foundations", "基础 RAG（LangChain）", "完成 PDF 加载、切分、向量化和 Top-K 检索。", "理解最小 RAG 数据流，并把它作为所有实验的对照组。"),
    "simple_rag_with_llamaindex": Lesson("01-foundations", "基础 RAG（LlamaIndex）", "使用 LlamaIndex 实现同一条基础流水线。", "比较框架抽象差异，不比较未经控制的最终答案。"),
    "simple_csv_rag": Lesson("01-foundations", "CSV RAG（LangChain）", "面向表格文本建立基础检索流程。", "区分向量检索与字段过滤、聚合计算的适用边界。"),
    "simple_csv_rag_with_llamaindex": Lesson("01-foundations", "CSV RAG（LlamaIndex）", "使用 LlamaIndex 处理 CSV 数据。", "检查表头、行结构和元数据是否在切分后保留。"),
    "reliable_rag": Lesson("01-foundations", "可靠 RAG", "增加相关性判断和回答约束。", "识别无答案问题，避免模型在证据不足时强行回答。"),
    "choose_chunk_size": Lesson("01-foundations", "Chunk 大小优化", "比较不同切分长度对检索的影响。", "使用固定测试集衡量召回效果、成本和延迟。"),
    "proposition_chunking": Lesson("01-foundations", "命题式切分", "把复杂段落拆分为可独立检索的原子事实。", "评估信息完整性与索引规模增加之间的权衡。"),
    "query_transformations": Lesson("02-query-enhancement", "查询转换", "演示重写、退一步提问和子问题拆解。", "保留原查询，并分别记录每种改写带来的召回变化。"),
    "HyDe_Hypothetical_Document_Embedding": Lesson("02-query-enhancement", "HyDE 假设文档嵌入", "先生成假设答案文档，再用其向量召回。", "关注生成偏差是否会把检索带离原始问题。"),
    "HyPE_Hypothetical_Prompt_Embeddings": Lesson("02-query-enhancement", "HyPE 假设提示嵌入", "使用假设问题或提示表示增强检索。", "比较 HyPE、HyDE 与直接查询的 Recall@K。"),
    "contextual_chunk_headers": Lesson("03-context-engineering", "上下文化 Chunk 标题", "为片段补充文档级标题和语境。", "验证额外上下文是否提升召回，同时避免引入错误信息。"),
    "relevant_segment_extraction": Lesson("03-context-engineering", "相关片段提取", "从较长文档中提取与问题相关的局部内容。", "区分召回文档与最终送入模型的证据片段。"),
    "context_enrichment_window_around_chunk": Lesson("03-context-engineering", "相邻窗口扩展（LangChain）", "命中 Chunk 后补充前后相邻片段。", "保留父文档和片段位置，避免上下文顺序错误。"),
    "context_enrichment_window_around_chunk_with_llamaindex": Lesson("03-context-engineering", "相邻窗口扩展（LlamaIndex）", "使用 LlamaIndex 实现句子窗口检索。", "比较相同数据和参数下的实现差异。"),
    "semantic_chunking": Lesson("03-context-engineering", "语义切分", "按语义变化而不是固定字符数确定边界。", "观察切分稳定性、索引规模和处理耗时。"),
    "contextual_compression": Lesson("03-context-engineering", "上下文压缩", "在生成前删除召回结果中的无关内容。", "检查压缩过程是否误删回答所需的关键证据。"),
    "document_augmentation": Lesson("03-context-engineering", "文档增强", "为文档生成问题、摘要或元数据以改善检索。", "区分原始事实与模型生成的辅助索引内容。"),
    "fusion_retrieval": Lesson("04-retrieval-ranking", "融合检索（LangChain）", "融合 BM25 与向量检索结果。", "理解分数归一化、权重和 Reciprocal Rank Fusion。"),
    "fusion_retrieval_with_llamaindex": Lesson("04-retrieval-ranking", "融合检索（LlamaIndex）", "使用 LlamaIndex 组合多路召回。", "使用同一测试集与 LangChain 版本对照。"),
    "reranking": Lesson("04-retrieval-ranking", "重排序（LangChain）", "对初步召回结果进行更精确的二次排序。", "分别记录初召回数量、重排数量、延迟和准确率。"),
    "reranking_with_llamaindex": Lesson("04-retrieval-ranking", "重排序（LlamaIndex）", "在 LlamaIndex 检索链中加入重排。", "避免只观察最终答案而忽略排序变化。"),
    "hierarchical_indices": Lesson("04-retrieval-ranking", "层次索引", "先检索摘要或主题，再定位细粒度片段。", "适合长文档；需要验证父子节点映射是否可靠。"),
    "dartboard": Lesson("04-retrieval-ranking", "Dartboard 多样性检索", "在相关性之外引入结果多样性。", "减少相似片段重复占据上下文窗口。"),
    "explainable_retrieval": Lesson("04-retrieval-ranking", "可解释检索", "解释召回内容与问题之间的关系。", "解释文本不能替代真实检索分数和来源元数据。"),
    "memorag": Lesson("04-retrieval-ranking", "MemoRAG 记忆增强检索", "通过记忆表示和代理查询扩大召回范围。", "与标准 RAG 对照其召回增益、索引开销和生成成本。"),
    "retrieval_with_feedback_loop": Lesson("05-adaptive-agentic", "带反馈循环的检索", "利用反馈调整后续检索或排序。", "明确反馈信号来源，防止错误反馈持续放大。"),
    "adaptive_retrieval": Lesson("05-adaptive-agentic", "自适应检索", "根据查询类型选择不同检索策略。", "先建立可解释的路由规则，再考虑使用模型路由。"),
    "self_rag": Lesson("05-adaptive-agentic", "Self-RAG", "动态决定是否检索并检查生成结果的支持度。", "记录每个判断节点，避免把失败隐藏在复杂流程中。"),
    "crag": Lesson("05-adaptive-agentic", "纠错 RAG（CRAG）", "评估召回质量，并在必要时改写或切换信息源。", "外部搜索应具有来源过滤、超时和失败回退。"),
    "Agentic_RAG": Lesson("05-adaptive-agentic", "Agentic RAG", "使用智能体规划复杂文档分析流程。", "重点评估可控性、调用次数、成本和中间状态。"),
    "graph_rag": Lesson("06-graph-hierarchical", "知识图谱 RAG", "结合实体关系和文本检索回答多跳问题。", "分别评估实体抽取、关系抽取和最终检索质量。"),
    "graphrag_with_milvus_vectordb": Lesson("06-graph-hierarchical", "Milvus Graph RAG", "使用 Milvus 存储文本和关系三元组。", "理解关系召回、文本召回和重排之间的协作。"),
    "Microsoft_GraphRag": Lesson("06-graph-hierarchical", "Microsoft GraphRAG", "构建实体、关系、社区及层次摘要。", "适合全局问题，但索引成本和更新复杂度较高。"),
    "raptor": Lesson("06-graph-hierarchical", "RAPTOR 树状检索", "递归聚类并生成多层摘要。", "检查摘要事实损失，以及局部与全局检索的路由策略。"),
    "json_rag": Lesson("07-multimodal-structured", "JSON RAG", "面向层次化 JSON 数据设计索引和查询。", "优先利用字段、路径和过滤，不要完全依赖向量相似度。"),
    "multi_model_rag_with_captioning": Lesson("07-multimodal-structured", "图片描述式多模态 RAG", "先为图片生成文字描述，再进入文本检索流程。", "图片描述质量决定可检索信息上限。"),
    "multi_model_rag_with_colpali": Lesson("07-multimodal-structured", "ColPali 文档图像检索", "直接从页面图像和版面中检索相关内容。", "适合扫描件、图表和复杂布局，资源需求高于文本 RAG。"),
    "define_evaluation_metrics": Lesson("08-evaluation", "定义 RAG 评估指标", "从正确性、忠实度和上下文相关性定义指标。", "先明确指标评价的是检索、生成还是完整链路。"),
    "evaluation_deep_eval": Lesson("08-evaluation", "DeepEval 评估", "使用 DeepEval 构造测试用例和评估指标。", "固定评审模型、提示和阈值，保证回归结果可比较。"),
    "evaluation_grouse": Lesson("08-evaluation", "GroUSE 评估", "评估基于上下文生成的可靠性。", "关注评审器本身的偏差和校准。"),
    "end-2-end_rag_evaluation": Lesson("08-evaluation", "端到端 RAG 评估", "组合数据集、评审指标和完整评估流水线。", "保留失败样本，定位问题发生在召回还是生成阶段。"),
    "open-rag-eval-example": Lesson("08-evaluation", "Open-RAG-Eval", "使用开放评估工具检查相关性、引用和幻觉。", "把自动指标与人工抽样复核结合使用。"),
}


CHAT_MODELS = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4-turbo-preview",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
)

EMBEDDING_MODELS = ("text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002")


def replace_runtime_config(source: str, *, evaluation: bool = False) -> str:
    """把常见硬编码替换为 Notebook/脚本启动单元提供的配置变量。"""
    source = source.replace("from helper_functions import", "from rag_techniques_zh.upstream_helpers import")
    source = source.replace("from evaluation.evalute_rag import", "from rag_techniques_zh.evaluation import")
    source = source.replace("../data/", "data/")
    source = source.replace("os.environ[\"OPENAI_API_KEY\"] = os.getenv('OPENAI_API_KEY')", "settings.require(\"OPENAI_API_KEY\")")
    source = source.replace('os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")', 'settings.require("OPENAI_API_KEY")')
    source = re.sub(
        r'os\.environ\["OPENAI_API_KEY"\]\s*=\s*input\([^\n]*\)',
        'settings.require("OPENAI_API_KEY")',
        source,
    )
    source = re.sub(
        r'os\.environ\["OPENAI_API_KEY"\]\s*=\s*"sk-[^"]*"',
        'settings.require("OPENAI_API_KEY")',
        source,
    )
    source = source.replace("genai.configure(api_key='your-api-key')", "genai.configure(api_key=settings.google.api_key)")
    source = source.replace('base_url = "https://api.contextual.ai/v1"', "base_url = settings.contextual.base_url")

    model_variable = "EVALUATION_MODEL" if evaluation else "CHAT_MODEL"
    for model in CHAT_MODELS:
        source = source.replace(f'"{model}"', model_variable)
        source = source.replace(f"'{model}'", model_variable)
    for model in EMBEDDING_MODELS:
        source = source.replace(f'"{model}"', "EMBEDDING_MODEL")
        source = source.replace(f"'{model}'", "EMBEDDING_MODEL")
    replacements = {
        '"command-r-plus"': "COHERE_CHAT_MODEL",
        "'command-r-plus'": "COHERE_CHAT_MODEL",
        '"embed-english-v3.0"': "COHERE_EMBEDDING_MODEL",
        "'embed-english-v3.0'": "COHERE_EMBEDDING_MODEL",
        '"rerank-english-v3.0"': "COHERE_RERANK_MODEL",
        "'rerank-english-v3.0'": "COHERE_RERANK_MODEL",
        '"gemini-1.5-flash"': "GOOGLE_CHAT_MODEL",
        "'gemini-1.5-flash'": "GOOGLE_CHAT_MODEL",
        '"llama-3.1-70b-versatile"': "GROQ_LARGE_MODEL",
        "'llama-3.1-70b-versatile'": "GROQ_LARGE_MODEL",
        '"llama-3.1-8b-instant"': "GROQ_FAST_MODEL",
        "'llama-3.1-8b-instant'": "GROQ_FAST_MODEL",
        '"mixtral-8x7b-32768"': "GROQ_FAST_MODEL",
        "'mixtral-8x7b-32768'": "GROQ_FAST_MODEL",
        '"nomic-embed-text:v1.5"': "OLLAMA_EMBEDDING_MODEL",
        "'nomic-embed-text:v1.5'": "OLLAMA_EMBEDDING_MODEL",
        '"all-MiniLM-L6-v2"': "SENTENCE_TRANSFORMER_MODEL",
        "'all-MiniLM-L6-v2'": "SENTENCE_TRANSFORMER_MODEL",
        '"cross-encoder/ms-marco-MiniLM-L-6-v2"': "CROSS_ENCODER_MODEL",
        "'cross-encoder/ms-marco-MiniLM-L-6-v2'": "CROSS_ENCODER_MODEL",
        '"ctxl-rerank-en-v1-instruct"': "CONTEXTUAL_RERANK_MODEL",
        "'ctxl-rerank-en-v1-instruct'": "CONTEXTUAL_RERANK_MODEL",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    source = source.replace(
        "os.environ[\"HF_token\"] = 'your-huggingface-api-key'",
        'settings.require("HF_TOKEN")',
    )
    source = source.replace("OpenAIEmbeddings()", "OpenAIEmbeddings(model=EMBEDDING_MODEL)")
    # 清理上游残留的 dotenv 与 Colab 配置代码：配置已由统一配置单元加载。
    source = source.replace("from dotenv import load_dotenv\n", "")
    source = source.replace("import dotenv\n", "")
    source = re.sub(r'^[ \t]*load_dotenv\(\)\s*(?:\n|$)', '', source, flags=re.MULTILINE)
    source = re.sub(r'^[ \t]*dotenv\.load_dotenv\(\)\s*(?:\n|$)', '', source, flags=re.MULTILINE)
    source = re.sub(r'^[ \t]*from google\.colab import[^\n]*\n', '', source, flags=re.MULTILINE)
    source = source.replace("# Load environment variables from a .env file\n", "")
    source = source.replace("# Load environment variables from '.env' file\n", "")
    return source


def bootstrap_source(evaluation: bool) -> str:
    return f"""# 统一配置入口：模型名和服务地址来自 config/models.toml，密钥来自 .env
from pathlib import Path
import os
import sys

_current = Path.cwd().resolve()
PROJECT_ROOT = next(
    candidate for candidate in (_current, *_current.parents)
    if (candidate / "pyproject.toml").exists()
)
_src = str(PROJECT_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from rag_techniques_zh.config import apply_runtime_environment
settings = apply_runtime_environment()
CHAT_MODEL = settings.openai.chat_model
EMBEDDING_MODEL = settings.openai.embedding_model
EVALUATION_MODEL = settings.openai.evaluation_model
OPENAI_API_KEY = settings.openai.api_key
OPENAI_BASE_URL = settings.openai.base_url
CONTEXTUAL_BASE_URL = settings.contextual.base_url
COHERE_CHAT_MODEL = settings.cohere.chat_model
COHERE_EMBEDDING_MODEL = settings.cohere.embedding_model
COHERE_RERANK_MODEL = settings.cohere.rerank_model
GOOGLE_CHAT_MODEL = settings.google.chat_model
GROQ_FAST_MODEL = settings.groq.fast_model
GROQ_LARGE_MODEL = settings.groq.large_model
OLLAMA_EMBEDDING_MODEL = settings.ollama.embedding_model
SENTENCE_TRANSFORMER_MODEL = settings.local_models.sentence_transformer_model
CROSS_ENCODER_MODEL = settings.local_models.cross_encoder_model
CONTEXTUAL_RERANK_MODEL = settings.contextual.rerank_model
if settings.cohere.api_key:
    os.environ.setdefault("CO_API_KEY", settings.cohere.api_key)

print("当前模型配置：", {{
    "chat": CHAT_MODEL,
    "embedding": EMBEDDING_MODEL,
    "evaluation": EVALUATION_MODEL,
    "base_url": OPENAI_BASE_URL,
}})
"""


def guide_text(name: str, lesson: Lesson, notebook_path: Path) -> str:
    stage_title = STAGES[lesson.stage][0]
    return f"""# {lesson.title}

> 学习阶段：{stage_title}  
> 上游文件：`{notebook_path.name}`  
> 配置入口：`config/models.toml` + `.env`

## 目标

{lesson.summary}

## 学习重点

{lesson.focus}

## 建议步骤

1. 先阅读本导读，明确该技术试图解决的基础 RAG 问题。
2. 复制 `.env.example` 为 `.env`，只填写当前案例实际需要的密钥。
3. 从项目根目录启动 JupyterLab，再打开同目录 Notebook。
4. 先使用默认小型数据和少量样本运行，确认流水线正确。
5. 与第一阶段的基础 RAG 使用同一测试问题进行对照。
6. 记录检索指标、最终答案、延迟、Token 或 API 成本以及失败案例。

## 验收问题

- 该技术新增了哪些处理步骤？
- 它改善的是召回、排序、上下文还是生成？
- 相比基础 RAG，指标是否稳定提升？
- 它增加了多少复杂度、延迟和外部依赖？
- 哪些问题或数据条件下不应使用它？

## 配置说明

Notebook 顶部已插入统一配置单元。不要在 Notebook 中直接填写 API Key、模型名或服务地址。
"""


def chinese_intro(name: str, lesson: Lesson, original: str) -> str:
    return f"""# {lesson.title}

> 中文学习版导读｜原始案例：`{original}`  
> 所属阶段：{STAGES[lesson.stage][0]}

## 本节目标

{lesson.summary}

## 阅读重点

{lesson.focus}

## 运行约定

- 从项目根目录启动 JupyterLab。
- 模型和服务地址统一读取 `config/models.toml`。
- API Key 等敏感信息只写入本地 `.env`。
- 本 Notebook 保留上游实现用于技术核对；新增中文导读负责说明学习顺序、配置方式和实验重点。
- 运行前阅读同目录的 `{name}.zh-CN.md`。
"""


def transform_notebook(source: Path, destination: Path, lesson: Lesson) -> None:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    cleaned: list[dict] = []
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        # 跳过上游遥测追踪图片与营销推广单元格。
        if "rag-techniques-views-tracker" in text or "diamant-ai.com" in text or "rag-made-simple" in text:
            continue
        if cell.get("cell_type") == "code":
            text = replace_runtime_config(text, evaluation=lesson.stage == "08-evaluation")
            cell["source"] = text.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
        # 清理后若单元格为空（代码或 markdown）则跳过。
        if not "".join(cell.get("source", [])).strip():
            continue
        cleaned.append(cell)
    notebook["cells"] = cleaned

    name = source.stem
    intro = {
        "cell_type": "markdown",
        "metadata": {"tags": ["中文导读"]},
        "source": chinese_intro(name, lesson, source.name).splitlines(keepends=True),
    }
    config = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["统一配置", "parameters"]},
        "outputs": [],
        "source": bootstrap_source(lesson.stage == "08-evaluation").splitlines(keepends=True),
    }
    notebook["cells"] = [intro, config, *notebook.get("cells", [])]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def script_header(lesson: Lesson) -> str:
    return f'''"""{lesson.title}。

学习目标：{lesson.summary}
阅读重点：{lesson.focus}

模型、API 地址和访问凭证由项目统一配置层提供，禁止在脚本中硬编码。
"""

from rag_techniques_zh.config import apply_runtime_environment

settings = apply_runtime_environment()
CHAT_MODEL = settings.openai.chat_model
EMBEDDING_MODEL = settings.openai.embedding_model
EVALUATION_MODEL = settings.openai.evaluation_model
COHERE_CHAT_MODEL = settings.cohere.chat_model
COHERE_EMBEDDING_MODEL = settings.cohere.embedding_model
COHERE_RERANK_MODEL = settings.cohere.rerank_model
GOOGLE_CHAT_MODEL = settings.google.chat_model
GROQ_FAST_MODEL = settings.groq.fast_model
GROQ_LARGE_MODEL = settings.groq.large_model
OLLAMA_EMBEDDING_MODEL = settings.ollama.embedding_model
SENTENCE_TRANSFORMER_MODEL = settings.local_models.sentence_transformer_model
CROSS_ENCODER_MODEL = settings.local_models.cross_encoder_model
CONTEXTUAL_RERANK_MODEL = settings.contextual.rerank_model

'''


def transform_python(source: Path, destination: Path, lesson: Lesson, *, evaluation: bool = False) -> None:
    text = source.read_text(encoding="utf-8")
    if text.startswith('"""'):
        end = text.find('"""', 3)
        if end != -1:
            text = text[end + 3 :].lstrip()
    text = replace_runtime_config(text, evaluation=evaluation)
    text = text.replace("from dotenv import load_dotenv\n", "")
    text = text.replace("load_dotenv()\n", "")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(script_header(lesson) + text, encoding="utf-8")


def write_stage_readmes(target: Path, grouped: dict[str, list[tuple[str, Lesson]]]) -> None:
    for stage, lessons in grouped.items():
        title, summary = STAGES[stage]
        lines = [
            f"# {title}",
            "",
            summary,
            "",
            "## 推荐顺序",
            "",
            "| 顺序 | 主题 | 作用 |",
            "| --- | --- | --- |",
        ]
        for index, (name, lesson) in enumerate(lessons, start=1):
            lines.append(f"| {index} | [{lesson.title}](./{name}.zh-CN.md) | {lesson.summary} |")
        lines.extend(
            [
                "",
                "完成本阶段后，请在自己的实验记录中保存：配置快照、测试问题、指标、失败案例和结论。",
                "",
            ]
        )
        (target / "learning-path" / stage / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_root_readme(target: Path, grouped: dict[str, list[tuple[str, Lesson]]]) -> None:
    stage_lines = []
    for stage, lessons in grouped.items():
        title, summary = STAGES[stage]
        stage_lines.append(f"- [{title}](./learning-path/{stage}/README.md)：{summary}（{len(lessons)} 个案例）")

    text = f"""# RAG Techniques 中文学习与改造版

本项目基于 [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)
的非商业学习许可进行重编排。上游快照版本记录在 `UPSTREAM_VERSION`。

本改造版完成三项工作：

1. 按由浅入深的学习顺序重新组织 42 个 Notebook。
2. 为每个案例增加中文导读、学习重点、验收问题和配置说明。
3. 将模型名、API Base URL 与访问凭证从案例代码中分离。

## 学习路径

{chr(10).join(stage_lines)}

## 快速开始

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
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
"""
    (target / "README.md").write_text(text, encoding="utf-8")


def build(source: Path, target: Path) -> None:
    source = source.resolve()
    target = target.resolve()
    if not (source / "all_rag_techniques").is_dir():
        raise FileNotFoundError(f"不是有效的上游仓库：{source}")
    if target == source or source in target.parents:
        raise ValueError("目标目录不能覆盖上游仓库。")

    target.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[str, Lesson]]] = {stage: [] for stage in STAGES}

    notebook_sources = list((source / "all_rag_techniques").glob("*.ipynb"))
    notebook_sources += list((source / "evaluation").glob("*.ipynb"))
    found = {path.stem for path in notebook_sources}
    missing = found - LESSONS.keys()
    stale = LESSONS.keys() - found
    if missing or stale:
        raise RuntimeError(f"案例映射不完整。未映射：{sorted(missing)}；不存在：{sorted(stale)}")

    for notebook_source in sorted(notebook_sources, key=lambda path: path.name.lower()):
        name = notebook_source.stem
        lesson = LESSONS[name]
        stage_dir = target / "learning-path" / lesson.stage
        destination = stage_dir / notebook_source.name
        transform_notebook(notebook_source, destination, lesson)
        (stage_dir / f"{name}.zh-CN.md").write_text(
            guide_text(name, lesson, destination),
            encoding="utf-8",
        )
        grouped[lesson.stage].append((name, lesson))

        script_source = source / "all_rag_techniques_runnable_scripts" / f"{name}.py"
        if script_source.exists():
            transform_python(script_source, stage_dir / "scripts" / script_source.name, lesson)

    # 公共辅助代码也接入统一配置，但尽量保持上游算法逻辑。
    helper_lesson = Lesson("01-foundations", "上游公共辅助函数", "提供文档处理、检索和问答公共函数。", "改造重点是统一 Embedding 和 API 配置。")
    transform_python(
        source / "helper_functions.py",
        target / "src" / "rag_techniques_zh" / "upstream_helpers.py",
        helper_lesson,
    )
    evaluation_source = source / "evaluation" / "evalute_rag.py"
    if evaluation_source.exists():
        evaluation_lesson = Lesson("08-evaluation", "上游评估辅助函数", "提供 RAG 自动评估函数。", "评审模型统一读取配置。")
        transform_python(
            evaluation_source,
            target / "src" / "rag_techniques_zh" / "evaluation.py",
            evaluation_lesson,
            evaluation=True,
        )

    for folder in ("data", "images"):
        source_folder = source / folder
        if source_folder.exists():
            shutil.copytree(source_folder, target / folder, dirs_exist_ok=True)

    shutil.copy2(source / "LICENSE", target / "LICENSE.UPSTREAM")
    shutil.copy2(source / "README.md", target / "UPSTREAM_README.md")
    version = "unknown"
    git_head = source / ".git" / "HEAD"
    if git_head.exists():
        import subprocess

        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
    (target / "UPSTREAM_VERSION").write_text(version + "\n", encoding="utf-8")

    for stage in grouped:
        grouped[stage].sort(key=lambda item: list(LESSONS).index(item[0]))
    write_stage_readmes(target, grouped)
    write_root_readme(target, grouped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()
    build(args.source, args.target)


if __name__ == "__main__":
    main()
