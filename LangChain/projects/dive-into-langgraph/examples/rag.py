"""
基于 OpenAI-compatible API 的 RAG 示例

参考: https://docs.langchain.com/oss/python/langchain/rag#build-a-rag-agent-with-langchain
"""

import bs4

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from bootstrap import create_chat_model, create_embeddings

# 加载模型
llm = create_chat_model(temperature=0.7)

# 初始化内存向量存储
embeddings = create_embeddings()
vector_store = InMemoryVectorStore(embedding=embeddings)

bs4_strainer = bs4.SoupStrainer(class_=("post"))
loader = WebBaseLoader(
    web_paths=("https://luochang212.github.io/posts/quick_bi_intro/",),
    bs_kwargs={"parse_only": bs4_strainer},
    requests_kwargs={
        "headers": {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        }
    },
)
docs = loader.load()

assert len(docs) == 1
# print(f"Total characters: {len(docs[0].page_content)}")
# print(docs[0].page_content[:500])

# 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

# print(f"Split blog post into {len(all_splits)} sub-documents.")

# 将文档添加到向量存储
document_ids = vector_store.add_documents(documents=all_splits)

# print(document_ids[:3])

# 创建上下文检索工具
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# 创建 ReAct Agent
agent = create_agent(
    llm,
    tools=[retrieve_context],
    system_prompt=(
        # If desired, specify custom instructions
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries."
    )
)

# 测试 ReAct Agent
query = "当前的 Agent 能力，有哪些局限？"

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
