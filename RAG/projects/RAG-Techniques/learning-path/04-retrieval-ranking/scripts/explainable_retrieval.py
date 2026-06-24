"""可解释检索。

学习目标：解释召回内容与问题之间的关系。
阅读重点：解释文本不能替代真实检索分数和来源元数据。

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

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path
from rag_techniques_zh.upstream_helpers import *
from rag_techniques_zh.evaluation import *

# Load environment variables from a .env file

# Set the OpenAI API key environment variable
settings.require("OPENAI_API_KEY")


# Define utility classes/functions
class ExplainableRetriever:
    def __init__(self, texts):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        self.llm = ChatOpenAI(temperature=0, model_name=CHAT_MODEL, max_tokens=4000)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        explain_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Analyze the relationship between the following query and the retrieved context.
            Explain why this context is relevant to the query and how it might help answer the query.

            Query: {query}

            Context: {context}

            Explanation:
            """
        )
        self.explain_chain = explain_prompt | self.llm

    def retrieve_and_explain(self, query):
        docs = self.retriever.get_relevant_documents(query)
        explained_results = []

        for doc in docs:
            input_data = {"query": query, "context": doc.page_content}
            explanation = self.explain_chain.invoke(input_data).content
            explained_results.append({
                "content": doc.page_content,
                "explanation": explanation
            })
        return explained_results


class ExplainableRAGMethod:
    def __init__(self, texts):
        self.explainable_retriever = ExplainableRetriever(texts)

    def run(self, query):
        return self.explainable_retriever.retrieve_and_explain(query)


# Argument Parsing
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Explainable RAG Method")
    parser.add_argument('--query', type=str, default='Why is the sky blue?', help="Query for the retriever")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Sample texts (these can be replaced by actual data)
    texts = [
        "The sky is blue because of the way sunlight interacts with the atmosphere.",
        "Photosynthesis is the process by which plants use sunlight to produce energy.",
        "Global warming is caused by the increase of greenhouse gases in Earth's atmosphere."
    ]

    explainable_rag = ExplainableRAGMethod(texts)
    results = explainable_rag.run(args.query)

    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Content: {result['content']}")
        print(f"Explanation: {result['explanation']}")
        print()
