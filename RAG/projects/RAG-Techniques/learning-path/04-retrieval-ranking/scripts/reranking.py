"""重排序（LangChain）。

学习目标：对初步召回结果进行更精确的二次排序。
阅读重点：分别记录初召回数量、重排数量、延迟和准确率。

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
from langchain_core.documents import Document
from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field
import argparse

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from rag_techniques_zh.upstream_helpers import *
from rag_techniques_zh.evaluation import *

# Load environment variables from a .env file

# Set the OpenAI API key environment variable
settings.require("OPENAI_API_KEY")


# Helper Classes and Functions

class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")


def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {doc}
        Relevance Score:"""
    )

    llm = ChatOpenAI(temperature=0, model_name=CHAT_MODEL, max_tokens=4000)
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)

    scored_docs = []
    for doc in docs:
        input_data = {"query": query, "doc": doc.page_content}
        score = llm_chain.invoke(input_data).relevance_score
        try:
            score = float(score)
        except ValueError:
            score = 0  # Default score if parsing fails
        scored_docs.append((doc, score))

    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]


class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str, num_docs=2) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=30)
        return rerank_documents(query, initial_docs, top_n=num_docs)


class CrossEncoderRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")
    cross_encoder: Any = Field(description="Cross-encoder model for reranking")
    k: int = Field(default=5, description="Number of documents to retrieve initially")
    rerank_top_k: int = Field(default=3, description="Number of documents to return after reranking")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.cross_encoder.predict(pairs)
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")


def compare_rag_techniques(query: str, docs: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)

    print("Comparison of Retrieval Techniques")
    print("==================================")
    print(f"Query: {query}\n")

    print("Baseline Retrieval Result:")
    baseline_docs = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(baseline_docs):
        print(f"\nDocument {i + 1}:")
        print(doc.page_content)

    print("\nAdvanced Retrieval Result:")
    custom_retriever = CustomRetriever(vectorstore=vectorstore)
    advanced_docs = custom_retriever.get_relevant_documents(query)
    for i, doc in enumerate(advanced_docs):
        print(f"\nDocument {i + 1}:")
        print(doc.page_content)


# Main class
class RAGPipeline:
    def __init__(self, path: str):
        self.vectorstore = encode_pdf(path)
        self.llm = ChatOpenAI(temperature=0, model_name=CHAT_MODEL)

    def run(self, query: str, retriever_type: str = "reranker"):
        if retriever_type == "reranker":
            retriever = CustomRetriever(vectorstore=self.vectorstore)
        elif retriever_type == "cross_encoder":
            cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            retriever = CrossEncoderRetriever(
                vectorstore=self.vectorstore,
                cross_encoder=cross_encoder,
                k=10,
                rerank_top_k=5
            )
        else:
            raise ValueError("Unknown retriever type. Use 'reranker' or 'cross_encoder'.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain({"query": query})

        print(f"\nQuestion: {query}")
        print(f"Answer: {result['result']}")
        print("\nRelevant source documents:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"\nDocument {i + 1}:")
            print(doc.page_content[:200] + "...")


# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--path", type=str, default="data/Understanding_Climate_Change.pdf", help="Path to the document")
    parser.add_argument("--query", type=str, default='What are the impacts of climate change?', help="Query to ask")
    parser.add_argument("--retriever_type", type=str, default="reranker", choices=["reranker", "cross_encoder"],
                        help="Type of retriever to use")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = RAGPipeline(path=args.path)
    pipeline.run(query=args.query, retriever_type=args.retriever_type)

    # Demonstrate the reranking comparison
    # Example that demonstrates why we should use reranking
    chunks = [
        "The capital of France is great.",
        "The capital of France is huge.",
        "The capital of France is beautiful.",
        """Have you ever visited Paris? It is a beautiful city where you can eat delicious food and see the Eiffel Tower. 
        I really enjoyed all the cities in France, but its capital with the Eiffel Tower is my favorite city.""",
        "I really enjoyed my trip to Paris, France. The city is beautiful and the food is delicious. I would love to visit again. Such a great capital city."
    ]
    docs = [Document(page_content=sentence) for sentence in chunks]

    compare_rag_techniques(query="what is the capital of france?", docs=docs)
