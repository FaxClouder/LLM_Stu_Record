"""上下文压缩。

学习目标：在生成前删除召回结果中的无关内容。
阅读重点：检查压缩过程是否误删回答所需的关键证据。

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
import time
import argparse
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA
from rag_techniques_zh.upstream_helpers import *
from rag_techniques_zh.evaluation import *

# Add the parent directory to the path since we work with notebooks
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Load environment variables from a .env file
settings.require("OPENAI_API_KEY")


class ContextualCompressionRAG:
    """
    A class to handle the process of creating a retrieval-based Question Answering system
    with a contextual compression retriever.
    """

    def __init__(self, path, model_name=CHAT_MODEL, temperature=0, max_tokens=4000):
        """
        Initializes the ContextualCompressionRAG by setting up the document store and retriever.

        Args:
            path (str): Path to the PDF file to process.
            model_name (str): The name of the language model to use (default: gpt-4o-mini).
            temperature (float): The temperature for the language model.
            max_tokens (int): The maximum tokens for the language model (default: 4000).
        """
        print("\n--- Initializing Contextual Compression RAG ---")
        self.path = path
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Step 1: Create a vector store
        self.vector_store = self._encode_document()

        # Step 2: Create a retriever
        self.retriever = self.vector_store.as_retriever()

        # Step 3: Initialize language model and create a contextual compressor
        self.llm = self._initialize_llm()
        self.compressor = LLMChainExtractor.from_llm(self.llm)

        # Step 4: Combine the retriever with the compressor
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.retriever
        )

        # Step 5: Create a QA chain with the compressed retriever
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.compression_retriever,
            return_source_documents=True
        )

    def _encode_document(self):
        """Helper function to encode the document into a vector store."""
        return encode_pdf(self.path)

    def _initialize_llm(self):
        """Helper function to initialize the language model."""
        return ChatOpenAI(temperature=self.temperature, model_name=self.model_name, max_tokens=self.max_tokens)

    def run(self, query):
        """
        Executes a query using the QA chain and prints the result.

        Args:
            query (str): The query to run against the document.
        """
        print("\n--- Running Query ---")
        start_time = time.time()
        result = self.qa_chain.invoke({"query": query})
        elapsed_time = time.time() - start_time

        # Display the result and the source documents
        print(f"Result: {result['result']}")
        print(f"Source Documents: {result['source_documents']}")
        print(f"Query Execution Time: {elapsed_time:.2f} seconds")
        return result, elapsed_time


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process a PDF document with contextual compression RAG.")
    parser.add_argument("--model_name", type=str, default=CHAT_MODEL,
                        help="Name of the language model to use (default: gpt-4o-mini).")
    parser.add_argument("--path", type=str, default="data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to process.")
    parser.add_argument("--query", type=str, default="What is the main topic of the document?",
                        help="Query to test the retriever (default: 'What is the main topic of the document?').")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Temperature setting for the language model (default: 0).")
    parser.add_argument("--max_tokens", type=int, default=4000,
                        help="Max tokens for the language model (default: 4000).")

    return parser.parse_args()


# Main function to run the RAG pipeline
def main(args):
    # Initialize ContextualCompressionRAG
    contextual_compression_rag = ContextualCompressionRAG(
        path=args.path,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    # Run a query
    contextual_compression_rag.run(args.query)


if __name__ == '__main__':
    # Call the main function with parsed arguments
    main(parse_args())
