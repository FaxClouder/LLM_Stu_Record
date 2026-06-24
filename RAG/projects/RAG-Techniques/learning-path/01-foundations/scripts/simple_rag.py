"""基础 RAG（LangChain）。

学习目标：完成 PDF 加载、切分、向量化和 Top-K 检索。
阅读重点：理解最小 RAG 数据流，并把它作为所有实验的对照组。

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
import argparse
import time

# Add the parent directory to the path since we work with notebooks
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from rag_techniques_zh.upstream_helpers import *
from rag_techniques_zh.evaluation import *

# Load environment variables from a .env file (e.g., OpenAI API key)
settings.require("OPENAI_API_KEY")


class SimpleRAG:
    """
    A class to handle the Simple RAG process for document chunking and query retrieval.
    """

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2):
        """
        Initializes the SimpleRAGRetriever by encoding the PDF document and creating the retriever.

        Args:
            path (str): Path to the PDF file to encode.
            chunk_size (int): Size of each text chunk (default: 1000).
            chunk_overlap (int): Overlap between consecutive chunks (default: 200).
            n_retrieved (int): Number of chunks to retrieve for each query (default: 2).
        """
        print("\n--- Initializing Simple RAG Retriever ---")

        # Encode the PDF document into a vector store using OpenAI embeddings
        start_time = time.time()
        self.vector_store = encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"Chunking Time: {self.time_records['Chunking']:.2f} seconds")

        # Create a retriever from the vector store
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        Retrieves and displays the context for the given query.

        Args:
            query (str): The query to retrieve context for.

        Returns:
            tuple: The retrieval time.
        """
        # Measure time for retrieval
        start_time = time.time()
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")

        # Display the retrieved context
        show_context(context)


# Function to validate command line inputs
def validate_args(args):
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved must be a positive integer.")
    return args


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Encode a PDF document and test a simple RAG.")
    parser.add_argument("--path", type=str, default="data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of each text chunk (default: 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap between consecutive chunks (default: 200).")
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="Number of chunks to retrieve for each query (default: 2).")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="Query to test the retriever (default: 'What is the main cause of climate change?').")
    parser.add_argument("--evaluate", action="store_true",
                        help="Whether to evaluate the retriever's performance (default: False).")

    # Parse and validate arguments
    return validate_args(parser.parse_args())


# Main function to handle argument parsing and call the SimpleRAGRetriever class
def main(args):
    # Initialize the SimpleRAGRetriever
    simple_rag = SimpleRAG(
        path=args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved
    )

    # Retrieve context based on the query
    simple_rag.run(args.query)

    # Evaluate the retriever's performance on the query (if requested)
    if args.evaluate:
        evaluate_rag(simple_rag.chunks_query_retriever)


if __name__ == '__main__':
    # Call the main function with parsed arguments
    main(parse_args())
