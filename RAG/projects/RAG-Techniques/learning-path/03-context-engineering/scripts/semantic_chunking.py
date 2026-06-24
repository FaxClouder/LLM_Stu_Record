"""语义切分。

学习目标：按语义变化而不是固定字符数确定边界。
阅读重点：观察切分稳定性、索引规模和处理耗时。

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

import time
import os
import sys
import argparse
from rag_techniques_zh.upstream_helpers import *
from langchain_experimental.text_splitter import SemanticChunker, BreakpointThresholdType
from langchain_openai.embeddings import OpenAIEmbeddings

# Add the parent directory to the path since we work with notebooks
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Load environment variables from a .env file (e.g., OpenAI API key)
settings.require("OPENAI_API_KEY")


# Function to run semantic chunking and return chunking and retrieval times
class SemanticChunkingRAG:
    """
    A class to handle the Semantic Chunking RAG process for document chunking and query retrieval.
    """

    def __init__(self, path, n_retrieved=2, embeddings=None, breakpoint_type: BreakpointThresholdType = "percentile",
                 breakpoint_amount=90):
        """
        Initializes the SemanticChunkingRAG by encoding the content using a semantic chunker.

        Args:
            path (str): Path to the PDF file to encode.
            n_retrieved (int): Number of chunks to retrieve for each query (default: 2).
            embeddings: Embedding model to use.
            breakpoint_type (str): Type of semantic breakpoint threshold.
            breakpoint_amount (float): Amount for the semantic breakpoint threshold.
        """
        print("\n--- Initializing Semantic Chunking RAG ---")
        # Read PDF to string
        content = read_pdf_to_string(path)

        # Use provided embeddings or initialize OpenAI embeddings
        self.embeddings = embeddings if embeddings else OpenAIEmbeddings(model=EMBEDDING_MODEL)

        # Initialize the semantic chunker
        self.semantic_chunker = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_amount
        )

        # Measure time for semantic chunking
        start_time = time.time()
        self.semantic_docs = self.semantic_chunker.create_documents([content])
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"Semantic Chunking Time: {self.time_records['Chunking']:.2f} seconds")

        # Create a vector store and retriever from the semantic chunks
        self.semantic_vectorstore = FAISS.from_documents(self.semantic_docs, self.embeddings)
        self.semantic_retriever = self.semantic_vectorstore.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        Retrieves and displays the context for the given query.

        Args:
            query (str): The query to retrieve context for.

        Returns:
            tuple: The retrieval time.
        """
        # Measure time for semantic retrieval
        start_time = time.time()
        semantic_context = retrieve_context_per_question(query, self.semantic_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Semantic Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")

        # Display the retrieved context
        show_context(semantic_context)
        return self.time_records


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a PDF document with semantic chunking RAG.")
    parser.add_argument("--path", type=str, default="data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="Number of chunks to retrieve for each query (default: 2).")
    parser.add_argument("--breakpoint_threshold_type", type=str,
                        choices=["percentile", "standard_deviation", "interquartile", "gradient"],
                        default="percentile",
                        help="Type of breakpoint threshold to use for chunking (default: percentile).")
    parser.add_argument("--breakpoint_threshold_amount", type=float, default=90,
                        help="Amount of the breakpoint threshold to use (default: 90).")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of each text chunk in simple chunking (default: 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap between consecutive chunks in simple chunking (default: 200).")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="Query to test the retriever (default: 'What is the main cause of climate change?').")
    parser.add_argument("--experiment", action="store_true",
                        help="Run the experiment to compare performance between semantic chunking and simple chunking.")

    return parser.parse_args()


# Main function to process PDF, chunk text, and test retriever
def main(args):
    # Initialize SemanticChunkingRAG
    semantic_rag = SemanticChunkingRAG(
        path=args.path,
        n_retrieved=args.n_retrieved,
        breakpoint_type=args.breakpoint_threshold_type,
        breakpoint_amount=args.breakpoint_threshold_amount
    )

    # Run a query
    semantic_rag.run(args.query)


if __name__ == '__main__':
    # Call the main function with parsed arguments
    main(parse_args())
