"""相邻窗口扩展（LangChain）。

学习目标：命中 Chunk 后补充前后相邻片段。
阅读重点：保留父文档和片段位置，避免上下文顺序错误。

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
from rag_techniques_zh.upstream_helpers import *
from rag_techniques_zh.evaluation import *
from typing import List

# Load environment variables from a .env file

# Set the OpenAI API key environment variable
settings.require("OPENAI_API_KEY")


# Function to split text into chunks with metadata of the chunk chronological index
def split_text_to_chunks_with_indices(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(Document(page_content=chunk, metadata={"index": len(chunks), "text": text}))
        start += chunk_size - chunk_overlap
    return chunks


# Function to retrieve a chunk from the vectorstore based on its index in the metadata
def get_chunk_by_index(vectorstore, target_index: int) -> Document:
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
    for doc in all_docs:
        if doc.metadata.get('index') == target_index:
            return doc
    return None


# Function that retrieves from the vectorstore based on semantic similarity and pads each retrieved chunk with its neighboring chunks
def retrieve_with_context_overlap(vectorstore, retriever, query: str, num_neighbors: int = 1, chunk_size: int = 200,
                                  chunk_overlap: int = 20) -> List[str]:
    relevant_chunks = retriever.get_relevant_documents(query)
    result_sequences = []

    for chunk in relevant_chunks:
        current_index = chunk.metadata.get('index')
        if current_index is None:
            continue

        # Determine the range of chunks to retrieve
        start_index = max(0, current_index - num_neighbors)
        end_index = current_index + num_neighbors + 1

        # Retrieve all chunks in the range
        neighbor_chunks = []
        for i in range(start_index, end_index):
            neighbor_chunk = get_chunk_by_index(vectorstore, i)
            if neighbor_chunk:
                neighbor_chunks.append(neighbor_chunk)

        # Sort chunks by their index to ensure correct order
        neighbor_chunks.sort(key=lambda x: x.metadata.get('index', 0))

        # Concatenate chunks, accounting for overlap
        concatenated_text = neighbor_chunks[0].page_content
        for i in range(1, len(neighbor_chunks)):
            current_chunk = neighbor_chunks[i].page_content
            overlap_start = max(0, len(concatenated_text) - chunk_overlap)
            concatenated_text = concatenated_text[:overlap_start] + current_chunk

        result_sequences.append(concatenated_text)

    return result_sequences


# Main class that encapsulates the RAG method
class RAGMethod:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = self._prepare_docs()
        self.vectorstore, self.retriever = self._prepare_retriever()

    def _prepare_docs(self) -> List[Document]:
        content = """
            Artificial Intelligence (AI) has a rich history dating back to the mid-20th century. The term "Artificial Intelligence" was coined in 1956 at the Dartmouth Conference, marking the field's official beginning.
            
            In the 1950s and 1960s, AI research focused on symbolic methods and problem-solving. The Logic Theorist, created in 1955 by Allen Newell and Herbert A. Simon, is often considered the first AI program.
            
            The 1960s saw the development of expert systems, which used predefined rules to solve complex problems. DENDRAL, created in 1965, was one of the first expert systems, designed to analyze chemical compounds.
            
            However, the 1970s brought the first "AI Winter," a period of reduced funding and interest in AI research, largely due to overpromised capabilities and underdelivered results.
            
            The 1980s saw a resurgence with the popularization of expert systems in corporations. The Japanese government's Fifth Generation Computer Project also spurred increased investment in AI research globally.
            
            Neural networks gained prominence in the 1980s and 1990s. The backpropagation algorithm, although discovered earlier, became widely used for training multi-layer networks during this time.
            
            The late 1990s and 2000s marked the rise of machine learning approaches. Support Vector Machines (SVMs) and Random Forests became popular for various classification and regression tasks.
            
            Deep Learning, a subset of machine learning using neural networks with many layers, began to show promising results in the early 2010s. The breakthrough came in 2012 when a deep neural network significantly outperformed other machine learning methods in the ImageNet competition.
            
            Since then, deep learning has revolutionized many AI applications, including image and speech recognition, natural language processing, and game playing. In 2016, Google's AlphaGo defeated a world champion Go player, a landmark achievement in AI.
            
            The current era of AI is characterized by the integration of deep learning with other AI techniques, the development of more efficient and powerful hardware, and the ethical considerations surrounding AI deployment.
            
            Transformers, introduced in 2017, have become a dominant architecture in natural language processing, enabling models like GPT (Generative Pre-trained Transformer) to generate human-like text.
            
            As AI continues to evolve, new challenges and opportunities arise. Explainable AI, robust and fair machine learning, and artificial general intelligence (AGI) are among the key areas of current and future research in the field.
            """
        return split_text_to_chunks_with_indices(content, self.chunk_size, self.chunk_overlap)

    def _prepare_retriever(self):
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(self.docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        return vectorstore, retriever

    def run(self, query: str, num_neighbors: int = 1):
        baseline_chunk = self.retriever.get_relevant_documents(query)
        enriched_chunks = retrieve_with_context_overlap(self.vectorstore, self.retriever, query, num_neighbors,
                                                        self.chunk_size, self.chunk_overlap)
        return baseline_chunk[0].page_content, enriched_chunks[0]


# Argument parsing function
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG method on a given PDF and query.")
    parser.add_argument("--query", type=str, default="When did deep learning become prominent in AI?",
                        help="Query to test the retriever (default: 'What is the main topic of the document?').")
    parser.add_argument('--chunk_size', type=int, default=400, help="Size of text chunks.")
    parser.add_argument('--chunk_overlap', type=int, default=200, help="Overlap between chunks.")
    parser.add_argument('--num_neighbors', type=int, default=1, help="Number of neighboring chunks for context.")
    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    args = parse_args()

    # Initialize and run the RAG method
    rag_method = RAGMethod(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    baseline, enriched = rag_method.run(args.query, num_neighbors=args.num_neighbors)

    print("Baseline Chunk:")
    print(baseline)

    print("\nEnriched Chunks:")
    print(enriched)
