"""HyDE 假设文档嵌入。

学习目标：先生成假设答案文档，再用其向量召回。
阅读重点：关注生成偏差是否会把检索带离原始问题。

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

# Add the parent directory to the path since we work with notebooks
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from rag_techniques_zh.upstream_helpers import *
from rag_techniques_zh.evaluation import *

# Load environment variables from a .env file

# Set the OpenAI API key environment variable
settings.require("OPENAI_API_KEY")

# Define the HyDe retriever class - creating vector store, generating hypothetical document, and retrieving
class HyDERetriever:
    def __init__(self, files_path, chunk_size=500, chunk_overlap=100):
        self.llm = ChatOpenAI(temperature=0, model_name=CHAT_MODEL, max_tokens=4000)
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = encode_pdf(files_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        self.hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth.
            The document size has to be exactly {chunk_size} characters.""",
        )
        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetical_document(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_chain.invoke(input_variables).content

    def retrieve(self, query, k=3):
        hypothetical_doc = self.generate_hypothetical_document(query)
        similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k=k)
        return similar_docs, hypothetical_doc


# Main class for running the retrieval process
class ClimateChangeRAG:
    def __init__(self, path, query):
        self.retriever = HyDERetriever(path)
        self.query = query

    def run(self):
        # Retrieve results and hypothetical document
        results, hypothetical_doc = self.retriever.retrieve(self.query)

        # Plot the hypothetical document and the retrieved documents
        docs_content = [doc.page_content for doc in results]

        print("Hypothetical document:\n")
        print(text_wrap(hypothetical_doc) + "\n")
        show_context(docs_content)


# Argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(description="Run the Climate Change RAG method.")
    parser.add_argument("--path", type=str, default="data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to process.")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="Query to test the retriever (default: 'What is the main topic of the document?').")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Create and run the RAG method instance
    rag_runner = ClimateChangeRAG(args.path, args.query)
    rag_runner.run()
