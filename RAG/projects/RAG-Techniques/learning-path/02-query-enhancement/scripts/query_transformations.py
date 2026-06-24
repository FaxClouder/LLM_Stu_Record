"""查询转换。

学习目标：演示重写、退一步提问和子问题拆解。
阅读重点：保留原查询，并分别记录每种改写带来的召回变化。

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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables from a .env file

# Set the OpenAI API key environment variable
settings.require("OPENAI_API_KEY")


# Function for rewriting a query to improve retrieval
def rewrite_query(original_query, llm_chain):
    """
    Rewrite the original query to improve retrieval.

    Args:
    original_query (str): The original user query
    llm_chain: The chain used to generate the rewritten query

    Returns:
    str: The rewritten query
    """
    response = llm_chain.invoke(original_query)
    return response.content


# Function for generating a step-back query to retrieve broader context
def generate_step_back_query(original_query, llm_chain):
    """
    Generate a step-back query to retrieve broader context.

    Args:
    original_query (str): The original user query
    llm_chain: The chain used to generate the step-back query

    Returns:
    str: The step-back query
    """
    response = llm_chain.invoke(original_query)
    return response.content


# Function for decomposing a query into simpler sub-queries
def decompose_query(original_query, llm_chain):
    """
    Decompose the original query into simpler sub-queries.

    Args:
    original_query (str): The original complex query
    llm_chain: The chain used to generate sub-queries

    Returns:
    List[str]: A list of simpler sub-queries
    """
    response = llm_chain.invoke(original_query).content
    sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Sub-queries:')]
    return sub_queries


# Main class for the RAG method
class RAGQueryProcessor:
    def __init__(self):
        # Initialize LLM models
        self.re_write_llm = ChatOpenAI(temperature=0, model_name=CHAT_MODEL, max_tokens=4000)
        self.step_back_llm = ChatOpenAI(temperature=0, model_name=CHAT_MODEL, max_tokens=4000)
        self.sub_query_llm = ChatOpenAI(temperature=0, model_name=CHAT_MODEL, max_tokens=4000)

        # Initialize prompt templates
        query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
        Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

        Original query: {original_query}

        Rewritten query:"""
        step_back_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
        Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

        Original query: {original_query}

        Step-back query:"""
        subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
        Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

        Original query: {original_query}

        example: What are the impacts of climate change on the environment?

        Sub-queries:
        1. What are the impacts of climate change on biodiversity?
        2. How does climate change affect the oceans?
        3. What are the effects of climate change on agriculture?
        4. What are the impacts of climate change on human health?"""

        # Create LLMChains
        self.query_rewriter = PromptTemplate(input_variables=["original_query"],
                                             template=query_rewrite_template) | self.re_write_llm
        self.step_back_chain = PromptTemplate(input_variables=["original_query"],
                                              template=step_back_template) | self.step_back_llm
        self.subquery_decomposer_chain = PromptTemplate(input_variables=["original_query"],
                                                        template=subquery_decomposition_template) | self.sub_query_llm

    def run(self, original_query):
        """
        Run the full RAG query processing pipeline.

        Args:
        original_query (str): The original query to be processed
        """
        # Rewrite the query
        rewritten_query = rewrite_query(original_query, self.query_rewriter)
        print("Original query:", original_query)
        print("\nRewritten query:", rewritten_query)

        # Generate step-back query
        step_back_query = generate_step_back_query(original_query, self.step_back_chain)
        print("\nStep-back query:", step_back_query)

        # Decompose the query into sub-queries
        sub_queries = decompose_query(original_query, self.subquery_decomposer_chain)
        print("\nSub-queries:")
        for i, sub_query in enumerate(sub_queries, 1):
            print(f"{i}. {sub_query}")


# Argument parsing
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Process a query using the RAG method.")
    parser.add_argument("--query", type=str, default='What are the impacts of climate change on the environment?',
                        help="The original query to be processed")
    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    args = parse_args()
    processor = RAGQueryProcessor()
    processor.run(args.query)
