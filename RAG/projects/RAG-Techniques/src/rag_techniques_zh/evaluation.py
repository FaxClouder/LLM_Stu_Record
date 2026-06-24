"""上游评估辅助函数。

学习目标：提供 RAG 自动评估函数。
阅读重点：评审模型统一读取配置。

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

import json
from typing import List, Tuple, Dict, Any

from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 09/15/24 kimmeyh Added path where helper functions is located to the path
# Add the parent directory to the path since we work with notebooks
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rag_techniques_zh.upstream_helpers import (
    create_question_answer_from_context_chain,
    answer_question_from_context,
    retrieve_context_per_question
)

def create_deep_eval_test_cases(
    questions: List[str],
    gt_answers: List[str],
    generated_answers: List[str],
    retrieved_documents: List[str]
) -> List[LLMTestCase]:
    """
    Create a list of LLMTestCase objects for evaluation.

    Args:
        questions (List[str]): List of input questions.
        gt_answers (List[str]): List of ground truth answers.
        generated_answers (List[str]): List of generated answers.
        retrieved_documents (List[str]): List of retrieved documents.

    Returns:
        List[LLMTestCase]: List of LLMTestCase objects.
    """
    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer,
            retrieval_context=retrieved_document
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]

# Define evaluation metrics
correctness_metric = GEval(
    name="Correctness",
    model=EVALUATION_MODEL,
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=EVALUATION_MODEL,
    include_reason=False
)

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model=EVALUATION_MODEL,
    include_reason=True
)

def evaluate_rag(retriever, num_questions: int = 5) -> Dict[str, Any]:
    """
    Evaluates a RAG system using predefined test questions and metrics.
    
    Args:
        retriever: The retriever component to evaluate
        num_questions: Number of test questions to generate
    
    Returns:
        Dict containing evaluation metrics
    """
    
    # Initialize LLM
    llm = ChatOpenAI(temperature=0, model_name=EVALUATION_MODEL)
    
    # Create evaluation prompt
    eval_prompt = PromptTemplate.from_template("""
    Evaluate the following retrieval results for the question.
    
    Question: {question}
    Retrieved Context: {context}
    
    Rate on a scale of 1-5 (5 being best) for:
    1. Relevance: How relevant is the retrieved information to the question?
    2. Completeness: Does the context contain all necessary information?
    3. Conciseness: Is the retrieved context focused and free of irrelevant information?
    
    Provide ratings in JSON format:
    """)
    
    # Create evaluation chain
    eval_chain = (
        eval_prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Generate test questions
    question_gen_prompt = PromptTemplate.from_template(
        "Generate {num_questions} diverse test questions about climate change:"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()
    
    questions = question_chain.invoke({"num_questions": num_questions}).split("\n")
    
    # Evaluate each question
    results = []
    for question in questions:
        # Get retrieval results
        context = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in context])
        
        # Evaluate results
        eval_result = eval_chain.invoke({
            "question": question,
            "context": context_text
        })
        results.append(eval_result)
    
    return {
        "questions": questions,
        "results": results,
        "average_scores": calculate_average_scores(results)
    }

def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    """Calculate average scores across all evaluation results."""
    # Implementation depends on the exact format of your results
    pass

if __name__ == "__main__":
    # Add any necessary setup or configuration here
    # Example: evaluate_rag(your_chunks_query_retriever_function)
    pass
