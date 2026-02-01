"""
Evaluation Module
Handles RAG evaluation metrics.
"""

from .metrics import (
    RAGEvaluator,
    EvaluationResult,
    evaluate_context_relevance,
    evaluate_faithfulness,
    evaluate_answer_correctness,
    evaluate_answer_relevance,
    run_evaluation
)

__all__ = [
    "RAGEvaluator",
    "EvaluationResult",
    "evaluate_context_relevance",
    "evaluate_faithfulness",
    "evaluate_answer_correctness",
    "evaluate_answer_relevance",
    "run_evaluation",
]
