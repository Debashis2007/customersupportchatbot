"""
RAG Evaluation Metrics
Implements evaluation metrics for RAG systems:
- Context Relevance: How relevant are retrieved documents?
- Faithfulness: Is the answer grounded in the context?
- Answer Correctness: How accurate is the answer?
- Answer Relevance: Does the answer address the question?
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Represents evaluation results for a single query."""
    query: str
    answer: str
    context: List[str]
    ground_truth: Optional[str] = None
    
    # Scores (0-1)
    context_relevance: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    answer_correctness: float = 0.0
    
    # Detailed feedback
    feedback: Dict[str, str] = field(default_factory=dict)
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        weights = {
            "context_relevance": 0.25,
            "faithfulness": 0.30,
            "answer_relevance": 0.25,
            "answer_correctness": 0.20
        }
        
        score = (
            self.context_relevance * weights["context_relevance"] +
            self.faithfulness * weights["faithfulness"] +
            self.answer_relevance * weights["answer_relevance"] +
            self.answer_correctness * weights["answer_correctness"]
        )
        
        return round(score, 3)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "scores": {
                "context_relevance": self.context_relevance,
                "faithfulness": self.faithfulness,
                "answer_relevance": self.answer_relevance,
                "answer_correctness": self.answer_correctness,
                "overall": self.overall_score
            },
            "feedback": self.feedback
        }


class RAGEvaluator:
    """
    RAG Evaluation Framework
    
    Evaluates RAG systems on multiple dimensions:
    
    1. Context Relevance
       - Are the retrieved documents relevant to the query?
       - Measured by semantic similarity and LLM judgment
    
    2. Faithfulness (Groundedness)
       - Is the answer supported by the retrieved context?
       - Checks for hallucinations
       - Verifies claims against source documents
    
    3. Answer Relevance
       - Does the answer actually address the question?
       - Measures if the response is on-topic
    
    4. Answer Correctness
       - Is the answer factually correct?
       - Requires ground truth for comparison
    
    Evaluation Methods:
    - LLM-as-a-Judge: Use LLM to evaluate responses
    - Embedding similarity: Compare embeddings
    - Lexical overlap: Token/n-gram matching
    - Human evaluation: Manual scoring
    """
    
    def __init__(
        self,
        llm_client,
        embedding_model=None,
        use_ragas: bool = False
    ):
        """
        Initialize the evaluator.
        
        Args:
            llm_client: LLM client for evaluation
            embedding_model: Optional embedding model for similarity
            use_ragas: Whether to use RAGAS library
        """
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.use_ragas = use_ragas
        
        if use_ragas:
            try:
                from ragas import evaluate
                from ragas.metrics import (
                    context_relevancy,
                    faithfulness,
                    answer_relevancy,
                    answer_correctness
                )
                self.ragas_evaluate = evaluate
                self.ragas_metrics = {
                    "context_relevancy": context_relevancy,
                    "faithfulness": faithfulness,
                    "answer_relevancy": answer_relevancy,
                    "answer_correctness": answer_correctness
                }
                logger.info("RAGAS evaluation enabled")
            except ImportError:
                logger.warning("RAGAS not installed. Using custom evaluation.")
                self.use_ragas = False
    
    def evaluate(
        self,
        query: str,
        answer: str,
        context: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a single RAG response.
        
        Args:
            query: User's question
            answer: Generated answer
            context: Retrieved context documents
            ground_truth: Optional correct answer for comparison
            
        Returns:
            EvaluationResult with scores
        """
        result = EvaluationResult(
            query=query,
            answer=answer,
            context=context,
            ground_truth=ground_truth
        )
        
        # Evaluate context relevance
        result.context_relevance, result.feedback["context_relevance"] = \
            self._evaluate_context_relevance(query, context)
        
        # Evaluate faithfulness
        result.faithfulness, result.feedback["faithfulness"] = \
            self._evaluate_faithfulness(answer, context)
        
        # Evaluate answer relevance
        result.answer_relevance, result.feedback["answer_relevance"] = \
            self._evaluate_answer_relevance(query, answer)
        
        # Evaluate answer correctness (if ground truth provided)
        if ground_truth:
            result.answer_correctness, result.feedback["answer_correctness"] = \
                self._evaluate_answer_correctness(answer, ground_truth)
        else:
            result.answer_correctness = 0.0
            result.feedback["answer_correctness"] = "No ground truth provided"
        
        return result
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple test cases.
        
        Args:
            test_cases: List of dicts with keys: query, answer, context, ground_truth
            
        Returns:
            List of EvaluationResults
        """
        results = []
        
        for i, case in enumerate(test_cases):
            logger.info(f"Evaluating case {i+1}/{len(test_cases)}")
            
            result = self.evaluate(
                query=case["query"],
                answer=case["answer"],
                context=case.get("context", []),
                ground_truth=case.get("ground_truth")
            )
            results.append(result)
        
        return results
    
    def _evaluate_context_relevance(
        self,
        query: str,
        context: List[str]
    ) -> tuple:
        """
        Evaluate context relevance.
        
        Measures how relevant the retrieved documents are to the query.
        """
        if not context:
            return 0.0, "No context provided"
        
        prompt = f"""Evaluate how relevant the following context documents are to the question.

Question: {query}

Context documents:
{chr(10).join([f"Document {i+1}: {doc[:500]}" for i, doc in enumerate(context)])}

Rate the overall relevance from 0 to 10, where:
- 0-2: Context is completely irrelevant
- 3-4: Context is mostly irrelevant with minor connections
- 5-6: Context is somewhat relevant but missing key information
- 7-8: Context is mostly relevant and useful
- 9-10: Context is highly relevant and comprehensive

Respond in this format:
Score: [number]
Reasoning: [brief explanation]"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=200, temperature=0)
            
            # Parse score
            score = self._extract_score(response)
            reasoning = self._extract_reasoning(response)
            
            return score / 10, reasoning
        except Exception as e:
            logger.error(f"Context relevance evaluation failed: {e}")
            return 0.5, f"Evaluation error: {e}"
    
    def _evaluate_faithfulness(
        self,
        answer: str,
        context: List[str]
    ) -> tuple:
        """
        Evaluate faithfulness (groundedness).
        
        Measures whether the answer is supported by the context.
        Detects hallucinations and unsupported claims.
        """
        if not context:
            return 0.0, "No context to verify against"
        
        context_text = "\n\n".join(context)
        
        prompt = f"""Evaluate if the following answer is faithful to and supported by the context.

Context:
{context_text[:3000]}

Answer:
{answer}

Analyze the answer for:
1. Claims that are directly supported by the context
2. Claims that are NOT supported (hallucinations)
3. Claims that contradict the context

Rate faithfulness from 0 to 10, where:
- 0-2: Answer mostly contains unsupported claims or hallucinations
- 3-4: Answer has significant unsupported claims
- 5-6: Answer is partially supported but has some unsupported claims
- 7-8: Answer is mostly supported with minor unsupported details
- 9-10: Answer is fully supported by the context

Respond in this format:
Score: [number]
Reasoning: [brief explanation including any hallucinations found]"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=300, temperature=0)
            
            score = self._extract_score(response)
            reasoning = self._extract_reasoning(response)
            
            return score / 10, reasoning
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return 0.5, f"Evaluation error: {e}"
    
    def _evaluate_answer_relevance(
        self,
        query: str,
        answer: str
    ) -> tuple:
        """
        Evaluate answer relevance.
        
        Measures whether the answer addresses the question.
        """
        prompt = f"""Evaluate how well the answer addresses the question.

Question: {query}

Answer: {answer}

Rate answer relevance from 0 to 10, where:
- 0-2: Answer is completely off-topic
- 3-4: Answer barely addresses the question
- 5-6: Answer partially addresses the question but misses key aspects
- 7-8: Answer addresses most of the question well
- 9-10: Answer fully and directly addresses all aspects of the question

Respond in this format:
Score: [number]
Reasoning: [brief explanation]"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=200, temperature=0)
            
            score = self._extract_score(response)
            reasoning = self._extract_reasoning(response)
            
            return score / 10, reasoning
        except Exception as e:
            logger.error(f"Answer relevance evaluation failed: {e}")
            return 0.5, f"Evaluation error: {e}"
    
    def _evaluate_answer_correctness(
        self,
        answer: str,
        ground_truth: str
    ) -> tuple:
        """
        Evaluate answer correctness.
        
        Compares the answer against the ground truth.
        """
        prompt = f"""Compare the generated answer with the ground truth answer.

Ground Truth: {ground_truth}

Generated Answer: {answer}

Evaluate correctness considering:
1. Factual accuracy
2. Completeness
3. Any incorrect information

Rate correctness from 0 to 10, where:
- 0-2: Answer is mostly incorrect
- 3-4: Answer has significant errors
- 5-6: Answer is partially correct
- 7-8: Answer is mostly correct with minor issues
- 9-10: Answer is fully correct

Respond in this format:
Score: [number]
Reasoning: [brief explanation of differences]"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=200, temperature=0)
            
            score = self._extract_score(response)
            reasoning = self._extract_reasoning(response)
            
            return score / 10, reasoning
        except Exception as e:
            logger.error(f"Answer correctness evaluation failed: {e}")
            return 0.5, f"Evaluation error: {e}"
    
    def _extract_score(self, response: str) -> float:
        """Extract numeric score from response."""
        import re
        
        # Look for "Score: X" pattern
        match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return min(max(score, 0), 10)
        
        # Fallback: look for any number at start
        match = re.search(r'^(\d+(?:\.\d+)?)', response.strip())
        if match:
            score = float(match.group(1))
            return min(max(score, 0), 10)
        
        return 5.0  # Default middle score
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response."""
        import re
        
        # Look for "Reasoning: ..." pattern
        match = re.search(r'Reasoning:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()[:500]
        
        return response[:500]
    
    def generate_report(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Generate evaluation report from results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Report dictionary with aggregate statistics
        """
        if not results:
            return {"error": "No results to report"}
        
        # Calculate averages
        avg_context_relevance = sum(r.context_relevance for r in results) / len(results)
        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_answer_relevance = sum(r.answer_relevance for r in results) / len(results)
        
        # Answer correctness only for those with ground truth
        with_ground_truth = [r for r in results if r.ground_truth]
        avg_correctness = (
            sum(r.answer_correctness for r in with_ground_truth) / len(with_ground_truth)
            if with_ground_truth else 0
        )
        
        avg_overall = sum(r.overall_score for r in results) / len(results)
        
        # Distribution analysis
        def get_distribution(scores: List[float]) -> Dict[str, int]:
            dist = {"low": 0, "medium": 0, "high": 0}
            for s in scores:
                if s < 0.4:
                    dist["low"] += 1
                elif s < 0.7:
                    dist["medium"] += 1
                else:
                    dist["high"] += 1
            return dist
        
        report = {
            "summary": {
                "total_evaluations": len(results),
                "with_ground_truth": len(with_ground_truth),
                "overall_score": round(avg_overall, 3)
            },
            "metrics": {
                "context_relevance": {
                    "average": round(avg_context_relevance, 3),
                    "distribution": get_distribution([r.context_relevance for r in results])
                },
                "faithfulness": {
                    "average": round(avg_faithfulness, 3),
                    "distribution": get_distribution([r.faithfulness for r in results])
                },
                "answer_relevance": {
                    "average": round(avg_answer_relevance, 3),
                    "distribution": get_distribution([r.answer_relevance for r in results])
                },
                "answer_correctness": {
                    "average": round(avg_correctness, 3),
                    "evaluated_count": len(with_ground_truth)
                }
            },
            "recommendations": self._generate_recommendations(
                avg_context_relevance,
                avg_faithfulness,
                avg_answer_relevance,
                avg_correctness
            ),
            "individual_results": [r.to_dict() for r in results]
        }
        
        return report
    
    def _generate_recommendations(
        self,
        context_relevance: float,
        faithfulness: float,
        answer_relevance: float,
        answer_correctness: float
    ) -> List[str]:
        """Generate improvement recommendations based on scores."""
        recommendations = []
        
        if context_relevance < 0.6:
            recommendations.append(
                "Improve retrieval quality: Consider tuning chunk size, using hybrid search, "
                "or implementing query expansion."
            )
        
        if faithfulness < 0.6:
            recommendations.append(
                "Reduce hallucinations: Strengthen prompt instructions to stay grounded in context, "
                "or implement fact-checking post-processing."
            )
        
        if answer_relevance < 0.6:
            recommendations.append(
                "Improve answer relevance: Enhance prompt engineering to better focus on the question, "
                "or use chain-of-thought prompting."
            )
        
        if answer_correctness < 0.6:
            recommendations.append(
                "Improve answer correctness: Review and update knowledge base, "
                "implement better source validation."
            )
        
        if not recommendations:
            recommendations.append("System performing well across all metrics.")
        
        return recommendations


# Convenience functions

def evaluate_context_relevance(
    llm_client,
    query: str,
    context: List[str]
) -> float:
    """Quick evaluation of context relevance."""
    evaluator = RAGEvaluator(llm_client)
    result = evaluator._evaluate_context_relevance(query, context)
    return result[0]


def evaluate_faithfulness(
    llm_client,
    answer: str,
    context: List[str]
) -> float:
    """Quick evaluation of faithfulness."""
    evaluator = RAGEvaluator(llm_client)
    result = evaluator._evaluate_faithfulness(answer, context)
    return result[0]


def evaluate_answer_relevance(
    llm_client,
    query: str,
    answer: str
) -> float:
    """Quick evaluation of answer relevance."""
    evaluator = RAGEvaluator(llm_client)
    result = evaluator._evaluate_answer_relevance(query, answer)
    return result[0]


def evaluate_answer_correctness(
    llm_client,
    answer: str,
    ground_truth: str
) -> float:
    """Quick evaluation of answer correctness."""
    evaluator = RAGEvaluator(llm_client)
    result = evaluator._evaluate_answer_correctness(answer, ground_truth)
    return result[0]


def run_evaluation(
    llm_client,
    test_cases: List[Dict[str, Any]],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run full evaluation and generate report.
    
    Args:
        llm_client: LLM client for evaluation
        test_cases: List of test cases
        output_file: Optional file to save report
        
    Returns:
        Evaluation report
    """
    evaluator = RAGEvaluator(llm_client)
    results = evaluator.evaluate_batch(test_cases)
    report = evaluator.generate_report(results)
    
    if output_file:
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_file}")
    
    return report
