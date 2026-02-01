"""
Example: RAG Evaluation

This script demonstrates how to:
1. Evaluate retrieval quality
2. Measure answer faithfulness
3. Calculate answer correctness
4. Run comprehensive RAG evaluation
"""

import asyncio
from typing import List, Dict

from src.evaluation.metrics import (
    RAGEvaluator,
    EvaluationResult
)
from src.generation.llm import OpenAIClient
from src.config import config


# Sample evaluation data
EVALUATION_CASES = [
    {
        "query": "What is the price of the Pro plan?",
        "retrieved_contexts": [
            "TechCloud Pro plan costs $99 per month and includes auto-scaling, priority support, and advanced analytics.",
            "All TechCloud plans come with a 14-day free trial.",
            "Enterprise pricing is customized based on your needs."
        ],
        "generated_answer": "The TechCloud Pro plan costs $99 per month. It includes features like auto-scaling, priority support, and advanced analytics. You can also try it free for 14 days.",
        "ground_truth": "The Pro plan costs $99/month"
    },
    {
        "query": "How do I enable two-factor authentication?",
        "retrieved_contexts": [
            "To enable 2FA, go to Settings > Security > Two-Factor Authentication. You can use an authenticator app or SMS.",
            "We recommend using an authenticator app like Google Authenticator for better security.",
            "Password requirements: minimum 12 characters, including uppercase, lowercase, and numbers."
        ],
        "generated_answer": "To enable two-factor authentication: 1) Go to Settings, 2) Click Security, 3) Select Two-Factor Authentication, 4) Choose either an authenticator app (recommended) or SMS verification. We recommend using an authenticator app like Google Authenticator for the best security.",
        "ground_truth": "Go to Settings > Security > Two-Factor Authentication and choose authenticator app or SMS"
    },
    {
        "query": "What is the refund policy?",
        "retrieved_contexts": [
            "Monthly subscriptions: Full refund within 14 days of purchase.",
            "Annual subscriptions: Full refund within 30 days, pro-rated refund within 90 days.",
            "Contact billing@techcorp.com to request a refund."
        ],
        "generated_answer": "Our refund policy offers full refunds within 14 days for monthly plans and 30 days for annual plans. Annual subscribers can get a pro-rated refund within 90 days. Contact billing@techcorp.com to request your refund.",
        "ground_truth": "14-day full refund for monthly, 30-day for annual subscriptions"
    },
    {
        # Example of a potentially unfaithful answer
        "query": "What programming languages does TechCloud support?",
        "retrieved_contexts": [
            "TechCloud supports Python, Node.js, Ruby, and Java applications.",
            "Docker containers can run any language that works in a Linux environment.",
            "Our documentation covers getting started guides for common frameworks."
        ],
        "generated_answer": "TechCloud supports all major programming languages including Python, Node.js, Ruby, Java, Go, Rust, and C++. With Docker support, you can run virtually any application.",
        "ground_truth": "Python, Node.js, Ruby, and Java"
    },
    {
        # Example with irrelevant context
        "query": "What is the SLA uptime guarantee?",
        "retrieved_contexts": [
            "Our customer success team is available 24/7.",
            "TechCloud has data centers in multiple regions.",
            "We offer a 99.99% uptime SLA for Pro and Enterprise plans."
        ],
        "generated_answer": "TechCloud offers a 99.99% uptime SLA guarantee for Pro and Enterprise plans. This means we guarantee your services will be available 99.99% of the time.",
        "ground_truth": "99.99% uptime for Pro and Enterprise plans"
    }
]


def demo_manual_evaluation():
    """Demonstrate manual evaluation without LLM."""
    print("\n" + "=" * 60)
    print("📊 MANUAL EVALUATION METRICS DEMO")
    print("=" * 60)
    
    print("\n📝 Evaluation concepts explained:")
    
    print("""
    1. CONTEXT RELEVANCE
       ─────────────────
       Measures how relevant the retrieved contexts are to the query.
       
       High relevance: Context directly addresses the user's question
       Low relevance: Context is off-topic or tangentially related
       
       Example:
       Query: "What is the price?"
       Good context: "The Pro plan costs $99/month"
       Bad context: "Our team is based in San Francisco"
    
    2. FAITHFULNESS
       ─────────────
       Measures whether the generated answer is factually consistent
       with the retrieved contexts (no hallucinations).
       
       Faithful: Every claim can be traced to the context
       Unfaithful: Contains information not in the context
       
       Example:
       Context: "TechCloud supports Python and Node.js"
       Faithful answer: "You can use Python and Node.js"
       Unfaithful answer: "You can use Python, Node.js, Go, and Rust"
    
    3. ANSWER CORRECTNESS
       ──────────────────
       Measures how well the generated answer matches the expected
       ground truth answer (semantic similarity).
       
       High: Answer conveys the same information as ground truth
       Low: Answer is different or missing key information
    """)


async def demo_context_relevance(evaluator: RAGEvaluator):
    """Demonstrate context relevance evaluation."""
    print("\n" + "=" * 60)
    print("🎯 CONTEXT RELEVANCE EVALUATION DEMO")
    print("=" * 60)
    
    print("\n📊 Evaluating context relevance for each query...\n")
    
    for i, case in enumerate(EVALUATION_CASES[:3], 1):
        print(f"Case {i}: {case['query']}")
        
        try:
            score = await evaluator.evaluate_context_relevance(
                query=case['query'],
                contexts=case['retrieved_contexts']
            )
            
            print(f"   Relevance Score: {score:.2f}")
            
            # Show context analysis
            for j, ctx in enumerate(case['retrieved_contexts'], 1):
                preview = ctx[:50] + "..." if len(ctx) > 50 else ctx
                print(f"   Context {j}: {preview}")
            
        except Exception as e:
            print(f"   ⚠️ Evaluation skipped: {e}")
        
        print()


async def demo_faithfulness(evaluator: RAGEvaluator):
    """Demonstrate faithfulness evaluation."""
    print("\n" + "=" * 60)
    print("✓ FAITHFULNESS EVALUATION DEMO")
    print("=" * 60)
    
    print("\n📊 Evaluating answer faithfulness...\n")
    
    for i, case in enumerate(EVALUATION_CASES, 1):
        print(f"Case {i}: {case['query']}")
        
        try:
            score = await evaluator.evaluate_faithfulness(
                answer=case['generated_answer'],
                contexts=case['retrieved_contexts']
            )
            
            faithfulness_label = "✅ Faithful" if score > 0.7 else "⚠️ Potentially unfaithful"
            print(f"   Faithfulness Score: {score:.2f} {faithfulness_label}")
            print(f"   Answer preview: {case['generated_answer'][:60]}...")
            
        except Exception as e:
            print(f"   ⚠️ Evaluation skipped: {e}")
        
        print()


async def demo_answer_correctness(evaluator: RAGEvaluator):
    """Demonstrate answer correctness evaluation."""
    print("\n" + "=" * 60)
    print("📏 ANSWER CORRECTNESS EVALUATION DEMO")
    print("=" * 60)
    
    print("\n📊 Evaluating answer correctness against ground truth...\n")
    
    for i, case in enumerate(EVALUATION_CASES[:3], 1):
        print(f"Case {i}: {case['query']}")
        
        try:
            score = await evaluator.evaluate_answer_correctness(
                answer=case['generated_answer'],
                ground_truth=case['ground_truth']
            )
            
            print(f"   Correctness Score: {score:.2f}")
            print(f"   Generated: {case['generated_answer'][:50]}...")
            print(f"   Expected:  {case['ground_truth']}")
            
        except Exception as e:
            print(f"   ⚠️ Evaluation skipped: {e}")
        
        print()


async def demo_comprehensive_evaluation(evaluator: RAGEvaluator):
    """Demonstrate comprehensive RAG evaluation."""
    print("\n" + "=" * 60)
    print("📋 COMPREHENSIVE EVALUATION DEMO")
    print("=" * 60)
    
    print("\n📊 Running full evaluation on all cases...\n")
    
    results = []
    
    for i, case in enumerate(EVALUATION_CASES, 1):
        print(f"Evaluating case {i}: {case['query'][:40]}...")
        
        try:
            result = await evaluator.evaluate(
                query=case['query'],
                contexts=case['retrieved_contexts'],
                answer=case['generated_answer'],
                ground_truth=case['ground_truth']
            )
            results.append(result)
            
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
    
    # Summary statistics
    if results:
        print("\n" + "=" * 60)
        print("📈 EVALUATION SUMMARY")
        print("=" * 60)
        
        avg_relevance = sum(r.context_relevance for r in results) / len(results)
        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_correctness = sum(r.answer_correctness for r in results) / len(results)
        
        print(f"""
        Total cases evaluated: {len(results)}
        
        Average Scores:
        ───────────────
        Context Relevance:  {avg_relevance:.2f}  {'✅' if avg_relevance > 0.7 else '⚠️'}
        Faithfulness:       {avg_faithfulness:.2f}  {'✅' if avg_faithfulness > 0.7 else '⚠️'}
        Answer Correctness: {avg_correctness:.2f}  {'✅' if avg_correctness > 0.7 else '⚠️'}
        
        Interpretation:
        ───────────────
        > 0.8: Excellent
        0.6-0.8: Good
        0.4-0.6: Needs improvement
        < 0.4: Poor
        """)
        
        # Detailed breakdown
        print("\n📊 Individual Results:")
        print("-" * 60)
        print(f"{'Query':<30} {'Relevance':>10} {'Faithful':>10} {'Correct':>10}")
        print("-" * 60)
        
        for case, result in zip(EVALUATION_CASES, results):
            query_preview = case['query'][:28] + ".." if len(case['query']) > 30 else case['query']
            print(f"{query_preview:<30} {result.context_relevance:>10.2f} {result.faithfulness:>10.2f} {result.answer_correctness:>10.2f}")


def demo_evaluation_best_practices():
    """Demonstrate evaluation best practices."""
    print("\n" + "=" * 60)
    print("💡 EVALUATION BEST PRACTICES")
    print("=" * 60)
    
    print("""
    1. USE DIVERSE TEST CASES
       ─────────────────────
       • Include various query types (factual, procedural, opinion)
       • Test edge cases and ambiguous queries
       • Include both easy and challenging examples
    
    2. ESTABLISH GROUND TRUTH CAREFULLY
       ────────────────────────────────
       • Have multiple annotators for consistency
       • Use domain experts for complex topics
       • Document your evaluation criteria
    
    3. MONITOR METRICS OVER TIME
       ─────────────────────────
       • Track metrics as you update your system
       • Set up automated evaluation pipelines
       • Alert on significant metric drops
    
    4. BALANCE METRICS
       ───────────────
       • High retrieval + low faithfulness = retrieval is good, generation needs work
       • Low retrieval + high faithfulness = need better retrieval
       • All low = fundamental issues in both components
    
    5. CONSIDER LATENCY AND COST
       ─────────────────────────
       • LLM-based evaluation can be slow and expensive
       • Use sampling for large-scale evaluation
       • Consider lighter-weight metrics for monitoring
    
    6. ITERATE AND IMPROVE
       ───────────────────
       • Focus on lowest-scoring areas first
       • A/B test changes with evaluation metrics
       • Continuously add new test cases from production
    """)


async def main():
    """Run all evaluation demos."""
    print("\n" + "📊" * 30)
    print("   RAG EVALUATION EXAMPLES")
    print("📊" * 30)
    
    # Check for API key
    if not config.openai.api_key:
        print("\n⚠️ OPENAI_API_KEY not set. Some demos will be limited.")
        print("   Set the key in your .env file for full functionality.\n")
        
        # Demo 1: Manual concepts
        demo_manual_evaluation()
        demo_evaluation_best_practices()
        
        print("\n" + "=" * 60)
        print("✅ Basic demos completed!")
        print("   Set OPENAI_API_KEY for LLM-powered evaluation.")
        print("=" * 60 + "\n")
        return
    
    # Create evaluator with LLM
    llm_client = OpenAIClient(api_key=config.openai.api_key)
    evaluator = RAGEvaluator(llm_client=llm_client)
    
    # Demo 1: Manual concepts
    demo_manual_evaluation()
    
    # Demo 2: Context Relevance
    await demo_context_relevance(evaluator)
    
    # Demo 3: Faithfulness
    await demo_faithfulness(evaluator)
    
    # Demo 4: Answer Correctness
    await demo_answer_correctness(evaluator)
    
    # Demo 5: Comprehensive Evaluation
    await demo_comprehensive_evaluation(evaluator)
    
    # Demo 6: Best Practices
    demo_evaluation_best_practices()
    
    print("\n" + "=" * 60)
    print("✅ All evaluation demos completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
