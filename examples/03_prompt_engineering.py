"""
Example: Prompt Engineering Strategies

This script demonstrates how to:
1. Use different prompting strategies (few-shot, zero-shot, chain-of-thought)
2. Create effective RAG prompts
3. Build customer support specific prompts
4. Implement role-based context
"""

import asyncio
from typing import List, Dict

from src.prompts.templates import (
    PromptTemplate,
    SystemPrompt,
    RAGPromptTemplate,
    CustomerSupportPrompt
)
from src.prompts.strategies import (
    FewShotStrategy,
    ZeroShotStrategy,
    ChainOfThoughtStrategy,
    RoleContextStrategy,
    PromptStrategy
)


def demo_basic_templates():
    """Demonstrate basic prompt templates."""
    print("\n" + "=" * 60)
    print("📝 BASIC PROMPT TEMPLATES DEMO")
    print("=" * 60)
    
    # Simple template
    simple = PromptTemplate(
        template="Hello {name}, how can I help you with {topic} today?"
    )
    result = simple.format(name="Alice", topic="billing")
    print(f"\n1. Simple Template:")
    print(f"   {result}")
    
    # System prompt
    system = SystemPrompt(
        role="customer support agent",
        capabilities=["answer questions", "troubleshoot issues", "process refunds"],
        constraints=["be polite", "stay on topic", "escalate complex issues"]
    )
    result = system.format()
    print(f"\n2. System Prompt:")
    print(f"   {result[:200]}...")


def demo_rag_prompts():
    """Demonstrate RAG-specific prompts."""
    print("\n" + "=" * 60)
    print("📚 RAG PROMPT TEMPLATES DEMO")
    print("=" * 60)
    
    # RAG template
    rag_template = RAGPromptTemplate(
        system_prompt="You are a helpful assistant that answers questions based on the provided context.",
        context_prefix="Relevant Information:",
        query_prefix="User Question:"
    )
    
    # Sample context (would come from retrieval)
    context = [
        {
            "content": "TechCloud pricing: Basic plan is $29/month, Pro is $99/month.",
            "metadata": {"source": "pricing.md"}
        },
        {
            "content": "All plans include 24/7 email support. Pro plans get priority support.",
            "metadata": {"source": "support.md"}
        }
    ]
    
    query = "What's included in the Pro plan?"
    
    prompt = rag_template.format(query=query, context=context)
    
    print(f"\n📄 Generated RAG Prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)


def demo_customer_support_prompts():
    """Demonstrate customer support specific prompts."""
    print("\n" + "=" * 60)
    print("🎧 CUSTOMER SUPPORT PROMPTS DEMO")
    print("=" * 60)
    
    cs_prompt = CustomerSupportPrompt(
        company_name="TechCorp",
        tone="friendly and professional",
        products=["TechCloud", "SecureShield", "DataSync Pro"]
    )
    
    # Different intent examples
    intents = [
        ("billing_inquiry", "Why was I charged twice this month?"),
        ("technical_support", "My dashboard is loading slowly"),
        ("general_question", "What products do you offer?"),
    ]
    
    for intent, question in intents:
        prompt = cs_prompt.format(
            intent=intent,
            query=question,
            context="Customer has been with us for 2 years, Pro plan subscriber."
        )
        print(f"\n🏷️ Intent: {intent}")
        print(f"   Question: {question}")
        print(f"   Prompt Preview: {prompt[:150]}...")


def demo_few_shot_prompting():
    """Demonstrate few-shot prompting strategy."""
    print("\n" + "=" * 60)
    print("📋 FEW-SHOT PROMPTING DEMO")
    print("=" * 60)
    
    # Create few-shot strategy with examples
    few_shot = FewShotStrategy(
        examples=[
            {
                "input": "How do I reset my password?",
                "output": "To reset your password: 1) Go to login page, 2) Click 'Forgot Password', 3) Enter your email, 4) Check your inbox for the reset link."
            },
            {
                "input": "I want to upgrade my plan",
                "output": "Great choice! To upgrade: 1) Log into your account, 2) Go to Settings > Billing, 3) Click 'Upgrade Plan', 4) Select your new plan and confirm."
            },
            {
                "input": "My app isn't working",
                "output": "I'm sorry to hear that! Let's troubleshoot: 1) First, try refreshing the page, 2) Clear your browser cache, 3) If the issue persists, can you describe what error you're seeing?"
            }
        ],
        instruction="You are a helpful customer support agent. Answer questions in a clear, step-by-step format."
    )
    
    # Generate prompt for new query
    new_query = "How do I cancel my subscription?"
    prompt = few_shot.create_prompt(new_query)
    
    print(f"\n📝 Few-Shot Prompt for: '{new_query}'")
    print("-" * 40)
    print(prompt)
    print("-" * 40)


def demo_zero_shot_prompting():
    """Demonstrate zero-shot prompting strategy."""
    print("\n" + "=" * 60)
    print("🎯 ZERO-SHOT PROMPTING DEMO")
    print("=" * 60)
    
    zero_shot = ZeroShotStrategy(
        instruction="""You are a TechCorp customer support agent. 
        
Your responsibilities:
- Answer customer questions accurately
- Be friendly and professional
- Provide step-by-step instructions when helpful
- Escalate to a human agent for complex issues

Always base your answers on the provided context. If you don't know something, say so."""
    )
    
    query = "What regions are available for deployment?"
    context = "TechCloud supports US-East, US-West, EU-West, and Asia-Pacific regions."
    
    prompt = zero_shot.create_prompt(query, context=context)
    
    print(f"\n📝 Zero-Shot Prompt for: '{query}'")
    print("-" * 40)
    print(prompt)
    print("-" * 40)


def demo_chain_of_thought():
    """Demonstrate chain-of-thought prompting."""
    print("\n" + "=" * 60)
    print("🧠 CHAIN-OF-THOUGHT PROMPTING DEMO")
    print("=" * 60)
    
    cot = ChainOfThoughtStrategy(
        instruction="You are a customer support agent helping diagnose technical issues.",
        reasoning_prefix="Let me think through this step by step:",
        conclusion_prefix="Based on this analysis, here's my recommendation:"
    )
    
    query = """My website is showing a 503 error. I'm on the Basic plan and 
    recently had a traffic spike from a news article about my company."""
    
    prompt = cot.create_prompt(query)
    
    print(f"\n📝 Chain-of-Thought Prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    # Simulated CoT response
    print("\n📝 Example CoT Response:")
    print("-" * 40)
    cot_response = """Let me think through this step by step:

1. The error is a 503 Service Unavailable
2. This typically indicates the server is overloaded
3. The customer mentioned a traffic spike from news coverage
4. They're on the Basic plan, which has limited resources
5. Basic plans don't include auto-scaling

Based on this analysis, here's my recommendation:

The 503 error is likely due to the traffic spike overwhelming your Basic plan's 
resources. I recommend:
1. Temporarily upgrade to the Pro plan which includes auto-scaling
2. Enable caching to reduce server load
3. Consider using a CDN for static assets

Would you like me to help you upgrade your plan or set up caching?"""
    print(cot_response)


def demo_role_context_strategy():
    """Demonstrate role and context-aware prompting."""
    print("\n" + "=" * 60)
    print("👤 ROLE-CONTEXT PROMPTING DEMO")
    print("=" * 60)
    
    role_strategy = RoleContextStrategy(
        roles={
            "new_user": "Be extra helpful and explain concepts in simple terms.",
            "power_user": "Be concise and technical. Skip basic explanations.",
            "enterprise_client": "Be formal. Mention dedicated support options."
        }
    )
    
    query = "How do I set up API access?"
    
    for role in ["new_user", "power_user", "enterprise_client"]:
        user_context = {"role": role, "plan": "Pro" if role != "enterprise_client" else "Enterprise"}
        prompt = role_strategy.create_prompt(query, user_context=user_context)
        
        print(f"\n👤 Role: {role}")
        print(f"   Prompt: {prompt[:150]}...")


def demo_prompt_composition():
    """Demonstrate composing multiple strategies."""
    print("\n" + "=" * 60)
    print("🔧 PROMPT COMPOSITION DEMO")
    print("=" * 60)
    
    # Combine RAG template with few-shot examples
    rag_template = RAGPromptTemplate(
        system_prompt="""You are a TechCorp support agent. Answer based on the context provided.
        
Here are examples of good responses:

Example 1:
Q: How do I reset my password?
A: Go to login page > Click 'Forgot Password' > Enter email > Check inbox for reset link.

Example 2:
Q: What's the refund policy?
A: We offer full refunds within 14 days for monthly plans, 30 days for annual plans.""",
        context_prefix="Knowledge Base:",
        query_prefix="Customer Question:"
    )
    
    context = [
        {"content": "TechCloud supports auto-scaling on Pro and Enterprise plans."},
        {"content": "Auto-scaling adjusts resources automatically based on traffic."}
    ]
    
    query = "Does my plan support auto-scaling?"
    prompt = rag_template.format(query=query, context=context)
    
    print(f"\n📝 Composed Prompt (RAG + Few-Shot):")
    print("-" * 40)
    print(prompt)
    print("-" * 40)


async def main():
    """Run all prompt engineering demos."""
    print("\n" + "✍️" * 30)
    print("   PROMPT ENGINEERING EXAMPLES")
    print("✍️" * 30)
    
    # Demo 1: Basic Templates
    demo_basic_templates()
    
    # Demo 2: RAG Prompts
    demo_rag_prompts()
    
    # Demo 3: Customer Support Prompts
    demo_customer_support_prompts()
    
    # Demo 4: Few-Shot
    demo_few_shot_prompting()
    
    # Demo 5: Zero-Shot
    demo_zero_shot_prompting()
    
    # Demo 6: Chain-of-Thought
    demo_chain_of_thought()
    
    # Demo 7: Role-Context
    demo_role_context_strategy()
    
    # Demo 8: Composition
    demo_prompt_composition()
    
    print("\n" + "=" * 60)
    print("✅ All prompt engineering demos completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
