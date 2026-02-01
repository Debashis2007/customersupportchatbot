"""
Tests for prompts module.
"""

import pytest
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
    RoleContextStrategy
)


class TestPromptTemplate:
    """Tests for basic PromptTemplate."""
    
    def test_simple_format(self):
        """Test simple variable substitution."""
        template = PromptTemplate(template="Hello {name}!")
        result = template.format(name="World")
        
        assert result == "Hello World!"
    
    def test_multiple_variables(self):
        """Test multiple variable substitution."""
        template = PromptTemplate(
            template="Hello {name}, welcome to {place}!"
        )
        result = template.format(name="Alice", place="TechCorp")
        
        assert "Alice" in result
        assert "TechCorp" in result
    
    def test_missing_variable(self):
        """Test handling of missing variables."""
        template = PromptTemplate(template="Hello {name}!")
        
        with pytest.raises(KeyError):
            template.format()
    
    def test_extra_variables_ignored(self):
        """Test that extra variables don't cause errors."""
        template = PromptTemplate(template="Hello {name}!")
        result = template.format(name="World", extra="ignored")
        
        assert result == "Hello World!"


class TestSystemPrompt:
    """Tests for SystemPrompt."""
    
    def test_basic_system_prompt(self):
        """Test basic system prompt generation."""
        prompt = SystemPrompt(
            role="assistant",
            capabilities=["answer questions"],
            constraints=["be polite"]
        )
        result = prompt.format()
        
        assert "assistant" in result.lower()
        assert "answer questions" in result
        assert "polite" in result
    
    def test_empty_capabilities(self):
        """Test with empty capabilities."""
        prompt = SystemPrompt(
            role="helper",
            capabilities=[],
            constraints=["be concise"]
        )
        result = prompt.format()
        
        assert "helper" in result.lower()
    
    def test_multiple_constraints(self):
        """Test with multiple constraints."""
        prompt = SystemPrompt(
            role="support agent",
            capabilities=["help users"],
            constraints=["be polite", "stay on topic", "don't make things up"]
        )
        result = prompt.format()
        
        assert "polite" in result
        assert "topic" in result


class TestRAGPromptTemplate:
    """Tests for RAGPromptTemplate."""
    
    def test_basic_rag_prompt(self):
        """Test basic RAG prompt generation."""
        template = RAGPromptTemplate(
            system_prompt="You are helpful.",
            context_prefix="Context:",
            query_prefix="Question:"
        )
        
        context = [
            {"content": "TechCloud costs $99/month."}
        ]
        
        result = template.format(query="What is the price?", context=context)
        
        assert "helpful" in result
        assert "TechCloud costs $99" in result
        assert "What is the price?" in result
    
    def test_multiple_contexts(self):
        """Test with multiple context documents."""
        template = RAGPromptTemplate(
            system_prompt="Answer based on context.",
            context_prefix="Info:",
            query_prefix="Q:"
        )
        
        context = [
            {"content": "Fact 1: A is true."},
            {"content": "Fact 2: B is also true."},
            {"content": "Fact 3: C is important."}
        ]
        
        result = template.format(query="Tell me about A, B, C", context=context)
        
        assert "Fact 1" in result
        assert "Fact 2" in result
        assert "Fact 3" in result
    
    def test_empty_context(self):
        """Test with no context."""
        template = RAGPromptTemplate(
            system_prompt="Help the user.",
            context_prefix="Context:",
            query_prefix="Question:"
        )
        
        result = template.format(query="Hello?", context=[])
        
        assert "Hello?" in result
    
    def test_context_with_metadata(self):
        """Test context with source metadata."""
        template = RAGPromptTemplate(
            system_prompt="Answer questions.",
            context_prefix="Sources:",
            query_prefix="Query:"
        )
        
        context = [
            {"content": "Important info.", "metadata": {"source": "doc.pdf"}}
        ]
        
        result = template.format(query="What's important?", context=context)
        
        assert "Important info" in result


class TestCustomerSupportPrompt:
    """Tests for CustomerSupportPrompt."""
    
    def test_basic_customer_support_prompt(self):
        """Test basic customer support prompt."""
        prompt = CustomerSupportPrompt(
            company_name="TechCorp",
            tone="friendly",
            products=["TechCloud", "SecureShield"]
        )
        
        result = prompt.format(
            intent="general_question",
            query="What products do you have?",
            context="Customer is interested in cloud services."
        )
        
        assert "TechCorp" in result
        assert "What products do you have?" in result
    
    def test_billing_intent(self):
        """Test billing inquiry handling."""
        prompt = CustomerSupportPrompt(
            company_name="TechCorp",
            tone="professional",
            products=["TechCloud"]
        )
        
        result = prompt.format(
            intent="billing_inquiry",
            query="Why was I charged?",
            context="Customer has Pro plan."
        )
        
        assert "charged" in result or "billing" in result.lower()
    
    def test_technical_support_intent(self):
        """Test technical support handling."""
        prompt = CustomerSupportPrompt(
            company_name="TechCorp",
            tone="helpful",
            products=["TechCloud"]
        )
        
        result = prompt.format(
            intent="technical_support",
            query="My app is not working",
            context=""
        )
        
        assert "not working" in result


class TestFewShotStrategy:
    """Tests for FewShotStrategy."""
    
    def test_few_shot_with_examples(self):
        """Test few-shot prompting with examples."""
        strategy = FewShotStrategy(
            examples=[
                {"input": "Hi", "output": "Hello! How can I help?"},
                {"input": "Bye", "output": "Goodbye! Have a great day!"}
            ],
            instruction="Be helpful."
        )
        
        result = strategy.create_prompt("Help me")
        
        assert "Hi" in result
        assert "Hello! How can I help?" in result
        assert "Bye" in result
        assert "Help me" in result
    
    def test_few_shot_empty_examples(self):
        """Test few-shot with no examples (degrades to zero-shot)."""
        strategy = FewShotStrategy(
            examples=[],
            instruction="Answer questions."
        )
        
        result = strategy.create_prompt("What is 2+2?")
        
        assert "What is 2+2?" in result
    
    def test_few_shot_preserves_format(self):
        """Test that few-shot maintains consistent format."""
        strategy = FewShotStrategy(
            examples=[
                {"input": "Q1", "output": "A1"},
                {"input": "Q2", "output": "A2"}
            ],
            instruction="Follow the format."
        )
        
        result = strategy.create_prompt("Q3")
        
        # Should have consistent formatting
        assert result.count("Q1") == 1
        assert result.count("A1") == 1


class TestZeroShotStrategy:
    """Tests for ZeroShotStrategy."""
    
    def test_zero_shot_basic(self):
        """Test basic zero-shot prompting."""
        strategy = ZeroShotStrategy(
            instruction="You are a helpful assistant."
        )
        
        result = strategy.create_prompt("What is Python?")
        
        assert "helpful assistant" in result
        assert "What is Python?" in result
    
    def test_zero_shot_with_context(self):
        """Test zero-shot with additional context."""
        strategy = ZeroShotStrategy(
            instruction="Answer based on the context provided."
        )
        
        result = strategy.create_prompt(
            "What is the price?",
            context="TechCloud costs $99/month."
        )
        
        assert "TechCloud costs $99" in result
        assert "What is the price?" in result


class TestChainOfThoughtStrategy:
    """Tests for ChainOfThoughtStrategy."""
    
    def test_cot_basic(self):
        """Test basic chain-of-thought prompting."""
        strategy = ChainOfThoughtStrategy(
            instruction="Solve problems step by step.",
            reasoning_prefix="Let me think:",
            conclusion_prefix="Therefore:"
        )
        
        result = strategy.create_prompt("What is 2+2?")
        
        assert "step by step" in result.lower() or "think" in result.lower()
        assert "What is 2+2?" in result
    
    def test_cot_includes_reasoning_prompt(self):
        """Test that CoT encourages reasoning."""
        strategy = ChainOfThoughtStrategy(
            instruction="Think carefully.",
            reasoning_prefix="Step by step reasoning:",
            conclusion_prefix="Final answer:"
        )
        
        result = strategy.create_prompt("Complex question here")
        
        # Should include reasoning guidance
        assert "step" in result.lower() or "think" in result.lower()


class TestRoleContextStrategy:
    """Tests for RoleContextStrategy."""
    
    def test_role_based_prompting(self):
        """Test role-based prompting."""
        strategy = RoleContextStrategy(
            roles={
                "new_user": "Explain in simple terms.",
                "expert": "Be technical and concise."
            }
        )
        
        # New user
        result1 = strategy.create_prompt(
            "What is an API?",
            user_context={"role": "new_user"}
        )
        
        # Expert
        result2 = strategy.create_prompt(
            "What is an API?",
            user_context={"role": "expert"}
        )
        
        assert "simple" in result1.lower()
        assert "technical" in result2.lower() or "concise" in result2.lower()
    
    def test_unknown_role_fallback(self):
        """Test handling of unknown roles."""
        strategy = RoleContextStrategy(
            roles={
                "admin": "Full access.",
                "user": "Limited access."
            }
        )
        
        # Unknown role should still produce a prompt
        result = strategy.create_prompt(
            "Hello",
            user_context={"role": "unknown_role"}
        )
        
        assert "Hello" in result


class TestPromptComposition:
    """Tests for composing multiple prompt strategies."""
    
    def test_combining_strategies(self):
        """Test combining RAG template with few-shot examples."""
        # This tests the concept of composing prompts
        few_shot = FewShotStrategy(
            examples=[{"input": "Hi", "output": "Hello!"}],
            instruction="Be helpful."
        )
        
        # Get the few-shot formatted query
        enhanced_query = few_shot.create_prompt("New question")
        
        # Use in RAG template
        rag = RAGPromptTemplate(
            system_prompt="You are an assistant.",
            context_prefix="Context:",
            query_prefix="Query:"
        )
        
        final_prompt = rag.format(
            query=enhanced_query,
            context=[{"content": "Some info"}]
        )
        
        assert "Hi" in final_prompt
        assert "Hello!" in final_prompt
        assert "Some info" in final_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
