"""
Prompt Engineering Strategies
Implements various prompting techniques for improved LLM performance.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json


class PromptStrategy(ABC):
    """Abstract base class for prompt strategies."""
    
    @abstractmethod
    def apply(self, base_prompt: str, **kwargs) -> str:
        """Apply the strategy to enhance the prompt."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        pass


class ZeroShotStrategy(PromptStrategy):
    """
    Zero-Shot Prompting
    
    Provides task description without examples.
    Relies on the model's pre-trained knowledge.
    
    Best for:
    - Simple, well-defined tasks
    - When examples aren't available
    - Quick prototyping
    
    Example:
    "Classify the sentiment of this review as positive, negative, or neutral.
    Review: 'Great product, fast shipping!'
    Sentiment:"
    """
    
    @property
    def name(self) -> str:
        return "zero_shot"
    
    def __init__(
        self,
        task_description: Optional[str] = None,
        output_format: Optional[str] = None
    ):
        self.task_description = task_description
        self.output_format = output_format
    
    def apply(self, base_prompt: str, **kwargs) -> str:
        """Apply zero-shot prompting."""
        parts = []
        
        if self.task_description:
            parts.append(f"Task: {self.task_description}")
            parts.append("")
        
        parts.append(base_prompt)
        
        if self.output_format:
            parts.append("")
            parts.append(f"Output format: {self.output_format}")
        
        return "\n".join(parts)


class FewShotStrategy(PromptStrategy):
    """
    Few-Shot Prompting
    
    Includes examples to guide the model's behavior.
    Improves accuracy for specific formats or styles.
    
    Best for:
    - Custom output formats
    - Domain-specific terminology
    - Complex classification tasks
    
    Example:
    "Classify the sentiment:
    
    Review: 'Terrible experience, avoid!'
    Sentiment: negative
    
    Review: 'Okay product, nothing special'
    Sentiment: neutral
    
    Review: 'Absolutely love it!'
    Sentiment: positive
    
    Review: '{user_input}'
    Sentiment:"
    """
    
    @property
    def name(self) -> str:
        return "few_shot"
    
    def __init__(
        self,
        examples: List[Dict[str, str]],
        example_template: str = "Input: {input}\nOutput: {output}"
    ):
        """
        Initialize few-shot strategy.
        
        Args:
            examples: List of example dicts with 'input' and 'output' keys
            example_template: Template for formatting examples
        """
        self.examples = examples
        self.example_template = example_template
    
    def apply(self, base_prompt: str, **kwargs) -> str:
        """Apply few-shot prompting with examples."""
        parts = [base_prompt, "", "Examples:"]
        
        for i, example in enumerate(self.examples):
            parts.append(f"\nExample {i + 1}:")
            example_text = self.example_template.format(**example)
            parts.append(example_text)
        
        parts.append("\nNow respond to this:")
        
        return "\n".join(parts)
    
    def add_example(self, input_text: str, output_text: str) -> None:
        """Add a new example."""
        self.examples.append({"input": input_text, "output": output_text})


class ChainOfThoughtStrategy(PromptStrategy):
    """
    Chain-of-Thought (CoT) Prompting
    
    Encourages step-by-step reasoning before the final answer.
    Improves performance on complex reasoning tasks.
    
    Variants:
    - Zero-shot CoT: "Let's think step by step"
    - Few-shot CoT: Examples with reasoning chains
    - Self-consistency: Multiple reasoning paths
    
    Best for:
    - Math problems
    - Logical reasoning
    - Multi-step analysis
    - Complex decision making
    
    Example:
    "Question: If a store has 3 boxes with 12 items each, and they sell
    half of all items, how many remain?
    
    Let's think step by step:
    1. Total items = 3 boxes × 12 items = 36 items
    2. Items sold = 36 ÷ 2 = 18 items
    3. Items remaining = 36 - 18 = 18 items
    
    Answer: 18 items remain."
    """
    
    @property
    def name(self) -> str:
        return "chain_of_thought"
    
    def __init__(
        self,
        reasoning_examples: Optional[List[Dict[str, str]]] = None,
        trigger_phrase: str = "Let's think step by step:",
        include_steps: bool = True
    ):
        """
        Initialize CoT strategy.
        
        Args:
            reasoning_examples: Examples showing reasoning chains
            trigger_phrase: Phrase to trigger step-by-step reasoning
            include_steps: Whether to explicitly ask for numbered steps
        """
        self.reasoning_examples = reasoning_examples or []
        self.trigger_phrase = trigger_phrase
        self.include_steps = include_steps
    
    def apply(self, base_prompt: str, **kwargs) -> str:
        """Apply chain-of-thought prompting."""
        parts = [base_prompt]
        
        # Add reasoning examples if available
        if self.reasoning_examples:
            parts.append("\nHere's how to approach similar problems:")
            for example in self.reasoning_examples:
                parts.append(f"\nQuestion: {example['question']}")
                parts.append(f"Reasoning: {example['reasoning']}")
                parts.append(f"Answer: {example['answer']}")
        
        # Add trigger phrase
        parts.append("")
        parts.append(self.trigger_phrase)
        
        if self.include_steps:
            parts.append("""
Break down your reasoning:
1. First, identify the key information
2. Then, consider the relationships or rules involved
3. Apply step-by-step logic
4. Verify your conclusion
5. Provide the final answer""")
        
        return "\n".join(parts)


class RoleBasedStrategy(PromptStrategy):
    """
    Role-Based Prompting
    
    Assigns a specific persona/role to the AI.
    Helps maintain consistent behavior and expertise.
    
    Common roles:
    - Expert: "You are an expert in {field}..."
    - Assistant: "You are a helpful assistant that..."
    - Critic: "You are a critical reviewer who..."
    - Teacher: "You are a patient teacher explaining..."
    
    Best for:
    - Maintaining consistent tone
    - Domain expertise simulation
    - Specific behavior requirements
    """
    
    @property
    def name(self) -> str:
        return "role_based"
    
    def __init__(
        self,
        role: str,
        expertise: Optional[List[str]] = None,
        behavior_traits: Optional[List[str]] = None,
        restrictions: Optional[List[str]] = None
    ):
        """
        Initialize role-based strategy.
        
        Args:
            role: The role/persona description
            expertise: Areas of expertise
            behavior_traits: Personality/behavior characteristics
            restrictions: Things the role should not do
        """
        self.role = role
        self.expertise = expertise or []
        self.behavior_traits = behavior_traits or []
        self.restrictions = restrictions or []
    
    def apply(self, base_prompt: str, **kwargs) -> str:
        """Apply role-based prompting."""
        parts = [f"You are {self.role}."]
        
        if self.expertise:
            parts.append(f"\nYour areas of expertise include: {', '.join(self.expertise)}.")
        
        if self.behavior_traits:
            parts.append("\nBehavior guidelines:")
            for trait in self.behavior_traits:
                parts.append(f"- {trait}")
        
        if self.restrictions:
            parts.append("\nImportant restrictions:")
            for restriction in self.restrictions:
                parts.append(f"- {restriction}")
        
        parts.append(f"\n{base_prompt}")
        
        return "\n".join(parts)


class UserContextStrategy(PromptStrategy):
    """
    User Context Prompting
    
    Personalizes prompts based on user information.
    Provides relevant, tailored responses.
    
    Context types:
    - User preferences
    - Conversation history
    - User profile/segment
    - Previous interactions
    
    Best for:
    - Personalized recommendations
    - Contextual customer support
    - Adaptive conversations
    """
    
    @property
    def name(self) -> str:
        return "user_context"
    
    def __init__(
        self,
        user_profile: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None,
        history_window: int = 5
    ):
        """
        Initialize user context strategy.
        
        Args:
            user_profile: User profile information
            preferences: User preferences
            history_window: Number of previous interactions to include
        """
        self.user_profile = user_profile or {}
        self.preferences = preferences or {}
        self.history_window = history_window
        self.conversation_history: List[Dict[str, str]] = []
    
    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        # Keep only recent history
        if len(self.conversation_history) > self.history_window * 2:
            self.conversation_history = self.conversation_history[-self.history_window * 2:]
    
    def apply(self, base_prompt: str, **kwargs) -> str:
        """Apply user context prompting."""
        parts = []
        
        # Add user context
        if self.user_profile:
            parts.append("User context:")
            for key, value in self.user_profile.items():
                parts.append(f"- {key}: {value}")
            parts.append("")
        
        # Add preferences
        if self.preferences:
            parts.append("User preferences:")
            for key, value in self.preferences.items():
                parts.append(f"- {key}: {value}")
            parts.append("")
        
        # Add conversation history
        if self.conversation_history:
            parts.append("Previous conversation:")
            for msg in self.conversation_history[-self.history_window * 2:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                parts.append(f"{role}: {msg['content']}")
            parts.append("")
        
        parts.append(base_prompt)
        
        return "\n".join(parts)


class RAFTStrategy(PromptStrategy):
    """
    RAFT (Retrieval Augmented Fine-Tuning) Strategy
    
    Implements RAFT-style prompting for improved RAG performance.
    Based on the paper: "RAFT: Adapting Language Model to Domain Specific RAG"
    
    Key concepts:
    1. Train model to distinguish relevant vs irrelevant documents
    2. Generate chain-of-thought reasoning over retrieved context
    3. Cite specific sources for factual claims
    
    Prompt structure:
    - Clearly mark relevant documents
    - Request explicit reasoning
    - Require source citations
    """
    
    @property
    def name(self) -> str:
        return "raft"
    
    def __init__(
        self,
        require_reasoning: bool = True,
        require_citations: bool = True,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize RAFT strategy.
        
        Args:
            require_reasoning: Whether to require chain-of-thought reasoning
            require_citations: Whether to require source citations
            confidence_threshold: Threshold for considering document relevant
        """
        self.require_reasoning = require_reasoning
        self.require_citations = require_citations
        self.confidence_threshold = confidence_threshold
    
    def apply(self, base_prompt: str, **kwargs) -> str:
        """Apply RAFT-style prompting."""
        context_docs = kwargs.get("context_docs", [])
        
        parts = ["""You are a helpful assistant that answers questions based on retrieved documents.
Some documents may be more relevant than others - identify and focus on the most relevant ones."""]
        
        # Format documents with relevance indicators
        if context_docs:
            parts.append("\nRetrieved Documents:")
            for i, doc in enumerate(context_docs):
                relevance = doc.get("score", 0)
                relevance_marker = "★" if relevance >= self.confidence_threshold else "○"
                parts.append(f"\n[Document {i+1}] {relevance_marker} (relevance: {relevance:.2f})")
                parts.append(doc.get("content", "")[:1000])
        
        parts.append(f"\nQuestion: {base_prompt}")
        
        # Add reasoning requirements
        if self.require_reasoning:
            parts.append("""
Before answering, analyze the documents:
1. Which documents are most relevant to the question?
2. What key information do they contain?
3. Are there any contradictions or gaps?
4. How confident are you in the answer?""")
        
        # Add citation requirements
        if self.require_citations:
            parts.append("""
When answering:
- Quote relevant passages with [Doc N] citations
- If information comes from multiple documents, cite all
- If no document contains the answer, say so clearly""")
        
        parts.append("\nAnswer:")
        
        return "\n".join(parts)


class StructuredOutputStrategy(PromptStrategy):
    """
    Structured Output Prompting
    
    Forces the model to output in a specific format (JSON, XML, etc.)
    Useful for programmatic processing of responses.
    """
    
    @property
    def name(self) -> str:
        return "structured_output"
    
    def __init__(
        self,
        output_schema: Dict[str, Any],
        format_type: str = "json"
    ):
        """
        Initialize structured output strategy.
        
        Args:
            output_schema: Schema defining expected output structure
            format_type: Output format (json, xml, yaml)
        """
        self.output_schema = output_schema
        self.format_type = format_type
    
    def apply(self, base_prompt: str, **kwargs) -> str:
        """Apply structured output prompting."""
        schema_str = json.dumps(self.output_schema, indent=2)
        
        prompt = f"""{base_prompt}

Respond ONLY with valid {self.format_type.upper()} matching this schema:
{schema_str}

Do not include any text before or after the {self.format_type.upper()}.

{self.format_type.upper()} Response:"""
        
        return prompt


class CompositeStrategy(PromptStrategy):
    """
    Combines multiple prompt strategies.
    Applies strategies in order.
    """
    
    @property
    def name(self) -> str:
        return "composite"
    
    def __init__(self, strategies: List[PromptStrategy]):
        self.strategies = strategies
    
    def apply(self, base_prompt: str, **kwargs) -> str:
        """Apply all strategies in sequence."""
        prompt = base_prompt
        for strategy in self.strategies:
            prompt = strategy.apply(prompt, **kwargs)
        return prompt


# Pre-built strategy configurations for customer support
CUSTOMER_SUPPORT_STRATEGIES = {
    "default": CompositeStrategy([
        RoleBasedStrategy(
            role="a friendly and knowledgeable customer support specialist",
            expertise=["product information", "troubleshooting", "account management"],
            behavior_traits=[
                "Patient and understanding",
                "Clear and concise in explanations",
                "Proactive in offering solutions"
            ],
            restrictions=[
                "Never make up information",
                "Don't share confidential data",
                "Escalate when needed"
            ]
        )
    ]),
    
    "technical": CompositeStrategy([
        RoleBasedStrategy(
            role="a technical support engineer",
            expertise=["technical troubleshooting", "system configuration", "debugging"],
            behavior_traits=[
                "Methodical problem-solving approach",
                "Detailed technical explanations"
            ]
        ),
        ChainOfThoughtStrategy(
            trigger_phrase="Let me diagnose this issue step by step:"
        )
    ]),
    
    "sales": CompositeStrategy([
        RoleBasedStrategy(
            role="a helpful sales consultant",
            expertise=["product features", "pricing", "recommendations"],
            behavior_traits=[
                "Enthusiastic but not pushy",
                "Focused on customer needs"
            ]
        ),
        FewShotStrategy([
            {
                "input": "What's your best product for small businesses?",
                "output": "For small businesses, I'd recommend our Starter Plan. It includes..."
            }
        ])
    ])
}


def apply_prompt_strategy(
    prompt: str,
    strategy_name: str = "default",
    **kwargs
) -> str:
    """
    Apply a named prompt strategy.
    
    Args:
        prompt: Base prompt to enhance
        strategy_name: Name of strategy to apply
        **kwargs: Additional arguments for the strategy
        
    Returns:
        Enhanced prompt
    """
    if strategy_name in CUSTOMER_SUPPORT_STRATEGIES:
        strategy = CUSTOMER_SUPPORT_STRATEGIES[strategy_name]
    else:
        # Map simple names to strategy classes
        strategy_map = {
            "zero_shot": ZeroShotStrategy(),
            "few_shot": FewShotStrategy(kwargs.pop("examples", [])),
            "cot": ChainOfThoughtStrategy(),
            "chain_of_thought": ChainOfThoughtStrategy(),
            "raft": RAFTStrategy(),
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = strategy_map[strategy_name]
    
    return strategy.apply(prompt, **kwargs)
