"""
Prompt Templates
Defines reusable prompt templates for the RAG system and customer support chatbot.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from string import Template
import json


@dataclass
class PromptTemplate:
    """
    Base prompt template class.
    Supports variable substitution using Python's Template syntax.
    """
    template: str
    input_variables: List[str] = field(default_factory=list)
    name: str = ""
    description: str = ""
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        # Validate all required variables are provided
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Use safe_substitute to handle missing variables gracefully
        template = Template(self.template)
        return template.safe_substitute(**kwargs)
    
    def __str__(self) -> str:
        return self.template


@dataclass
class SystemPrompt:
    """
    System prompt for defining AI persona and behavior.
    """
    role: str
    context: str
    instructions: List[str]
    constraints: List[str] = field(default_factory=list)
    tone: str = "professional"
    
    def format(self) -> str:
        """Generate the system prompt string."""
        parts = [
            f"You are {self.role}.",
            "",
            "Context:",
            self.context,
            "",
            "Instructions:"
        ]
        
        for i, instruction in enumerate(self.instructions, 1):
            parts.append(f"{i}. {instruction}")
        
        if self.constraints:
            parts.append("")
            parts.append("Constraints:")
            for constraint in self.constraints:
                parts.append(f"- {constraint}")
        
        parts.append("")
        parts.append(f"Maintain a {self.tone} tone in all responses.")
        
        return "\n".join(parts)


class RAGPromptTemplate:
    """
    Specialized prompt template for RAG applications.
    Handles context injection, source citation, and answer formatting.
    """
    
    DEFAULT_TEMPLATE = """Answer the question based on the provided context. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
$context

Question: $question

Instructions:
- Base your answer only on the provided context
- Cite specific sources when possible using [Source: filename]
- If multiple sources support your answer, mention all of them
- Be concise but comprehensive
- If the context is insufficient, explain what information is missing

Answer:"""
    
    WITH_HISTORY_TEMPLATE = """Answer the question based on the conversation history and provided context.

Previous conversation:
$history

Context from knowledge base:
$context

Current question: $question

Instructions:
- Consider the conversation history for context
- Base your answer primarily on the provided context
- Cite sources using [Source: filename]
- Be concise but thorough

Answer:"""
    
    CITATION_TEMPLATE = """Based on the provided context, answer the question and cite your sources.

Context:
$context

Question: $question

Provide your answer in the following format:
1. Main answer
2. Sources used (list each source with the relevant information it provided)
3. Confidence level (high/medium/low based on context coverage)

Answer:"""
    
    def __init__(
        self,
        template: Optional[str] = None,
        include_history: bool = False,
        require_citations: bool = True
    ):
        if template:
            self.template = template
        elif include_history:
            self.template = self.WITH_HISTORY_TEMPLATE
        elif require_citations:
            self.template = self.CITATION_TEMPLATE
        else:
            self.template = self.DEFAULT_TEMPLATE
    
    def format(
        self,
        question: str,
        context: str,
        history: Optional[str] = None,
        **kwargs
    ) -> str:
        """Format the RAG prompt."""
        template = Template(self.template)
        
        variables = {
            "question": question,
            "context": context,
            "history": history or "",
            **kwargs
        }
        
        return template.safe_substitute(**variables)
    
    @staticmethod
    def format_context(
        documents: List[Dict[str, Any]],
        max_tokens: int = 4000,
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of document dicts with 'content' and 'metadata'
            max_tokens: Approximate maximum tokens for context
            include_metadata: Whether to include source metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Format document
            if include_metadata:
                source = metadata.get("source", f"Document {i}")
                # Extract just filename if it's a path
                if "/" in source:
                    source = source.split("/")[-1]
                doc_text = f"[Source: {source}]\n{content}"
            else:
                doc_text = content
            
            # Check length (rough approximation: 1 token ≈ 4 characters)
            doc_tokens = len(doc_text) // 4
            if current_length + doc_tokens > max_tokens:
                # Truncate if needed
                available = (max_tokens - current_length) * 4
                if available > 100:
                    doc_text = doc_text[:available] + "..."
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            current_length += doc_tokens
        
        return "\n\n---\n\n".join(context_parts)


class CustomerSupportPrompt:
    """
    Specialized prompts for customer support scenarios.
    """
    
    SYSTEM_PROMPT = SystemPrompt(
        role="a helpful customer support assistant for our company",
        context="""You help customers with their questions, issues, and requests.
You have access to the company's knowledge base and can provide accurate information about products, 
services, policies, and procedures.""",
        instructions=[
            "Greet customers warmly and professionally",
            "Listen carefully to understand their needs",
            "Provide accurate information based on the knowledge base",
            "If you're unsure about something, acknowledge it and offer to escalate",
            "Suggest relevant follow-up actions or resources",
            "End conversations with a helpful closing"
        ],
        constraints=[
            "Never make up information not in the knowledge base",
            "Don't share internal or confidential information",
            "Always be polite, even with frustrated customers",
            "Respect customer privacy"
        ],
        tone="friendly and professional"
    )
    
    # Intent classification prompt
    INTENT_CLASSIFICATION = PromptTemplate(
        template="""Classify the customer's intent from the following message.

Customer message: $message

Possible intents:
- QUESTION: Customer is asking for information
- ISSUE: Customer has a problem that needs solving
- REQUEST: Customer wants to perform an action
- FEEDBACK: Customer is providing feedback
- COMPLAINT: Customer is expressing dissatisfaction
- OTHER: None of the above

Respond with only the intent label.

Intent:""",
        input_variables=["message"],
        name="intent_classification",
        description="Classify customer message intent"
    )
    
    # Sentiment analysis prompt
    SENTIMENT_ANALYSIS = PromptTemplate(
        template="""Analyze the sentiment of this customer message.

Customer message: $message

Rate the sentiment on this scale:
- VERY_NEGATIVE: Angry, frustrated, very unhappy
- NEGATIVE: Unhappy, disappointed, concerned
- NEUTRAL: Neither positive nor negative
- POSITIVE: Satisfied, pleased
- VERY_POSITIVE: Delighted, enthusiastic, very happy

Respond with only the sentiment label.

Sentiment:""",
        input_variables=["message"],
        name="sentiment_analysis",
        description="Analyze customer message sentiment"
    )
    
    # Response generation with context
    RESPONSE_WITH_CONTEXT = PromptTemplate(
        template="""You are a customer support assistant. Use the context below to answer the customer's question.

Context from knowledge base:
$context

Customer's question: $question

Customer's sentiment: $sentiment

Instructions:
- Answer based on the context provided
- Match your tone to the customer's sentiment (be extra empathetic if negative)
- Provide specific, actionable information
- If you can't fully answer from the context, acknowledge what you don't know
- Suggest next steps if appropriate

Response:""",
        input_variables=["context", "question", "sentiment"],
        name="response_with_context",
        description="Generate customer support response with context"
    )
    
    # Escalation check prompt
    ESCALATION_CHECK = PromptTemplate(
        template="""Determine if this customer conversation should be escalated to a human agent.

Customer message: $message
Conversation context: $context
Customer sentiment: $sentiment

Escalation criteria:
- Customer explicitly requests human agent
- Issue is too complex for automated support
- Customer is very frustrated (repeated negative sentiment)
- Legal, security, or safety concerns
- Unable to resolve after multiple attempts

Should this be escalated? Respond with YES or NO and a brief reason.

Escalation decision:""",
        input_variables=["message", "context", "sentiment"],
        name="escalation_check",
        description="Check if conversation needs escalation"
    )
    
    # Follow-up suggestions
    FOLLOW_UP_SUGGESTIONS = PromptTemplate(
        template="""Based on the conversation, suggest helpful follow-up actions or information.

Customer's original question: $question
Answer provided: $answer
Topic area: $topic

Generate 2-3 relevant follow-up suggestions that might help the customer.
Format as a bulleted list.

Follow-up suggestions:""",
        input_variables=["question", "answer", "topic"],
        name="follow_up_suggestions",
        description="Generate follow-up suggestions"
    )
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the formatted system prompt."""
        return cls.SYSTEM_PROMPT.format()
    
    @classmethod
    def classify_intent(cls, message: str) -> str:
        """Get the intent classification prompt."""
        return cls.INTENT_CLASSIFICATION.format(message=message)
    
    @classmethod
    def analyze_sentiment(cls, message: str) -> str:
        """Get the sentiment analysis prompt."""
        return cls.SENTIMENT_ANALYSIS.format(message=message)
    
    @classmethod
    def generate_response(
        cls,
        question: str,
        context: str,
        sentiment: str = "NEUTRAL"
    ) -> str:
        """Get the response generation prompt."""
        return cls.RESPONSE_WITH_CONTEXT.format(
            context=context,
            question=question,
            sentiment=sentiment
        )


# Pre-defined prompt templates
PROMPT_TEMPLATES = {
    "rag_default": RAGPromptTemplate(),
    "rag_with_history": RAGPromptTemplate(include_history=True),
    "rag_citations": RAGPromptTemplate(require_citations=True),
    "customer_support": CustomerSupportPrompt,
}


def get_prompt_template(name: str) -> Any:
    """
    Get a pre-defined prompt template by name.
    
    Args:
        name: Template name
        
    Returns:
        Prompt template instance
    """
    if name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown template: {name}. Available: {list(PROMPT_TEMPLATES.keys())}")
    
    return PROMPT_TEMPLATES[name]
