"""
Prompts Module
Handles prompt engineering strategies and templates.
"""

from .templates import (
    PromptTemplate,
    SystemPrompt,
    RAGPromptTemplate,
    CustomerSupportPrompt,
    get_prompt_template
)
from .strategies import (
    PromptStrategy,
    ZeroShotStrategy,
    FewShotStrategy,
    ChainOfThoughtStrategy,
    RoleBasedStrategy,
    UserContextStrategy,
    RAFTStrategy,
    apply_prompt_strategy
)

__all__ = [
    # Templates
    "PromptTemplate",
    "SystemPrompt",
    "RAGPromptTemplate",
    "CustomerSupportPrompt",
    "get_prompt_template",
    # Strategies
    "PromptStrategy",
    "ZeroShotStrategy",
    "FewShotStrategy",
    "ChainOfThoughtStrategy",
    "RoleBasedStrategy",
    "UserContextStrategy",
    "RAFTStrategy",
    "apply_prompt_strategy",
]
