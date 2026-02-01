"""
Chatbot Module
Provides chat interface and conversation management.
"""

from .chat_handler import (
    CustomerSupportChatbot,
    ConversationManager,
    Message,
    Conversation
)

__all__ = [
    "CustomerSupportChatbot",
    "ConversationManager",
    "Message",
    "Conversation"
]
