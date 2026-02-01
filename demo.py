#!/usr/bin/env python3
"""
Demo mode for Customer Support Chatbot
Works without external API keys using local models.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def run_demo():
    """Run the chatbot in demo mode with mock responses."""
    print("\n" + "="*60)
    print("🤖 Customer Support Chatbot - DEMO MODE")
    print("="*60)
    print("\nThis demo shows the chatbot architecture without requiring API keys.")
    print("For full functionality, configure your .env file with API keys.\n")
    
    # Load knowledge base for context
    kb_path = Path(__file__).parent / "knowledge_base"
    
    print("📚 Loading Knowledge Base...")
    knowledge_docs = []
    for md_file in kb_path.glob("*.md"):
        content = md_file.read_text()
        knowledge_docs.append({
            "source": md_file.name,
            "content": content[:500] + "..." if len(content) > 500 else content
        })
        print(f"   ✓ Loaded: {md_file.name}")
    
    print(f"\n✅ Loaded {len(knowledge_docs)} knowledge base documents\n")
    
    # Simple keyword-based responses for demo
    demo_responses = {
        "return": """Based on our return policy:
        
📦 **Return Policy Summary:**
- Returns accepted within 30 days of purchase
- Items must be unused and in original packaging
- Refunds processed within 5-7 business days
- Free return shipping for defective items

Would you like me to help you start a return?""",
        
        "shipping": """Here's our shipping information:

🚚 **Shipping Options:**
- Standard Shipping: 5-7 business days ($5.99)
- Express Shipping: 2-3 business days ($12.99)
- Overnight Shipping: Next business day ($24.99)
- Free standard shipping on orders over $50

Is there a specific order you'd like to track?""",
        
        "password": """To reset your password:

🔐 **Password Reset Steps:**
1. Go to the login page
2. Click "Forgot Password"
3. Enter your email address
4. Check your inbox for the reset link
5. Create a new password (min 8 characters)

Need help with anything else?""",

        "contact": """You can reach us through:

📞 **Contact Options:**
- Phone: 1-800-SUPPORT (Mon-Fri, 9am-6pm EST)
- Email: support@example.com
- Live Chat: Available on our website
- Response time: Within 24 hours

How can I assist you today?""",

        "order": """I can help with your order!

📋 **Order Assistance:**
- To track: Provide your order number
- To modify: Changes within 1 hour of placing
- To cancel: Before shipping confirmation
- Issues: We'll make it right!

What's your order number?""",

        "default": """Thank you for contacting Customer Support! 

I'm a demo version of the chatbot. In the full version with API keys configured, I can:
- Answer questions using RAG (Retrieval Augmented Generation)
- Search through product documentation
- Provide personalized assistance
- Remember our conversation context

Try asking about: returns, shipping, passwords, orders, or contact info."""
    }
    
    def get_response(user_input):
        """Get a demo response based on keywords."""
        user_lower = user_input.lower()
        for keyword, response in demo_responses.items():
            if keyword in user_lower:
                return response
        return demo_responses["default"]
    
    # Interactive chat loop
    print("-" * 60)
    print("💬 Chat started! Type 'quit' or 'exit' to end.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n🤖 Bot: Thank you for using Customer Support! Have a great day! 👋\n")
                break
            
            response = get_response(user_input)
            print(f"\n🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

def show_architecture():
    """Display the chatbot architecture."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Customer Support Chatbot Architecture              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ┌─────────────┐     ┌─────────────────────────────────┐    ║
║  │   User      │────▶│  Document Processing             │    ║
║  │   Query     │     │  • PDF/DOCX/HTML parsing         │    ║
║  └─────────────┘     │  • Text chunking strategies      │    ║
║                      └───────────────┬─────────────────┘    ║
║                                      │                       ║
║                                      ▼                       ║
║  ┌─────────────────────────────────────────────────────┐    ║
║  │              Indexing & Embeddings                   │    ║
║  │  • OpenAI / Sentence Transformers embeddings         │    ║
║  │  • ChromaDB / FAISS vector stores                    │    ║
║  └───────────────────────┬─────────────────────────────┘    ║
║                          │                                   ║
║                          ▼                                   ║
║  ┌─────────────────────────────────────────────────────┐    ║
║  │                  Retrieval                           │    ║
║  │  • Vector search (semantic similarity)               │    ║
║  │  • Hybrid search (BM25 + vector)                     │    ║
║  │  • Cross-encoder reranking                           │    ║
║  └───────────────────────┬─────────────────────────────┘    ║
║                          │                                   ║
║                          ▼                                   ║
║  ┌─────────────────────────────────────────────────────┐    ║
║  │            Prompt Engineering & Generation           │    ║
║  │  • Few-shot / Chain-of-thought prompting             │    ║
║  │  • OpenAI GPT-4 / Anthropic Claude                   │    ║
║  │  • Context-aware response generation                 │    ║
║  └───────────────────────┬─────────────────────────────┘    ║
║                          │                                   ║
║                          ▼                                   ║
║  ┌─────────────┐     ┌─────────────────────────────────┐    ║
║  │  Response   │◀────│  Evaluation & Quality            │    ║
║  │             │     │  • Relevance scoring              │    ║
║  │             │     │  • Faithfulness checking          │    ║
║  └─────────────┘     └─────────────────────────────────┘    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Customer Support Chatbot Demo")
    parser.add_argument("--architecture", "-a", action="store_true", 
                        help="Show the system architecture")
    args = parser.parse_args()
    
    if args.architecture:
        show_architecture()
    else:
        show_architecture()
        run_demo()
