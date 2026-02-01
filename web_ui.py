#!/usr/bin/env python3
"""
Customer Support Chatbot - Web UI
A simple Flask-based web interface for the RAG chatbot.
"""

from flask import Flask, render_template_string, request, jsonify
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__)

# Global components
chatbot_instance = None


def initialize_chatbot():
    """Initialize the chatbot components."""
    global chatbot_instance
    
    if chatbot_instance is not None:
        return chatbot_instance
    
    print("🚀 Initializing Customer Support Chatbot...")
    
    from src.generation.llm import OllamaClient
    from src.indexing.embeddings import SentenceTransformerEmbeddings
    from src.indexing.vector_stores import ChromaVectorStore
    from src.generation.rag_pipeline import RAGPipeline, RAGConfig
    from src.document_processing.parsers import TextParser
    
    # Create LLM client (Ollama - free and local)
    print("  📦 Setting up LLM client (Ollama)...")
    llm_client = OllamaClient(model="llama3.2:1b")
    
    # Create embedding model (Sentence Transformers - free and local)
    print("  🔤 Setting up embedding model...")
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    print("  💾 Setting up vector store...")
    persist_dir = Path("./data/chroma_db")
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    vector_store = ChromaVectorStore(
        embedding_model=embedding_model,
        collection_name="customer_support_ui",
        persist_directory=str(persist_dir)
    )
    
    # Load knowledge base
    print("  📚 Loading knowledge base...")
    kb_path = Path("./data/knowledge_base")
    if kb_path.exists():
        parser = TextParser()
        documents = []
        for file_path in kb_path.glob("*.md"):
            try:
                parsed = parser.parse(str(file_path))
                content = parsed.content if hasattr(parsed, 'content') else str(parsed)
                documents.append({
                    "content": content,
                    "metadata": {"source": file_path.name}
                })
                print(f"    ✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"    ✗ Error: {file_path.name}: {e}")
        
        if documents:
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            vector_store.add_documents(texts, metadatas=metadatas)
            print(f"  ✅ Indexed {len(documents)} documents")
    
    # Create RAG pipeline
    print("  ⚙️ Creating RAG pipeline...")
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_client=llm_client,
        config=RAGConfig(
            top_k=3,
            temperature=0.7,
            max_tokens=512
        )
    )
    
    chatbot_instance = {
        "llm_client": llm_client,
        "vector_store": vector_store,
        "rag_pipeline": rag_pipeline
    }
    
    print("✅ Chatbot initialized successfully!\n")
    return chatbot_instance


# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TechCorp Customer Support</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .chat-container {
            width: 100%;
            max-width: 800px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }
        
        .chat-header h1 {
            font-size: 1.8em;
            margin-bottom: 10px;
        }
        
        .chat-header p {
            opacity: 0.9;
            font-size: 0.95em;
        }
        
        .chat-messages {
            height: 450px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 75%;
            padding: 12px 18px;
            border-radius: 18px;
            line-height: 1.5;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 5px;
        }
        
        .message.bot .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .message-avatar {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2em;
            margin: 0 10px;
        }
        
        .message.bot .message-avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #eee;
        }
        
        .input-wrapper {
            display: flex;
            gap: 10px;
        }
        
        #user-input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 1em;
            outline: none;
            transition: border-color 0.3s;
        }
        
        #user-input:focus {
            border-color: #667eea;
        }
        
        #send-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        #send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        #send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .quick-questions {
            padding: 15px 20px;
            background: #f8f9fa;
            border-top: 1px solid #eee;
        }
        
        .quick-questions h4 {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 10px;
        }
        
        .quick-btn {
            padding: 8px 15px;
            margin: 3px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 20px;
            font-size: 0.85em;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .quick-btn:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        
        .typing-indicator {
            display: none;
            padding: 12px 18px;
            background: white;
            border-radius: 18px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #667eea;
            border-radius: 50%;
            margin: 0 2px;
            animation: bounce 1.4s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        
        .footer {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            color: #666;
            font-size: 0.8em;
        }
        
        /* Scrollbar styling */
        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-messages::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        .chat-messages::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 TechCorp Customer Support</h1>
            <p>AI-powered assistant • RAG Technology • Local & Free</p>
        </div>
        
        <div class="chat-messages" id="chat-messages">
            <div class="message bot">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    Hello! I'm your TechCorp support assistant. I can help you with:
                    <br><br>
                    📦 <b>Product Information</b> - TechCloud, SecureShield, DataSync<br>
                    🔧 <b>Technical Troubleshooting</b> - Setup, configuration, issues<br>
                    💳 <b>Billing & Accounts</b> - Payments, subscriptions<br>
                    📋 <b>Policies</b> - Returns, warranties, privacy
                    <br><br>
                    How can I help you today?
                </div>
            </div>
        </div>
        
        <div class="quick-questions">
            <h4>💡 Quick Questions:</h4>
            <button class="quick-btn" onclick="askQuestion('What is your return policy?')">Return Policy</button>
            <button class="quick-btn" onclick="askQuestion('How do I reset my password?')">Reset Password</button>
            <button class="quick-btn" onclick="askQuestion('What products do you offer?')">Products</button>
            <button class="quick-btn" onclick="askQuestion('How do I contact support?')">Contact Support</button>
        </div>
        
        <div class="chat-input-container">
            <div class="input-wrapper">
                <input type="text" id="user-input" placeholder="Type your question here..." onkeypress="handleKeyPress(event)">
                <button id="send-btn" onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <div class="footer">
            🔒 Running locally with Ollama • 🆓 No API costs • 📚 RAG-powered responses
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chat-messages');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        
        function addMessage(content, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            if (!isUser) {
                messageDiv.innerHTML = `
                    <div class="message-avatar">🤖</div>
                    <div class="message-content">${content.replace(/\\n/g, '<br>')}</div>
                `;
            } else {
                messageDiv.innerHTML = `
                    <div class="message-content">${content}</div>
                `;
            }
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function showTyping() {
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message bot';
            typingDiv.id = 'typing-indicator';
            typingDiv.innerHTML = `
                <div class="message-avatar">🤖</div>
                <div class="typing-indicator" style="display: block;">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            `;
            chatMessages.appendChild(typingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function hideTyping() {
            const typing = document.getElementById('typing-indicator');
            if (typing) typing.remove();
        }
        
        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage(message, true);
            userInput.value = '';
            sendBtn.disabled = true;
            
            // Show typing indicator
            showTyping();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message }),
                });
                
                const data = await response.json();
                hideTyping();
                addMessage(data.response, false);
            } catch (error) {
                hideTyping();
                addMessage('Sorry, I encountered an error. Please try again.', false);
            }
            
            sendBtn.disabled = false;
            userInput.focus();
        }
        
        function askQuestion(question) {
            userInput.value = question;
            sendMessage();
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Focus input on load
        userInput.focus();
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    
    if not message.strip():
        return jsonify({'response': 'Please enter a message.'})
    
    try:
        bot = initialize_chatbot()
        
        # Query RAG pipeline
        response = bot["rag_pipeline"].query(question=message)
        
        return jsonify({'response': response.answer})
        
    except Exception as e:
        return jsonify({'response': f'I apologize, but I encountered an error: {str(e)}'})


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════╗
║       TechCorp Customer Support Chatbot - Web UI             ║
╠══════════════════════════════════════════════════════════════╣
║  🌐 Starting web interface...                                ║
║  📍 Open http://localhost:8888 in your browser               ║
║  🛑 Press Ctrl+C to stop                                     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize chatbot on startup
    initialize_chatbot()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=8888, debug=False)
