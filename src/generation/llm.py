"""
LLM Client Implementations
Provides unified interface for various LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # system, user, assistant
    content: str


@dataclass
class LLMResponse:
    """Represents an LLM response."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str = ""


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Chat with the LLM using message history."""
        pass
    
    @abstractmethod
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream responses from the LLM."""
        pass


class OpenAIClient(LLMClient):
    """
    OpenAI API Client
    
    Supports:
    - GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
    - Chat completions
    - Streaming responses
    - Function calling
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        organization: Optional[str] = None
    ):
        self.model = model
        
        try:
            from openai import OpenAI
            
            self.client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                organization=organization
            )
            
            logger.info(f"Initialized OpenAI client with model: {model}")
            
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate a response."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Chat with message history."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        choice = response.choices[0]
        
        return LLMResponse(
            content=choice.message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            finish_reason=choice.finish_reason
        )
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream responses."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def generate_with_functions(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate with function calling capability."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            functions=functions,
            function_call="auto",
            **kwargs
        )
        
        choice = response.choices[0]
        
        if choice.message.function_call:
            import json
            return {
                "type": "function_call",
                "function": choice.message.function_call.name,
                "arguments": json.loads(choice.message.function_call.arguments)
            }
        else:
            return {
                "type": "message",
                "content": choice.message.content
            }


class AnthropicClient(LLMClient):
    """
    Anthropic API Client
    
    Supports:
    - Claude 3 (Opus, Sonnet, Haiku)
    - Claude 2, Claude Instant
    - Streaming responses
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229"
    ):
        self.model = model
        
        try:
            from anthropic import Anthropic
            
            self.client = Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            
            logger.info(f"Initialized Anthropic client with model: {model}")
            
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate a response."""
        message_params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            message_params["system"] = system_prompt
        
        if temperature != 0.7:  # Anthropic uses different default
            message_params["temperature"] = temperature
        
        response = self.client.messages.create(**message_params)
        
        return response.content[0].text
    
    def chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Chat with message history."""
        # Separate system message
        system = None
        chat_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                chat_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        message_params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": chat_messages
        }
        
        if system:
            message_params["system"] = system
        
        response = self.client.messages.create(**message_params)
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            finish_reason=response.stop_reason
        )
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream responses."""
        message_params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            message_params["system"] = system_prompt
        
        with self.client.messages.stream(**message_params) as stream:
            for text in stream.text_stream:
                yield text


class HuggingFaceClient(LLMClient):
    """
    HuggingFace Inference Client
    
    Supports local models and HuggingFace Inference API.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_api: bool = True,
        api_key: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.use_api = use_api
        
        if use_api:
            try:
                from huggingface_hub import InferenceClient
                
                self.client = InferenceClient(
                    model=model_name,
                    token=api_key or os.getenv("HUGGINGFACE_API_KEY")
                )
            except ImportError:
                raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")
        else:
            # Local model loading
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                ).to(device)
                self.device = device
                
            except ImportError:
                raise ImportError("transformers not installed. Run: pip install transformers")
        
        logger.info(f"Initialized HuggingFace client: {model_name}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate a response."""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\n\nAssistant:"
        
        if self.use_api:
            response = self.client.text_generation(
                full_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response
        else:
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                **kwargs
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(full_prompt):]
    
    def chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Chat with message history."""
        # Format messages
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(msg.content)
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            else:
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n\n".join(prompt_parts)
        
        response = self.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
        
        return LLMResponse(
            content=response,
            model=self.model_name,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream responses (simulated for local models)."""
        response = self.generate(prompt, system_prompt, **kwargs)
        for word in response.split():
            yield word + " "


class OllamaClient(LLMClient):
    """
    Ollama Client for local LLM inference.
    
    Supports:
    - Various open-source models (Llama, Mistral, etc.)
    - Completely free and local
    - No API key required
    """
    
    def __init__(
        self,
        model: str = "llama3.2:1b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
        
        try:
            import requests
            self.requests = requests
            # Test connection
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Initialized Ollama client with model: {model}")
            else:
                logger.warning("Ollama server may not be running")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate a response using Ollama."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                logger.error(f"Ollama error: {response.text}")
                return "I apologize, but I'm having trouble generating a response."
                
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return "I apologize, but I'm having trouble generating a response."
    
    def chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Chat with Ollama using message history."""
        try:
            ollama_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            response = self.requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return LLMResponse(
                    content=result["message"]["content"],
                    model=self.model,
                    usage={
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                    }
                )
            else:
                return LLMResponse(
                    content="I apologize, but I'm having trouble processing your request.",
                    model=self.model,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                )
                
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return LLMResponse(
                content="I apologize, but I'm having trouble processing your request.",
                model=self.model,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream responses from Ollama."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True
                },
                stream=True,
                timeout=120
            )
            
            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
                        
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield "I apologize, but I'm having trouble generating a response."


def get_llm_client(
    provider: str = "openai",
    **kwargs
) -> LLMClient:
    """
    Factory function to get an LLM client.
    
    Args:
        provider: LLM provider (openai, anthropic, huggingface, ollama)
        **kwargs: Additional arguments for the client
        
    Returns:
        LLMClient instance
    """
    providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "huggingface": HuggingFaceClient,
        "ollama": OllamaClient,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown LLM provider: {provider}. Available: {list(providers.keys())}")
    
    return providers[provider](**kwargs)
