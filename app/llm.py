# app/llm.py

import os
from typing import Optional, Dict, Any
from langchain_google_vertexai import ChatVertexAI
from langchain_core.language_models.chat_models import BaseChatModel

# Constants
LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.0-flash-001"

def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    streaming: bool = False,
    tools: Optional[list] = None,
    **kwargs
) -> BaseChatModel:
    """
    Factory function that creates a new instance of the LLM with custom parameters.
    
    Args:
        model: The model name to use
        temperature: Sampling temperature between 0 and 1
        max_tokens: Maximum number of tokens to generate
        streaming: Whether to enable streaming responses
        tools: Optional list of tools to bind
        **kwargs: Additional parameters to pass to the LLM
        
    Returns:
        A configured LLM instance
    """
    llm = ChatVertexAI(
        model=model,
        location=LOCATION,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        **kwargs
    )
    
    # Bind tools if provided
    if tools:
        return llm.bind_tools(tools)
    
    return llm

# Create a default instance
llm = get_llm()

# Create specialized instances for common use cases
memory_llm = get_llm(temperature=0.2)  # Lower temperature for memory operations
classifier_llm = get_llm(temperature=0.1, max_tokens=256)  # Precise classification
creative_llm = get_llm(temperature=0.9)  # Higher temperature for creative tasks