"""
Chatbot model utilities for the stock_chatbot project.

This module provides a function to initialize a language model for chatbot interactions
using the LangChain Ollama integration.

Functions:
- init_llm: Initializes and returns a ChatOllama language model instance.
"""

import os

from langchain_ollama import ChatOllama

def init_llm():
    """
    Initializes and returns a ChatOllama language model instance.

    Returns:
        ChatOllama: An instance of the ChatOllama language model.
    """
    llm = ChatOllama(
        base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        model=os.getenv('OLLAMA_MODEL', 'mistral-small:22b-instruct-2409-q5_K_M'),
    )
    return llm
