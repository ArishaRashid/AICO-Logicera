"""
AICO Agent Package

This package provides AI-powered web browsing and summarization capabilities.
It includes tools for fetching web content, intelligent summarization, and
conversational AI agents with memory.
"""

__version__ = "1.0.0"
__author__ = "AICO Team"

# Import main components for easy access
from .agent_orchestrator import build_web_summarization_agent, chat_with_agent_simple, summarize_web_content, get_summarization_techniques
from .web_scraper import WebBrowserTool

__all__ = ["build_web_summarization_agent", "chat_with_agent_simple", "summarize_web_content", "get_summarization_techniques", "WebBrowserTool"]
