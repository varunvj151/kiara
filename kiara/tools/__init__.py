"""
kiara/tools — Agentic Tools and Superpowers for Kiara
======================================================
Provides:
  - Autonomous tool selection and dispatch (agent_engine)
  - Live web search & URL reading (web_agent)
  - Safe Python code execution sandbox (code_runner)
"""

from .web_agent import extract_urls, fetch_url_content, search_wikipedia, get_web_grounding_context, decide_action, AgentAction
from .code_runner import execute_python_code
from .agent_engine import run_agentic_loop, classify_request, validate_url, AgentResponse, ToolName

__all__ = [
    # web_agent
    "extract_urls",
    "fetch_url_content",
    "search_wikipedia",
    "get_web_grounding_context",
    "decide_action",
    "AgentAction",
    # code_runner
    "execute_python_code",
    # agent_engine
    "run_agentic_loop",
    "classify_request",
    "validate_url",
    "AgentResponse",
    "ToolName",
]
