"""
kiara/tools/agent_engine.py — Autonomous Tool-Selection & Dispatch Engine
==========================================================================
Core of Kiara's agentic loop.

Flow:
  1. classify_request(prompt) → ToolPlan
  2. run_agentic_loop() → dispatches each tool → AgentResponse
  3. Server injects AgentResponse.grounding_context into model prompt

Security:
  - SSRF protection: blocks RFC-1918 private IPs, loopback, cloud metadata
  - URL scheme whitelist: only http / https
  - Tool-call budget: max 4 per request
  - Duplicate-search detection
  - Safe AST calculator (no exec/eval on arbitrary code)
  - All tool output is untrusted data, never instructions
"""

from __future__ import annotations

import ast
import operator
import re
import socket
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─── Tool Registry ─────────────────────────────────────────────────────────────

class ToolName(str, Enum):
    ANSWER       = "answer"
    WEB_SEARCH   = "web_search"
    OPEN_URL     = "open_url"
    CALCULATOR   = "calculator"
    CODE_EXEC    = "code_execution"
    FINAL_ANSWER = "final_answer"
    CLARIFY      = "clarify"


# ─── Security: SSRF & URL Validation ──────────────────────────────────────────

_BLOCKED_IP_PREFIXES = (
    "10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.",
    "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.",
    "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
    "192.168.", "127.", "0.", "169.254.",
)
_BLOCKED_HOSTS = {
    "localhost", "metadata.google.internal", "169.254.169.254",
}
_ALLOWED_SCHEMES = {"http", "https"}


def validate_url(url: str) -> tuple[bool, str]:
    """Validate URL for SSRF safety. Returns (is_safe, reason)."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        return False, f"URL parse error: {e}"

    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return False, f"Blocked scheme '{parsed.scheme}' — only http/https allowed."

    host = (parsed.hostname or "").lower()
    if not host:
        return False, "No hostname."
    if host in _BLOCKED_HOSTS:
        return False, f"Blocked host '{host}' (internal/metadata)."

    try:
        ip = socket.gethostbyname(host)
        for prefix in _BLOCKED_IP_PREFIXES:
            if ip.startswith(prefix):
                return False, f"Blocked IP {ip} (private/internal network)."
    except socket.gaierror:
        pass  # unresolvable → let fetch fail naturally

    return True, ""


# ─── Signal Patterns for Autonomous Classification ─────────────────────────────

_LIVE_DATA_RE = re.compile(
    r"\b("
    r"latest|current|recent|today|now|this year|this month|this week|"
    r"2024|2025|2026|yesterday|just announced|just released|breaking|"
    r"trending|live|real.?time|up.?to.?date|upcoming|"
    r"price of|cost of|how much is|what does .+ cost|net worth|"
    r"who is the (current|new|present)|who is (now|currently)|"
    r"who is elected|who won|"
    r"chief minister|\bcm\b|\bcm of\b|prime minister|\bpm\b|\bpm of\b|president|ceo|chancellor|governor|"
    r"stock|crypto|bitcoin|ethereum|market cap|exchange rate|"
    r"weather|forecast|temperature in|"
    r"score|result|standings|ranking|leaderboard|"
    r"release notes|changelog|version \d|latest version|newest version|"
    r"news|headline|announced|launched|update on"
    r")\b",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+")

_CALC_RE = re.compile(
    r"\b(calculate|compute|what is \d|how much is \d|\d+\s*[\+\-\*\/\^]\s*\d|"
    r"percent|percentage|discount|tax|interest|convert \d|"
    r"\d+\s*%\s*of|\d+\s*(kg|km|lb|mile|celsius|fahrenheit))\b",
    re.IGNORECASE,
)

_CODE_EXEC_RE = re.compile(
    r"\b(run|execute|test this|verify|check if .+ works|output of|result of running|"
    r"does this code|will this work|debug this)\b",
    re.IGNORECASE,
)

_KNOWLEDGE_RE = re.compile(
    r"^(hi|hello|hey|greetings|thanks|thank you|"
    r"explain|define|describe|summarize|compare|contrast|"
    r"what is (a|an|the concept of|the meaning of)?|"
    r"what are|what was|"
    r"how (do|does|did|can|would|to)|"
    r"why (do|does|did|is|are)|"
    r"tell me about|walk me through|help me (understand|with)|"
    r"can you (explain|help|tell|write|give)|"
    r"give me (an example|a list|steps)|"
    r"write|create|generate|make|code|build|implement|design|draft|"
    r"solve|calculate|compute|simplify|evaluate|"
    r"who are you|who made you|who created you|what is kiara)",
    re.IGNORECASE,
)

_PURE_MATH_RE = re.compile(r"^[\d\s\+\-\*\/\^\(\)\.=<>%!,]+$")


def _is_pure_math(text: str) -> bool:
    s = text.strip()
    if not any(c.isdigit() for c in s):
        return False
    return bool(_PURE_MATH_RE.match(s))


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    tool: ToolName
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ToolResult:
    tool: ToolName
    success: bool
    content: str
    source_url: str = ""
    sources: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class ToolPlan:
    calls: list[ToolCall] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class AgentResponse:
    grounding_context: str | None
    status_steps: list[str]
    sources: list[str]
    tool_used: list[str]
    plan_reasoning: str


# ─── Classifier ───────────────────────────────────────────────────────────────

def classify_request(
    prompt: str,
    conversation_history: list[dict] | None = None,
    user_hint_search: bool | None = None,
    tool_budget: int = 4,
) -> ToolPlan:
    """
    Autonomously select tools for this prompt.

    user_hint_search:
        None  → fully autonomous (default)
        True  → user explicitly requested search
        False → user explicitly disabled search (other tools still allowed)
    """
    p = prompt.strip()
    calls: list[ToolCall] = []
    used: set[ToolName] = set()

    # 1. URL in prompt → open_url
    urls = _URL_RE.findall(p)
    if urls and user_hint_search is not False:
        raw = urls[0].rstrip(".,;!?:)'\"")
        target = raw if raw.startswith("http") else "https://" + raw
        ok, _ = validate_url(target)
        if ok:
            calls.append(ToolCall(ToolName.OPEN_URL, {"url": target}, "URL in prompt"))
            used.add(ToolName.OPEN_URL)

    # 2. Pure arithmetic → calculator
    if _is_pure_math(p):
        calls.append(ToolCall(ToolName.CALCULATOR, {"expression": p}, "Pure arithmetic"))
        used.add(ToolName.CALCULATOR)

    # 3. Decide web search
    needs_search = False
    if user_hint_search is True:
        needs_search = True
    elif user_hint_search is False:
        needs_search = False
    elif _LIVE_DATA_RE.search(p):
        needs_search = True
    elif (
        not _KNOWLEDGE_RE.match(p)
        and ToolName.OPEN_URL not in used
        and ToolName.CALCULATOR not in used
    ):
        words = p.split()
        if 2 <= len(words) <= 8:
            needs_search = True

    if needs_search and ToolName.WEB_SEARCH not in used and ToolName.CALCULATOR not in used:
        calls.append(ToolCall(ToolName.WEB_SEARCH, {"query": p}, "Live data required"))
        used.add(ToolName.WEB_SEARCH)

    # 4. Arithmetic calculation request (only if not a complex word problem)
    if (
        _CALC_RE.search(p)
        and ToolName.CALCULATOR not in used
        and not re.search(r"\b(step by step|story|word problem|if .+ buys)\b", p, re.IGNORECASE)
    ):
        calls.append(ToolCall(ToolName.CALCULATOR, {"expression": p}, "Arithmetic detected"))
        used.add(ToolName.CALCULATOR)

    # 5. Code execution (only if a code block is present)
    if _CODE_EXEC_RE.search(p) and "```" in p and ToolName.CODE_EXEC not in used:
        calls.append(ToolCall(ToolName.CODE_EXEC, {"code": p}, "Code execution requested"))
        used.add(ToolName.CODE_EXEC)

    calls = calls[:tool_budget]
    names = [c.tool.value for c in calls] if calls else ["knowledge"]
    return ToolPlan(calls=calls, reasoning=f"Plan: {', '.join(names)}")


# ─── Tool Dispatcher ──────────────────────────────────────────────────────────

def dispatch_tool(call: ToolCall, seen_queries: set[str]) -> ToolResult:
    """Execute one validated tool call and return its real result."""
    try:
        if call.tool == ToolName.WEB_SEARCH:
            return _exec_web_search(call.args, seen_queries)
        elif call.tool == ToolName.OPEN_URL:
            return _exec_open_url(call.args)
        elif call.tool == ToolName.CALCULATOR:
            return _exec_calculator(call.args)
        elif call.tool == ToolName.CODE_EXEC:
            return _exec_code(call.args)
        else:
            return ToolResult(call.tool, False, "", error=f"Unknown tool: {call.tool}")
    except Exception as e:
        return ToolResult(call.tool, False, "", error=f"Tool error: {e}")


def _exec_web_search(args: dict, seen: set[str]) -> ToolResult:
    from kiara.tools.web_agent import search_duckduckgo, search_wikipedia, _make_search_query
    raw = args.get("query", "")
    query = _make_search_query(raw)
    key = query.lower().strip()

    if key in seen:
        return ToolResult(ToolName.WEB_SEARCH, False, "",
                          error="Duplicate search — skipped.")
    seen.add(key)

    ddg = search_duckduckgo(query, max_results=4)
    wiki = search_wikipedia(query, max_results=2)

    if len(ddg) < 2:
        words = query.split()
        refined = " ".join(words[:3]) if len(words) > 2 else query
        ddg += search_duckduckgo(refined, max_results=2)
        wiki += search_wikipedia(refined, max_results=1)

    lines: list[str] = []
    if ddg:
        lines.append("Web Search Results:")
        for r in ddg:
            src = f" [{r['url']}]" if r.get("url") else ""
            lines.append(f"- {r['snippet']}{src}")
    if wiki:
        lines.append("Encyclopedia:")
        for r in wiki:
            lines.append(f"- {r['title']}: {r['snippet']}")

    if not lines:
        return ToolResult(ToolName.WEB_SEARCH, False, "",
                          error=f"No results for '{query}'")
    found_urls = [r["url"] for r in ddg if r.get("url")]
    return ToolResult(
        ToolName.WEB_SEARCH,
        True,
        "\n".join(lines),
        source_url=found_urls[0] if found_urls else "",
        sources=found_urls[:3],
    )


def _exec_open_url(args: dict) -> ToolResult:
    from kiara.tools.web_agent import fetch_url_content
    url = args.get("url", "")
    ok, reason = validate_url(url)
    if not ok:
        return ToolResult(ToolName.OPEN_URL, False, "", error=f"URL blocked: {reason}")
    content = fetch_url_content(url, max_chars=1500)
    if content.startswith("[Error") or content.startswith("[Unsupported"):
        return ToolResult(ToolName.OPEN_URL, False, "", source_url=url, error=content)
    return ToolResult(ToolName.OPEN_URL, True,
                      f"Content from {url}:\n{content}", source_url=url)


def _exec_calculator(args: dict) -> ToolResult:
    """Safe math evaluation via AST — no exec/eval on arbitrary code."""
    _OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported: {type(node).__name__}")

    raw = args.get("expression", "")
    expr = _extract_math(raw)
    if not expr:
        return ToolResult(ToolName.CALCULATOR, False, "",
                          error="No math expression found.")
    try:
        result = _eval(ast.parse(expr, mode="eval").body)
        if isinstance(result, float) and result == int(result):
            result_str = str(int(result))
        else:
            result_str = f"{result:.6g}"
        return ToolResult(ToolName.CALCULATOR, True,
                          f"Calculator: {expr} = {result_str}")
    except Exception as e:
        return ToolResult(ToolName.CALCULATOR, False, "",
                          error=f"Could not evaluate '{expr}': {e}")


def _extract_math(text: str) -> str:
    m = re.search(r"([\d\s\+\-\*\/\^\(\)\.%]+)", text)
    if m:
        candidate = re.sub(r"\s+", "", m.group(1).strip())
        if any(c.isdigit() for c in candidate):
            return candidate
    return re.sub(r"[^\d\s\+\-\*\/\^\(\)\.%]", "", text).strip()


def _exec_code(args: dict) -> ToolResult:
    from kiara.tools.code_runner import execute_python_code
    code = args.get("code", "")
    m = re.search(r"```(?:python|py)?\n([\s\S]+?)```", code)
    if m:
        code = m.group(1).strip()
    if not code.strip():
        return ToolResult(ToolName.CODE_EXEC, False, "", error="No code provided.")
    res = execute_python_code(code, timeout=5.0)
    if res["success"]:
        return ToolResult(ToolName.CODE_EXEC, True, f"Code Output:\n{res['output']}")
    return ToolResult(ToolName.CODE_EXEC, False, res.get("output", ""),
                      error=res.get("error", "Execution failed"))


# ─── Agentic Loop ─────────────────────────────────────────────────────────────

_STATUS_LABELS = {
    ToolName.WEB_SEARCH: "Searching the web",
    ToolName.OPEN_URL:   "Reading webpage",
    ToolName.CALCULATOR: "Calculating",
    ToolName.CODE_EXEC:  "Running code",
}


def run_agentic_loop(
    prompt: str,
    conversation_history: list[dict] | None = None,
    user_hint_search: bool | None = None,
    tool_budget: int = 4,
) -> AgentResponse:
    """
    Full autonomous tool-use loop for one user request.
    Called by the server before model generation.
    Returns AgentResponse with grounding context and status steps.
    """
    plan = classify_request(
        prompt=prompt,
        conversation_history=conversation_history,
        user_hint_search=user_hint_search,
        tool_budget=tool_budget,
    )

    if not plan.calls:
        return AgentResponse(
            grounding_context=None,
            status_steps=[],
            sources=[],
            tool_used=[],
            plan_reasoning=plan.reasoning,
        )

    seen_queries: set[str] = set()
    results: list[ToolResult] = []
    status_steps: list[str] = []
    sources: list[str] = []

    for call in plan.calls:
        label = _STATUS_LABELS.get(call.tool, "Thinking")
        if call.tool == ToolName.WEB_SEARCH:
            from kiara.tools.web_agent import _make_search_query
            q = _make_search_query(call.args.get("query", ""))
            status_steps.append(f"{label}: '{q[:40]}'")
        elif call.tool == ToolName.OPEN_URL:
            status_steps.append(f"{label}: {call.args.get('url', '')[:50]}")
        else:
            status_steps.append(label)

        result = dispatch_tool(call, seen_queries)
        results.append(result)
        if result.source_url and result.source_url not in sources:
            sources.append(result.source_url)
        for s in result.sources:
            if s and s not in sources:
                sources.append(s)

    # Build grounding context — include failures so model knows what was tried
    blocks: list[str] = []
    for r in results:
        if r.success and r.content:
            blocks.append(r.content)
        elif not r.success and r.error:
            blocks.append(f"[{r.tool.value} failed: {r.error}]")

    return AgentResponse(
        grounding_context="\n\n".join(blocks) if blocks else None,
        status_steps=status_steps,
        sources=sources,
        tool_used=[c.tool.value for c in plan.calls],
        plan_reasoning=plan.reasoning,
    )
