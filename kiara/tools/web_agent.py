"""
kiara/tools/web_agent.py — Agentic Web Search & Reasoning Engine
=================================================================
Implements the full agentic decision loop:
  1. Kiara interprets the task and what information it needs
  2. Kiara selects its next action:
       - Use model knowledge (no search needed)
       - Browse the web (live facts, current events, unknown entities)
       - Use a tool (math, code execution)
       - Gather more evidence (search again with refined query)
  3. Execute action, inspect results
  4. Produce final grounded answer

Zero API keys required. Uses DuckDuckGo HTML + Wikipedia API.
"""

from __future__ import annotations
import json
import re
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 KiaraAI/1.0"
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

# ─── Patterns that signal "needs live web data" ────────────────────────────────
_NEEDS_SEARCH_PATTERNS = [
    r'\b(latest|current|recent|today|now|2024|2025|2026|this year|this month|this week)\b',
    r'\b(who is|what is|where is|when did|when was|how many|price of|cost of)\b',
    r'\b(news|update|release|version|score|result|winner|election|weather)\b',
    r'\b(cm of|chief minister|prime minister|president|ceo of|founder of)\b',
    r'\b(stock|crypto|bitcoin|market|inflation|rate|dollar|rupee)\b',
    r'\b(movie|film|album|book|series|episode|season)\b',
    r'\b(died|born|age of|nationality of|married|divorce)\b',
]
_NEEDS_SEARCH_RE = re.compile("|".join(_NEEDS_SEARCH_PATTERNS), re.IGNORECASE)

# ─── Patterns that can be answered from model knowledge ───────────────────────
_PURE_KNOWLEDGE_PATTERNS = [
    r'^(hi|hello|hey|how are you|what can you do|help me|tell me a joke)',
    r'^(what is [a-z ]+ algorithm|explain [a-z ]+|define [a-z ]+|how does [a-z ]+ work)',
    r'^(write a|create a|generate a|make a|code a|build a)',
    r'^(translate|convert|calculate|compute|solve|simplify)',
]
_PURE_KNOWLEDGE_RE = re.compile("|".join(_PURE_KNOWLEDGE_PATTERNS), re.IGNORECASE)

# ─── Pure math / arithmetic detection ─────────────────────────────────────────
_MATH_RE = re.compile(r'^[\d\s\+\-\*\/\^\(\)\.=<>%!,]+$')


def _is_pure_math(text: str) -> bool:
    stripped = re.sub(r'[^\d\s\+\-\*\/\^\(\)\.=<>%!]', '', text).strip()
    return bool(stripped) and bool(_MATH_RE.match(stripped))


def extract_urls(text: str) -> list[str]:
    """Find all HTTP/HTTPS URLs present in text."""
    urls = URL_REGEX.findall(text)
    # Normalize www. URLs to https://
    normalized = []
    for u in urls:
        u_clean = u.rstrip(".,;!?:)'\"]}")
        if u_clean.startswith("www."):
            u_clean = "https://" + u_clean
        normalized.append(u_clean)
    return normalized


def fetch_url_content(url: str, max_chars: int = 1500, timeout: int = 6) -> str:
    """Fetch a web page, strip HTML markup, and return clean readable content."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" in content_type or "image" in content_type or "zip" in content_type:
                return f"[Unsupported binary content: {content_type}]"
            html = response.read().decode("utf-8", errors="ignore")

        # Strip scripts, styles, forms, and headers
        clean = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<style.*?</style>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<nav.*?</nav>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<footer.*?</footer>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<header.*?</header>', '', clean, flags=re.DOTALL | re.IGNORECASE)

        # Strip all HTML tags
        clean = re.sub(r'<[^>]+>', ' ', clean)
        # Normalize whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()

        if len(clean) > max_chars:
            clean = clean[:max_chars] + "... [truncated]"
        return clean or "[No readable text found on page]"
    except Exception as e:
        return f"[Error accessing {url}: {e}]"


def search_duckduckgo(query: str, max_results: int = 4, timeout: int = 6) -> list[dict[str, str]]:
    """Search DuckDuckGo HTML for real-time live web search snippets and links."""
    try:
        url = "https://html.duckduckgo.com/html/"
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://html.duckduckgo.com/",
        }
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        matches = re.findall(r'<a class="result__snippet"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html, re.DOTALL)
        results = []
        for raw_url, snippet_html in matches[:max_results]:
            clean_s = re.sub(r'<[^>]+>', '', snippet_html).strip()
            clean_s = re.sub(r'\s+', ' ', clean_s)
            clean_s = urllib.parse.unquote(clean_s)
            clean_s = clean_s.replace("&#x27;", "'").replace("&quot;", '"').replace("&amp;", "&")
            if clean_s:
                results.append({
                    "snippet": clean_s,
                    "url": raw_url,
                })
        return results
    except Exception as e:
        print(f"[Kiara Web Agent] DuckDuckGo search notice: {e}")
        return []


def search_wikipedia(query: str, max_results: int = 2, timeout: int = 5) -> list[dict[str, str]]:
    """Search Wikipedia API for high-density factual snippets."""
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded}&utf8=&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        search_items = data.get("query", {}).get("search", [])
        results = []
        for item in search_items[:max_results]:
            snippet_clean = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
            snippet_clean = re.sub(r'\s+', ' ', snippet_clean).strip()
            snippet_clean = snippet_clean.replace("&#x27;", "'").replace("&quot;", '"').replace("&amp;", "&")
            results.append({
                "title": item.get("title", ""),
                "snippet": snippet_clean,
            })
        return results
    except Exception as e:
        print(f"[Kiara Web Agent] Wikipedia search notice: {e}")
        return []


def _make_search_query(prompt: str) -> str:
    """Strip filler words from a prompt to make a clean, effective search query."""
    q = prompt.strip()
    for prefix in [
        "who is the", "who is", "what is the", "what is", "where is the",
        "where is", "when did", "when was", "how many", "how much",
        "tell me about", "explain", "describe", "search for",
        "search", "show me", "can you tell me", "do you know",
        "i want to know", "what are", "give me information about",
    ]:
        if q.lower().startswith(prefix):
            q = q[len(prefix):].strip()
            break
    if q.lower().startswith("the "):
        q = q[4:].strip()
    q = q.rstrip("?.!")
    return q or prompt.strip()


class AgentAction:
    """Agentic action constants."""
    USE_KNOWLEDGE = "knowledge"
    WEB_SEARCH = "web_search"
    READ_URL = "read_url"
    MATH_TOOL = "math"


def decide_action(prompt: str, enable_search: bool = False) -> str:
    """
    Kiara meta-reasoning: decide the best action for this prompt.
    Returns one of AgentAction constants.
    """
    p = prompt.strip()

    # 1. URL in prompt -> read it
    if extract_urls(p):
        return AgentAction.READ_URL

    # 2. Pure arithmetic -> model handles it; no search
    if _is_pure_math(p):
        return AgentAction.MATH_TOOL

    # 3. Search disabled -> knowledge only
    if not enable_search:
        return AgentAction.USE_KNOWLEDGE

    # 4. Patterns that clearly need live data
    if _NEEDS_SEARCH_RE.search(p):
        return AgentAction.WEB_SEARCH

    # 5. Pure generation / creative / coding prompts -> knowledge
    if _PURE_KNOWLEDGE_RE.match(p):
        return AgentAction.USE_KNOWLEDGE

    # 6. Short factual entity queries (2-6 words) -> search
    words = p.split()
    if 2 <= len(words) <= 6:
        return AgentAction.WEB_SEARCH

    return AgentAction.USE_KNOWLEDGE


def get_web_grounding_context(
    prompt: str,
    enable_search: bool = False,
    max_search_rounds: int = 2,
) -> tuple[str | None, str | None]:
    """
    Full agentic tool-use loop:
      Step 1: Decide action (knowledge / search / url / math)
      Step 2: Execute chosen action
      Step 3: Inspect results; refine + search again if thin
      Step 4: Return grounding context for final answer generation

    Returns:
        tuple of (context_string, status_message) or (None, None)
    """
    action = decide_action(prompt, enable_search=enable_search)

    # ── READ URL ─────────────────────────────────────────────────────────────
    if action == AgentAction.READ_URL:
        urls = extract_urls(prompt)
        target_url = urls[0]
        status = f"Reading webpage: {target_url[:50]}"
        content = fetch_url_content(target_url, max_chars=1200)
        context = f"Live Webpage Content from ({target_url}):\n{content}\n"
        return context, status

    # ── MATH or KNOWLEDGE ────────────────────────────────────────────────────
    if action in (AgentAction.MATH_TOOL, AgentAction.USE_KNOWLEDGE):
        return None, None

    # ── WEB SEARCH (multi-round agentic) ─────────────────────────────────────
    clean_query = _make_search_query(prompt)
    status = f"Searching '{clean_query[:40]}'"

    ddg_results = search_duckduckgo(clean_query, max_results=3)
    wiki_results = search_wikipedia(clean_query, max_results=2)

    # Round 2: if results are thin, refine the query and search again
    if len(ddg_results) < 2 and max_search_rounds > 1:
        words = clean_query.split()
        refined = " ".join(words[:3]) if len(words) > 2 else clean_query + " Wikipedia"
        extra_ddg = search_duckduckgo(refined, max_results=2)
        extra_wiki = search_wikipedia(refined, max_results=1)
        ddg_results.extend(extra_ddg)
        wiki_results.extend(extra_wiki)

    context_lines: list[str] = []
    if ddg_results:
        context_lines.append("Live Web Search Results:")
        for r in ddg_results:
            src = f" (Source: {r['url']})" if r.get("url") else ""
            context_lines.append(f"- {r['snippet']}{src}")

    if wiki_results:
        context_lines.append("Encyclopedia Summary:")
        for r in wiki_results:
            context_lines.append(f"- {r['title']}: {r['snippet']}")

    if context_lines:
        return "\n".join(context_lines) + "\n", status

    # Nothing found -> fall back to model knowledge silently
    return None, None
