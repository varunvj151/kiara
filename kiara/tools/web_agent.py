"""
kiara/tools/web_agent.py — Real-Time Web Scraping and Search Grounding
======================================================================
Gives Kiara live internet capabilities:
- URL reader: extracts clean, readable text from any website link.
- Wikipedia & encyclopedia lookup: retrieves grounded factual context.
- Zero API keys required; works instantly using standard HTTPS requests.
"""

from __future__ import annotations
import json
import re
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 KiaraAI/1.0"
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')


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
            results.append({
                "title": item.get("title", ""),
                "snippet": snippet_clean,
            })
        return results
    except Exception as e:
        print(f"[Kiara Web Agent] Wikipedia search notice: {e}")
        return []


def get_web_grounding_context(prompt: str, enable_search: bool = False) -> tuple[str | None, str | None]:
    """Inspect prompt for URLs or explicit web search request, and retrieve ground truth.
    
    Returns:
        tuple of (context_string, status_message) or (None, None)
    """
    urls = extract_urls(prompt)
    if urls:
        target_url = urls[0]
        status = f"Reading webpage: {target_url}"
        content = fetch_url_content(target_url, max_chars=1200)
        context = f"Live Webpage Content from ({target_url}):\n{content}\n"
        return context, status

    if enable_search:
        # Strip common prompt filler to form search query
        clean_query = prompt
        for prefix in ["who is", "what is", "tell me about", "explain", "search", "search for"]:
            if clean_query.lower().startswith(prefix):
                clean_query = clean_query[len(prefix):].strip()
                break

        status = f"Searching live knowledge for '{clean_query[:40]}...'"
        wiki_results = search_wikipedia(clean_query, max_results=2)
        if wiki_results:
            context_lines = ["Live Knowledge Base:"]
            for r in wiki_results:
                context_lines.append(f"- {r['title']}: {r['snippet']}")
            return "\n".join(context_lines) + "\n", status

    return None, None
