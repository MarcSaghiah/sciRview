"""
Web search module via DuckDuckGo (no API key required).
Used to enrich LLM context with fresh web results.
"""
from typing import List, Dict

def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web via DuckDuckGo.
    Returns a list of {title, url, snippet}.
    """
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
        return results
    except ImportError:
        return []
    except Exception:
        return []
