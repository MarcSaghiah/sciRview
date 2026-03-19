"""
Module for searching and retrieving articles from ArXiv.
"""
import arxiv
from typing import List, Dict

def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search ArXiv for articles matching a query.
    Returns a list of dicts with title, authors, summary, PDF URL.
    """
    results = []
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    for result in search.results():
        results.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "summary": result.summary,
            "pdf_url": result.pdf_url,
            "published": result.published.strftime("%Y-%m-%d")
        })
    return results
