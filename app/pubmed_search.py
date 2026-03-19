"""
Module for searching and retrieving articles from PubMed.
"""
from typing import List, Dict
from Bio import Entrez
import os
from dotenv import load_dotenv
load_dotenv()

Entrez.email = os.getenv("PUBMED_EMAIL", "your.email@example.com")

def search_pubmed(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search PubMed for articles matching a query.
    Returns a list of dicts with title, authors, abstract, PMID.
    """
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    ids = record["IdList"]
    handle.close()
    if not ids:
        return []
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    from Bio import Medline
    records = list(Medline.parse(handle))
    handle.close()
    results = []
    for rec in records:
        results.append({
            "title": rec.get("TI", ""),
            "authors": rec.get("AU", []),
            "summary": rec.get("AB", ""),
            "pmid": rec.get("PMID", ""),
            "published": rec.get("DP", "")
        })
    return results
