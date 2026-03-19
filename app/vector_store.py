"""
Module for vectorising text and indexing/searching in ChromaDB.
"""
import os
import warnings
import logging
import chromadb
from typing import List, Dict

# Suppress non-critical HuggingFace / transformers warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer, CrossEncoder

# Absolute path to the local vector database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma_db")
os.makedirs(DB_PATH, exist_ok=True)

# Initialise persistent ChromaDB (new API)
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Scientific literature embedding model (trained on paper citations)
EMBEDDING_MODEL = "allenai-specter"
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Cross-encoder reranker for precision (lazy-loaded)
_reranker = None
def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

COLLECTION_NAME = "articles"

# Create or retrieve collection with cosine similarity
collection = chroma_client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def reset_collection():
    """Delete and recreate the collection (wipes all indexed documents)."""
    chroma_client.delete_collection(COLLECTION_NAME)
    global collection
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def add_documents(docs: List[Dict]):
    """
    Add a list of documents (dict: id, title, text, meta) to the vector store.
    """
    texts = [doc['text'] for doc in docs]
    ids = [doc['id'] for doc in docs]
    metadatas = [{k: v for k, v in doc.items() if k not in ['id', 'text']} for doc in docs]
    embeddings = embedder.encode(texts).tolist()
    # upsert avoids errors if a document with the same ID already exists
    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

def library_count() -> int:
    """Return the number of documents in the library."""
    return collection.count()

def search_similar(query: str, n_results: int = 5, threshold: float = 0.4, rerank: bool = True) -> List[Dict]:
    """
    Semantic search + cross-encoder reranking.
    - threshold=0.4 for RAG, threshold=0.0 for display scoring
    - rerank=True: applies the cross-encoder on top candidates
    """
    if collection.count() == 0:
        return []
    # Fetch more candidates so the reranker has enough material
    candidate_n = min(n_results * 4, collection.count())
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=candidate_n)
    hits = []
    for i in range(len(results['ids'][0])):
        distance = results['distances'][0][i]
        similarity = 1 - distance
        if similarity >= threshold:
            hits.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'score': similarity,
                'meta': results['metadatas'][0][i],
                'distance': distance
            })
    if not hits:
        return []
    # Reranking cross-encoder
    if rerank and len(hits) > 1:
        reranker = _get_reranker()
        pairs = [(query, h['text'][:512]) for h in hits]  # truncated for speed
        rerank_scores = reranker.predict(pairs)
        for i, h in enumerate(hits):
            h['rerank_score'] = float(rerank_scores[i])
        hits.sort(key=lambda x: x['rerank_score'], reverse=True)
    else:
        for h in hits:
            h['rerank_score'] = h['score']
    return hits[:n_results]
