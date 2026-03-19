"""
LLM module — Groq (primary, cloud) + Ollama (local fallback).
"""
import os
import json
import hashlib
import requests
from typing import List, Generator

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"

# Model configurable via .env (default: phi3:mini — fast on CPU)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Max passages and max chars per passage to limit context size
MAX_PASSAGES = 3
MAX_PASSAGE_CHARS = 800

# Simple in-memory cache: {hash: summary}
_summary_cache: dict = {}

def is_model_available(model: str = OLLAMA_MODEL) -> bool:
    """Check if a model is available in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            return any(model in m for m in models)
    except Exception:
        pass
    return False

def pull_model(model: str = OLLAMA_MODEL) -> bool:
    """Download a model into Ollama if not present."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model},
            timeout=300
        )
        return response.status_code == 200
    except Exception:
        return False

def warmup_model(model: str = OLLAMA_MODEL):
    """Warm up the Ollama model (load into memory) by sending a minimal prompt."""
    try:
        requests.post(OLLAMA_URL, json={"model": model, "prompt": "ok", "stream": False}, timeout=60)
    except Exception:
        pass

def summarize_abstract(abstract: str, model: str = OLLAMA_MODEL) -> str:
    """
    Generate a short summary of a scientific abstract.
    Always responds via LLM, even if the abstract is short or missing.
    """
    text = abstract.strip() if abstract else ""

    cache_key = hashlib.md5(("abstract_summary" + text).encode()).hexdigest()
    if cache_key in _summary_cache:
        return _summary_cache[cache_key]

    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    except Exception:
        return "Ollama is not reachable. Make sure Ollama is running."

    if text:
        prompt = (
            f"You are an expert scientific assistant. Respond in the same language as the abstract.\n\n"
            f"Summarize the following scientific abstract in 3 clear sentences, highlighting the main findings:\n\n{text[:600]}"
        )
    else:
        prompt = (
            "You are an expert scientific assistant. The user requested a summary but no abstract text is available for this article. "
            "Respond helpfully: acknowledge that no abstract is available, and suggest the user open the full PDF to read more."
        )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 120, "temperature": 0.3}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json().get("response", "[No summary generated]")
        _summary_cache[cache_key] = result
        return result
    except requests.exceptions.Timeout:
        return "[Timeout. Please retry in a few seconds.]"
    except Exception as e:
        return f"[Error: {e}]"


def generate_summary(passages: List[str], question: str = None, model: str = OLLAMA_MODEL,
                     web_results: List[dict] = None) -> str:
    """
    Answer a question by reasoning freely, drawing on:
    - article excerpts found (RAG)
    - web search results (optional)
    - the model's own knowledge
    """
    truncated = [p[:MAX_PASSAGE_CHARS] for p in passages[:MAX_PASSAGES]] if passages else []

    cache_key = hashlib.md5((str(question) + "".join(truncated) + str(web_results)).encode()).hexdigest()
    if cache_key in _summary_cache:
        return _summary_cache[cache_key]

    # Check if Ollama is reachable
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    except Exception:
        return "Ollama is not reachable. Make sure Ollama is running."

    if not is_model_available(model):
        pulled = pull_model(model)
        if not pulled:
            return f"[Model '{model}' is not available. Run: ollama pull {model}]"

    # Build the prompt
    system = (
        "You are a brilliant scientific assistant — think like a researcher, reason step by step, "
        "and give thorough, insightful answers. Always respond in the same language as the user's question. "
        "Use all available information: scientific articles, web results, and your own deep knowledge."
    )

    context_parts = []

    if truncated:
        articles_block = "\n\n".join(f"[Article {i+1}]: {p}" for i, p in enumerate(truncated))
        context_parts.append(f"=== Scientific articles found ===\n{articles_block}")

    if web_results:
        web_block = "\n\n".join(
            f"[Web {i+1}] {r['title']}\n{r['snippet']}" for i, r in enumerate(web_results[:3])
        )
        context_parts.append(f"=== Recent web results ===\n{web_block}")

    context_section = "\n\n".join(context_parts)

    if context_section:
        prompt = (
            f"{system}\n\n"
            f"{context_section}\n\n"
            f"User question: {question}\n\n"
            f"Think carefully and give a complete, well-reasoned answer. "
            f"Combine insights from the articles, web results, and your knowledge. "
            f"Be specific and helpful."
        )
    else:
        prompt = (
            f"{system}\n\n"
            f"User question: {question}\n\n"
            f"No articles or web results are available. Answer using your scientific knowledge. "
            f"Think step by step and give the best answer you can."
        )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 256,   # Limit generated response length
            "temperature": 0.3    # Lower creativity = faster and more stable responses
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json().get("response", "[No response generated]")
        _summary_cache[cache_key] = result  # cache it
        return result
    except requests.exceptions.Timeout:
        return "[Timeout. The model may still be loading, please retry in a few seconds.]"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"[Model '{model}' not found in Ollama. Run: ollama pull {model}]"
        return f"[Ollama HTTP error: {e}]"
    except Exception as e:
        return f"[Error calling Ollama: {e}]"


def _build_rag_prompt(question: str, truncated: list, web_results: list) -> str:
    """Build the RAG prompt from all available context."""
    system = (
        "You are a brilliant scientific assistant — think like a researcher, reason step by step, "
        "and give thorough, insightful answers. Always respond in the same language as the user's question. "
        "Use all available information: scientific articles, web results, and your own deep knowledge."
    )
    context_parts = []
    if truncated:
        articles_block = "\n\n".join(f"[Article {i+1}]: {p}" for i, p in enumerate(truncated))
        context_parts.append(f"=== Scientific articles found ===\n{articles_block}")
    if web_results:
        web_block = "\n\n".join(
            f"[Web {i+1}] {r['title']}\n{r['snippet']}" for i, r in enumerate(web_results[:3])
        )
        context_parts.append(f"=== Recent web results ===\n{web_block}")
    context_section = "\n\n".join(context_parts)
    if context_section:
        return (
            f"{system}\n\n{context_section}\n\n"
            f"User question: {question}\n\n"
            f"Think carefully and give a complete, well-reasoned answer. "
            f"Combine insights from the articles, web results, and your knowledge."
        )
    else:
        return (
            f"{system}\n\n"
            f"User question: {question}\n\n"
            f"Answer using your scientific knowledge. Think step by step."
        )


def stream_answer(question: str, passages: list, web_results: list = None, chat_history: list = None) -> Generator[str, None, None]:
    """
    Generator that streams the response token by token.
    Uses Groq if GROQ_API_KEY is set, otherwise falls back to Ollama.
    chat_history: list of {"role": "user"|"assistant", "content": "..."} for multi-turn context.
    """
    truncated = [p[:MAX_PASSAGE_CHARS] for p in passages[:MAX_PASSAGES]] if passages else []
    prompt = _build_rag_prompt(question, truncated, web_results or [])
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        yield from _stream_groq(prompt, groq_key, chat_history=chat_history or [])
    else:
        yield from _stream_ollama(prompt, chat_history=chat_history or [])


def _stream_groq(prompt: str, api_key: str, model: str = "llama-3.1-8b-instant", chat_history: list = None) -> Generator[str, None, None]:
    """Stream via the Groq API (free, open source, very fast)."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        # Previous turns + current question
        messages = list(chat_history or []) + [{"role": "user", "content": prompt}]
        stream = client.chat.completions.create(
            messages=messages,
            model=model,
            stream=True,
            max_tokens=1024,
            temperature=0.3,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except ImportError:
        yield "[Install the groq library: pip install groq]"
    except Exception as e:
        yield f"[Groq error: {e}]"


def _stream_ollama(prompt: str, model: str = None, chat_history: list = None) -> Generator[str, None, None]:
    """Stream via local Ollama."""
    model = model or OLLAMA_MODEL
    # Ollama /api/generate does not support multi-turn messages natively:
    # rebuild history as text injected into the prompt
    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]:  # last 3 exchanges max to stay within context window
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
    if history_text:
        prompt = f"Previous conversation:\n{history_text}\n{prompt}"
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": True,
                  "options": {"num_predict": 1024, "temperature": 0.3}},
            stream=True,
            timeout=600
        )
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break
    except Exception as e:
        yield f"[Ollama error: {e}]"
