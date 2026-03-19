# sciRview

Worked on: September 2025

Stack: Python, Streamlit, Groq API, Ollama, ChromaDB, Hugging Face Transformers, ArXiv API, PubMed API, DuckDuckGo Search

**Open-source scientific research assistant — ArXiv · PubMed**

## 🚀 Overview

Ask a scientific question in any language. sciRview searches for articles, ranks them by semantic relevance, and answers via a context-aware LLM using the found abstracts and live web results.

---

## ✨ Features

- **Multi-source search**: ArXiv and PubMed simultaneously via their official APIs
- **Auto-translation**: ask in any language, the search runs in English
- **Scientific semantic scoring**: articles ranked by true relevance using [`allenai-specter`](https://huggingface.co/sentence-transformers/allenai-specter) (trained on millions of paper citations)
- **Cross-encoder reranking**: retrieved passages are reranked by [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) to maximise context quality injected into the LLM
- **Results sorted by relevance**: ArXiv and PubMed merged and ranked from most to least relevant
- **Full PDF indexing**: "📚 Index PDF" button per ArXiv article — downloads the PDF, splits it into ~400-word overlapping chunks, and indexes them in ChromaDB for much richer RAG context than abstracts alone
- **Persistent library**: ChromaDB keeps all indexed documents between sessions; each new search enriches the database without erasing previous ones. A "🗑️ Clear library" button in the sidebar resets everything
- **AI summary per article**: "✨ AI Summary" button on each article for a natural language summary
- **Multi-turn RAG chat**: contextualised discussion with conversation memory — the LLM answers using reranked indexed passages + live DuckDuckGo web results
- **RAG metrics**: after each response, displays average context score, number of passages, and web results used
- **Dual LLM engine**:
  - **Groq** (recommended): free cloud, ~2 second response, `llama-3.1-8b-instant` model
  - **Ollama**: local CPU/GPU fallback, no internet connection required
- **PDF access**: direct link to the PDF for each ArXiv article

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Embeddings | `allenai-specter` via SentenceTransformers |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` via SentenceTransformers |
| Vector store | ChromaDB (PersistentClient, cosine similarity, persistent) |
| PDF chunking | Sliding window 400 words / 80-word overlap |
| Primary LLM | Groq API — `llama-3.1-8b-instant` (free) |
| Fallback LLM | Local Ollama (`phi3:mini` by default) |
| Web search | DuckDuckGo (`ddgs`, no API key) |
| Scientific APIs | `arxiv`, `biopython` (PubMed/NCBI) |
| PDF extraction | PyMuPDF |
| Translation | `deep-translator` (Google Translate) |

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- Git

### 1. Clone the repo

```bash
git clone https://github.com/your-account/sciRview.git
cd sciRview
```

### 2. Create and activate a virtual environment

**Windows:**

```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> On first launch, two models are automatically downloaded from Hugging Face:
> - `allenai-specter` (~440 MB) — scientific embeddings
> - `cross-encoder/ms-marco-MiniLM-L-6-v2` (~91 MB) — reranking

### 4. Configure the `.env` file

Create a `.env` file at the project root:

```env
# Email required for NCBI/PubMed API (required)
PUBMED_EMAIL=your@email.com

# Ollama model (only used if GROQ_API_KEY is empty)
# CPU only    → phi3:mini  (recommended, fast)
# GPU         → mistral or llama3:8b
OLLAMA_MODEL=phi3:mini

# Groq API — free, 10x faster than local Ollama
# Create an account at https://console.groq.com and generate a key
GROQ_API_KEY=gsk_your_key_here
```

---

## 🚀 Launch

### Windows (recommended)

Double-click `run.bat` or in a PowerShell terminal:

```powershell
.\run.bat
```

This script activates the venv, clears the Python cache (.pyc), and launches Streamlit.

### All platforms

```bash
streamlit run ui/app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

---

## ⚡ Groq Configuration

Without Groq, the LLM runs locally via Ollama — ~2 minutes on CPU.
With Groq (free, open source), responses come in ~2 seconds.

1. Create a free account at [console.groq.com](https://console.groq.com)
2. Go to **API Keys** → **Create API Key**
3. Copy the key (`gsk_...`) and add it to `.env`:

   ```env
   GROQ_API_KEY=gsk_your_key_here
   ```

4. Restart the app

Once configured, the assistant header shows: `⚡ Engine: Groq (llama3, fast)`

---

## 🦙 Ollama Configuration

If you prefer a fully local LLM with no data sent externally:

1. Install [Ollama](https://ollama.com/download)
2. Download a model:

   ```bash
   ollama pull phi3:mini   # CPU, ~2 GB
   # or
   ollama pull mistral     # GPU, ~4 GB
   ```

3. Leave `GROQ_API_KEY` empty in `.env`

---

## 📖 Usage

### Search

1. Enter a question or keywords (any language)
2. Choose the number of results per source (1–20)
3. Click **Search**

Articles are displayed **sorted by semantic relevance score** (most relevant first), all sources merged (ArXiv + PubMed).

| Score | Interpretation |
|---|---|
| `0.8 – 1.0` | Highly relevant |
| `0.5 – 0.8` | Relevant |
| `0.3 – 0.5` | Indirect link |
| `< 0.3` | Weakly related |

### AI summary per article

Click **✨ AI Summary** on an article to get a clear natural language summary of its abstract. The summary is generated once and cached for the session.

### Full PDF indexing

For ArXiv articles, click **📚 Index PDF** to download and split the PDF into chunks (~400 words) and index them in ChromaDB. This greatly enriches the RAG context beyond the abstract alone. Chunks persist between sessions.

The sidebar shows the total number of indexed documents and offers a **🗑️ Clear library** button to start over.

### Scientific chat (RAG)

The **🧠 Scientific Assistant** section at the bottom lets you chat with the LLM:

- The most relevant passages are retrieved from ChromaDB then **reranked** by the cross-encoder before injection
- A DuckDuckGo web search enriches context in real time
- The indicator `📚 Injected context: X article(s) + Y web result(s)` shows what was used
- After each response: `📊 Context: avg score X.XX · N passage(s) · M web`
- The chat is **multi-turn**: follow-up questions take previous exchanges into account
- If no context is found: `⚠️ No context found — response based on general knowledge only`
- Click **🗑️ Clear conversation** to start over

### Example questions

```
What are the side effects of semaglutide on the cardiovascular system?
Impact of deep learning in medical radiology in 2024?
CRISPR gene editing in cancer therapy recent advances
What biomarkers are associated with early Alzheimer's disease?
```

---

## 📂 Project Structure

```
sciRview/
├── ui/
│   └── app.py                  # Main Streamlit interface
├── app/
│   ├── arxiv_search.py         # ArXiv API search
│   ├── pubmed_search.py        # PubMed/NCBI API search
│   ├── vector_store.py         # ChromaDB + allenai-specter + cross-encoder reranking
│   ├── llm_ollama.py           # LLM: Groq (primary) + Ollama (fallback), multi-turn
│   ├── web_search.py           # DuckDuckGo web search (ddgs)
│   └── pdf_utils.py            # PDF extraction + chunking (sliding window)
├── data/
│   └── chroma_db/              # Local vector database (not committed)
├── .env                        # Environment variables (not committed)
├── requirements.txt
├── run.bat                     # Quick Windows launcher (terminal visible)
└── launch.vbs                  # Silent Windows launcher (no terminal window)
```

---

## 🔧 Troubleshooting

**`ModuleNotFoundError` on launch**
Make sure the venv is activated and `pip install -r requirements.txt` has been run.

**`⚠️ No context found` in chat**
Run a search first, then ask your question in the chat.

**LLM responds in 2 minutes**
Ollama is running on CPU. Add a `GROQ_API_KEY` in `.env` to switch to Groq (~2s).

**PubMed returns 0 results**
Check that `PUBMED_EMAIL` is set in `.env`.

**Warnings in the terminal (Windows symlinks, embeddings.position_ids)**
These messages have no functional impact and can be ignored.

**Import error for `duckduckgo_search`**
The package was renamed. Run: `pip install ddgs`

---

## 📄 License

MIT — Open source project, contributions welcome.

---

**Author: Marc Saghiah**
