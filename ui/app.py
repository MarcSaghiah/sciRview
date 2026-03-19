# Streamlit entry point for sciRview
import streamlit as st
import sys
import os
import hashlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deep_translator import GoogleTranslator
from app.arxiv_search import search_arxiv
from app.pubmed_search import search_pubmed
from app.vector_store import add_documents, search_similar, reset_collection, library_count
from app.llm_ollama import stream_answer, summarize_abstract, warmup_model
from app.web_search import search_web
from app.pdf_utils import extract_and_chunk_pdf

st.set_page_config(
    page_title="sciRview — Scientific Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Remove default Streamlit padding */
#root > div:first-child { padding-top: 0 !important; }
.block-container { padding-top: 1.4rem !important; padding-bottom: 1rem !important; }
header[data-testid="stHeader"] { height: 2.5rem !important; }
/* Reduce divider and title margins */
[data-testid="stMarkdownContainer"] h1 { margin-bottom: 0 !important; padding-bottom: 0 !important; }
[data-testid="stCaptionContainer"] { margin-top: 0 !important; padding-top: 0 !important; }
hr { margin-top: 0.5rem !important; margin-bottom: 0.5rem !important; }
/* Badges source */
.badge-arxiv {
    background-color: #b31b1b; color: white;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.4px;
}
.badge-pubmed {
    background-color: #0071bc; color: white;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.4px;
}
/* Badges score */
.score-high { background:#1a7f4b; color:white; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.score-mid  { background:#c96a00; color:white; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.score-low  { background:#555;    color:white; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
/* Titre article */
.art-title { font-size:1.05rem; font-weight:700; line-height:1.4; }
/* Meta info */
.art-meta  { color:#888; font-size:0.82rem; margin-top:0.3rem; }
/* Welcome card */
.welcome-card {
    text-align:center; padding:2rem 2rem;
    border-radius:12px; margin-top:0.5rem;
}
.welcome-icon { font-size:3.5rem; margin-bottom:0.5rem; }
.welcome-title { font-size:1.4rem; font-weight:700; margin-bottom:0.4rem; }
.welcome-sub   { font-size:0.95rem; color:#888; }
</style>
""", unsafe_allow_html=True)

# --- Session state ---

if "arxiv_results" not in st.session_state:
    st.session_state.arxiv_results = []
if "pubmed_results" not in st.session_state:
    st.session_state.pubmed_results = []
if "score_map" not in st.session_state:
    st.session_state.score_map = {}
if "article_summaries" not in st.session_state:
    st.session_state.article_summaries = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_warmed" not in st.session_state:
    warmup_model()
    st.session_state.model_warmed = True

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🔬 sciRview")
    st.caption("Scientific research assistant")
    st.divider()

    with st.form("search_form"):
        query = st.text_input("Question or keywords", placeholder="e.g. CRISPR cancer therapy 2024")
        max_results = st.slider("Results per source", 1, 20, 5)
        submitted = st.form_submit_button("🔍 Search", use_container_width=True)

    st.divider()

    # LLM engine status
    groq_configured = bool(os.getenv("GROQ_API_KEY", "").strip())
    if groq_configured:
        st.success("⚡ Groq active — fast responses")
    else:
        st.warning("🖥️ Local Ollama\nAdd GROQ_API_KEY for instant responses")

    # Session metrics
    n_articles = len(st.session_state.arxiv_results) + len(st.session_state.pubmed_results)
    if n_articles > 0:
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Articles", n_articles)
        col2.metric("Exchanges", len(st.session_state.chat_history) // 2)

    # Persistent library
    lib_count = library_count()
    if lib_count > 0:
        st.divider()
        st.metric("📚 Library", f"{lib_count} indexed docs")
        if st.button("🗑️ Clear library", use_container_width=True):
            reset_collection()
            st.session_state.score_map = {}
            st.rerun()

# --- Main header ---
st.markdown("# sciRview 🔬")
st.caption("Scientific research assistant — ArXiv · PubMed · RAG · LLM")
st.divider()

# --- Search ---

if submitted and query:
    # Auto-translate query to English
    try:
        translated_query = GoogleTranslator(source='auto', target='en').translate(query)
    except Exception as e:
        st.warning(f"Auto-translation failed: {e}")
        translated_query = query

    if translated_query.strip().lower() != query.strip().lower():
        st.info(f"🌐 Translated query: **{translated_query}**")

    with st.spinner("Searching ArXiv and PubMed..."):
        st.session_state.arxiv_results = search_arxiv(translated_query, max_results)
        st.session_state.pubmed_results = search_pubmed(translated_query, max_results)
        st.session_state.article_summaries = {}

    c1, c2, c3 = st.columns(3)
    c1.metric("ArXiv", len(st.session_state.arxiv_results))
    c2.metric("PubMed", len(st.session_state.pubmed_results))
    c3.metric("Total", len(st.session_state.arxiv_results) + len(st.session_state.pubmed_results))

    # Vectorise abstracts and score articles semantically
    docs = []
    for art in st.session_state.arxiv_results:
        if art.get('summary'):
            docs.append({'id': art['pdf_url'], 'text': art['summary'],
                         'title': art['title'], 'source': 'arxiv', 'published': art['published']})
    for art in st.session_state.pubmed_results:
        if art.get('summary') and art.get('pmid'):
            docs.append({'id': art['pmid'], 'text': art['summary'],
                         'title': art['title'], 'source': 'pubmed', 'published': art['published']})

    if docs:
        with st.spinner("Vectorising and scoring..."):
            add_documents(docs)
            hits = search_similar(translated_query, n_results=max_results * 2, threshold=0.0)
            st.session_state.score_map = {hit['id']: hit['score'] for hit in hits}
    else:
        st.warning("No abstracts available for vectorisation.")
        st.session_state.score_map = {}


def _score_badge(score):
    if score is None:
        return ""
    if score >= 0.65:
        return f'<span class="score-high">★ {score:.2f}</span>'
    elif score >= 0.4:
        return f'<span class="score-mid">◈ {score:.2f}</span>'
    else:
        return f'<span class="score-low">○ {score:.2f}</span>'

def _source_badge(source):
    if source == "arxiv":
        return '<span class="badge-arxiv">ArXiv</span>'
    return '<span class="badge-pubmed">PubMed</span>'

def render_article(art_id, title, authors, published, summary, link_label, link_url, pdf_url=None, source_prefix="", source=""):
    """Render an article card with score, abstract and action buttons."""
    score = st.session_state.score_map.get(art_id)

    with st.container(border=True):
        # Row 1: badges + title
        authors_short = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        st.markdown(
            f"{_source_badge(source)} &nbsp; {_score_badge(score)}<br>"
            f'<span class="art-title">{title}</span><br>'
            f'<span class="art-meta">📅 {published} &nbsp;·&nbsp; 👥 {authors_short}</span>',
            unsafe_allow_html=True
        )

        with st.expander("📄 Abstract"):
            st.markdown(summary)

        btn_cols = st.columns([1, 1, 1, 3])
        summary_key = f"synth_{source_prefix}_{art_id}"

        with btn_cols[0]:
            if st.button("✨ AI Summary", key=f"btn_synth_{source_prefix}_{hashlib.md5(art_id.encode()).hexdigest()[:10]}"):
                with st.spinner("Generating..."):
                    result = summarize_abstract(summary)
                    st.session_state.article_summaries[summary_key] = result

        if pdf_url:
            with btn_cols[1]:
                st.link_button("📄 PDF", url=pdf_url)

        if pdf_url and source == "arxiv":
            with btn_cols[2]:
                pdf_idx_key = f"pdf_idx_{hashlib.md5(art_id.encode()).hexdigest()[:10]}"
                if st.button("📚 Index PDF", key=pdf_idx_key):
                    with st.spinner("Downloading and indexing chunks..."):
                        chunks = extract_and_chunk_pdf(pdf_url)
                        if chunks:
                            chunk_docs = [
                                {'id': f"{art_id}_chunk_{i}", 'text': c,
                                 'title': title, 'source': 'arxiv_pdf', 'published': published}
                                for i, c in enumerate(chunks)
                            ]
                            add_documents(chunk_docs)
                            st.success(f"✅ {len(chunks)} chunks indexed")
                        else:
                            st.warning("PDF not accessible or empty")

        if summary_key in st.session_state.article_summaries:
            st.info("**✨ AI Summary**\n\n" + st.session_state.article_summaries[summary_key])


# --- Combined results, sorted by semantic score ---
all_articles = []
for art in st.session_state.arxiv_results:
    all_articles.append({**art, "_id": art['pdf_url'], "_source": "arxiv"})
for art in st.session_state.pubmed_results:
    all_articles.append({**art, "_id": art.get('pmid', ''), "_source": "pubmed"})

if all_articles:
    score_map = st.session_state.score_map
    # Sort by descending score (articles with no score at the end)
    all_articles.sort(key=lambda a: score_map.get(a["_id"], -1), reverse=True)

    n = len(all_articles)
    st.markdown(f"### 📑 {n} article{'s' if n > 1 else ''} — sorted by relevance")

    for art in all_articles:
        if art["_source"] == "arxiv":
            render_article(
                art_id=art['pdf_url'],
                title=art['title'],
                authors=art['authors'],
                published=art['published'],
                summary=art['summary'],
                link_label="Lien PDF ArXiv",
                link_url=art['pdf_url'],
                pdf_url=art['pdf_url'],
                source_prefix="arxiv",
                source="arxiv"
            )
        else:
            render_article(
                art_id=art['pmid'],
                title=art['title'],
                authors=art['authors'],
                published=art['published'],
                summary=art['summary'],
                link_label="PubMed link",
                link_url=f"https://pubmed.ncbi.nlm.nih.gov/{art['pmid']}/",
                pdf_url=art.get('pdf_url'),
                source_prefix="pubmed",
                source="pubmed"
            )

# --- Welcome screen ---
if not st.session_state.arxiv_results and not st.session_state.pubmed_results:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">🔬</div>
        <div class="welcome-title">Ready to explore the scientific literature</div>
        <div class="welcome-sub">Enter your question in the sidebar to get started</div>
    </div>
    """, unsafe_allow_html=True)

# --- RAG chat interface ---
st.divider()
st.markdown("## 🧠 Scientific Assistant")

import os
groq_configured = bool(os.getenv("GROQ_API_KEY", "").strip())
if groq_configured:
    st.caption("⚡ Engine: **Groq** (llama3, fast) — indexed articles + web search + general knowledge")
else:
    st.caption("🖥️ Engine: **Local Ollama** — add GROQ_API_KEY in .env for instant responses")

# Display conversation history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if rag_question := st.chat_input("🔬 Ask your question (e.g. What side effects are mentioned?)..."):
    # Display the question
    with st.chat_message("user"):
        st.markdown(rag_question)
    st.session_state.chat_history.append({"role": "user", "content": rag_question})

    # Retrieve context
    try:
        translated_q = GoogleTranslator(source='auto', target='en').translate(rag_question)
    except Exception:
        translated_q = rag_question

    hits = search_similar(translated_q, n_results=5, threshold=0.4)  # permissive threshold for RAG
    passages = [hit['text'] for hit in hits] if hits else []
    web_results = search_web(translated_q, max_results=4)

    # Context indicator
    context_info = []
    if hits:
        context_info.append(f"{len(hits)} indexed article(s)")
    if web_results:
        context_info.append(f"{len(web_results)} web result(s)")
    if context_info:
        st.caption(f"📚 Injected context: {' + '.join(context_info)}")
    else:
        st.caption("⚠️ No context found — response based on general knowledge only")

    with st.chat_message("assistant"):
        response = st.write_stream(stream_answer(rag_question, passages, web_results, chat_history=st.session_state.chat_history))

    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # RAG metrics
    if hits:
        avg_score = sum(h.get('rerank_score', h['score']) for h in hits) / len(hits)
        st.caption(f"📊 Context: avg score `{avg_score:.2f}` · {len(hits)} passage(s) · {len(web_results)} web")

# Clear conversation button
if st.session_state.chat_history:
    if st.button("🗑️ Clear conversation", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

