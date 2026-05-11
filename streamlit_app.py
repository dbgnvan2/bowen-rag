#!/usr/bin/env python3
"""
Bowen Theory RAG — Streamlit web app
Same backend as the desktop app; designed for Railway deployment.
"""

import json
import logging
import os
import re
from pathlib import Path

# Load .env so local runs pick up API keys without manual export
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Suppress verbose startup logs from PyTorch / transformers / sentence-transformers
import warnings
warnings.filterwarnings("ignore")
for _noisy in ("transformers", "sentence_transformers", "torch", "filelock",
               "urllib3", "huggingface_hub", "tqdm"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.WARNING)

import numpy as np
import streamlit as st
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
REFS_DIR = BASE_DIR / "rag-document-search" / "references"

# ── Authority tiers ─────────────────────────────────────────────────────────
_AUTHORITY_TIERS_DEFAULT = [
    ("Family Therapy in_Clinical_Practice_Chapter", 3.0),
    ("Family Evaluation",                            3.0),
    ("Bowen Basic Series Tape",                      3.0),
    ("BOWEN-KERR INTERVIEW SERIES",                  3.0),
    ("Bowen Family Systems Theory",                  3.0),
    ("Bowen on Triangles",                           3.0),
    ("Bowen Theory and Therapy",                     3.0),
    ("Chronic Anxiety and Defining",                 3.0),
    ("Cancer and the Emotional System",              3.0),
    ("Family and Society Kerr",                      3.0),
    ("Family as a System Kerr",                      3.0),
    ("Family Systems and Therapy Kerr",              3.0),
    ("Physical Illness as the Family Emotional",     3.0),
    ("Psychotherapy Past Present Future",            3.0),
    ("FSJ ",                                         1.3),
    ("Copy of ",                                     1.3),
    ("Family Center Reports",                        1.3),
    ("Papero",                                       1.15),
    ("Friedman",                                     1.15),
    ("Fogarty",                                      1.15),
    ("Guerin",                                       1.15),
    ("Toman",                                        1.15),
]


def _load_authority_tiers() -> list:
    config = BASE_DIR / "authority_tiers.yml"
    if not config.exists():
        return _AUTHORITY_TIERS_DEFAULT
    try:
        import yaml
        with open(config, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return [(t["pattern"], float(t["multiplier"]))
                for t in data.get("tiers", []) if "pattern" in t]
    except Exception:
        return _AUTHORITY_TIERS_DEFAULT


AUTHORITY_TIERS = _load_authority_tiers()


def authority_boost(doc_name: str) -> float:
    dn = doc_name.lower()
    for pattern, mult in AUTHORITY_TIERS:
        if pattern.lower() in dn:
            return mult
    return 1.0


def _load_author_map() -> list:
    config = BASE_DIR / "author_map.yml"
    if not config.exists():
        return []
    try:
        import yaml
        with open(config, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return [(a["pattern"], a["author"])
                for a in data.get("authors", []) if "pattern" in a and "author" in a]
    except Exception:
        return []


AUTHOR_MAP = _load_author_map()


def doc_author(doc_name: str) -> str:
    dn = doc_name.lower()
    for pattern, author in AUTHOR_MAP:
        if pattern.lower() in dn:
            return author
    return "Unknown"


def all_known_authors() -> list:
    seen: list = []
    for _, author in AUTHOR_MAP:
        if author not in seen:
            seen.append(author)
    return seen


# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a research assistant helping analyse the Bowen Family Systems Theory literature. "
    "STRICT RULES — you must follow these without exception:\n"
    "1. Use ONLY the source excerpts provided in the user message. "
    "Do not draw on any prior training knowledge, general knowledge, or outside information.\n"
    "2. Do not infer, extrapolate, or fill gaps with assumptions. "
    "If the provided excerpts do not address something, say so explicitly rather than guessing.\n"
    "3. Every claim or statement in your response must be directly traceable to a specific excerpt. "
    "Cite the source document in brackets immediately after the claim, e.g. [Document Name].\n"
    "4. If sources conflict or are ambiguous, note the conflict and quote both — do not resolve it yourself.\n"
    "5. Do not add introductory or concluding remarks that go beyond what the sources say.\n"
    "6. If asked about something not covered in the provided excerpts, respond: "
    "'The provided sources do not contain information on this point.'"
)

CLAUDE_MODELS = [
    "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5",
    "claude-opus-4-6", "claude-opus-4-5", "claude-sonnet-4-5",
]
OPENAI_MODELS    = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini"]
DEEPSEEK_MODELS  = ["deepseek-v4-flash", "deepseek-v4-pro"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"


# ══════════════════════════════════════════════════════════════════════════════
# IndexManager  (identical to bowen_rag_gui.py — no tkinter dependency)
# ══════════════════════════════════════════════════════════════════════════════

class IndexManager:
    def __init__(self):
        self.chunks: list = []
        self.matrix       = None
        self.vectorizer   = None
        self.bm25         = None
        self.embed_matrix = None
        self.embed_model  = None
        self.loaded       = False

    def load(self, refs_dir: Path = REFS_DIR) -> dict:
        meta_path  = refs_dir / "chunk_metadata.json"
        matrix_npz = refs_dir / "tfidf_matrix.npz"
        matrix_npy = refs_dir / "tfidf_matrix.npy"

        if not meta_path.exists():
            raise FileNotFoundError(f"Index not found at {refs_dir}.")

        with open(meta_path) as f:
            self.chunks = json.load(f)

        if matrix_npz.exists():
            self.matrix = sp_sparse.load_npz(str(matrix_npz)).toarray()
        else:
            self.matrix = np.load(str(matrix_npy))

        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            max_features=8000, stop_words="english",
            lowercase=True, ngram_range=(1, 2),
            min_df=2, sublinear_tf=True
        )
        self.vectorizer.fit(texts)
        self.loaded = True

        self._doc_chunk_ids: dict = {}
        for i, c in enumerate(self.chunks):
            self._doc_chunk_ids.setdefault(c["doc_name"], []).append(i)

        docs = len(set(c["doc_name"] for c in self.chunks))

        embed_npy = refs_dir / "embed_matrix.npy"
        if EMBEDDING_AVAILABLE and embed_npy.exists():
            self.embed_matrix = np.load(str(embed_npy))

        if BM25_AVAILABLE:
            tokenized = [self._tokenize(c["text"]) for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized)

        return {"chunks": len(self.chunks), "documents": docs,
                "embeddings": self.embed_matrix is not None}

    def get_context_window(self, chunk_id: int, window: int = 2) -> list:
        doc_name = self.chunks[chunk_id]["doc_name"]
        doc_ids  = self._doc_chunk_ids.get(doc_name, [])
        try:
            pos = doc_ids.index(chunk_id)
        except ValueError:
            return [self.chunks[chunk_id]["text"]]
        start = max(0, pos - window)
        end   = min(len(doc_ids), pos + window + 1)
        return [self.chunks[doc_ids[j]]["text"] for j in range(start, end)]

    def semantic_search(self, query: str, top_k: int, use_boost: bool = True) -> list:
        if not self.loaded:
            return []
        qvec     = self.vectorizer.transform([query])
        raw      = cosine_similarity(qvec, self.matrix)[0]
        boost_fn = authority_boost if use_boost else (lambda _: 1.0)
        boosted  = np.array([raw[i] * boost_fn(self.chunks[i]["doc_name"])
                             for i in range(len(self.chunks))])
        idx = boosted.argsort()[::-1][:top_k]
        return [
            {**self.chunks[i],
             "score":       float(boosted[i]),
             "score_label": f"{boosted[i]*100:.0f}% ★"
                            if use_boost and authority_boost(self.chunks[i]["doc_name"]) > 1.0
                            else f"{boosted[i]*100:.0f}%",
             "mode": "semantic"}
            for i in idx if boosted[i] > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> list:
        return [w for w in re.split(r'\W+', text.lower())
                if len(w) > 2 and w not in IndexManager._STOP]

    _STOP = frozenset({
        "the","and","for","are","but","not","you","all","can","had","her","was",
        "one","our","out","day","get","has","him","his","how","man","new","now",
        "old","see","two","way","who","boy","did","its","let","put","say","she",
        "too","use","what","with","this","that","have","from","they","will","been",
        "more","when","than","them","were","said","each","which","about","there",
        "their","would","make","like","into","time","look","just","come","could",
        "also","some","then","these","many","well","only","over","such","after",
        "most","very","even","back","any","good","know","same","tell","does",
        "bowen","kerr","theory","family","therapy","systems","system",
        "murray","michael","dr","said","think","know","people","things",
    })

    @staticmethod
    def _stems(word: str) -> list:
        variants = [word]
        if word.endswith("ing") and len(word) > 5:
            variants.append(word[:-3])
        if word.endswith("ed") and len(word) > 4:
            variants.append(word[:-2])
            variants.append(word[:-1])
        if word.endswith("ies") and len(word) > 4:
            variants.append(word[:-3] + "y")
        if word.endswith("es") and len(word) > 4:
            variants.append(word[:-2])
        if word.endswith("s") and len(word) > 3:
            variants.append(word[:-1])
        return variants

    def keyword_search(self, query: str, top_k: int, use_boost: bool = True) -> list:
        if not self.loaded:
            return []
        raw_terms = [t.lower() for t in query.split()
                     if len(t) > 2 and t.lower() not in self._STOP]
        if not raw_terms:
            return []
        term_sets = [set(self._stems(t)) for t in raw_terms]
        boost_fn  = authority_boost if use_boost else (lambda _: 1.0)
        doc_best: dict = {}
        for c in self.chunks:
            tl   = c["text"].lower()
            hits = sum(max(tl.count(v) for v in variants) for variants in term_sets)
            if hits == 0:
                continue
            dn    = c["doc_name"]
            score = hits * boost_fn(dn)
            label = f"{hits} hits" + (" ★" if use_boost and authority_boost(dn) > 1.0 else "")
            if dn not in doc_best or score > doc_best[dn]["score"]:
                doc_best[dn] = {**c, "score": score, "score_label": label, "mode": "keyword"}
        out = sorted(doc_best.values(), key=lambda x: x["score"], reverse=True)
        return out[:top_k]

    def combined_search(self, query: str, top_k: int, use_boost: bool = True) -> list:
        sem = {r["id"]: r for r in self.semantic_search(query, top_k, use_boost=use_boost)}
        kw  = {r["id"]: r for r in self.keyword_search(query, top_k, use_boost=use_boost)}
        merged = {**kw, **sem}
        return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    def bm25_search(self, query: str, top_k: int, use_boost: bool = True) -> list:
        if not self.loaded or self.bm25 is None:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        boost_fn = authority_boost if use_boost else (lambda _: 1.0)
        raw      = self.bm25.get_scores(tokens)
        boosted  = np.array([raw[i] * boost_fn(self.chunks[i]["doc_name"])
                             for i in range(len(self.chunks))])
        idx = boosted.argsort()[::-1][:top_k]
        return [
            {**self.chunks[i],
             "score":       float(boosted[i]),
             "score_label": f"{boosted[i]:.2f} ★"
                            if use_boost and authority_boost(self.chunks[i]["doc_name"]) > 1.0
                            else f"{boosted[i]:.2f}",
             "mode": "bm25"}
            for i in idx if boosted[i] > 0
        ]

    def hybrid_search(self, query: str, top_k: int, use_boost: bool = True) -> list:
        if self.embed_matrix is None:
            raise RuntimeError(
                "Embedding index not built. Run 'Build Embeddings' locally and commit embed_matrix.npy.")
        if len(self.embed_matrix) != len(self.chunks):
            raise RuntimeError(
                f"Embedding index is stale ({len(self.embed_matrix)} vs {len(self.chunks)} chunks).")
        pool         = min(top_k * 4, len(self.chunks))
        bm25_results  = self.bm25_search(query, pool, use_boost=use_boost)
        embed_results = self.embedding_search(query, pool, use_boost=use_boost)
        bm25_rank  = {r["id"]: i for i, r in enumerate(bm25_results)}
        embed_rank = {r["id"]: i for i, r in enumerate(embed_results)}
        K      = 60
        all_ids = set(bm25_rank) | set(embed_rank)
        rrf = {cid: (1.0 / (K + bm25_rank[cid])  if cid in bm25_rank  else 0.0)
                   + (1.0 / (K + embed_rank[cid]) if cid in embed_rank else 0.0)
               for cid in all_ids}
        top_ids      = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:top_k]
        chunk_lookup = {r["id"]: r for r in embed_results + bm25_results}
        max_rrf      = 2.0 / K
        results = []
        for cid in top_ids:
            base = chunk_lookup[cid].copy()
            pct  = rrf[cid] / max_rrf * 100
            boosted_flag = use_boost and authority_boost(base["doc_name"]) > 1.0
            base["score"]       = rrf[cid]
            base["score_label"] = f"{pct:.0f}% ⬡" + (" ★" if boosted_flag else "")
            base["mode"]        = "hybrid"
            results.append(base)
        return results

    def embedding_search(self, query: str, top_k: int, use_boost: bool = True) -> list:
        if not self.loaded or self.embed_matrix is None:
            raise RuntimeError(
                "Embedding index not available. Build it locally and commit embed_matrix.npy.")
        if self.embed_model is None:
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        if len(self.embed_matrix) != len(self.chunks):
            raise RuntimeError("Embedding index is stale — rebuild it.")
        qvec    = self.embed_model.encode([query])
        raw     = cosine_similarity(qvec, self.embed_matrix)[0]
        boost_fn = authority_boost if use_boost else (lambda _: 1.0)
        boosted = np.array([raw[i] * boost_fn(self.chunks[i]["doc_name"])
                            for i in range(len(self.chunks))])
        idx = boosted.argsort()[::-1][:top_k]
        return [
            {**self.chunks[i],
             "score":       float(boosted[i]),
             "score_label": f"{boosted[i]*100:.0f}% ✦"
                            if use_boost and authority_boost(self.chunks[i]["doc_name"]) > 1.0
                            else f"{boosted[i]*100:.0f}%",
             "mode": "embedding"}
            for i in idx if boosted[i] > 0
        ]

    def top_docs_search(self, query: str, top_chunks: int = 300,
                        top_docs: int = 30, use_boost: bool = True) -> list:
        if not self.loaded:
            return []
        qvec     = self.vectorizer.transform([query])
        raw      = cosine_similarity(qvec, self.matrix)[0]
        boost_fn = authority_boost if use_boost else (lambda _: 1.0)
        doc_chunks: dict = {}
        for i, score in enumerate(raw):
            if score <= 0:
                continue
            dn = self.chunks[i]["doc_name"]
            doc_chunks.setdefault(dn, []).append((score, i))
        doc_scores: dict = {}
        for dn, pairs in doc_chunks.items():
            pairs.sort(reverse=True)
            top3_sum  = sum(s for s, _ in pairs[:3])
            agg_score = top3_sum * boost_fn(dn)
            best_idx  = pairs[0][1]
            label     = f"{agg_score*100:.0f}%"
            if use_boost and authority_boost(dn) > 1.0:
                label += " ★"
            doc_scores[dn] = {
                **self.chunks[best_idx],
                "score":       agg_score,
                "score_label": label,
                "mode":        "semantic",
            }
        out = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return out[:top_docs]

    def list_documents(self) -> list:
        seen, docs = set(), []
        for c in self.chunks:
            if c["doc_name"] not in seen:
                seen.add(c["doc_name"])
                docs.append(c["doc_name"])
        return sorted(docs)


# ══════════════════════════════════════════════════════════════════════════════
# LLM streaming
# ══════════════════════════════════════════════════════════════════════════════

def _llm_stream(messages: list, system: str):
    """Generator that yields tokens from the configured LLM provider."""
    ss       = st.session_state
    provider = ss.get("provider", "claude")

    if provider == "claude":
        key = ss.get("claude_key", "")
        if not key:
            raise RuntimeError("Claude API key not set — go to Settings.")
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        with client.messages.stream(
            model=ss.get("claude_model", "claude-sonnet-4-6"),
            max_tokens=16000, system=system, messages=messages
        ) as s:
            for token in s.text_stream:
                yield token

    elif provider == "openai":
        key = ss.get("openai_key", "")
        if not key:
            raise RuntimeError("OpenAI API key not set — go to Settings.")
        import openai
        client = openai.OpenAI(api_key=key)
        full = [{"role": "system", "content": system}] + messages
        with client.chat.completions.create(
            model=ss.get("openai_model", "gpt-4o"),
            max_tokens=16000, messages=full, stream=True
        ) as s:
            for chunk in s:
                t = chunk.choices[0].delta.content or ""
                if t:
                    yield t

    elif provider == "deepseek":
        key = ss.get("deepseek_key", "")
        if not key:
            raise RuntimeError("DeepSeek API key not set — go to Settings.")
        import anthropic
        client = anthropic.Anthropic(api_key=key, base_url=DEEPSEEK_BASE_URL)
        with client.messages.stream(
            model=ss.get("deepseek_model", "deepseek-v4-flash"),
            max_tokens=8000, system=system, messages=messages
        ) as s:
            for token in s.text_stream:
                yield token

    else:  # ollama
        import requests
        url   = ss.get("ollama_url", "http://localhost:11434")
        model = ss.get("ollama_model", "qwen2.5:7b")
        full  = [{"role": "system", "content": system}] + messages
        r = requests.post(f"{url.rstrip('/')}/api/chat",
                          json={"model": model, "messages": full, "stream": True},
                          stream=True, timeout=300)
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                d = json.loads(line)
                t = d.get("message", {}).get("content", "")
                if t:
                    yield t
                if d.get("done"):
                    break


# ══════════════════════════════════════════════════════════════════════════════
# App bootstrap
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading index…")
def _get_index() -> IndexManager:
    idx = IndexManager()
    idx.load(REFS_DIR)
    return idx


def _init_session():
    defaults = {
        "search_results":  [],
        "staged_chunks":   [],
        "chat_history":    [],
        "last_rpt_context":  "",
        "last_rpt_appendix": "",
        "last_report":       "",
        "provider":        os.environ.get("LLM_PROVIDER", "claude"),
        "claude_key":      os.environ.get("ANTHROPIC_API_KEY", ""),
        "claude_model":    os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "openai_key":      os.environ.get("OPENAI_API_KEY", ""),
        "openai_model":    os.environ.get("OPENAI_MODEL", "gpt-4o"),
        "ollama_url":      os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        "ollama_model":    os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
        "deepseek_key":    os.environ.get("DEEPSEEK_API_KEY", ""),
        "deepseek_model":  os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "system_prompt":       SYSTEM_PROMPT,
        "default_search_mode": "hybrid",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _check_auth():
    required = os.environ.get("APP_PASSWORD", "")
    if not required:
        return
    if st.session_state.get("authenticated"):
        return
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.title("Bowen Theory RAG")
        st.caption("Enter the access password to continue.")
        pwd = st.text_input("Password", type="password")
        if st.button("Enter", type="primary", use_container_width=True):
            if pwd == required:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()


def _gather_chunks(idx: IndexManager, query: str) -> list:
    ss        = st.session_state
    mode      = ss.get("rpt_mode", "top-docs")
    k         = ss.get("rpt_k", 30)
    use_boost = ss.get("rpt_boost", True)

    if "top-docs" in mode:
        fresh = idx.top_docs_search(query, top_chunks=300, top_docs=k, use_boost=use_boost)
    elif mode == "semantic":
        fresh = idx.semantic_search(query, k, use_boost=use_boost)
    elif mode == "keyword":
        fresh = idx.keyword_search(query, k, use_boost=use_boost)
    elif mode == "embedding":
        fresh = idx.embedding_search(query, k, use_boost=use_boost)
    elif mode == "hybrid":
        fresh = idx.hybrid_search(query, k, use_boost=use_boost)
    else:
        fresh = idx.combined_search(query, k, use_boost=use_boost)

    staged = ss.get("staged_chunks", [])
    if not staged:
        return fresh

    seen: set = set()
    merged: list = []
    for c in staged + fresh:
        key = c.get("id") if c.get("id") is not None else c.get("doc_name", "")
        if key in seen:
            continue
        seen.add(key)
        merged.append(c)
    return merged


def _format_chunk_text(text: str) -> str:
    """Return chunk text cleaned up for readable markdown display."""
    # Strip [Section Title]\n\n prefix added by the indexer
    text = re.sub(r'^\[[^\]]+\]\s*\n\n', '', text, count=1)
    # Split on paragraph breaks (2+ newlines)
    paragraphs = re.split(r'\n{2,}', text)
    cleaned = []
    for para in paragraphs:
        # Collapse soft-wrap newlines within a paragraph (common in PDF text)
        para = re.sub(r'[ \t]*\n[ \t]*', ' ', para)
        # Collapse multiple spaces
        para = re.sub(r' {2,}', ' ', para).strip()
        if para:
            cleaned.append(para)
    return '\n\n'.join(cleaned)


@st.dialog("Section text", width="large")
def _show_section_dialog() -> None:
    result = st.session_state.get("_dialog_result", {})
    doc    = result.get("doc_name", "")
    title  = result.get("section_title", "")
    page   = result.get("page")
    chunk_pos   = result.get("chunk_pos")
    doc_total   = result.get("doc_chunk_count", 1)

    parts = [doc]
    if title:
        parts.append(title)
    if page:
        parts.append(f"p. {page}")
    elif chunk_pos and doc_total > 1:
        parts.append(f"~{round(chunk_pos / doc_total * 100)}% through document")
    st.caption(" · ".join(parts))
    st.divider()
    st.markdown(_format_chunk_text(result.get("text", "")))


def _score_color(result: dict) -> str:
    score = result.get("score", 0)
    if result.get("mode") == "keyword":
        return "#7c3aed"
    if isinstance(score, float):
        if score > 0.4:
            return "#16a34a"
        if score > 0.15:
            return "#ca8a04"
    return "#6b7280"


def _result_card(result: dict, checkbox_key: str):
    """Render a single search result card with checkbox."""
    author  = doc_author(result["doc_name"])
    color   = _score_color(result)
    excerpt = re.sub(r'\s+', ' ', result["text"][:220]).strip()
    page    = result.get("page")

    col_cb, col_body, col_view = st.columns([0.04, 0.88, 0.08])
    with col_cb:
        checked = st.checkbox("Select", key=checkbox_key, label_visibility="collapsed")
    with col_body:
        badges = (
            f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:11px;margin-right:4px">{result["score_label"]}</span>'
        )
        if author != "Unknown":
            badges += (
                f'<span style="background:#7c3aed;color:white;padding:2px 8px;'
                f'border-radius:4px;font-size:11px;margin-right:4px">{author}</span>'
            )
        if page:
            badges += (
                f'<span style="background:#475569;color:white;padding:2px 8px;'
                f'border-radius:4px;font-size:11px;margin-right:4px">p.{page}</span>'
            )
        else:
            chunk_pos = result.get("chunk_pos")
            doc_total = result.get("doc_chunk_count", 1)
            if chunk_pos and doc_total > 1:
                pct = round(chunk_pos / doc_total * 100)
                badges += (
                    f'<span style="background:#64748b;color:white;padding:2px 8px;'
                    f'border-radius:4px;font-size:11px;margin-right:4px">~{pct}%</span>'
                )
        st.markdown(
            f'{badges}<strong style="font-size:13px">{result["doc_name"]}</strong>',
            unsafe_allow_html=True
        )
        st.caption(excerpt + "…")
    with col_view:
        if st.button("↗", key=f"view_{checkbox_key}", use_container_width=True):
            st.session_state["_dialog_result"] = result
            _show_section_dialog()
    return checked


# ══════════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════════

def page_search(idx: IndexManager):
    st.header("Search")

    # 40% controls panel | 60% results — wide enough for help= icons to render
    ctrl, results_col = st.columns([2, 3])

    with ctrl:
        query = st.text_area(
            "Query", height=100, placeholder="Enter your search query…",
            help="Enter keywords or a natural-language question about Bowen theory.",
        )

        mode_options = [
            ("Top Docs (recommended)", "top-docs"),
            ("Semantic (TF-IDF)",      "semantic"),
            ("Keyword",                "keyword"),
            ("Both",                   "both"),
        ]
        if EMBEDDING_AVAILABLE and idx.embed_matrix is not None:
            mode_options.append(("Embedding", "embedding"))
        if EMBEDDING_AVAILABLE and BM25_AVAILABLE and idx.embed_matrix is not None:
            mode_options.append(("Hybrid (BM25 + Embedding)", "hybrid"))

        mode_labels = [m[0] for m in mode_options]
        mode_values = [m[1] for m in mode_options]
        default_mode = st.session_state.get("default_search_mode", "hybrid")
        default_idx  = mode_values.index(default_mode) if default_mode in mode_values else 0

        mode_idx = st.selectbox(
            "Mode", range(len(mode_labels)),
            format_func=lambda i: mode_labels[i],
            index=default_idx,
            help=(
                "**Top Docs** — aggregates per document; best for most queries.\n\n"
                "**Semantic** — TF-IDF cosine similarity; fast, exact-vocabulary.\n\n"
                "**Keyword** — exact word matching with stemming; good for names.\n\n"
                "**Both** — merges semantic + keyword.\n\n"
                "**Embedding** — conceptual similarity regardless of exact wording.\n\n"
                "**Hybrid** — BM25 + Embedding via Reciprocal Rank Fusion; usually best overall."
            ),
        )
        mode = mode_values[mode_idx]

        top_k = st.number_input(
            "Results", min_value=1, max_value=200, value=15,
            help="Maximum number of results to return.",
        )
        use_boost = st.checkbox(
            "Authority boost", value=True,
            help=(
                "Multiplies scores for primary Bowen/Kerr sources (3×), "
                "FSJ articles (1.3×), and other named theorists (1.15×)."
            ),
        )

        authors       = ["All authors"] + all_known_authors()
        author_filter = st.selectbox(
            "Author filter", authors,
            help="Narrow results to a specific author.",
        )

        search_clicked = st.button("Search", type="primary", use_container_width=True)

        st.divider()
        staged = st.session_state.get("staged_chunks", [])
        if staged:
            st.success(f"{len(staged)} chunks staged for Report")
            if st.button("Clear staged", use_container_width=True):
                st.session_state.staged_chunks = []
                st.rerun()

    if search_clicked and query.strip():
        st.session_state.last_search_query = query.strip()
        with st.spinner("Searching…"):
            try:
                if mode == "top-docs":
                    results = idx.top_docs_search(query, top_chunks=300,
                                                  top_docs=top_k, use_boost=use_boost)
                elif mode == "semantic":
                    results = idx.semantic_search(query, top_k, use_boost=use_boost)
                elif mode == "keyword":
                    results = idx.keyword_search(query, top_k, use_boost=use_boost)
                elif mode == "embedding":
                    results = idx.embedding_search(query, top_k, use_boost=use_boost)
                elif mode == "hybrid":
                    results = idx.hybrid_search(query, top_k, use_boost=use_boost)
                else:
                    results = idx.combined_search(query, top_k, use_boost=use_boost)
            except RuntimeError as e:
                st.error(str(e))
                results = []

        if author_filter != "All authors":
            results = [r for r in results if doc_author(r["doc_name"]) == author_filter]

        for key in list(st.session_state.keys()):
            if key.startswith("sel_"):
                del st.session_state[key]

        st.session_state.search_results = results

    with results_col:
        results = st.session_state.get("search_results", [])
        if not results:
            st.info("Run a search to see results.")
        else:
            sel_col1, sel_col2, sel_col3 = st.columns(3)
            with sel_col1:
                if st.button("Select All"):
                    for i, r in enumerate(results):
                        st.session_state[f"sel_{r.get('id', i)}"] = True
                    st.rerun()
            with sel_col2:
                if st.button("Clear selection"):
                    for i, r in enumerate(results):
                        st.session_state[f"sel_{r.get('id', i)}"] = False
                    st.rerun()
            with sel_col3:
                if st.button(
                    "Stage selected for Report", type="primary",
                    help="Save checked results to merge into a Report alongside fresh retrieval.",
                ):
                    selected = [results[i] for i, r in enumerate(results)
                                if st.session_state.get(f"sel_{r.get('id', i)}", False)]
                    if selected:
                        st.session_state.staged_chunks = selected
                        st.success(f"{len(selected)} chunks staged.")
                    else:
                        st.warning("Select at least one result first.")

            st.caption(f"{len(results)} results")
            st.divider()

            for i, result in enumerate(results):
                cb_key = f"sel_{result.get('id', i)}"
                _result_card(result, cb_key)


def page_chat(idx: IndexManager):
    st.header("Chat")

    # Single compact row — all options on one line
    cc1, cc2, cc3, cc4, cc5, cc6 = st.columns([2, 1, 1, 2, 1, 1])

    chat_mode_opts = ["top-docs", "semantic", "keyword", "both"]
    if EMBEDDING_AVAILABLE and idx.embed_matrix is not None:
        chat_mode_opts.insert(0, "embedding")
        if BM25_AVAILABLE:
            chat_mode_opts.insert(1, "hybrid")
    default_mode     = st.session_state.get("default_search_mode", "hybrid")
    chat_default_idx = chat_mode_opts.index(default_mode) if default_mode in chat_mode_opts else 0

    with cc1:
        chat_mode = st.selectbox("Mode", chat_mode_opts, index=chat_default_idx,
                                 key="chat_mode_sel")
    with cc2:
        chat_k = st.number_input("Chunks", min_value=3, max_value=100, value=12,
                                 key="chat_k_inp")
    with cc3:
        chat_boost = st.checkbox("Boost", value=True, key="chat_boost_cb")
    with cc4:
        chat_authors = ["All authors"] + all_known_authors()
        chat_author  = st.selectbox("Author", chat_authors, key="chat_author_sel")
    with cc5:
        st.write("")
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with cc6:
        st.write("")
        if st.button("?", key="chat_help_btn", use_container_width=True):
            st.session_state["show_chat_help"] = not st.session_state.get("show_chat_help", False)

    if st.session_state.get("show_chat_help", False):
        st.info(
            "**Mode** — how source chunks are retrieved "
            "(Hybrid = BM25 + Embedding, best overall).\n\n"
            "**Chunks** — number of source passages retrieved per question.\n\n"
            "**Boost** — apply authority weighting (primary sources rank 3× higher).\n\n"
            "**Author** — restrict retrieval to a specific author's works."
        )

    # Display history
    for msg_idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])} docs)"):
                    for src_idx, src in enumerate(msg["sources"]):
                        sc, vc = st.columns([5, 1])
                        with sc:
                            st.caption(f"**{src['doc']}** — {src['excerpt']}…")
                        with vc:
                            chunk = next((c for c in idx.chunks
                                          if c["doc_name"] == src["doc"]), None)
                            if chunk and st.button("↗",
                                                   key=f"chat_view_{msg_idx}_{src_idx}"):
                                st.session_state["_dialog_result"] = chunk
                                _show_section_dialog()

    # Input
    if prompt := st.chat_input("Ask about Bowen theory…"):
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve chunks
        try:
            if chat_mode == "top-docs":
                chunks = idx.top_docs_search(prompt, top_chunks=300,
                                             top_docs=chat_k, use_boost=chat_boost)
            elif chat_mode == "semantic":
                chunks = idx.semantic_search(prompt, chat_k, use_boost=chat_boost)
            elif chat_mode == "keyword":
                chunks = idx.keyword_search(prompt, chat_k, use_boost=chat_boost)
            elif chat_mode == "embedding":
                chunks = idx.embedding_search(prompt, chat_k, use_boost=chat_boost)
            elif chat_mode == "hybrid":
                chunks = idx.hybrid_search(prompt, chat_k, use_boost=chat_boost)
            else:
                chunks = idx.combined_search(prompt, chat_k, use_boost=chat_boost)
        except RuntimeError as e:
            st.error(str(e))
            return

        if chat_author != "All authors":
            chunks = [c for c in chunks if doc_author(c["doc_name"]) == chat_author]
            if not chunks:
                st.error(f"No chunks found for author: {chat_author}")
                return

        # Build context
        docs: dict = {}
        for c in chunks:
            cid  = c.get("id")
            txts = (idx.get_context_window(cid, window=1)
                    if cid is not None and hasattr(idx, "_doc_chunk_ids")
                    else [c["text"]])
            seen = set(docs.get(c["doc_name"], []))
            for t in txts:
                if t not in seen:
                    docs.setdefault(c["doc_name"], []).append(t)
                    seen.add(t)

        context = "\n\n---\n\n".join(
            f"### [{dn}]\n" + "\n…\n".join(txts) for dn, txts in docs.items()
        )
        current_content = (
            f"[Retrieved {len(chunks)} chunks from {len(docs)} documents]\n\n"
            f"{context}\n\n---\nQuestion: {prompt}"
        )
        messages_to_send = [{"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat_history] + [
            {"role": "user", "content": current_content}
        ]

        system = st.session_state.get("system_prompt", SYSTEM_PROMPT)

        with st.chat_message("assistant"):
            try:
                response = st.write_stream(_llm_stream(messages_to_send, system))
            except Exception as e:
                st.error(f"LLM error: {e}")
                return

            doc_names = sorted(set(c["doc_name"] for c in chunks))
            sources   = [{"doc": d,
                          "excerpt": re.sub(r'\s+', ' ', next(
                              (c["text"][:150] for c in chunks if c["doc_name"] == d), ""))
                          } for d in doc_names]
            # Key index this message will have once appended to history:
            # chat_history currently has N items; user appended first → N,
            # assistant appended second → N+1.
            cur_msg_idx = len(st.session_state.chat_history) + 1
            with st.expander(f"Sources used ({len(doc_names)} docs)"):
                for src_idx, src in enumerate(sources):
                    sc, vc = st.columns([5, 1])
                    with sc:
                        st.caption(f"**{src['doc']}** — {src['excerpt']}…")
                    with vc:
                        chunk = next((c for c in idx.chunks
                                      if c["doc_name"] == src["doc"]), None)
                        if chunk and st.button("↗",
                                               key=f"chat_view_{cur_msg_idx}_{src_idx}"):
                            st.session_state["_dialog_result"] = chunk
                            _show_section_dialog()

        # Bare Q&A stored in history (no chunks)
        st.session_state.chat_history.append({"role": "user",      "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response,
                                               "sources": sources})


def page_report(idx: IndexManager):
    st.header("Report Generator")

    query = st.text_area(
        "Topic / Question", height=80,
        value=st.session_state.get("last_search_query", ""),
        placeholder='e.g. "What does Bowen theory say about triangles?"',
        help="The question or topic the report will address. Pre-filled from your last search.",
    )

    # Single row: Retrieve | Mode | Target words | Chunks per source
    rc1, rc2, rc3, rc4 = st.columns([1, 2, 1, 1])
    with rc1:
        rpt_k = st.number_input(
            "Retrieve top", min_value=5, max_value=150, value=30,
            help=(
                "How many source chunks to retrieve and pass to the LLM. "
                "More = broader coverage but slower and costlier. "
                "30 is a good starting point; use 50+ for broad topics."
            ),
        )
        st.session_state.rpt_k = rpt_k
    with rc2:
        rpt_mode_opts = ["top-docs (recommended)", "semantic", "keyword", "both"]
        if EMBEDDING_AVAILABLE and idx.embed_matrix is not None:
            rpt_mode_opts.append("embedding")
            if BM25_AVAILABLE:
                rpt_mode_opts.append("hybrid")
        default_mode    = st.session_state.get("default_search_mode", "hybrid")
        rpt_mode_values = [o.split(" ")[0] for o in rpt_mode_opts]
        rpt_default_idx = rpt_mode_values.index(default_mode) if default_mode in rpt_mode_values else 0
        rpt_mode = st.selectbox(
            "Mode", rpt_mode_opts, index=rpt_default_idx,
            help=(
                "Search method used to retrieve source chunks.\n\n"
                "**Hybrid** — best overall coverage.\n\n"
                "**Top Docs** — fast; good for well-indexed topics.\n\n"
                "**Keyword** — good for names and specific terms."
            ),
        )
        st.session_state.rpt_mode = rpt_mode.split(" ")[0]
    with rc3:
        target_words = st.number_input(
            "Target words", min_value=500, max_value=10000, value=2000, step=500,
            help=(
                "Approximate minimum word count for the generated report. "
                "2000 is a solid summary; 4000+ for a deep dive."
            ),
        )
    with rc4:
        cpd = st.number_input(
            "Chunks per source", min_value=1, max_value=20, value=5,
            help=(
                "How many text chunks from each document are included as context "
                "(sliding window around the top chunk). "
                "Higher = more context per document but larger LLM input."
            ),
        )

    cb1, cb2 = st.columns(2)
    with cb1:
        rpt_boost = st.checkbox(
            "Authority boost", value=True,
            help=(
                "Apply authority weighting when retrieving chunks. "
                "Primary Bowen/Kerr sources 3×, FSJ articles 1.3×, other named theorists 1.15×."
            ),
        )
        st.session_state.rpt_boost = rpt_boost
    with cb2:
        include_appendix = st.checkbox(
            "Include sources as Appendix",
            help="Appends the full formatted text of every cited source after the report body.",
        )

    staged = st.session_state.get("staged_chunks", [])
    if staged:
        st.info(f"{len(staged)} chunks staged from Search will be merged with fresh retrieval.")
        if st.button("Clear staged"):
            st.session_state.staged_chunks = []
            st.rerun()

    col_gen, col_clear = st.columns([1, 4])
    with col_gen:
        generate = st.button("Generate Report", type="primary")

    if generate:
        if not query.strip():
            st.warning("Enter a topic or question.")
            return

        with st.spinner("Gathering sources…"):
            try:
                chunks = _gather_chunks(idx, query)
            except RuntimeError as e:
                st.error(str(e))
                return

        if not chunks:
            st.warning("No relevant sources found.")
            return

        # Build context
        docs: dict = {}
        window = max(0, (cpd - 1) // 2)
        for c in chunks:
            cid = c.get("id")
            expanded = (idx.get_context_window(cid, window=window)
                        if cid is not None and hasattr(idx, "_doc_chunk_ids")
                        else [c["text"]])
            existing = set(docs.get(c["doc_name"], []))
            for t in expanded:
                if t not in existing:
                    docs.setdefault(c["doc_name"], []).append(t)
                    existing.add(t)

        ref_map  = {name: i + 1 for i, name in enumerate(sorted(docs))}
        refs_md  = "\n".join(f"{num}. {name}"
                             for name, num in sorted(ref_map.items(), key=lambda x: x[1]))
        context_parts = [
            f"### [{ref_map[dn]}] {dn}\n" + "\n…\n".join(txts)
            for dn, txts in docs.items()
        ]
        context = "\n\n---\n\n".join(context_parts)
        st.session_state.last_rpt_context = context

        prompt = f"""Write a detailed report on the following topic using ONLY the source excerpts provided below.

**Topic / Question:** {query}

---

## SOURCE EXCERPTS ({len(docs)} documents)

{context}

---

## STRICT INSTRUCTIONS

- **Use only the excerpts above.** Do not add any information from outside these sources.
- **Do not infer, assume, or extrapolate.** If the sources do not explicitly address a point, write: "The provided sources do not address this point."
- **Every factual claim must be cited** immediately after the claim using the reference number in brackets, e.g. [1] or [3].
- Write at least {target_words} words. Develop each section fully using evidence from the excerpts.

## REPORT STRUCTURE

1. **Introduction & Definition**
2. **Theoretical Foundations**
3. **Key Dimensions**
4. **Relationship to Other Bowen Concepts**
5. **Clinical Presentation**
6. **Clinical Implications & Therapeutic Approach**
7. **Direct Quotations & Illustrations**
8. **Gaps & Limitations**
9. **References** — reproduce the numbered reference list verbatim

## References
{refs_md}
"""

        st.subheader("References")
        st.text(refs_md)
        st.divider()

        st.subheader("Report")
        system = st.session_state.get("system_prompt", SYSTEM_PROMPT)
        try:
            result = st.write_stream(_llm_stream(
                [{"role": "user", "content": prompt}], system))
            st.session_state.last_report = result
        except Exception as e:
            st.error(f"LLM error: {e}")
            return

        # Build appendix from source texts
        if include_appendix and docs:
            appendix_parts = ["\n\n---\n\n## Appendix: Source Texts\n"]
            for doc_name in sorted(docs, key=lambda d: ref_map[d]):
                appendix_parts.append(f"\n### [{ref_map[doc_name]}] {doc_name}\n")
                for txt in docs[doc_name]:
                    appendix_parts.append(_format_chunk_text(txt) + "\n")
                appendix_parts.append("\n---\n")
            st.session_state.last_rpt_appendix = "\n".join(appendix_parts)
        else:
            st.session_state.last_rpt_appendix = ""

    # Show chunks used (after generation)
    if st.session_state.get("last_rpt_context"):
        with st.expander("Audit: show chunks sent to LLM"):
            sections = re.split(r'\n\n---\n\n', st.session_state.last_rpt_context)
            for section in sections:
                lines     = section.strip().split('\n', 1)
                hdr       = lines[0].lstrip('#').strip()
                body      = re.sub(r'\n{3,}', '\n\n', lines[1].strip()) if len(lines) > 1 else ""
                st.markdown(f"**{hdr}**")
                st.text(body)
                st.divider()

    # Appendix (shown after audit, before download)
    if st.session_state.get("last_rpt_appendix"):
        with st.expander("Appendix: Source Texts"):
            st.markdown(st.session_state.last_rpt_appendix)

    # Download button — includes appendix when present
    if st.session_state.get("last_report"):
        full_download = st.session_state.last_report
        if st.session_state.get("last_rpt_appendix"):
            full_download += "\n\n" + st.session_state.last_rpt_appendix
        st.download_button(
            "Download report as .md",
            data=full_download,
            file_name="bowen_report.md",
            mime="text/markdown",
        )


def page_settings():
    st.header("Settings")

    # ── Search defaults ──────────────────────────────────────────────────────
    st.subheader("Search Defaults")

    all_modes = [
        ("Hybrid (BM25 + Embedding) — recommended", "hybrid"),
        ("Top Docs",                                 "top-docs"),
        ("Semantic (TF-IDF)",                        "semantic"),
        ("Keyword",                                  "keyword"),
        ("Both (Semantic + Keyword)",                "both"),
        ("Embedding",                                "embedding"),
    ]
    all_mode_labels = [m[0] for m in all_modes]
    all_mode_values = [m[1] for m in all_modes]
    cur_default     = st.session_state.get("default_search_mode", "hybrid")
    cur_idx         = all_mode_values.index(cur_default) if cur_default in all_mode_values else 0

    chosen = st.selectbox(
        "Default search mode",
        range(len(all_mode_labels)),
        format_func=lambda i: all_mode_labels[i],
        index=cur_idx,
        help="Pre-selected mode on the Search, Report, and Chat pages when you first load them. "
             "Hybrid and Embedding require the embedding index to be built.",
    )
    st.session_state.default_search_mode = all_mode_values[chosen]

    if all_mode_values[chosen] in ("hybrid", "embedding") and not (
        EMBEDDING_AVAILABLE and getattr(_get_index(), "embed_matrix", None) is not None
    ):
        st.warning("Hybrid and Embedding modes require the embedding index. "
                   "If it is not available the pages will fall back to Top Docs.")

    st.divider()
    st.subheader("LLM Provider")
    provider = st.radio("Provider", ["claude", "openai", "deepseek", "ollama"],
                        index=["claude", "openai", "deepseek", "ollama"].index(
                            st.session_state.get("provider", "claude")),
                        horizontal=True)
    st.session_state.provider = provider

    def _key_input(label: str, ss_key: str):
        existing = st.session_state.get(ss_key, "")
        if existing:
            st.caption(f"Key saved: ···{existing[-6:]}")
        new_val = st.text_input(
            label, value="", type="password",
            placeholder="Paste new key to update…" if existing else "Paste key here…",
        )
        if new_val:
            st.session_state[ss_key] = new_val

    if provider == "claude":
        st.subheader("Claude (Anthropic)")
        _key_input("API Key", "claude_key")
        model = st.selectbox("Model", CLAUDE_MODELS,
                             index=CLAUDE_MODELS.index(
                                 st.session_state.get("claude_model", "claude-sonnet-4-6"))
                             if st.session_state.get("claude_model") in CLAUDE_MODELS else 0)
        st.session_state.claude_model = model

    elif provider == "openai":
        st.subheader("OpenAI")
        _key_input("API Key", "openai_key")
        model = st.selectbox("Model", OPENAI_MODELS,
                             index=OPENAI_MODELS.index(
                                 st.session_state.get("openai_model", "gpt-4o"))
                             if st.session_state.get("openai_model") in OPENAI_MODELS else 0)
        st.session_state.openai_model = model

    elif provider == "deepseek":
        st.subheader("DeepSeek")
        _key_input("API Key", "deepseek_key")
        model = st.selectbox("Model", DEEPSEEK_MODELS,
                             index=DEEPSEEK_MODELS.index(
                                 st.session_state.get("deepseek_model", "deepseek-v4-flash"))
                             if st.session_state.get("deepseek_model") in DEEPSEEK_MODELS else 0)
        st.session_state.deepseek_model = model
        st.caption(f"Endpoint: {DEEPSEEK_BASE_URL}")

    else:
        st.subheader("Ollama (self-hosted)")
        st.info("Ollama must be running and accessible from the server. "
                "On Railway this requires a separately hosted Ollama instance.")
        url = st.text_input("Server URL", value=st.session_state.get("ollama_url",
                                                                       "http://localhost:11434"))
        st.session_state.ollama_url = url
        model = st.text_input("Model", value=st.session_state.get("ollama_model", "qwen2.5:7b"))
        st.session_state.ollama_model = model

    st.divider()
    st.subheader("System Prompt")
    sp = st.text_area("System Prompt", value=st.session_state.get("system_prompt", SYSTEM_PROMPT),
                      height=200)
    st.session_state.system_prompt = sp

    st.divider()
    if st.button("Test connection", type="primary"):
        with st.spinner("Testing…"):
            try:
                result = "".join(_llm_stream(
                    [{"role": "user", "content": "Reply with exactly: OK"}],
                    "You are a test assistant."
                ))
                st.success(f"Connected — response: {result[:80]}")
            except Exception as e:
                st.error(f"Connection failed: {e}")


def page_index(idx: IndexManager):
    st.header("Index")

    if idx.loaded:
        docs  = len(set(c["doc_name"] for c in idx.chunks))
        st.metric("Documents", docs)
        st.metric("Chunks", f"{len(idx.chunks):,}")
        st.metric("Embeddings", "loaded" if idx.embed_matrix is not None else "not available")
        st.metric("BM25", "loaded" if idx.bm25 is not None else "not available")
    else:
        st.error("Index not loaded.")

    st.divider()
    st.info(
        "**To update the index:** rebuild locally with the desktop app "
        "(Index tab → Rebuild Index), then commit and push the updated "
        "`rag-document-search/references/` files. Railway redeploys automatically."
    )

    with st.expander("Document list"):
        for doc in idx.list_documents():
            author = doc_author(doc)
            boost  = authority_boost(doc)
            badge  = f" ★ {boost}×" if boost > 1.0 else ""
            st.caption(f"**{doc}**  —  {author}{badge}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Bowen Theory RAG",
        layout="wide",
        initial_sidebar_state="expanded",
    )


    _init_session()
    _check_auth()

    idx = _get_index()

    _NAV = [
        ("Search",   "Find and browse source passages in the index. "
                     "Check results and stage them to carry into a Report."),
        ("Chat",     "Conversational Q&A with the literature. "
                     "Ask follow-up questions; each turn retrieves fresh source chunks."),
        ("Report",   "Generate a structured, cited research report on a topic. "
                     "Best for getting a comprehensive, referenced answer. "
                     "This is the core workflow for most users."),
        ("Index",    "Admin — view index statistics (document count, chunk count, embedding status). "
                     "Do not modify unless you are rebuilding the index."),
        ("Settings", "Admin — configure the LLM provider, API keys, default search mode, "
                     "and system prompt. Do not change unless you know what you are doing."),
    ]

    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Search"

    with st.sidebar:
        st.markdown("### Bowen Theory RAG")
        _provider = st.session_state.get("provider", "claude")
        _model = {
            "claude":   st.session_state.get("claude_model",   "claude-sonnet-4-6"),
            "openai":   st.session_state.get("openai_model",   "gpt-4o"),
            "deepseek": st.session_state.get("deepseek_model", "deepseek-v4-flash"),
            "ollama":   st.session_state.get("ollama_model",   "qwen2.5:7b"),
        }.get(_provider, _provider)
        if idx.loaded:
            docs  = len(set(c["doc_name"] for c in idx.chunks))
            embed = f" · {len(idx.embed_matrix):,} embeddings" if idx.embed_matrix is not None else ""
            st.markdown(
                f'<div style="font-size:12px;color:#4b5563;line-height:1.4;margin:0">'
                f'{docs} docs · {len(idx.chunks):,} chunks{embed}<br>'
                f'🤖 {_provider} · {_model}</div>',
                unsafe_allow_html=True,
            )
        st.divider()
        for _name, _desc in _NAV:
            _selected = st.session_state["nav_page"] == _name
            _nc, _hc = st.columns([5, 1])
            with _nc:
                if st.button(
                    ("▶ " if _selected else "") + _name,
                    use_container_width=True,
                    key=f"nav_{_name}",
                    type="primary" if _selected else "secondary",
                ):
                    st.session_state["nav_page"] = _name
                    st.session_state[f"show_help_{_name}"] = False
                    st.rerun()
            with _hc:
                if st.button("?", key=f"help_{_name}", use_container_width=True):
                    _tog = f"show_help_{_name}"
                    st.session_state[_tog] = not st.session_state.get(_tog, False)
                    st.rerun()
            if st.session_state.get(f"show_help_{_name}", False):
                st.info(_desc)
        st.divider()
        st.caption("Bowen Family Systems Theory research tool")

    page = st.session_state["nav_page"]

    if page == "Search":
        page_search(idx)
    elif page == "Chat":
        page_chat(idx)
    elif page == "Report":
        page_report(idx)
    elif page == "Index":
        page_index(idx)
    else:
        page_settings()


if __name__ == "__main__":
    main()
