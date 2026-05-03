#!/usr/bin/env python3
"""
Bowen Theory RAG  —  Document Search & Analysis GUI
Search, index management, and LLM-powered report generation.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import importlib.util
import json
import threading
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──────────────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle (.app)
    _BUNDLE_DIR = Path(sys._MEIPASS)
    BASE_DIR    = Path.home() / "Documents" / "BowenRAG"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
else:
    _BUNDLE_DIR = Path(__file__).parent
    BASE_DIR    = Path(__file__).parent

SOURCE_DIR     = BASE_DIR / "source_files"
REFS_DIR       = _BUNDLE_DIR / "rag-document-search" / "references"
BUILD_PY       = _BUNDLE_DIR / "rag-document-search" / "scripts" / "build_index.py"
TRANSCRIPTS_PY = _BUNDLE_DIR / "process_transcripts.py"
OUT_DIR        = BASE_DIR / "outputs"
USER_REFS_DIR  = BASE_DIR / "references"   # writable; rebuild target in frozen mode
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

CLAUDE_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
]

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "o1",
    "o1-mini",
]

OLLAMA_MODELS = [
    "qwen3.5:latest",
    "qwen3.5:4b",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "llama3.2:latest",
    "mistral:latest",
]

# ── Authority tiers ────────────────────────────────────────────────────────────
# Patterns matched against doc_name (case-insensitive prefix/substring).
# Multiplier applied to raw TF-IDF score before ranking.
AUTHORITY_TIERS = [
    # Tier 1 — Primary Bowen & Kerr sources (3.0×)
    ("Family Therapy in_Clinical_Practice_Chapter",  3.0),  # FTCP book chapters
    ("Bowen Basic Series Tape",                       3.0),  # Bowen lecture tapes
    ("BOWEN-KERR INTERVIEW SERIES",                   3.0),  # Bowen-Kerr interviews
    ("Bowen Family Systems Theory",                   3.0),  # Bowen theory docs
    ("Bowen on Triangles",                            3.0),
    ("Bowen Theory and Therapy",                      3.0),
    ("Chronic Anxiety and Defining",                  3.0),  # Kerr Atlantic article
    ("Cancer and the Emotional System",               3.0),  # Kerr
    ("Family and Society Kerr",                       3.0),
    ("Family as a System Kerr",                       3.0),
    ("Family Systems and Therapy Kerr",               3.0),
    ("Physical Illness as the Family Emotional",      3.0),  # Kerr
    ("Psychotherapy Past Present Future",             3.0),  # Bowen
    # Tier 2 — Family Systems Journal & Family Center Reports (1.3×)
    ("Copy of ",                                      1.3),  # FSJ article PDFs
    ("Family Center Reports",                         1.3),
    # Tier 3 — Other named theorist papers (1.15×)
    ("Papero",                                        1.15),
    ("Friedman",                                      1.15),
    ("Fogarty",                                       1.15),
    ("Guerin",                                        1.15),
    ("Toman",                                         1.15),
    # Everything else stays at 1.0×
]

def authority_boost(doc_name: str) -> float:
    dn = doc_name.lower()
    for pattern, mult in AUTHORITY_TIERS:
        if pattern.lower() in dn:
            return mult
    return 1.0

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


# ══════════════════════════════════════════════════════════════════════════════
# Tooltip helper
# ══════════════════════════════════════════════════════════════════════════════

class Tooltip:
    """Small popup that appears when a widget is clicked."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text   = text
        self.win    = None
        widget.bind("<Button-1>", self._show)

    def _show(self, event=None):
        if self.win:
            self._hide()
            return

        x = self.widget.winfo_rootx() + 24
        y = self.widget.winfo_rooty() + 24

        self.win = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)

        frame = tk.Frame(tw, bg="#1e293b", bd=1, relief="solid")
        frame.pack()

        tk.Label(
            frame, text=self.text,
            bg="#1e293b", fg="white",
            font=("Helvetica", 11),
            wraplength=340, justify="left",
            padx=12, pady=10
        ).pack()

        # Close on click anywhere or after 8 s
        tw.bind("<Button-1>", lambda e: self._hide())
        self.widget.after(8000, self._hide)

    def _hide(self):
        if self.win:
            self.win.destroy()
            self.win = None


def help_btn(parent, text: str, bg: str) -> tk.Label:
    """Return a small clickable ⓘ label that shows a Tooltip."""
    lbl = tk.Label(parent, text=" ⓘ ", fg="#2563eb", bg=bg,
                   font=("Helvetica", 12, "bold"), cursor="hand2")
    Tooltip(lbl, text)
    return lbl


# ══════════════════════════════════════════════════════════════════════════════
# Index logic
# ══════════════════════════════════════════════════════════════════════════════

class IndexManager:
    def __init__(self):
        self.chunks: list  = []
        self.matrix        = None   # np.ndarray (N, features)
        self.vectorizer    = None   # TfidfVectorizer, fitted
        self.loaded        = False

    def load(self, refs_dir: Path = REFS_DIR) -> dict:
        meta_path   = refs_dir / "chunk_metadata.json"
        matrix_npz  = refs_dir / "tfidf_matrix.npz"
        matrix_npy  = refs_dir / "tfidf_matrix.npy"

        if not meta_path.exists():
            raise FileNotFoundError(f"Index not found at {refs_dir}. Rebuild first.")

        with open(meta_path) as f:
            self.chunks = json.load(f)

        if matrix_npz.exists():
            self.matrix = sp_sparse.load_npz(str(matrix_npz)).toarray()
        else:
            self.matrix = np.load(str(matrix_npy))

        # Re-fit vectorizer on stored texts (preserves vocab / IDF ordering)
        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            max_features=8000, stop_words="english",
            lowercase=True, ngram_range=(1, 2),
            min_df=2, sublinear_tf=True
        )
        self.vectorizer.fit(texts)
        self.loaded = True

        # Build per-document ordered chunk index for context-window expansion
        self._doc_chunk_ids: dict = {}
        for i, c in enumerate(self.chunks):
            self._doc_chunk_ids.setdefault(c["doc_name"], []).append(i)

        docs = len(set(c["doc_name"] for c in self.chunks))
        return {"chunks": len(self.chunks), "documents": docs}

    def get_context_window(self, chunk_id: int, window: int = 2) -> list:
        """Return ordered texts of chunks within ±window of chunk_id in the same doc."""
        doc_name = self.chunks[chunk_id]["doc_name"]
        doc_ids  = self._doc_chunk_ids.get(doc_name, [])
        try:
            pos = doc_ids.index(chunk_id)
        except ValueError:
            return [self.chunks[chunk_id]["text"]]
        start = max(0, pos - window)
        end   = min(len(doc_ids), pos + window + 1)
        return [self.chunks[doc_ids[j]]["text"] for j in range(start, end)]

    # ── Search ────────────────────────────────────────────────────────────────

    def semantic_search(self, query: str, top_k: int) -> list:
        if not self.loaded:
            return []
        qvec   = self.vectorizer.transform([query])
        raw    = cosine_similarity(qvec, self.matrix)[0]
        # Apply authority boost then re-rank
        boosted = np.array([
            raw[i] * authority_boost(self.chunks[i]["doc_name"])
            for i in range(len(self.chunks))
        ])
        idx = boosted.argsort()[::-1][:top_k]
        return [
            {**self.chunks[i],
             "score":       float(boosted[i]),
             "score_label": f"{boosted[i]*100:.0f}% ★" if authority_boost(self.chunks[i]["doc_name"]) > 1.0 else f"{boosted[i]*100:.0f}%",
             "mode":        "semantic"}
            for i in idx if boosted[i] > 0
        ]

    # Common English stop words + corpus-specific noise words
    _STOP = frozenset({
        # English stop words
        "the","and","for","are","but","not","you","all","can","had","her","was",
        "one","our","out","day","get","has","him","his","how","man","new","now",
        "old","see","two","way","who","boy","did","its","let","put","say","she",
        "too","use","what","with","this","that","have","from","they","will","been",
        "more","when","than","them","were","said","each","which","about","there",
        "their","would","make","like","into","time","look","just","come","could",
        "also","some","then","these","many","well","only","over","such","after",
        "most","very","even","back","any","good","know","same","tell","does",
        # Corpus-specific: appear in nearly every document so carry no signal
        "bowen","kerr","theory","family","therapy","systems","system",
        "murray","michael","dr","said","think","know","people","things",
    })

    @staticmethod
    def _stems(word: str) -> list:
        """Return the word plus simple stem variants (strips s, es, ing, ed)."""
        variants = [word]
        if word.endswith("ing") and len(word) > 5:
            variants.append(word[:-3])          # running → run
        if word.endswith("ed") and len(word) > 4:
            variants.append(word[:-2])           # talked → talk
            variants.append(word[:-1])           # talked → talke (covers 'e' drop)
        if word.endswith("ies") and len(word) > 4:
            variants.append(word[:-3] + "y")     # families → family
        if word.endswith("es") and len(word) > 4:
            variants.append(word[:-2])            # processes → process
        if word.endswith("s") and len(word) > 3:
            variants.append(word[:-1])            # networks → network
        return variants

    def keyword_search(self, query: str, top_k: int) -> list:
        if not self.loaded:
            return []

        # Extract meaningful terms: strip stop words, length > 2
        raw_terms = [t.lower() for t in query.split()
                     if len(t) > 2 and t.lower() not in self._STOP]
        if not raw_terms:
            return []

        # Expand each term with stem variants
        term_sets = [set(self._stems(t)) for t in raw_terms]

        # Score: per document, count hits for any variant, then pick best chunk
        doc_best: dict = {}
        for c in self.chunks:
            tl   = c["text"].lower()
            hits = sum(
                max(tl.count(variant) for variant in variants)
                for variants in term_sets
            )
            if hits == 0:
                continue
            dn      = c["doc_name"]
            boost   = authority_boost(dn)
            score   = hits * boost
            label   = f"{hits} hits" + (" ★" if boost > 1.0 else "")
            if dn not in doc_best or score > doc_best[dn]["score"]:
                doc_best[dn] = {**c, "score": score,
                                "score_label": label, "mode": "keyword"}

        out = sorted(doc_best.values(), key=lambda x: x["score"], reverse=True)
        return out[:top_k]

    def combined_search(self, query: str, top_k: int) -> list:
        sem = {r["id"]: r for r in self.semantic_search(query, top_k)}
        kw  = {r["id"]: r for r in self.keyword_search(query, top_k)}
        merged = {**kw, **sem}   # semantic wins on overlap
        out = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return out[:top_k]

    def top_docs_search(self, query: str, top_chunks: int = 300,
                        top_docs: int = 30) -> list:
        """
        Score documents by summing their top-3 chunk scores (× authority boost),
        then return the best single chunk from each document as the representative.
        Aggregating across chunks means a book chapter with 8 relevant passages
        beats a journal article with 1 concentrated paragraph.
        """
        if not self.loaded:
            return []

        qvec = self.vectorizer.transform([query])
        raw  = cosine_similarity(qvec, self.matrix)[0]

        # Group all above-zero chunks by document
        doc_chunks: dict = {}
        for i, score in enumerate(raw):
            if score <= 0:
                continue
            dn = self.chunks[i]["doc_name"]
            doc_chunks.setdefault(dn, []).append((score, i))

        # For each document: aggregate score = sum of top-3 raw chunk scores × boost
        doc_scores: dict = {}
        for dn, pairs in doc_chunks.items():
            pairs.sort(reverse=True)
            top3_sum  = sum(s for s, _ in pairs[:3])
            agg_score = top3_sum * authority_boost(dn)
            best_idx  = pairs[0][1]   # best single chunk for display
            boost     = authority_boost(dn)
            label     = f"{agg_score*100:.0f}%"
            if boost > 1.0:
                label += " ★"
            doc_scores[dn] = {
                **self.chunks[best_idx],
                "score":       agg_score,
                "score_label": label,
                "mode":        "semantic",
            }

        out = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return out[:top_docs]

    # ── Utility ───────────────────────────────────────────────────────────────

    def list_documents(self) -> list:
        seen, docs = set(), []
        for c in self.chunks:
            if c["doc_name"] not in seen:
                seen.add(c["doc_name"])
                docs.append(c["doc_name"])
        return sorted(docs)


# ══════════════════════════════════════════════════════════════════════════════
# LLM client
# ══════════════════════════════════════════════════════════════════════════════

class LLMClient:

    @staticmethod
    def call_claude(prompt: str, api_key: str, model: str,
                    system: str, on_chunk=None) -> str:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("Run: pip install anthropic")

        client = anthropic.Anthropic(api_key=api_key)
        msgs   = [{"role": "user", "content": prompt}]

        if on_chunk:
            buf = []
            with client.messages.stream(
                model=model, max_tokens=16000, system=system, messages=msgs
            ) as stream:
                for token in stream.text_stream:
                    buf.append(token)
                    on_chunk(token)
            return "".join(buf)

        r = client.messages.create(
            model=model, max_tokens=16000, system=system, messages=msgs)
        return r.content[0].text

    @staticmethod
    def call_ollama(prompt: str, url: str, model: str,
                    system: str, on_chunk=None) -> str:
        try:
            import requests as req
        except ImportError:
            raise RuntimeError("Run: pip install requests")

        payload = {"model": model, "prompt": prompt,
                   "system": system, "stream": bool(on_chunk)}
        r = req.post(f"{url.rstrip('/')}/api/generate",
                     json=payload, stream=bool(on_chunk), timeout=300)
        r.raise_for_status()

        if on_chunk:
            buf = []
            for line in r.iter_lines():
                if line:
                    d = json.loads(line)
                    t = d.get("response", "")
                    buf.append(t)
                    on_chunk(t)
                    if d.get("done"):
                        break
            return "".join(buf)

        return r.json().get("response", "")

    @staticmethod
    def call_openai(prompt: str, api_key: str, model: str,
                    system: str, on_chunk=None) -> str:
        try:
            import openai
        except ImportError:
            raise RuntimeError("Run: pip install openai")

        client = openai.OpenAI(api_key=api_key)
        msgs   = [{"role": "system", "content": system},
                  {"role": "user",   "content": prompt}]

        if on_chunk:
            buf = []
            with client.chat.completions.create(
                model=model, max_tokens=16000, messages=msgs, stream=True
            ) as stream:
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        buf.append(token)
                        on_chunk(token)
            return "".join(buf)

        r = client.chat.completions.create(
            model=model, max_tokens=16000, messages=msgs)
        return r.choices[0].message.content

    @staticmethod
    def list_ollama_models(url: str) -> list:
        try:
            import requests as req
            r = req.get(f"{url.rstrip('/')}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []


# ══════════════════════════════════════════════════════════════════════════════
# GUI App
# ══════════════════════════════════════════════════════════════════════════════

class App:
    PAD = 8

    def __init__(self, root: tk.Tk):
        self.root  = root
        self.index = IndexManager()
        self.llm   = LLMClient()

        # Mutable state
        self.search_results: list  = []
        self.checked_vars:   list  = []   # tk.BooleanVar per result row
        self._staged_chunks: list  = []   # chunks sent from Search → Report
        self._last_report:   str   = ""

        # Settings
        self.provider    = tk.StringVar(value="openai")
        self.claude_key  = tk.StringVar()
        self.claude_mdl  = tk.StringVar(value="claude-opus-4-6")
        self.openai_key  = tk.StringVar()
        self.openai_mdl  = tk.StringVar(value="gpt-4o")
        self.ollama_url  = tk.StringVar(value="http://localhost:11434")
        self.ollama_mdl  = tk.StringVar(value="qwen3.5:latest")
        self.top_k       = tk.IntVar(value=15)
        self.srch_mode   = tk.StringVar(value="top-docs")
        self.index_stats = tk.StringVar(value="Loading index…")
        self._status     = tk.StringVar(value="Initializing…")

        root.title("Bowen Theory RAG  ·  Search & Analysis")
        root.geometry("1340x900")
        root.minsize(960, 660)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        self._apply_theme()
        self._build_header()
        self._build_notebook()
        self._build_statusbar()

        # Load index on a background thread
        threading.Thread(target=self._bg_load_index, daemon=True).start()

    # ── Theme ──────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        s = ttk.Style(self.root)
        s.theme_use("clam")

        BG   = "#f4f6fa"
        FG   = "#1a1a2e"
        ACC  = "#2563eb"
        HDR  = "#e8ecf8"

        self._bg  = BG
        self._acc = ACC

        s.configure("TFrame",           background=BG)
        s.configure("TLabel",           background=BG, foreground=FG, font=("Helvetica", 12))
        s.configure("TNotebook",        background=BG, borderwidth=0)
        s.configure("TNotebook.Tab",    font=("Helvetica", 12, "bold"), padding=[14, 7])
        s.configure("TButton",          font=("Helvetica", 12), padding=6)
        s.configure("TCheckbutton",     background=BG, foreground=FG, font=("Helvetica", 11))
        s.configure("TRadiobutton",     background=BG, foreground=FG, font=("Helvetica", 11))
        s.configure("TEntry",           font=("Helvetica", 12))
        s.configure("TCombobox",        font=("Helvetica", 12))
        s.configure("TSpinbox",         font=("Helvetica", 12))
        s.configure("TLabelframe",      background=BG, foreground=FG)
        s.configure("TLabelframe.Label",font=("Helvetica", 12, "bold"), foreground=ACC)
        s.configure("TSeparator",       background="#d1d5db")

        s.configure("Accent.TButton",
                    foreground="white", background=ACC,
                    font=("Helvetica", 12, "bold"))
        s.map("Accent.TButton",
              background=[("active", "#1d4ed8"), ("pressed", "#1e40af")])

        s.configure("Danger.TButton",
                    foreground="white", background="#dc2626",
                    font=("Helvetica", 12, "bold"))
        s.map("Danger.TButton",
              background=[("active", "#b91c1c")])

        s.configure("Header.TLabel",
                    font=("Helvetica", 20, "bold"), foreground="#1a1a2e",
                    background=HDR)
        s.configure("Sub.TLabel",
                    font=("Helvetica", 11), foreground="#6b7280",
                    background=HDR)
        s.configure("Stat.TLabel",
                    font=("Helvetica", 12, "bold"), foreground=ACC,
                    background=HDR)
        s.configure("Info.TLabel",
                    font=("Helvetica", 11), foreground=ACC, background=BG)

        self.root.configure(bg=BG)
        self._hdr_bg = HDR

    # ── Header ─────────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=self._hdr_bg, pady=10)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.columnconfigure(0, weight=1)

        ttk.Label(hdr, text="Bowen Theory RAG",
                  style="Header.TLabel").grid(row=0, column=0, sticky="w", padx=16)
        ttk.Label(hdr,
                  text="Document Search  ·  Index Manager  ·  LLM Analysis  ·  Reports",
                  style="Sub.TLabel").grid(row=1, column=0, sticky="w", padx=16)
        ttk.Label(hdr, textvariable=self.index_stats,
                  style="Stat.TLabel").grid(row=0, column=1, sticky="e", padx=16)

    # ── Notebook ────────────────────────────────────────────────────────────────

    def _build_notebook(self):
        self._nb = ttk.Notebook(self.root)
        self._nb.grid(row=1, column=0, sticky="nsew", padx=6, pady=(4, 0))

        self._tab_search()
        self._tab_index()
        self._tab_llm()
        self._tab_report()

        self._nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, event):
        """When switching to Report tab, copy the current search query."""
        if self._nb.index("current") == 3:   # Report tab
            query = self._srch_q.get("1.0", "end-1c").strip()
            if query:
                self._rpt_q.delete("1.0", "end")
                self._rpt_q.insert("1.0", query)

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        sep = ttk.Separator(self.root)
        sep.grid(row=2, column=0, sticky="ew")

        ttk.Label(self.root, textvariable=self._status,
                  font=("Helvetica", 10), foreground="#6b7280",
                  background=self._bg, anchor="w").grid(
            row=3, column=0, sticky="ew", padx=14, pady=3)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1  —  Search
    # ══════════════════════════════════════════════════════════════════════════

    def _tab_search(self):
        f = ttk.Frame(self._nb)
        self._nb.add(f, text="  Search  ")
        f.columnconfigure(1, weight=1)
        f.rowconfigure(0, weight=1)

        # ── Left controls ────────────────────────────────────────────────────
        lf = ttk.LabelFrame(f, text="Query", padding=10)
        lf.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        ttk.Label(lf, text="Query (Ctrl+Enter to search):").pack(anchor="w")
        self._srch_q = tk.Text(lf, width=30, height=5, font=("Helvetica", 12),
                                wrap="word", relief="solid", borderwidth=1)
        self._srch_q.pack(fill="x", pady=(2, 10))
        self._srch_q.bind("<Control-Return>", lambda e: self._do_search())

        mode_row = tk.Frame(lf, bg=self._bg)
        mode_row.pack(fill="x")
        ttk.Label(mode_row, text="Mode:").pack(side="left")
        help_btn(mode_row,
            "Search modes:\n\n"
            "• Top Docs (recommended) — scores and ranks entire documents by "
            "aggregating their best matching passages. Primary sources (FTCP book "
            "chapters, Bowen/Kerr tapes) are boosted. One result per document. "
            "Best for 'what documents cover X' queries.\n\n"
            "• Semantic (TF-IDF) — finds passages that are conceptually similar "
            "to your query using word-frequency analysis. Good for broad concepts. "
            "Multiple passages from the same document can appear.\n\n"
            "• Keyword — searches for the exact words in your query (plus simple "
            "variants: singular/plural, -ing, -ed). Common English words and "
            "corpus-wide words like 'bowen', 'family', 'theory' are ignored as "
            "they appear everywhere. Use this for specific terms, names, or "
            "phrases not well captured by TF-IDF. One result per document.\n\n"
            "• Both — merges Semantic and Keyword results.",
            self._bg).pack(side="left", padx=2)

        for label, val in [("Top Docs  (1 result per document)", "top-docs"),
                            ("Semantic (TF-IDF)",                 "semantic"),
                            ("Keyword",                           "keyword"),
                            ("Both",                              "both")]:
            ttk.Radiobutton(lf, text=label, variable=self.srch_mode,
                            value=val).pack(anchor="w")

        ttk.Separator(lf).pack(fill="x", pady=8)
        tk.Frame(lf, bg=self._bg).pack()
        row = ttk.Frame(lf)
        row.pack(fill="x")
        ttk.Label(row, text="Results:").pack(side="left")
        ttk.Spinbox(row, from_=1, to=200, textvariable=self.top_k,
                    width=5).pack(side="left", padx=4)

        ttk.Button(lf, text="Search", style="Accent.TButton",
                   command=self._do_search).pack(fill="x", pady=(12, 4))

        ttk.Separator(lf).pack(fill="x", pady=6)
        ttk.Label(lf, text="Selection:").pack(anchor="w")
        ttk.Button(lf, text="Select All",
                   command=self._sel_all).pack(fill="x", pady=2)
        ttk.Button(lf, text="Clear",
                   command=self._sel_clear).pack(fill="x", pady=2)

        self._sel_lbl = ttk.Label(lf, text="0 selected", style="Info.TLabel")
        self._sel_lbl.pack(anchor="w", pady=4)

        ttk.Button(lf, text="→ Send to Report",
                   style="Accent.TButton",
                   command=self._send_to_report).pack(fill="x", pady=(6, 2))

        # ── Right results + preview ──────────────────────────────────────────
        rf = ttk.Frame(f)
        rf.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        rf.columnconfigure(0, weight=1)
        rf.rowconfigure(0, weight=3)
        rf.rowconfigure(1, weight=1)

        # Scrollable results list
        res_lf = ttk.LabelFrame(rf, text="Results", padding=4)
        res_lf.grid(row=0, column=0, sticky="nsew")
        res_lf.columnconfigure(0, weight=1)
        res_lf.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(res_lf, bg="#fafafa", highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(res_lf, orient="vertical",
                             command=self._canvas.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self._canvas.configure(yscrollcommand=vsb.set)

        self._inner = tk.Frame(self._canvas, bg="#fafafa")
        self._win   = self._canvas.create_window((0, 0), window=self._inner,
                                                   anchor="nw")
        self._inner.bind("<Configure>",
                         lambda e: self._canvas.configure(
                             scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>",
                          lambda e: self._canvas.itemconfig(
                              self._win, width=e.width))
        # Mouse wheel
        self._canvas.bind("<MouseWheel>",
                          lambda e: self._canvas.yview_scroll(
                              -1*(e.delta//120), "units"))
        self._canvas.bind("<Button-4>",
                          lambda e: self._canvas.yview_scroll(-1, "units"))
        self._canvas.bind("<Button-5>",
                          lambda e: self._canvas.yview_scroll(1, "units"))

        # Preview pane
        prev_lf = ttk.LabelFrame(rf, text="Preview", padding=4)
        prev_lf.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        prev_lf.columnconfigure(0, weight=1)
        prev_lf.rowconfigure(0, weight=1)

        self._preview = scrolledtext.ScrolledText(
            prev_lf, height=8, wrap="word",
            font=("Helvetica", 11), state="disabled",
            relief="flat", bg="#f8f9fa")
        self._preview.grid(row=0, column=0, sticky="nsew")

    def _do_search(self):
        query = self._srch_q.get("1.0", "end-1c").strip()
        if not query:
            return
        if not self.index.loaded:
            self._set_status("Index not loaded — rebuild it in the Index tab.")
            return

        self._set_status("Searching…")
        mode = self.srch_mode.get()
        k    = self.top_k.get()

        if mode == "top-docs":
            results = self.index.top_docs_search(query, top_chunks=300, top_docs=k)
        elif mode == "semantic":
            results = self.index.semantic_search(query, k)
        elif mode == "keyword":
            results = self.index.keyword_search(query, k)
        else:
            results = self.index.combined_search(query, k)

        self.search_results = results
        self._render_results(results)
        self._set_status(f"{len(results)} results for: \"{query}\"")

    def _render_results(self, results: list):
        for w in self._inner.winfo_children():
            w.destroy()
        self.checked_vars = []

        if not results:
            ttk.Label(self._inner, text="No results found.",
                      foreground="#9ca3af").pack(padx=10, pady=10)
            self._update_sel_lbl()
            return

        for res in results:
            var = tk.BooleanVar()
            self.checked_vars.append(var)

            card = tk.Frame(self._inner, bg="white", relief="flat",
                            highlightthickness=1,
                            highlightbackground="#e5e7eb")
            card.pack(fill="x", padx=6, pady=3)

            # Top row: checkbox, score badge, doc name
            top = tk.Frame(card, bg="white")
            top.pack(fill="x", padx=8, pady=(6, 2))

            tk.Checkbutton(top, variable=var, bg="white",
                           command=self._update_sel_lbl).pack(side="left")

            score_val = res.get("score", 0)
            if isinstance(score_val, float):          # semantic
                badge_col = ("#16a34a" if score_val > 0.4 else
                             "#ca8a04" if score_val > 0.15 else "#6b7280")
            else:                                     # keyword hits
                badge_col = "#7c3aed"

            tk.Label(top, text=res["score_label"], fg="white",
                     bg=badge_col, font=("Helvetica", 9, "bold"),
                     padx=5, pady=2).pack(side="left", padx=(0, 6))

            mode_badge = res.get("mode", "")
            if mode_badge:
                tk.Label(top, text=mode_badge, fg="#6b7280",
                         font=("Helvetica", 9), bg="white").pack(side="left", padx=2)

            tk.Label(top, text=res["doc_name"],
                     fg="#1d4ed8", font=("Helvetica", 11, "bold"), bg="white",
                     cursor="hand2", anchor="w", justify="left",
                     wraplength=700).pack(side="left", padx=6, fill="x")

            # Excerpt
            excerpt = res["text"][:220].replace("\n", " ") + "…"
            body = tk.Label(card, text=excerpt, fg="#374151",
                            font=("Helvetica", 10), bg="white",
                            wraplength=760, justify="left", anchor="w")
            body.pack(fill="x", padx=14, pady=(0, 8))

            # Click anywhere on card to preview
            for w in (card, top, body):
                w.bind("<Button-1>", lambda e, r=res: self._show_preview(r))

        self._update_sel_lbl()

    def _show_preview(self, res: dict):
        self._preview.config(state="normal")
        self._preview.delete("1.0", "end")
        header = f"Document: {res['doc_name']}\n{'─'*60}\n\n"
        self._preview.insert("1.0", header + res["text"])
        self._preview.config(state="disabled")

    def _sel_all(self):
        for v in self.checked_vars: v.set(True)
        self._update_sel_lbl()

    def _sel_clear(self):
        for v in self.checked_vars: v.set(False)
        self._update_sel_lbl()

    def _update_sel_lbl(self):
        n = sum(1 for v in self.checked_vars if v.get())
        self._sel_lbl.config(text=f"{n} selected")

    def _send_to_report(self):
        chunks = [self.search_results[i]
                  for i, v in enumerate(self.checked_vars) if v.get()]
        if not chunks:
            messagebox.showinfo("No Selection",
                                "Check some results first, then click Send.")
            return
        self._staged_chunks = chunks
        self._nb.select(3)   # jump to Report tab
        self._rpt_staged_lbl.config(
            text=f"{len(chunks)} chunks pre-loaded from Search ({len(set(c['doc_name'] for c in chunks))} docs)")
        self._set_status(f"Sent {len(chunks)} chunks to Report tab.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2  —  Index Manager
    # ══════════════════════════════════════════════════════════════════════════

    def _tab_index(self):
        f = ttk.Frame(self._nb)
        self._nb.add(f, text="  Index  ")
        f.columnconfigure(0, weight=1)
        f.rowconfigure(2, weight=1)

        # Controls
        cf = ttk.LabelFrame(f, text="Index Management", padding=12)
        cf.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        cf.columnconfigure(1, weight=1)

        ttk.Label(cf, text="Source directory:").grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        self._src_var = tk.StringVar(value=str(SOURCE_DIR))
        ttk.Entry(cf, textvariable=self._src_var).grid(
            row=0, column=1, sticky="ew")
        ttk.Button(cf, text="Browse…",
                   command=lambda: self._browse_dir(self._src_var)).grid(
            row=0, column=2, padx=(4, 0))

        ttk.Label(cf, text="Index directory:").grid(
            row=1, column=0, sticky="w", pady=(6, 0), padx=(0, 8))
        self._idx_var = tk.StringVar(value=str(USER_REFS_DIR if getattr(sys, 'frozen', False) else REFS_DIR))
        ttk.Entry(cf, textvariable=self._idx_var).grid(
            row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(cf, text="Browse…",
                   command=lambda: self._browse_dir(self._idx_var)).grid(
            row=1, column=2, padx=(4, 0), pady=(6, 0))

        ttk.Label(cf, text="Transcripts directory:").grid(
            row=2, column=0, sticky="w", pady=(6, 0), padx=(0, 8))
        self._trans_var = tk.StringVar(
            value=str(Path.home() / "transcripts" / "projects"))
        ttk.Entry(cf, textvariable=self._trans_var).grid(
            row=2, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(cf, text="Browse…",
                   command=lambda: self._browse_dir(self._trans_var)).grid(
            row=2, column=2, padx=(4, 0), pady=(6, 0))

        btn_row = tk.Frame(cf, bg=self._bg)
        btn_row.grid(row=3, column=0, columnspan=3, pady=(12, 4))
        ttk.Button(btn_row, text="Reload Index",
                   command=lambda: threading.Thread(
                       target=self._bg_load_index, daemon=True).start()
                   ).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Import Transcripts",
                   command=self._import_transcripts).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Import + Rebuild",
                   style="Accent.TButton",
                   command=self._import_and_rebuild).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Rebuild Index  (all .txt + .pdf)",
                   command=self._rebuild_index).pack(side="left", padx=4)

        self._idx_stat_lbl = ttk.Label(cf, text="", style="Info.TLabel")
        self._idx_stat_lbl.grid(row=4, column=0, columnspan=3, sticky="w")

        # Progress bar
        self._progress = ttk.Progressbar(f, mode="indeterminate")
        self._progress.grid(row=1, column=0, sticky="ew", padx=8)

        # Build log
        lf = ttk.LabelFrame(f, text="Build Log", padding=6)
        lf.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        lf.columnconfigure(0, weight=1)
        lf.rowconfigure(0, weight=1)

        self._build_log = scrolledtext.ScrolledText(
            lf, height=18, font=("Courier", 11),
            state="disabled", bg="#1e1e2e", fg="#a6e3a1",
            insertbackground="white", relief="flat")
        self._build_log.grid(row=0, column=0, sticky="nsew")

    def _browse_dir(self, var: tk.StringVar):
        d = filedialog.askdirectory(initialdir=var.get())
        if d:
            var.set(d)

    def _bg_load_index(self):
        try:
            refs = Path(self._idx_var.get()) if hasattr(self, "_idx_var") else REFS_DIR
            # Fall back to bundled index if the user refs dir has no index yet
            if not (refs / "chunk_metadata.json").exists() and refs != REFS_DIR:
                refs = REFS_DIR
            stats = self.index.load(refs)
            self.root.after(0, self._on_index_loaded, stats)
        except Exception as e:
            self.root.after(0, self._on_index_error, str(e))

    def _on_index_loaded(self, stats: dict):
        msg = f"✓  {stats['documents']} documents  ·  {stats['chunks']:,} chunks"
        self.index_stats.set(msg)
        if hasattr(self, "_idx_stat_lbl"):
            self._idx_stat_lbl.config(text=msg)
        self._set_status("Index loaded.")
        self._log(f"[{_ts()}] Index loaded: {msg}\n")

    def _on_index_error(self, err: str):
        self.index_stats.set(f"⚠  {err}")
        self._set_status(f"Index error: {err}")
        if hasattr(self, "_idx_stat_lbl"):
            self._idx_stat_lbl.config(text=f"⚠  {err}")
        self._log(f"[{_ts()}] ERROR: {err}\n")

    def _rebuild_index(self):
        src = self._src_var.get()
        out = self._idx_var.get()
        if not Path(src).is_dir():
            messagebox.showerror("Error", f"Source directory not found:\n{src}")
            return

        self._log(f"\n[{_ts()}] Starting rebuild…\n  Source: {src}\n  Output: {out}\n\n")
        self._progress.start(10)
        self._set_status("Rebuilding index…")

        def _work():
            import builtins
            _orig_print = builtins.print
            try:
                def _gui_print(*args, **kwargs):
                    self.root.after(0, self._log, " ".join(str(a) for a in args) + "\n")
                builtins.print = _gui_print

                spec = importlib.util.spec_from_file_location("build_index", str(BUILD_PY))
                bi   = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bi)

                indexer = bi.DocumentIndexer(src)
                indexer.build_index()
                indexer.save_index(out)
                self.root.after(0, self._on_rebuild_done)
            except Exception as e:
                self.root.after(0, self._log, f"\n[EXCEPTION] {e}\n")
                self.root.after(0, self._progress.stop)
                self.root.after(0, self._set_status, "Rebuild failed.")
            finally:
                builtins.print = _orig_print

        threading.Thread(target=_work, daemon=True).start()

    def _import_transcripts(self, then_rebuild=False):
        """Run process_transcripts.py to copy formatted transcripts into source_files/."""
        trans_dir = self._trans_var.get()
        src_dir   = self._src_var.get()

        if not Path(trans_dir).is_dir():
            messagebox.showerror("Error", f"Transcripts directory not found:\n{trans_dir}")
            return
        if not TRANSCRIPTS_PY.exists():
            messagebox.showerror("Error", f"process_transcripts.py not found at:\n{TRANSCRIPTS_PY}")
            return

        self._log(f"\n[{_ts()}] Importing transcripts…\n"
                  f"  From: {trans_dir}\n  To:   {src_dir}\n\n")
        self._progress.start(10)
        self._set_status("Importing transcripts…")

        def _work():
            import builtins
            _orig_print = builtins.print
            try:
                def _gui_print(*args, **kwargs):
                    self.root.after(0, self._log, " ".join(str(a) for a in args) + "\n")
                builtins.print = _gui_print

                spec = importlib.util.spec_from_file_location("process_transcripts", str(TRANSCRIPTS_PY))
                pt   = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(pt)

                old_argv = sys.argv
                sys.argv = [
                    "process_transcripts.py",
                    "--transcripts-dir", trans_dir,
                    "--source-dir", src_dir,
                ]
                try:
                    pt.main()
                finally:
                    sys.argv = old_argv

                if then_rebuild:
                    self.root.after(0, self._log, f"\n[{_ts()}] Import done. Starting rebuild…\n\n")
                    self.root.after(0, self._rebuild_index)
                else:
                    self.root.after(0, self._progress.stop)
                    self.root.after(0, self._set_status, "Import complete.")
                    self.root.after(0, self._log, f"\n[{_ts()}] Import complete.\n")
            except SystemExit:
                pass  # argparse calls sys.exit(0) on --help; treat as success
            except Exception as e:
                self.root.after(0, self._log, f"\n[EXCEPTION] {e}\n")
                self.root.after(0, self._progress.stop)
                self.root.after(0, self._set_status, "Import failed.")
            finally:
                builtins.print = _orig_print

        threading.Thread(target=_work, daemon=True).start()

    def _import_and_rebuild(self):
        """Import transcripts then immediately rebuild the full index."""
        self._import_transcripts(then_rebuild=True)

    def _on_rebuild_done(self):
        self._progress.stop()
        self._log(f"\n[{_ts()}] Rebuild complete. Reloading index…\n")
        threading.Thread(target=self._bg_load_index, daemon=True).start()

    def _log(self, text: str):
        if not hasattr(self, "_build_log"):
            return
        self._build_log.config(state="normal")
        self._build_log.insert("end", text)
        self._build_log.see("end")
        self._build_log.config(state="disabled")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3  —  LLM Settings
    # ══════════════════════════════════════════════════════════════════════════

    def _tab_llm(self):
        f = ttk.Frame(self._nb)
        self._nb.add(f, text="  LLM Settings  ")
        f.columnconfigure(0, weight=1)

        # Provider
        pf = ttk.LabelFrame(f, text="Provider", padding=12)
        pf.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        ttk.Radiobutton(pf, text="OpenAI  (GPT-4o / o1)",
                        variable=self.provider, value="openai",
                        command=self._toggle_provider).pack(anchor="w", pady=2)
        ttk.Radiobutton(pf, text="Claude  (Anthropic API)",
                        variable=self.provider, value="claude",
                        command=self._toggle_provider).pack(anchor="w", pady=2)
        ttk.Radiobutton(pf, text="Ollama  (local · Qwen / Llama / Mistral / …)",
                        variable=self.provider, value="ollama",
                        command=self._toggle_provider).pack(anchor="w", pady=2)

        # OpenAI
        self._openai_pane = ttk.LabelFrame(f, text="OpenAI Settings", padding=12)
        self._openai_pane.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        self._openai_pane.columnconfigure(1, weight=1)

        ttk.Label(self._openai_pane, text="API Key:").grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        self._oai_key_entry = ttk.Entry(self._openai_pane, textvariable=self.openai_key,
                                         show="•", width=55)
        self._oai_key_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(self._openai_pane, text="Show/Hide",
                   command=lambda: self._oai_key_entry.config(
                       show="" if self._oai_key_entry["show"] == "•" else "•")
                   ).grid(row=0, column=2, padx=4)

        ttk.Label(self._openai_pane, text="Model:").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(self._openai_pane, textvariable=self.openai_mdl,
                     values=OPENAI_MODELS, width=25).grid(
            row=1, column=1, sticky="w", pady=(8, 0))

        # Claude
        self._claude_pane = ttk.LabelFrame(f, text="Claude Settings", padding=12)
        self._claude_pane.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        self._claude_pane.columnconfigure(1, weight=1)

        ttk.Label(self._claude_pane, text="API Key:").grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        self._key_entry = ttk.Entry(self._claude_pane, textvariable=self.claude_key,
                                     show="•", width=55)
        self._key_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(self._claude_pane, text="Show/Hide",
                   command=self._toggle_key).grid(row=0, column=2, padx=4)

        ttk.Label(self._claude_pane, text="Model:").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(self._claude_pane, textvariable=self.claude_mdl,
                     values=CLAUDE_MODELS, width=35).grid(
            row=1, column=1, sticky="w", pady=(8, 0))

        # Ollama
        self._ollama_pane = ttk.LabelFrame(f, text="Ollama Settings", padding=12)
        self._ollama_pane.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        self._ollama_pane.columnconfigure(1, weight=1)

        ttk.Label(self._ollama_pane, text="Server URL:").grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(self._ollama_pane, textvariable=self.ollama_url,
                  width=40).grid(row=0, column=1, sticky="w")

        ttk.Label(self._ollama_pane, text="Model:").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        self._ollama_combo = ttk.Combobox(
            self._ollama_pane, textvariable=self.ollama_mdl,
            values=OLLAMA_MODELS, width=32)
        self._ollama_combo.grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Button(self._ollama_pane, text="Fetch Available",
                   command=self._fetch_ollama).grid(
            row=1, column=2, padx=4, pady=(8, 0))

        # System prompt
        sp = ttk.LabelFrame(f, text="System Prompt (used for all LLM calls)", padding=10)
        sp.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        sp.columnconfigure(0, weight=1)

        self._sys_prompt = tk.Text(sp, height=5, font=("Helvetica", 11),
                                    wrap="word", relief="solid", borderwidth=1)
        self._sys_prompt.pack(fill="x")
        self._sys_prompt.insert("1.0", SYSTEM_PROMPT)

        # Test
        tf = ttk.Frame(f)
        tf.grid(row=4, column=0, sticky="w", padx=8, pady=(12, 0))
        ttk.Button(tf, text="Test Connection",
                   command=self._test_llm).pack(side="left")
        self._test_lbl = ttk.Label(tf, text="", width=70)
        self._test_lbl.pack(side="left", padx=10)

        self._toggle_provider()

    def _toggle_provider(self):
        p = self.provider.get()
        self._openai_pane.grid() if p == "openai" else self._openai_pane.grid_remove()
        self._claude_pane.grid() if p == "claude" else self._claude_pane.grid_remove()
        self._ollama_pane.grid() if p == "ollama" else self._ollama_pane.grid_remove()

    def _toggle_key(self):
        self._key_entry.config(
            show="" if self._key_entry["show"] == "•" else "•")

    def _fetch_ollama(self):
        models = self.llm.list_ollama_models(self.ollama_url.get())
        if models:
            self._ollama_combo["values"] = models
            self._test_lbl.config(text=f"✓ Found {len(models)} models.",
                                   foreground="green")
        else:
            self._test_lbl.config(text="No models found. Is Ollama running?",
                                   foreground="red")

    def _test_llm(self):
        self._test_lbl.config(text="Testing…", foreground="#6b7280")
        self.root.update_idletasks()

        def _work():
            try:
                sys_p = self._get_system_prompt()
                p = self.provider.get()
                if p == "openai":
                    r = self.llm.call_openai(
                        "Reply with exactly: OK",
                        self.openai_key.get(), self.openai_mdl.get(), sys_p)
                elif p == "claude":
                    r = self.llm.call_claude(
                        "Reply with exactly: OK",
                        self.claude_key.get(), self.claude_mdl.get(), sys_p)
                else:
                    r = self.llm.call_ollama(
                        "Reply with exactly: OK",
                        self.ollama_url.get(), self.ollama_mdl.get(), sys_p)
                self.root.after(0, self._test_lbl.config,
                                {"text": f"✓  {r[:80]}", "foreground": "green"})
            except Exception as e:
                self.root.after(0, self._test_lbl.config,
                                {"text": f"✗  {e}", "foreground": "red"})

        threading.Thread(target=_work, daemon=True).start()

    def _get_system_prompt(self) -> str:
        if hasattr(self, "_sys_prompt"):
            return self._sys_prompt.get("1.0", "end-1c").strip()
        return SYSTEM_PROMPT

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4  —  Report Generator
    # ══════════════════════════════════════════════════════════════════════════

    def _tab_report(self):
        f = ttk.Frame(self._nb)
        self._nb.add(f, text="  Report Generator  ")
        f.columnconfigure(0, weight=1)
        f.rowconfigure(2, weight=1)

        # Query controls
        qf = ttk.LabelFrame(f, text="Report Query", padding=12)
        qf.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        qf.columnconfigure(1, weight=1)

        ttk.Label(qf, text="Topic / Question:").grid(
            row=0, column=0, sticky="nw", padx=(0, 10))
        self._rpt_q = tk.Text(qf, height=3, font=("Helvetica", 12),
                               wrap="word", relief="solid", borderwidth=1)
        self._rpt_q.grid(row=0, column=1, columnspan=4, sticky="ew")
        ex = ('e.g. "Provide a list of documents on nodal events and generate a '
              'report on this concept"')
        ttk.Label(qf, text=ex, foreground="#9ca3af",
                  font=("Helvetica", 10)).grid(
            row=1, column=1, columnspan=4, sticky="w", pady=(2, 6))

        # Row 2 — Retrieve top + mode
        r2 = tk.Frame(qf, bg=self._bg)
        r2.grid(row=2, column=0, columnspan=5, sticky="w", pady=4)

        ttk.Label(r2, text="Retrieve top:").pack(side="left")
        self._rpt_k = tk.IntVar(value=30)
        ttk.Spinbox(r2, from_=5, to=150, textvariable=self._rpt_k,
                    width=6).pack(side="left", padx=(4, 0))
        help_btn(r2,
            "How many source documents to pull into the report.\n\n"
            "Higher = more breadth and more source material for the LLM to draw on, "
            "but the prompt grows larger and generation takes longer.\n\n"
            "Recommended: 20–40 for most topics. Use 50+ for broad surveys.",
            self._bg).pack(side="left", padx=(2, 16))

        ttk.Label(r2, text="Mode:").pack(side="left")
        self._rpt_mode = tk.StringVar(value="top-docs")
        ttk.Combobox(r2, textvariable=self._rpt_mode,
                     values=["top-docs (recommended)", "semantic", "keyword", "both"],
                     width=22).pack(side="left", padx=(4, 0))
        help_btn(r2,
            "How documents are retrieved and ranked:\n\n"
            "• top-docs (recommended) — fetches 300 chunks, scores each document by "
            "aggregating its best chunks, then picks the top N documents. "
            "Primary sources (FTCP, tapes) are boosted. Best for most queries.\n\n"
            "• semantic — TF-IDF cosine similarity. Good for conceptual queries.\n\n"
            "• keyword — counts exact word matches. Good for proper names or "
            "specific terms not captured by TF-IDF.\n\n"
            "• both — merges semantic and keyword results.",
            self._bg).pack(side="left", padx=2)

        # Row 3 — Target length + chunks per source
        r3 = tk.Frame(qf, bg=self._bg)
        r3.grid(row=3, column=0, columnspan=5, sticky="w", pady=4)

        ttk.Label(r3, text="Target length:").pack(side="left")
        self._rpt_words = tk.IntVar(value=2000)
        ttk.Spinbox(r3, from_=500, to=10000, increment=500,
                    textvariable=self._rpt_words, width=7).pack(side="left", padx=(4, 0))
        ttk.Label(r3, text=" words").pack(side="left")
        help_btn(r3,
            "The minimum number of words the LLM is instructed to write.\n\n"
            "• 1000–1500 — concise overview\n"
            "• 2000–3000 — standard scholarly report (recommended)\n"
            "• 4000–5000 — in-depth treatment with extensive quotation\n"
            "• 6000+ — comprehensive literature review\n\n"
            "Note: actual output depends on how much relevant source material exists. "
            "If the sources are thin on a topic the model will flag gaps rather than pad.",
            self._bg).pack(side="left", padx=(2, 20))

        ttk.Label(r3, text="Chunks per source:").pack(side="left")
        self._rpt_chunks_per_doc = tk.IntVar(value=5)
        ttk.Spinbox(r3, from_=1, to=20, textvariable=self._rpt_chunks_per_doc,
                    width=4).pack(side="left", padx=(4, 0))
        help_btn(r3,
            "Controls the context window around each retrieved passage.\n\n"
            "Each chunk is ~1500 characters (~250 words) of the original text.\n\n"
            "The value sets the total span: the best-matching chunk plus "
            "neighbouring chunks before and after it from the same document, "
            "so you get the full surrounding argument — not just the sentence "
            "that matched the query.\n\n"
            "• 1 — matched chunk only\n"
            "• 3 — matched chunk ± 1 neighbour (~750 words per source)\n"
            "• 5 — matched chunk ± 2 neighbours (~1250 words per source)\n"
            "• 7 — matched chunk ± 3 neighbours (~1750 words per source)\n\n"
            "Recommended: 5–7 for rich reports. Higher values increase prompt "
            "size and may slow generation on local (Ollama) models.",
            self._bg).pack(side="left", padx=2)

        self._rpt_staged_lbl = ttk.Label(qf, text="", style="Info.TLabel")
        self._rpt_staged_lbl.grid(row=4, column=1, columnspan=4, sticky="w",
                                   pady=(4, 0))

        btn_row = tk.Frame(qf, bg=self._bg)
        btn_row.grid(row=5, column=0, columnspan=5, pady=(10, 0))
        ttk.Button(btn_row, text="Generate Report",
                   style="Accent.TButton",
                   command=self._generate_report).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Clear staged chunks",
                   command=self._clear_staged).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Save as .md",
                   command=self._save_report).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Copy to Clipboard",
                   command=self._copy_report).pack(side="left", padx=4)

        # Reference list
        rf = ttk.LabelFrame(f, text="Reference List  (documents used)", padding=6)
        rf.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 0))
        rf.columnconfigure(0, weight=1)

        self._ref_box = scrolledtext.ScrolledText(
            rf, height=5, font=("Helvetica", 11), wrap="word",
            state="disabled", relief="flat", bg="#f8f9fa")
        self._ref_box.grid(row=0, column=0, sticky="ew")

        # Report output
        out_f = ttk.LabelFrame(f, text="Report Output  (Markdown)", padding=6)
        out_f.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        out_f.columnconfigure(0, weight=1)
        out_f.rowconfigure(0, weight=1)

        self._rpt_out = scrolledtext.ScrolledText(
            out_f, font=("Courier", 12), wrap="word",
            relief="flat", bg="#f8f9fa")
        self._rpt_out.grid(row=0, column=0, sticky="nsew")

    def _clear_staged(self):
        self._staged_chunks = []
        self._rpt_staged_lbl.config(text="")

    def _gather_chunks(self, query: str) -> list:
        """Use staged chunks if available, else search fresh."""
        if self._staged_chunks:
            return self._staged_chunks

        if not self.index.loaded:
            raise RuntimeError("Index not loaded.")

        mode = self._rpt_mode.get()
        k    = self._rpt_k.get()
        if "top-docs" in mode:
            return self.index.top_docs_search(query, top_chunks=300, top_docs=k)
        if mode == "semantic":
            return self.index.semantic_search(query, k)
        if mode == "keyword":
            return self.index.keyword_search(query, k)
        return self.index.combined_search(query, k)

    def _generate_report(self):
        query = self._rpt_q.get("1.0", "end-1c").strip()
        if not query:
            messagebox.showinfo("Empty Query", "Enter a topic or question.")
            return

        # Gather chunks
        try:
            chunks = self._gather_chunks(query)
        except RuntimeError as e:
            messagebox.showerror("Error", str(e))
            return

        if not chunks:
            messagebox.showinfo("No Results", "No relevant chunks found.")
            return

        # Build reference list (grouped by doc)
        docs: dict[str, list] = {}
        cpd = self._rpt_chunks_per_doc.get()
        window = max(0, (cpd - 1) // 2)   # e.g. cpd=6 → window=2 → 5 chunks

        for c in chunks:
            chunk_id = c.get("id")
            if chunk_id is not None and hasattr(self.index, "_doc_chunk_ids"):
                expanded = self.index.get_context_window(chunk_id, window=window)
            else:
                expanded = [c["text"]]
            # Deduplicate: only add texts not already in the list
            existing = set(docs.get(c["doc_name"], []))
            for t in expanded:
                if t not in existing:
                    docs.setdefault(c["doc_name"], []).append(t)
                    existing.add(t)

        # Show reference list
        refs_numbered = "\n".join(
            f"{i+1}. {name}" for i, name in enumerate(sorted(docs)))
        self._ref_box.config(state="normal")
        self._ref_box.delete("1.0", "end")
        self._ref_box.insert("1.0", refs_numbered)
        self._ref_box.config(state="disabled")

        # Build context — each doc's texts are already the expanded window
        context_parts = []
        for doc_name, texts in docs.items():
            combined = "\n…\n".join(texts)
            context_parts.append(f"### [{doc_name}]\n{combined}")
        context = "\n\n---\n\n".join(context_parts)

        refs_md   = "\n".join(f"- {n}" for n in sorted(docs))
        target_wc = self._rpt_words.get()

        prompt = f"""Write a detailed report on the following topic using ONLY the source excerpts provided below.

**Topic / Question:** {query}

---

## SOURCE EXCERPTS ({len(docs)} documents)

{context}

---

## STRICT INSTRUCTIONS

- **Use only the excerpts above.** Do not add any information from outside these sources.
- **Do not infer, assume, or extrapolate.** If the sources do not explicitly address a point, write: "The provided sources do not address this point."
- **Every factual claim must be cited** immediately after the claim using [Document Name].
- **Do not paraphrase without attribution.** If you summarise a source's position, cite it.
- If sources disagree or use different language for the same idea, quote both and note the difference — do not resolve it yourself.
- Write at least {target_wc} words. Develop each section fully using evidence from the excerpts.

## REPORT STRUCTURE

Produce a well-structured Markdown report with these sections, each developed in depth:

1. **Introduction & Definition** — what do the sources say this concept is?
2. **Theoretical Foundations** — how do the sources describe its origins and place in Bowen theory?
3. **Key Dimensions** — what distinct aspects or components do the sources identify?
4. **Relationship to Other Bowen Concepts** — what connections do the sources explicitly draw?
5. **Clinical Presentation** — how do the sources describe this appearing in families or individuals?
6. **Clinical Implications & Therapeutic Approach** — what do the sources say about working with this clinically?
7. **Direct Quotations & Illustrations** — include key verbatim or near-verbatim passages from the sources
8. **Gaps & Limitations** — what does this topic lack coverage on in the provided sources?
9. **References** — numbered list of every source document used

Use Markdown headings (##, ###), bullet lists where appropriate, and **bold** for key terms from the sources.

## References
{refs_md}
"""

        # Clear output area
        self._rpt_out.delete("1.0", "end")
        self._rpt_out.insert("1.0",
            f"# Generating report…\n_Topic: {query}_\n_Sources: {len(docs)} documents_\n\n")
        self._set_status("Generating report…")
        self._last_report = ""

        def _on_token(t: str):
            self.root.after(0, self._append_rpt, t)

        def _work():
            try:
                sys_p = self._get_system_prompt()
                p = self.provider.get()
                if p == "openai":
                    if not self.openai_key.get():
                        raise RuntimeError("OpenAI API key not set — go to LLM Settings.")
                    result = self.llm.call_openai(
                        prompt, self.openai_key.get(),
                        self.openai_mdl.get(), sys_p, _on_token)
                elif p == "claude":
                    if not self.claude_key.get():
                        raise RuntimeError("Claude API key not set — go to LLM Settings.")
                    result = self.llm.call_claude(
                        prompt, self.claude_key.get(),
                        self.claude_mdl.get(), sys_p, _on_token)
                else:
                    result = self.llm.call_ollama(
                        prompt, self.ollama_url.get(),
                        self.ollama_mdl.get(), sys_p, _on_token)
                self._last_report = result
                self.root.after(0, self._set_status, "Report complete.")
            except Exception as e:
                self.root.after(0, self._append_rpt, f"\n\n**[ERROR]** {e}\n")
                self.root.after(0, self._set_status, f"Error: {e}")

        threading.Thread(target=_work, daemon=True).start()

    def _append_rpt(self, text: str):
        self._rpt_out.insert("end", text)
        self._rpt_out.see("end")

    def _save_report(self):
        content = self._rpt_out.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo("Nothing to Save", "Generate a report first.")
            return

        topic = (self._rpt_q.get("1.0", "end-1c").strip()[:50]
                 .replace(" ", "_").replace("/", "-").replace(":", ""))
        default_name = f"report_{topic}_{datetime.now():%Y%m%d_%H%M}.md"

        path = filedialog.asksaveasfilename(
            initialdir=str(OUT_DIR),
            initialfile=default_name,
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All", "*")])

        if not path:
            return

        refs = self._ref_box.get("1.0", "end").strip()
        full  = f"---\ntopic: {topic}\ngenerated: {datetime.now():%Y-%m-%d %H:%M}\n---\n\n"
        if refs:
            full += f"## Source Documents\n\n{refs}\n\n---\n\n"
        full += content

        Path(path).write_text(full, encoding="utf-8")
        self._set_status(f"Saved: {path}")
        messagebox.showinfo("Saved", f"Report saved to:\n{path}")

    def _copy_report(self):
        content = self._rpt_out.get("1.0", "end").strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self._set_status("Report copied to clipboard.")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._status.set(msg)

    def _get_system_prompt(self) -> str:
        if hasattr(self, "_sys_prompt"):
            return self._sys_prompt.get("1.0", "end-1c").strip()
        return SYSTEM_PROMPT


# ──────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main():
    root = tk.Tk()
    try:
        # macOS: use the Aqua icon approach to suppress default Python icon
        root.tk.call("wm", "iconphoto", root._w, tk.PhotoImage(file=""))
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
