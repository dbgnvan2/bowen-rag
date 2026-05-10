#!/usr/bin/env python3
"""
Build semantic search index from text and PDF documents.
Chunks documents and creates embeddings for RAG search.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyPDF2 not installed. PDF indexing will be skipped.")

class DocumentIndexer:
    def __init__(self, doc_dir: str, chunk_size: int = 1500, overlap: int = 200):
        """
        Initialize indexer.

        Args:
            doc_dir: Directory containing text documents
            chunk_size: Approximate characters per chunk
            overlap: Character overlap between chunks
        """
        self.doc_dir = doc_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []
        self.metadata = []
        self.vectorizer = None
        self.tfidf_matrix = None

    def load_documents(self) -> List[Tuple[str, str]]:
        """Load all text and PDF files from directory."""
        documents = []

        # Load .txt files
        txt_files = sorted(Path(self.doc_dir).glob("*.txt"))
        for filepath in txt_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append((filepath.stem, content))
            except UnicodeDecodeError:
                try:
                    with open(filepath, 'r', encoding='utf-16') as f:
                        content = f.read()
                    documents.append((filepath.stem, content))
                    print(f"  Note: read {filepath.name} as UTF-16")
                except Exception:
                    try:
                        with open(filepath, 'r', encoding='utf-16', errors='replace') as f:
                            content = f.read()
                        documents.append((filepath.stem, content))
                        print(f"  Note: read {filepath.name} as UTF-16 with replacement characters (file may be truncated)")
                    except Exception:
                        try:
                            with open(filepath, 'r', encoding='latin-1') as f:
                                content = f.read()
                            documents.append((filepath.stem, content))
                            print(f"  Note: read {filepath.name} as latin-1 fallback")
                        except Exception as e:
                            print(f"Error reading {filepath}: {e}")
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

        # Load .pdf files if PyPDF2 is available
        if PDF_SUPPORT:
            pdf_files = sorted(Path(self.doc_dir).glob("*.pdf"))
            for filepath in pdf_files:
                try:
                    text = self._extract_pdf_text(filepath)
                    if text.strip():
                        documents.append((filepath.stem, text))
                except Exception as e:
                    print(f"Error reading PDF {filepath}: {e}")

        return documents

    def _extract_pdf_text(self, filepath: Path) -> str:
        """Extract text from PDF file, inserting [PDF_PAGE:N] markers between pages."""
        parts = []
        try:
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        parts.append(f"[PDF_PAGE:{i + 1}]{page_text}")
        except Exception as e:
            print(f"  Warning: Could not fully extract text from {filepath.name}: {e}")

        return "\n".join(parts)

    def chunk_document(self, doc_name: str, content: str) -> List[Dict]:
        """
        Split document into chunks.
        If the document contains '## Section N –' headings (formatted transcripts),
        each section becomes its own chunk.  Otherwise fall back to overlapping
        word-count chunks.
        """
        if re.search(r'^## Section \d+', content, re.MULTILINE):
            return self._chunk_by_sections(doc_name, content)
        return self._chunk_by_wordcount(doc_name, content)

    def _chunk_by_sections(self, doc_name: str, content: str) -> List[Dict]:
        """One chunk per ## Section heading — preserves semantic boundaries."""
        parts = re.split(r'\n(## Section \d+ – [^\n]+)\n', content)
        chunks = []
        it = iter(parts[1:])   # skip content before first heading
        for heading, body in zip(it, it):
            m = re.match(r'## Section \d+ – (.+?)(?:\s*\(\[[\d:]+\]\))?\.?\s*$', heading)
            section_title = m.group(1).strip() if m else heading.strip()
            body_clean = body.strip()
            if not body_clean:
                continue
            full_text = f"[{section_title}]\n\n{body_clean}"
            chunks.append({
                "doc_name": doc_name,
                "section_title": section_title,
                "text": full_text,
                "char_count": len(full_text),
                "page": None,
            })
        return chunks

    _PDF_PAGE_RE = re.compile(r'\[PDF_PAGE:(\d+)\]')

    def _chunk_by_wordcount(self, doc_name: str, content: str) -> List[Dict]:
        """Split document into overlapping chunks at sentence boundaries.
        Tracks PDF page numbers from [PDF_PAGE:N] markers inserted by _extract_pdf_text.
        """
        # Split content into (page_num_or_None, segment_text) pairs
        segments: List[Tuple[Optional[int], str]] = []
        current_page: Optional[int] = None
        last_pos = 0
        for m in self._PDF_PAGE_RE.finditer(content):
            if m.start() > last_pos:
                segments.append((current_page, content[last_pos:m.start()]))
            current_page = int(m.group(1))
            last_pos = m.end()
        if last_pos < len(content):
            segments.append((current_page, content[last_pos:]))

        # Build a flat list of (sentence, page_num) pairs
        sentences_with_pages: List[Tuple[str, Optional[int]]] = []
        for page_num, seg_text in segments:
            for sent in re.split(r'(?<=[.!?])\s+', seg_text):
                if sent.strip():
                    sentences_with_pages.append((sent, page_num))

        # Produce overlapping chunks, recording the page of the first sentence
        chunks = []
        chunk_buf: List[Tuple[str, Optional[int]]] = []
        current_chars = 0

        for sentence, page_num in sentences_with_pages:
            projected = current_chars + len(sentence) + (1 if chunk_buf else 0)
            if projected > self.chunk_size and chunk_buf:
                chunk_text = " ".join(s for s, _ in chunk_buf).strip()
                chunks.append({
                    "doc_name": doc_name,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "page": chunk_buf[0][1],
                })
                # overlap: carry last N chars of sentences forward
                overlap_buf: List[Tuple[str, Optional[int]]] = []
                overlap_chars = 0
                for s, p in reversed(chunk_buf):
                    if overlap_chars + len(s) < self.overlap:
                        overlap_buf.insert(0, (s, p))
                        overlap_chars += len(s) + 1
                    else:
                        break
                chunk_buf = overlap_buf + [(sentence, page_num)]
                current_chars = sum(len(s) + 1 for s, _ in chunk_buf)
            else:
                chunk_buf.append((sentence, page_num))
                current_chars += len(sentence) + (1 if current_chars else 0)

        if chunk_buf:
            chunk_text = " ".join(s for s, _ in chunk_buf).strip()
            chunks.append({
                "doc_name": doc_name,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "page": chunk_buf[0][1],
            })

        return chunks

    def build_index(self) -> Dict:
        """Build complete search index."""
        print("Loading documents...")
        documents = self.load_documents()

        if not documents:
            raise ValueError(f"No .txt files found in {self.doc_dir}")

        print(f"Found {len(documents)} documents")

        # Chunk all documents
        print("Chunking documents...")
        for doc_name, content in documents:
            chunks = self.chunk_document(doc_name, content)
            self.chunks.extend(chunks)

        print(f"Created {len(self.chunks)} chunks")

        # Build TF-IDF vectors for similarity
        print("Building TF-IDF index...")
        chunk_texts = [c["text"] for c in self.chunks]

        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)

        # Build per-document chunk position map: chunk_index → (1-based pos, total in doc)
        doc_seq: Dict[str, List[int]] = {}
        for i, chunk in enumerate(self.chunks):
            doc_seq.setdefault(chunk["doc_name"], []).append(i)
        pos_map: Dict[int, Tuple[int, int]] = {}
        for ids in doc_seq.values():
            total = len(ids)
            for pos, idx in enumerate(ids):
                pos_map[idx] = (pos + 1, total)

        # Create metadata without embedding vectors (they're sparse and large)
        print("Saving metadata...")
        self.metadata = [
            {
                "id": i,
                "doc_name": chunk["doc_name"],
                "section_title": chunk.get("section_title", ""),
                "text": chunk["text"],
                "char_count": chunk["char_count"],
                "page": chunk.get("page"),
                "chunk_pos": pos_map[i][0],
                "doc_chunk_count": pos_map[i][1],
                "preview": chunk["text"][:150] + "..."
            }
            for i, chunk in enumerate(self.chunks)
        ]

        return {
            "num_documents": len(documents),
            "num_chunks": len(self.chunks),
            "vectorizer_features": len(self.vectorizer.get_feature_names_out())
        }

    def save_index(self, output_dir: str):
        """Save index to disk."""
        os.makedirs(output_dir, exist_ok=True)

        # Save metadata
        metadata_path = os.path.join(output_dir, "chunk_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Save vectorizer metadata (don't serialize full params, just feature names)
        vectorizer_path = os.path.join(output_dir, "vectorizer.json")
        vectorizer_data = {
            "feature_names": self.vectorizer.get_feature_names_out().tolist(),
            "max_features": 500,
            "ngram_range": [1, 2]
        }
        with open(vectorizer_path, 'w') as f:
            json.dump(vectorizer_data, f, indent=2)

        # Save matrix in sparse format (much smaller than dense)
        matrix_path = os.path.join(output_dir, "tfidf_matrix.npz")
        sp_sparse.save_npz(matrix_path, self.tfidf_matrix)

        print(f"Index saved to {output_dir}")


if __name__ == "__main__":
    import sys

    doc_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    indexer = DocumentIndexer(doc_dir)
    stats = indexer.build_index()
    print(f"Index stats: {stats}")
    indexer.save_index(output_dir)
