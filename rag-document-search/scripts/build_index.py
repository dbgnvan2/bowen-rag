#!/usr/bin/env python3
"""
Build semantic search index from text and PDF documents.
Chunks documents and creates embeddings for RAG search.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
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
        """Extract text from PDF file."""
        text = []
        try:
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
        except Exception as e:
            print(f"  Warning: Could not fully extract text from {filepath.name}: {e}")

        return "\n".join(text)

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
        # Split on section headings, keeping the heading text
        parts = re.split(r'\n(## Section \d+ – [^\n]+)\n', content)
        chunks = []
        it = iter(parts[1:])   # skip content before first heading
        for heading, body in zip(it, it):
            m = re.match(r'## Section \d+ – (.+?)(?:\s*\(\[[\d:]+\]\))?\.?\s*$', heading)
            section_title = m.group(1).strip() if m else heading.strip()
            body_clean = body.strip()
            if not body_clean:
                continue
            # Prepend section title so TF-IDF indexes it as a search term
            full_text = f"[{section_title}]\n\n{body_clean}"
            chunks.append({
                "doc_name": doc_name,
                "section_title": section_title,
                "text": full_text,
                "char_count": len(full_text),
            })
        return chunks

    def _chunk_by_wordcount(self, doc_name: str, content: str) -> List[Dict]:
        """
        Split document into overlapping chunks at sentence boundaries.
        """
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', content)

        current_chunk = ""
        chunk_sentences = []

        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(test_chunk) > self.chunk_size and chunk_sentences:
                # Save current chunk
                chunk_text = " ".join(chunk_sentences)
                chunks.append({
                    "doc_name": doc_name,
                    "text": chunk_text.strip(),
                    "char_count": len(chunk_text)
                })

                # Start new chunk with overlap
                overlap_sentences = []
                overlap_chars = 0
                for sent in reversed(chunk_sentences):
                    if overlap_chars + len(sent) < self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_chars += len(sent) + 1
                    else:
                        break

                chunk_sentences = overlap_sentences + [sentence]
                current_chunk = " ".join(chunk_sentences)
            else:
                chunk_sentences.append(sentence)
                current_chunk = test_chunk

        # Add final chunk
        if chunk_sentences:
            chunks.append({
                "doc_name": doc_name,
                "text": " ".join(chunk_sentences).strip(),
                "char_count": len(" ".join(chunk_sentences))
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

        # Create metadata without embedding vectors (they're sparse and large)
        print("Saving metadata...")
        self.metadata = [
            {
                "id": i,
                "doc_name": chunk["doc_name"],
                "section_title": chunk.get("section_title", ""),
                "text": chunk["text"],
                "char_count": chunk["char_count"],
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
