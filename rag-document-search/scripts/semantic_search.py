#!/usr/bin/env python3
"""
Semantic search against indexed documents.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearcher:
    def __init__(self, index_dir: str):
        """Load prebuilt index from disk."""
        self.index_dir = index_dir
        self.metadata = []
        self.tfidf_matrix = None
        self.vectorizer = None
        self._load_index()

    def _load_index(self):
        """Load metadata and index from disk."""
        metadata_path = os.path.join(self.index_dir, "chunk_metadata.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Index not found at {self.index_dir}. "
                "Run build_index.py first."
            )

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load TF-IDF matrix
        matrix_path = os.path.join(self.index_dir, "tfidf_matrix.npy")
        self.tfidf_matrix = np.load(matrix_path)

        # Reconstruct vectorizer from saved params
        vectorizer_path = os.path.join(self.index_dir, "vectorizer.json")
        with open(vectorizer_path, 'r') as f:
            vectorizer_data = json.load(f)

        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )

        # Fit on feature names to recreate the same feature space
        # This is a workaround since we need the same vocabulary
        chunk_texts = [m["text"] for m in self.metadata]
        self.vectorizer.fit(chunk_texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant chunks.

        Args:
            query: Natural language search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with scores
        """
        if not query.strip():
            return []

        # Vectorize query
        query_vector = self.vectorizer.transform([query])

        # Compute similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include above-zero scores
                results.append({
                    "id": self.metadata[idx]["id"],
                    "doc_name": self.metadata[idx]["doc_name"],
                    "text": self.metadata[idx]["text"],
                    "preview": self.metadata[idx]["preview"],
                    "relevance_score": float(similarities[idx]),
                    "relevance_percent": int(similarities[idx] * 100)
                })

        return results

    def get_document_summary(self, doc_name: str) -> Dict:
        """Get metadata about a specific document."""
        chunks_in_doc = [m for m in self.metadata if m["doc_name"] == doc_name]
        return {
            "doc_name": doc_name,
            "num_chunks": len(chunks_in_doc),
            "total_chars": sum(m["char_count"] for m in chunks_in_doc)
        }


def format_results(results: List[Dict], query: str = "") -> str:
    """Format search results for display."""
    if not results:
        return "No results found."

    output = []
    if query:
        output.append(f"Search results for: \"{query}\"\n")

    for i, result in enumerate(results, 1):
        output.append(f"Result {i}: {result['doc_name']}")
        output.append(f"Relevance: {result['relevance_percent']}%")
        output.append(f"\n{result['text']}\n")

    return "\n".join(output)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: semantic_search.py <index_dir> <query> [top_k]")
        sys.exit(1)

    index_dir = sys.argv[1]
    query = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    searcher = SemanticSearcher(index_dir)
    results = searcher.search(query, top_k=top_k)
    print(format_results(results, query))
