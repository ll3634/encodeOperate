#!/usr/bin/env python3
"""
Search tool with BM25/TF-IDF retrieval.
Minimal implementation without heavy dependencies.
"""

import json
import math
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
from dataclasses import dataclass


STOPWORDS = frozenset({
    'a', 'an', 'the', 'is', 'was', 'were', 'are', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'am', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'out', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'not', 'so',
    'yet', 'both', 'either', 'neither', 'each', 'every', 'all', 'any',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'only', 'own',
    'same', 'than', 'too', 'very', 'just', 'because', 'if', 'when',
    'where', 'how', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
    'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
    'it', 'its', 'they', 'them', 'their', 'theirs', 'about', 'up',
})


def stem(word: str) -> str:
    """Simple suffix stripping for English words.

    Handles common morphological variants so that, e.g., 'directed' and
    'director' share the same index token, improving BM25 recall for
    property-attribute queries (director/directed, author/authored, etc.).
    Longer suffixes are checked first to avoid partial over-stripping.
    """
    if len(word) <= 4:
        return word
    for suffix in ('tion', 'sion', 'ness', 'ment', 'ing'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    for suffix in ('ed', 'er', 'or', 'ly', 'al'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def tokenize(text: str) -> List[str]:
    """Tokenize text: lowercase, split on non-alphanumeric, remove stopwords, apply stemming."""
    return [stem(w) for w in re.findall(r'\w+', text.lower()) if w not in STOPWORDS]


@dataclass
class Document:
    """A document in the corpus."""
    id: str
    title: str
    text: str
    tokens: List[str] = None
    
    def __post_init__(self):
        if self.tokens is None:
            self.tokens = tokenize(self.title + " " + self.text)


class BM25Searcher:
    """
    Simple BM25 searcher.
    No external dependencies - pure Python implementation.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.doc_freqs: Dict[str, int] = {}  # term -> doc count
        self.avg_doc_len: float = 0.0
        self.doc_lens: List[int] = []
        self._indexed = False
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the index."""
        self.documents.extend(documents)
        self._indexed = False
    
    def build_index(self):
        """Build the BM25 index."""
        if not self.documents:
            return
        
        self.doc_freqs = {}
        self.doc_lens = []
        
        for doc in self.documents:
            self.doc_lens.append(len(doc.tokens))
            seen_terms = set()
            for term in doc.tokens:
                if term not in seen_terms:
                    self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                    seen_terms.add(term)
        
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 1.0
        self._indexed = True
    
    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a document."""
        doc = self.documents[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        term_freqs = Counter(doc.tokens)
        
        score = 0.0
        n_docs = len(self.documents)
        
        for term in query_tokens:
            if term not in self.doc_freqs:
                continue
            
            df = self.doc_freqs[term]
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            
            tf = term_freqs.get(term, 0)
            tf_component = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            )
            
            score += idf * tf_component
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for documents matching the query."""
        if not self._indexed:
            self.build_index()
        
        if not self.documents:
            return []
        
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        
        scores = [(i, self._score(query_tokens, i)) for i in range(len(self.documents))]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_idx, score in scores[:top_k]:
            if score > 0:
                results.append((self.documents[doc_idx], score))
        
        return results


class SearchTool:
    """
    Search tool for ReAct agent.
    Uses BM25 for retrieval from a local corpus.
    """
    
    def __init__(
        self,
        corpus_path: Optional[str] = None,
        top_k: int = 5,
        max_chars: int = 500
    ):
        self.searcher = BM25Searcher()
        self.top_k = top_k
        self.max_chars = max_chars
        self.corpus_loaded = False
        
        if corpus_path:
            self.load_corpus(corpus_path)
    
    def load_corpus(self, path: str):
        """Load corpus from JSONL file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Corpus file not found: {path}\n"
                f"Please prepare the corpus first. See README.md for instructions."
            )
        
        documents = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                doc = Document(
                    id=data.get("id", str(len(documents))),
                    title=data.get("title", ""),
                    text=data.get("text", data.get("content", ""))
                )
                documents.append(doc)
        
        self.searcher.add_documents(documents)
        self.searcher.build_index()
        self.corpus_loaded = True
        print(f"Loaded {len(documents)} documents from {path}")
    
    def __call__(self, query: str) -> str:
        """Execute search and return formatted results."""
        if not self.corpus_loaded:
            return "Error: No corpus loaded. Please load a corpus first."
        
        results = self.searcher.search(query, self.top_k)
        
        if not results:
            return "No relevant documents found."
        
        output_parts = []
        for i, (doc, score) in enumerate(results, 1):
            text = doc.text[:self.max_chars]
            if len(doc.text) > self.max_chars:
                text += "..."
            output_parts.append(f"[{i}] {doc.title}: {text}")
        
        return "\n\n".join(output_parts)

