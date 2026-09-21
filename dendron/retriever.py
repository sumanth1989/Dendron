"""
RAG (Retrieval-Augmented Generation) based tool retrieval for Dendron.
Enables agents and users to semantically query the execution tree and find
the best available tool for their specific task or usecase.

Zero-dependency by default: includes an internal TF-IDF & subword vector space
engine with cosine similarity, while supporting optional dense vector embedding
functions (e.g. OpenAI, Hugging Face, Ollama, etc.).
"""

from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Common English stopwords to ignore during lexical tokenization
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}


@dataclass
class RAGSearchResult:
    """Represents a retrieved tool match with relevance score and rationale."""
    node: Any  # DendronNode (avoid circular import)
    score: float
    document: str
    match_reasons: List[str] = field(default_factory=list)

    @property
    def tool_name(self) -> str:
        return self.node.tool.name

    @property
    def description(self) -> str:
        return self.node.tool.description

    def __repr__(self) -> str:
        return f"<RAGSearchResult(tool='{self.tool_name}', score={self.score:.3f})>"


class DendronRetriever:
    """
    Retrieval-Augmented Generation (RAG) engine for Dendron tool execution trees.
    Indexes tool nodes and allows semantic/lexical querying.
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None
    ) -> None:
        """
        Initializes the retriever.
        :param embedding_fn: Optional custom dense embedding function that takes
                             a list of strings and returns a list of float vectors.
                             If None, uses built-in zero-dependency vector space engine.
        """
        self.embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = embedding_fn

        # Internal index structures
        self._nodes: Dict[str, Any] = {}  # node_id -> DendronNode
        self._documents: Dict[str, str] = {}  # node_id -> search document text
        self._tokenized_docs: Dict[str, List[str]] = {}  # node_id -> list of tokens
        self._dense_embeddings: Dict[str, List[float]] = {}  # node_id -> vector (if embedding_fn)

        # Lexical statistics
        self._doc_frequencies: Counter[str] = Counter()
        self._total_docs: int = 0

    # MARK: - Tokenization & Vector Space

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenizes text into words and subword n-grams for semantic closeness."""
        cleaned = re.sub(r"[^a-zA-Z0-9_\s]", " ", text.lower())
        words = [w for w in cleaned.split() if w and w not in STOPWORDS]
        tokens = list(words)
        # Add character 3-grams for words >= 4 chars to handle fuzzy/stem matches
        for word in words:
            if len(word) >= 4:
                for i in range(len(word) - 2):
                    tokens.append(f"ngram:{word[i:i+3]}")
        return tokens

    def _compute_tf_idf_vector(self, tokens: List[str]) -> Dict[str, float]:
        """Computes sublinear TF-IDF vector with length normalization."""
        counts = Counter(tokens)
        vector: Dict[str, float] = {}
        for token, count in counts.items():
            tf = 1.0 + math.log(count)
            df = self._doc_frequencies.get(token, 1)
            # Smooth IDF
            idf = math.log(1.0 + (self._total_docs + 1) / (df + 0.5))
            vector[token] = tf * idf

        # L2 norm normalization
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {k: v / norm for k, v in vector.items()}

    @staticmethod
    def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        """Calculates cosine similarity between two dense float vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1e-9
        norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1e-9
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    # MARK: - Indexing

    def index_node(self, node: Any) -> None:
        """Indexes or updates a single DendronNode in the retriever."""
        # Generate search document text
        if hasattr(node, "to_search_document"):
            doc_text = node.to_search_document()
        else:
            doc_text = f"Tool: {node.tool.name}\nDescription: {node.tool.description}"

        old_tokens = self._tokenized_docs.get(node.id)
        if old_tokens:
            # Decrement old DF
            unique_old = set(old_tokens)
            for t in unique_old:
                self._doc_frequencies[t] -= 1
                if self._doc_frequencies[t] <= 0:
                    del self._doc_frequencies[t]
        else:
            self._total_docs += 1

        tokens = self._tokenize(doc_text)
        for t in set(tokens):
            self._doc_frequencies[t] += 1

        self._nodes[node.id] = node
        self._documents[node.id] = doc_text
        self._tokenized_docs[node.id] = tokens

        # Compute dense embedding if function is present
        if self.embedding_fn:
            try:
                embeddings = self.embedding_fn([doc_text])
                if embeddings and len(embeddings) > 0:
                    self._dense_embeddings[node.id] = embeddings[0]
            except Exception:
                pass

    def index_tree(self, tree: Any) -> None:
        """Indexes all nodes in a Dendron tree."""
        nodes = tree.search_bfs() if hasattr(tree, "search_bfs") else []
        for node in nodes:
            self.index_node(node)

    def remove_node(self, node_id: str) -> None:
        """Removes a node from the index."""
        if node_id in self._nodes:
            del self._nodes[node_id]
            doc_tokens = self._tokenized_docs.pop(node_id, [])
            for t in set(doc_tokens):
                self._doc_frequencies[t] -= 1
                if self._doc_frequencies[t] <= 0:
                    del self._doc_frequencies[t]
            self._documents.pop(node_id, None)
            self._dense_embeddings.pop(node_id, None)
            self._total_docs = max(0, self._total_docs - 1)

    # MARK: - Retrieval

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0
    ) -> List[RAGSearchResult]:
        """
        Retrieves the top-k most relevant tools for a given user/agent query.
        :param query: Natural language description of the intended task or usecase.
        :param top_k: Maximum number of tools to return.
        :param min_score: Minimum relevance score threshold (0.0 to 1.0).
        :return: List of RAGSearchResult objects sorted by descending relevance.
        """
        if not self._nodes or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        query_vector = self._compute_tf_idf_vector(query_tokens)

        # Dense query vector if embedding_fn is present
        dense_query_vec: Optional[List[float]] = None
        if self.embedding_fn:
            try:
                embeddings = self.embedding_fn([query])
                if embeddings and len(embeddings) > 0:
                    dense_query_vec = embeddings[0]
            except Exception:
                dense_query_vec = None

        results: List[RAGSearchResult] = []
        lower_query = query.lower()

        for node_id, node in self._nodes.items():
            doc_tokens = self._tokenized_docs[node_id]
            doc_vector = self._compute_tf_idf_vector(doc_tokens)

            # 1. Lexical / sparse cosine similarity
            lexical_score = sum(
                weight * doc_vector.get(token, 0.0)
                for token, weight in query_vector.items()
            )

            # 2. Exact match bonuses
            match_reasons: List[str] = []
            bonus = 0.0

            # Direct tool name match
            if node.tool.name.lower() in lower_query:
                bonus += 0.25
                match_reasons.append(f"Tool name '{node.tool.name}' matched query directly")

            # Tag match
            for tag in node.tool.tags:
                if tag.lower() in lower_query:
                    bonus += 0.15
                    match_reasons.append(f"Matched tag '{tag}'")

            # Parameter match
            for param_name in node.tool.parameters:
                if param_name.lower() in lower_query:
                    bonus += 0.10
                    match_reasons.append(f"Matched parameter '{param_name}'")

            # Experience note match
            for note in node.experience_notes:
                if any(w in note.lower() for w in query_tokens if not w.startswith("ngram:")):
                    bonus += 0.10
                    match_reasons.append("Matched learned experience note")
                    break

            combined_score = min(1.0, lexical_score + bonus)

            # 3. Blend with dense embedding score if available
            if dense_query_vec and node_id in self._dense_embeddings:
                dense_score = self._cosine_similarity(dense_query_vec, self._dense_embeddings[node_id])
                # 60% dense vector, 40% lexical + exact match
                combined_score = (0.60 * dense_score) + (0.40 * combined_score)
                match_reasons.append(f"Dense semantic similarity: {dense_score:.2f}")

            if combined_score >= min_score:
                node.record_access()
                results.append(RAGSearchResult(
                    node=node,
                    score=combined_score,
                    document=self._documents.get(node_id, ""),
                    match_reasons=match_reasons
                ))

        # Sort descending by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def retrieve_best(self, query: str, min_score: float = 0.0) -> Optional[Any]:
        """Convenience method returning the single best DendronNode or None."""
        results = self.retrieve(query=query, top_k=1, min_score=min_score)
        return results[0].node if results else None


# Backwards compatibility alias
RAGToolRetriever = DendronRetriever

