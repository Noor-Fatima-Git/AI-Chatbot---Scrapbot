"""
Text Embeddings Module for RAG (Retrieval-Augmented Generation).

This module converts text into vector embeddings for similarity search.
Embeddings are numerical representations of text that capture semantic meaning.

Why embeddings are needed:
- Text similarity: Find similar content in knowledge base
- Semantic search: "car" matches "automobile", "vehicle"
- Vector operations: Fast similarity computation using cosine distance

How embeddings work:
1. Text is converted to dense vector (e.g., 384-dimensional)
2. Similar texts have similar vectors (close in vector space)
3. Cosine similarity measures how similar two texts are
4. Used in RAG to find relevant chunks from knowledge base

Embedding models:
- sentence-transformers: Pre-trained models optimized for semantic similarity
- Alternative: OpenAI embeddings, Cohere embeddings (require API keys)
"""

from typing import List
import numpy as np
from utils.logger import logger


class EmbeddingGenerator:
    """
    Generates text embeddings for semantic similarity search.
    
    Uses sentence-transformers library for generating embeddings.
    Falls back to TF-IDF if sentence-transformers is not available.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of sentence-transformers model
                       "all-MiniLM-L6-v2" is fast and efficient
        """
        self.model = None
        self.model_name = model_name
        
        # Try to load sentence-transformers model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.logger.info(f"Embedding model '{model_name}' loaded successfully")
        except ImportError:
            logger.logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers\n"
                "Falling back to TF-IDF embeddings."
            )
            self.model = None
        except Exception as e:
            logger.log_error(e, "EmbeddingGenerator.__init__")
            self.model = None
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings (shape: [len(texts), embedding_dim])
        """
        if not texts:
            return np.array([])
        
        try:
            if self.model:
                # Use sentence-transformers for high-quality embeddings
                # Returns numpy array of shape [num_texts, embedding_dim]
                embeddings = self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                return embeddings
            else:
                # Fallback to simple TF-IDF-like approach
                # This is a basic implementation - not as good as sentence-transformers
                logger.logger.warning("Using fallback embedding method (TF-IDF)")
                return self._fallback_embeddings(texts)
                
        except Exception as e:
            logger.log_error(e, "EmbeddingGenerator.generate_embeddings")
            return self._fallback_embeddings(texts)
    
    def _fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Fallback embedding method using simple word frequency.
        
        This is a basic implementation that doesn't capture semantics well.
        Should be replaced with proper sentence-transformers in production.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Simple bag-of-words style embeddings
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=100)
            embeddings = vectorizer.fit_transform(texts).toarray()
            return embeddings
        except Exception as e:
            logger.log_error(e, "EmbeddingGenerator._fallback_embeddings (sklearn)")
        # Pure-Python bag-of-words when sklearn is missing or broken
        dim = 100
        vocab: dict = {}
        rows = []
        for text in texts:
            counts: dict = {}
            for token in text.lower().split():
                token = "".join(c for c in token if c.isalnum())
                if not token:
                    continue
                if token not in vocab and len(vocab) < dim:
                    vocab[token] = len(vocab)
                if token in vocab:
                    counts[vocab[token]] = counts.get(vocab[token], 0) + 1
            row = np.zeros(dim, dtype=np.float64)
            for idx, cnt in counts.items():
                row[idx] = float(cnt)
            norm = np.linalg.norm(row)
            if norm > 0:
                row /= norm
            rows.append(row)
        return np.vstack(rows) if rows else np.zeros((len(texts), dim))
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Cosine similarity measures angle between vectors:
        - 1.0: Identical (same direction)
        - 0.0: Orthogonal (perpendicular)
        - -1.0: Opposite directions
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1.0 to 1.0)
        """
        # Normalize vectors to unit length
        # This makes cosine similarity = dot product of normalized vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Cosine similarity = dot product of normalized vectors
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Clamp to [-1, 1] to handle floating point errors
        return float(np.clip(similarity, -1.0, 1.0))


# Create singleton instance
embedding_generator = EmbeddingGenerator()
