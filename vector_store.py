"""
Vector Store for RAG - Manages document embeddings and similarity search.

This module implements a vector store using FAISS (Facebook AI Similarity Search)
for efficient similarity search over large document collections.

Why FAISS is used:
- Fast: Optimized C++ backend for vector operations
- Scalable: Handles millions of vectors efficiently
- Memory efficient: Supports both CPU and GPU operations
- Open source: No API costs, runs locally

Vector similarity search in simple terms:
1. Documents are converted to vectors (embeddings)
2. User query is also converted to a vector
3. Find documents whose vectors are "close" to query vector
4. "Close" means high cosine similarity (similar meaning)
5. Return top-k most similar documents as context

Fallback to ChromaDB:
If FAISS is not available, falls back to ChromaDB or simple
in-memory vector storage.
"""

import os
import pickle
from typing import List, Tuple, Optional
import numpy as np
from utils.logger import logger


class VectorStore:
    """
    Vector store for storing and searching document embeddings.
    
    Uses FAISS for efficient similarity search. Stores:
    - Document chunks (text segments)
    - Corresponding embeddings (vector representations)
    - Metadata (optional, for filtering)
    """
    
    def __init__(self, store_path: Optional[str] = None):
        """
        Initialize vector store.
        
        Args:
            store_path: Path to save/load vector store from disk
        """
        self.store_path = store_path
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        
        # Try to load FAISS
        try:
            import faiss
            self.faiss_available = True
            logger.logger.info("FAISS available for vector search")
        except ImportError:
            self.faiss_available = False
            logger.logger.warning(
                "FAISS not installed. Install with: pip install faiss-cpu\n"
                "Using fallback vector search."
            )
        
        # Load existing store if path provided
        if store_path and os.path.exists(store_path):
            self.load(store_path)
    
    def add_documents(self, chunks: List[str], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the store.
        
        Args:
            chunks: List of text chunks (document segments)
            embeddings: NumPy array of embeddings (shape: [num_chunks, embedding_dim])
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        self.chunks = chunks
        self.embeddings = embeddings.astype('float32')  # FAISS requires float32
        
        # Build FAISS index if available
        if self.faiss_available and len(embeddings) > 0:
            try:
                import faiss
                
                # Get embedding dimension
                dimension = embeddings.shape[1]
                
                # Create FAISS index
                # L2 (Euclidean) distance index - can also use cosine similarity
                # For cosine similarity, normalize vectors first
                self.faiss_index = faiss.IndexFlatL2(dimension)
                
                # Add embeddings to index
                self.faiss_index.add(self.embeddings)
                
                logger.logger.info(f"Added {len(chunks)} documents to FAISS index")
            except Exception as e:
                logger.log_error(e, "VectorStore.add_documents")
                self.faiss_index = None
        else:
            logger.logger.info(f"Added {len(chunks)} documents (FAISS not used)")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for similar documents using query embedding.
        
        Process:
        1. Convert query to embedding (done by caller)
        2. Find top-k most similar documents using FAISS or cosine similarity
        3. Return chunks with similarity scores
        
        Args:
            query_embedding: Embedding vector of query (shape: [embedding_dim])
            top_k: Number of top results to return
            
        Returns:
            List of tuples (chunk_text, similarity_score)
        """
        if self.embeddings is None or len(self.chunks) == 0:
            logger.logger.warning("Vector store is empty")
            return []
        
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        try:
            if self.faiss_available and self.faiss_index is not None:
                # Use FAISS for fast search
                # Returns distances (lower is better), so we convert to similarities
                distances, indices = self.faiss_index.search(query_embedding, top_k)
                
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.chunks):
                        # Convert distance to similarity (1 / (1 + distance))
                        # This gives similarity score between 0 and 1
                        similarity = 1.0 / (1.0 + distances[0][i])
                        results.append((self.chunks[idx], float(similarity)))
                
                return results
            else:
                # Fallback: Use cosine similarity
                return self._cosine_search(query_embedding, top_k)
                
        except Exception as e:
            logger.log_error(e, "VectorStore.search")
            return self._cosine_search(query_embedding, top_k)
    
    def _cosine_search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """
        Fallback search using cosine similarity.
        
        Computes cosine similarity between query and all documents,
        then returns top-k most similar.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (chunk, similarity) tuples
        """
        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Normalize all document embeddings
        doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarities (dot product of normalized vectors)
        similarities = np.dot(doc_norms, query_norm.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))
        
        return results
    
    def save(self, path: Optional[str] = None):
        """
        Save vector store to disk.
        
        Args:
            path: Path to save to (uses self.store_path if None)
        """
        save_path = path or self.store_path
        if not save_path:
            logger.logger.warning("No path provided for saving vector store")
            return
        
        try:
            data = {
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'faiss_index': self.faiss_index
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.logger.info(f"Vector store saved to {save_path}")
        except Exception as e:
            logger.log_error(e, "VectorStore.save")
    
    def load(self, path: Optional[str] = None):
        """
        Load vector store from disk.
        
        Args:
            path: Path to load from (uses self.store_path if None)
        """
        load_path = path or self.store_path
        if not load_path or not os.path.exists(load_path):
            logger.logger.warning(f"Vector store file not found: {load_path}")
            return
        
        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            
            self.chunks = data.get('chunks', [])
            self.embeddings = data.get('embeddings')
            self.faiss_index = data.get('faiss_index')
            
            logger.logger.info(f"Vector store loaded from {load_path}")
        except Exception as e:
            logger.log_error(e, "VectorStore.load")
    
    def get_size(self) -> int:
        """
        Get number of documents in store.
        
        Returns:
            Number of stored documents
        """
        return len(self.chunks)


# Create singleton instance
vector_store = VectorStore()
