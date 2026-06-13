"""
RAG Engine - Retrieval-Augmented Generation for Scrapbot.

This module implements the full RAG pipeline:
1. Document chunking: Split documents into manageable pieces
2. Embedding generation: Convert chunks to vectors
3. Vector similarity search: Find relevant chunks for query
4. Response generation: Generate answer from retrieved context

Retrieval-Augmented Generation concept:
- Traditional chatbots: Generate answers from training data only
- RAG: Retrieve relevant information from knowledge base, then generate answer
- Benefits: Can answer questions about information not in training data
- Use case: Knowledge questions, factual queries, domain-specific information

Why RAG is used instead of pure generation:
- Accuracy: Answers are grounded in retrieved facts, not hallucinated
- Updatability: Knowledge base can be updated without retraining model
- Transparency: Can show sources of information
- Cost: No need for large language models, uses smaller embedding models

Fallback logic from intent classifier → RAG:
1. User asks a question
2. Intent classifier tries to classify intent
3. If confidence is low OR question is knowledge-seeking → Use RAG
4. RAG retrieves relevant chunks from knowledge base
5. Return retrieved information as answer
"""

import os
from typing import Optional, List, Tuple, Any, Dict
from pathlib import Path
from rag.vector_store import vector_store
from nlp.embeddings import embedding_generator
from utils.logger import logger
from utils.helpers import get_project_root


class RAGEngine:
    """
    RAG Engine for retrieving and generating answers from knowledge base.
    
    This class manages the complete RAG pipeline:
    - Loading and chunking documents
    - Generating embeddings
    - Storing in vector database
    - Retrieving relevant chunks for queries
    """
    
    def __init__(self, knowledge_base_path: str = None, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize RAG engine.
        
        Args:
            knowledge_base_path: Directory containing knowledge base files (default: project data/)
            chunk_size: Size of text chunks (characters)
            chunk_overlap: Overlap between chunks (for context preservation)
        """
        if knowledge_base_path is None:
            knowledge_base_path = os.path.join(get_project_root(), "data")
        self.knowledge_base_path = Path(knowledge_base_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.initialized = False
        self._initialization_attempted = False
        
        # Don't initialize on import - wait until first query
        # This prevents errors during module import
    
    def _initialize_knowledge_base(self):
        """
        Initialize knowledge base by loading and indexing documents.
        
        Process:
        1. Load all text files from knowledge base directory
        2. Split into chunks
        3. Generate embeddings
        4. Store in vector database
        """
        try:
            # Load documents from knowledge base
            documents = self._load_documents()
            
            if not documents:
                logger.logger.warning("No documents found in knowledge base")
                return
            
            # Split documents into chunks
            # Chunking is important because:
            # - Large documents don't fit in embedding models
            # - Smaller chunks allow more precise retrieval
            # - Overlap preserves context across chunk boundaries
            chunks = self._chunk_documents(documents)
            
            if not chunks:
                logger.logger.warning("No chunks created from documents")
                return
            
            # Generate embeddings for chunks
            # Embeddings convert text to numerical vectors for similarity search
            logger.logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = embedding_generator.generate_embeddings(chunks)
            
            # Add to vector store
            vector_store.add_documents(chunks, embeddings)
            
            self.initialized = True
            logger.logger.info(f"RAG engine initialized with {len(chunks)} chunks")
            
        except Exception as e:
            logger.log_error(e, "RAGEngine._initialize_knowledge_base")
            self.initialized = False
    
    def _load_documents(self) -> List[str]:
        """
        Load all text documents from knowledge base directory.
        
        Supports .txt files. Can be extended to support PDF, DOCX, etc.
        
        Returns:
            List of document texts
        """
        documents = []
        
        # Look for .txt files in knowledge base directory
        txt_files = list(self.knowledge_base_path.glob("*.txt"))
        
        # Also check for a notes.txt file (from test files)
        if not txt_files:
            # Try common knowledge base file names
            common_files = ["notes.txt", "knowledge.txt", "kb.txt"]
            for filename in common_files:
                filepath = self.knowledge_base_path / filename
                if filepath.exists():
                    txt_files.append(filepath)
        
        for filepath in txt_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        documents.append(content)
                        logger.logger.info(f"Loaded document: {filepath.name}")
            except Exception as e:
                logger.log_error(e, f"RAGEngine._load_documents ({filepath})")
        
        return documents
    
    def _chunk_documents(self, documents: List[str]) -> List[str]:
        """
        Split documents into smaller chunks.
        
        Chunking strategy:
        - Split by sentences first (preserves meaning)
        - Then combine sentences into chunks of desired size
        - Overlap chunks to preserve context
        
        Args:
            documents: List of full document texts
            
        Returns:
            List of text chunks
        """
        import re
        
        all_chunks = []
        
        for doc in documents:
            # Split into sentences
            # Simple sentence splitting - can be improved with NLTK/spaCy
            sentences = re.split(r'[.!?]+\s+', doc)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Combine sentences into chunks
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # If adding this sentence exceeds chunk size, save current chunk
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    all_chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    # Overlap: keep last few sentences for context
                    overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                    current_chunk = overlap_sentences.copy()
                    current_length = sum(len(s) for s in current_chunk)
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                all_chunks.append(chunk_text)
        
        return all_chunks
    
    def query(self, question: str, top_k: int = 3) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Query the RAG system with a user question.

        Process:
        1. Generate embedding for question
        2. Search vector store for similar chunks
        3. Combine top-k chunks as context
        4. Return context as answer and grounding reason for explain-why

        Args:
            question: User's question
            top_k: Number of top chunks to retrieve

        Returns:
            (Answer text or None, reason dict with chunks/query for grounding)
        """
        # Lazy initialization - only initialize on first query
        # This prevents errors during module import
        if not self._initialization_attempted:
            self._initialization_attempted = True
            if self.knowledge_base_path.exists():
                self._initialize_knowledge_base()

        if not self.initialized:
            # RAG not available - return None to allow fallback
            return None, None

        try:
            # Generate embedding for question
            question_embedding = embedding_generator.generate_embeddings([question])[0]

            # Search for similar chunks
            logger.log_rag_query(question, top_k)
            results = vector_store.search(question_embedding, top_k=top_k)

            if not results:
                logger.logger.warning(f"No results found for query: {question}")
                return None, None

            # Combine top chunks as answer
            # Higher similarity chunks are more relevant
            answer_parts = []
            for chunk, similarity in results:
                if similarity > 0.3:  # Minimum similarity threshold
                    answer_parts.append(chunk)

            if not answer_parts:
                return None, None

            # Combine chunks into answer
            # Simple concatenation - can be improved with LLM generation
            answer = "\n\n".join(answer_parts)
            reason = {"chunks": len(answer_parts), "query": question[:100]}
            logger.logger.info(f"GROUNDING RAG: {reason}")
            logger.logger.info(f"RAG query successful: {len(answer_parts)} chunks retrieved")
            return answer, reason

        except Exception as e:
            logger.log_error(e, "RAGEngine.query")
            return None, None


# Create singleton instance
rag_engine = RAGEngine()


def rag_answer(question: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Convenience function for RAG querying.

    This function is called from chatbot.py when RAG is needed.

    Args:
        question: User's question

    Returns:
        (Answer from RAG system or None, reason dict for grounding)
    """
    return rag_engine.query(question)
