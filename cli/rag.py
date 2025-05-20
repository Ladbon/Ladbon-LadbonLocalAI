from typing import List, Dict, Optional
import os

class DocumentIndex:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """Initialize a FAISS index for document retrieval"""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
            
            self.embedder = SentenceTransformer(embedding_model)
            self.documents = []
            self.faiss_index = None
            self.dimension = None
            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "Required libraries not found. Please install with:\n"
                "pip install sentence-transformers faiss-cpu"
            )
    
    def add_documents(self, documents: List[str]):
        """Add documents to the index"""
        if not documents:
            return False
            
        self.documents.extend(documents)
        
        # Create embeddings
        embeddings = self.embedder.encode(documents, convert_to_numpy=True)
        
        # First time setup
        if self.faiss_index is None:
            self.faiss_index = self._faiss.IndexFlatIP(self.dimension)
        
        self._faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        return True
        return True
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search for relevant documents"""
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            return []
            
        # Encode and normalize query
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        self._faiss.normalize_L2(q_emb)
        
        # Search
        scores, indices = self.faiss_index.search(q_emb, k)
        
        # Return matched documents
        return [self.documents[int(idx)] for idx in indices[0]]