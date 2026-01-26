import numpy as np
import faiss
from typing import List, Optional
import os
import logging
import pickle
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel

from config import IterRetGenConfig

# --- Component 1: Retriever (BGE-M3 + FAISS) ---
class VectorDatabase:
    def __init__(self, config: IterRetGenConfig, documents: List[str]):
        self.config = config
        self.documents = documents
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Loading BGE-M3 Retriever from: {config.retriever_model_name}")
        # use_fp16=True for speed, BGE-M3 supports it
        self.encoder = BGEM3FlagModel(config.retriever_model_name, use_fp16=True)
        self.logger.info("BGE-M3 Retriever loaded successfully")
        
        # Try to load pre-indexed database, otherwise build and save

        print(config.index_path)
        print(self._index_exists(config.index_path))

        if config.index_path and self._index_exists(config.index_path):
            self.logger.info(f"Loading pre-indexed FAISS database from: {config.index_path}")
            self.index, loaded_documents = self._load_index(config.index_path)
            self.documents = loaded_documents
            self.logger.info(f"Loaded FAISS Index with {self.index.ntotal} vectors")
            if len(loaded_documents) != len(documents):
                self.logger.warning(f"Loaded {len(loaded_documents)} documents from index, but {len(documents)} were passed. Using loaded documents.")
        else:
            self.logger.info(f"Building FAISS Index for {len(documents)} documents...")
            self.index = self._build_index(documents)
            self.logger.info("FAISS Index built successfully")
            
            # Save the index if path is provided
            if config.index_path:
                self.logger.info(f"Saving FAISS Index to: {config.index_path}")
                self._save_index(config.index_path, self.index, self.documents)
                self.logger.info("FAISS Index saved successfully")

    def _build_index(self, docs):
        # BGE-M3 Dense Retrieval
        self.logger.debug(f"Encoding {len(docs)} documents with batch_size=12")
        embeddings = self.encoder.encode(docs, batch_size=12)['dense_vecs']
        self.logger.debug(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
        
        # Convert to float32 and ensure contiguous array (FAISS requirement)
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))
        
        # Using FlatIP (Inner Product) for cosine similarity behavior
        index = faiss.IndexFlatIP(self.config.embedding_dim)
        index.add(embeddings)
        self.logger.info(f"FAISS Index created with {index.ntotal} vectors, dimension: {self.config.embedding_dim}")
        return index
    
    def _index_exists(self, index_path: str) -> bool:
        """Check if pre-indexed database files exist"""
        index_file = f"{index_path}.index"
        docs_file = f"{index_path}.docs.pkl"
        return os.path.exists(index_file) and os.path.exists(docs_file)
    
    def _save_index(self, index_path: str, index: faiss.Index, documents: List[str]):
        """Save FAISS index and documents to disk"""
        index_file = f"{index_path}.index"
        docs_file = f"{index_path}.docs.pkl"
        
        # Create directory if it doesn't exist
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, index_file)
        
        # Save documents list
        with open(docs_file, 'wb') as f:
            pickle.dump(documents, f)
    
    def _load_index(self, index_path: str):
        """Load FAISS index and documents from disk"""
        index_file = f"{index_path}.index"
        docs_file = f"{index_path}.docs.pkl"
        
        # Load FAISS index
        index = faiss.read_index(index_file)
        
        # Load documents list
        with open(docs_file, 'rb') as f:
            documents = pickle.load(f)
        
        return index, documents

    def search(self, query: str, k: int = None, return_ids: bool = False):
        """
        Search for documents similar to query.
        
        Args:
            query: Search query
            k: Number of results to return
            return_ids: If True, return (documents, ids, scores). If False, return documents only.
        
        Returns:
            If return_ids=False: List[str] of documents
            If return_ids=True: Tuple of (List[str], List[int], List[float]) - (documents, ids, scores)
        """
        k = k or self.config.top_k
        self.logger.debug(f"Searching for query: '{query[:100]}...' (k={k})")
        
        query_embedding = self.encoder.encode([query])['dense_vecs']
        
        # Convert to float32 and ensure contiguous array (FAISS requirement)
        query_embedding = np.ascontiguousarray(query_embedding.astype('float32'))
        
        D, I = self.index.search(query_embedding, k)
        
        retrieved_docs = [self.documents[i] for i in I[0]]
        retrieved_ids = I[0].tolist()
        scores = D[0].tolist()
        
        self.logger.debug(f"Retrieved {len(retrieved_docs)} documents. Top similarity scores: {scores[:3]}")
        
        if return_ids:
            return retrieved_docs, retrieved_ids, scores
        return retrieved_docs
