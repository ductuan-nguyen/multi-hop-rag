from dataclasses import dataclass
from typing import Optional

# --- Configuration & Hyperparameters ---
@dataclass
class IterRetGenConfig:
    # Model Configs
    # retriever_model_name: str = "BAAI/bge-m3"
    retriever_model_name: str = "BAAI/bge-m3"
    llm_model_name: str = "gemini-2.5-flash-lite"
    # llm_model_name: str = "gemini-2.5-flash"
    
    # Algorithm Hyperparameters (from Paper)
    max_iterations: int = 3        # Paper suggests T=2 is optimal [cite: 26]
    top_k: int = 5                 # Paper retrieves top-5 paragraphs 

    # Evaluation config
    eval_recall_k: int = 3         # default Recall@3 (can change to 5, 10, ...)
    
    # FAISS Config
    embedding_dim: int = 1024      # BGE-M3 dimension
    index_path: Optional[str] = '/self_ask/faiss/faiss_index.index'  # Path to save/load pre-indexed database
    
    # Generation Config
    temperature: float = 0.0       # Greedy decoding as per paper [cite: 106]
