import time
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from FlagEmbedding import BGEM3FlagModel
from collections import Counter
import string
import os
import logging
import pickle
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# --- Configuration & Hyperparameters ---
@dataclass
class IterRetGenConfig:
    # Model Configs
    # retriever_model_name: str = "BAAI/bge-m3"
    retriever_model_name: str = "/home4/vietld/hf_model/bge-m3"
    llm_model_name: str = "gemini-2.5-flash-lite"
    # llm_model_name: str = "gemini-2.5-flash"
    
    # Algorithm Hyperparameters (from Paper)
    max_iterations: int = 3        # Paper suggests T=2 is optimal [cite: 26]
    top_k: int = 5                 # Paper retrieves top-5 paragraphs 
    
    # FAISS Config
    embedding_dim: int = 1024      # BGE-M3 dimension
    index_path: Optional[str] = '/home3/tuannd/llm/faiss_index.index'  # Path to save/load pre-indexed database
    
    # Generation Config
    temperature: float = 0.0       # Greedy decoding as per paper [cite: 106]

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

    def search(self, query: str, k: int = None) -> List[str]:
        k = k or self.config.top_k
        self.logger.debug(f"Searching for query: '{query[:100]}...' (k={k})")
        
        query_embedding = self.encoder.encode([query])['dense_vecs']
        
        # Convert to float32 and ensure contiguous array (FAISS requirement)
        query_embedding = np.ascontiguousarray(query_embedding.astype('float32'))
        
        D, I = self.index.search(query_embedding, k)
        
        retrieved_docs = [self.documents[i] for i in I[0]]
        self.logger.debug(f"Retrieved {len(retrieved_docs)} documents. Top similarity scores: {D[0][:3]}")
        return retrieved_docs

# --- Component 2: Generator (Gemini) ---
class LLMGenerator:
    def __init__(self, config: IterRetGenConfig, api_key: str):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing LLM Generator with model: {config.llm_model_name}")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.llm_model_name)
        self.config = config
        self.logger.info("LLM Generator initialized successfully")

    def generate(self, prompt: str) -> str:
        # Paper uses greedy decoding (temp=0) [cite: 106]
        self.logger.debug(f"Generating response with temperature={self.config.temperature}")
        self.logger.debug(f"Prompt length: {len(prompt)} characters")
        
        generation_config = genai.types.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=4096
        )
        response = self.model.generate_content(prompt, generation_config=generation_config)
        generated_text = response.text.strip()
        self.logger.debug(f"Generated response length: {len(generated_text)} characters")
        return generated_text

# --- Core Logic: ITER-RETGEN Class ---
class IterRetGen:
    def __init__(self, config: IterRetGenConfig, documents: List[str], api_key: str):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.logger.info(f"Initializing IterRetGen with {len(documents)} documents, max_iterations={config.max_iterations}")
        self.retriever = VectorDatabase(config, documents)
        self.llm = LLMGenerator(config, api_key)
        self.logger.info("IterRetGen initialized successfully")

    def _construct_prompt(self, query: str, context: List[str]) -> str:
        # Chain-of-Thought Prompt structure inspired by paper [cite: 515]
        context_str = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context)])
        return f"""
Answer the question based on the context provided. 
Reason step-by-step before providing the final answer.

Context:
{context_str}

Question: {query}

Let's think step by step. Answer in Vietnamese.
"""

    def solve(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.info(f"Starting solve for query: '{query}'")
        
        # --- Iteration 0: Initial Retrieval ---
        # Query = Original Question
        self.logger.info("Iteration 0: Initial retrieval")
        current_context = self.retriever.search(query)
        self.logger.info(f"Retrieved {len(current_context)} documents in initial retrieval")
        
        # Initial Generation
        prompt = self._construct_prompt(query, current_context)
        current_generation = self.llm.generate(prompt)
        self.logger.info(f"Iteration 0: Generated initial answer (length: {len(current_generation)} chars)")
        
        all_generations = [current_generation]
        
        # --- Iteration 1 to T: Iterative Retrieval-Generation ---
        # The loop runs T times. Paper recommends T=2.
        for t in range(1, self.config.max_iterations + 1):
            self.logger.info(f"Iteration {t}/{self.config.max_iterations}: Starting retrieval-generation")
            
            # NEW QUERY: Concatenate original query + previous generation
            # This is the key "Synergy" step described in Eq (1) [cite: 59, 125]
            augmented_query = f"{query} {current_generation}"
            self.logger.debug(f"Augmented query length: {len(augmented_query)} characters")
            
            # Retrieve using augmented query
            new_docs = self.retriever.search(augmented_query)
            self.logger.info(f"Iteration {t}: Retrieved {len(new_docs)} new documents")
            
            # Combine old and new context.
            # The paper emphasizes processing "all retrieved knowledge as a whole" [cite: 14]
            # We use a set to avoid duplicates while maintaining order
            combined_context = list(dict.fromkeys(current_context + new_docs))
            self.logger.info(f"Iteration {t}: Combined context size: {len(combined_context)} documents (was {len(current_context)})")
            
            # Generate new response
            prompt = self._construct_prompt(query, combined_context)
            current_generation = self.llm.generate(prompt)
            self.logger.info(f"Iteration {t}: Generated answer (length: {len(current_generation)} chars)")
            
            all_generations.append(current_generation)
            current_context = combined_context

        end_time = time.time()
        latency = end_time - start_time
        
        self.logger.info(f"Solve completed in {latency:.2f}s. Final context: {len(current_context)} documents, "
                        f"Total generations: {len(all_generations)}")
        
        return {
            "final_answer": current_generation,
            "all_generations": all_generations,
            "latency": latency,
            "final_context": current_context
        }

# --- Evaluation Metrics ---
class Evaluator:
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return " ".join([word for word in text.split() if word not in ["a", "an", "the"]])
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def exact_match_score(self, prediction, ground_truth):
        return (self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0: return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def model_based_accuracy(self, prediction, ground_truth, llm_generator):
        # Implementation of Acc† (Section 4.2 in paper) 
        prompt = f"""
        Question: (Implicit in context)
        Prediction: {prediction}
        Ground-truth Answer: {ground_truth}
        
        Does the Prediction imply the Ground-truth Answer? Output Yes or No:
        """
        judgment = llm_generator.generate(prompt)
        return 1 if "yes" in judgment.lower() else 0

    def evaluate(self, results, ground_truths, llm_generator):
        logger = logging.getLogger(__name__)
        logger.info(f"Starting evaluation on {len(results)} results")
        metrics = {"EM": [], "F1": [], "Acc_dagger": [], "Latency": []}
        
        for idx, (res, gt) in enumerate(zip(results, ground_truths)):
            pred = res['final_answer']
            em = self.exact_match_score(pred, gt)
            f1 = self.f1_score(pred, gt)
            metrics["EM"].append(em)
            metrics["F1"].append(f1)
            # Acc† requires LLM call
            logger.debug(f"Evaluating result {idx+1}/{len(results)}: EM={em}, F1={f1:.3f}")
            acc_dagger = self.model_based_accuracy(pred, gt, llm_generator)
            metrics["Acc_dagger"].append(acc_dagger)
            metrics["Latency"].append(res['latency'])
            
        final_metrics = {k: np.mean(v) for k, v in metrics.items()}
        logger.info(f"Evaluation completed. Final metrics: {final_metrics}")
        return final_metrics

# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Mock Data
    # my_documents = [
    #     "The 2015 AFL Rising Star award was won by Jesse Hogan.",
    #     "Jesse Hogan is a professional Australian rules footballer.",
    #     "Jesse Hogan is 195 cm tall.",
    #     "The AFL Rising Star is an annual award given to a standout young player."
    # ]
    import json
    document_file_path = '/home4/vietld/master/web_mining/playground/dichvucong_iHanoi_v1.0.jsonl'
    my_documents = []
    with open(document_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)['contents']
            my_documents.append(data)
    print(len(my_documents))
    print(my_documents[0])
    
    # Set index path based on document file (saves to same directory)
    # index_path = os.path.join(os.path.dirname(document_file_path), 'faiss_index')
    index_path = 'faiss_index'
    my_config = IterRetGenConfig(index_path=index_path)
    
    # Initialize
    # Replace with your actual API key
    # iter_ret_gen = IterRetGen(my_config, my_documents, api_key="YOUR_GEMINI_KEY")
    api_key = os.getenv("GEMINI_API_KEY")
    # print(f"API Key: {api_key}")
    iter_ret_gen = IterRetGen(my_config, my_documents, api_key=api_key)
    
    # Run
    with open('/home3/tuannd/llm/multi_hop_ihanoi_v1.0.json', 'r') as f:
        data = json.load(f)
        data = data[:3]
    
    results = []
    ground_truths = []
    for i,item in enumerate(data):
        # with open(f'result_iter/result_{i}.json', 'w') as f:
        query = item['multi_hop_question']
        result = iter_ret_gen.solve(query)
        results.append(result)
        ground_truths.append(item['answer'])
        # print(f"Final Answer: {result['final_answer']}")
        # print(f"Time Taken: {result['latency']:.2f}s")
        # item['final_answer'] = result['final_answer']
        # item['latency'] = result['latency']
        # json.dump(item, f, indent=4, ensure_ascii=False)
    
    evaluator = Evaluator()
    metrics = evaluator.evaluate(results, ground_truths, iter_ret_gen.llm)
    print(metrics)
    # query = "Cơ quan chịu trách nhiệm cấp phép triển lãm và phản hồi bằng văn bản trong vòng 15 ngày làm việc đối với các trường hợp phức tạp thì cần bao lâu để xem xét cấp phép cho người nước ngoài nghiên cứu, sưu tầm, tư liệu hóa di sản văn hóa phi vật thể?"
    # result = iter_ret_gen.solve(query)
    
    # print(f"Final Answer: {result['final_answer']}")
    # print(f"Time Taken: {result['latency']:.2f}s")