"""
IterRetGen with BGE-M3 Classifier and Query Refiner Integration.

This module extends the iteretgen approach by:
1. Using a classifier to filter contexts (Irrelevant/Relevant/Contain Answer)
2. Using a query refiner to generate refined queries instead of LLM at each iteration

Pipeline logic:
- If "Contain Answer" is found: Stop and generate final answer with all relevant contexts
- If only "Relevant" contexts: Use query refiner to refine query, then search with refined query
- If no relevant contexts: Fallback to original iteretgen (LLM generates answer, concat with query)
- At max iterations: Generate final answer with all accumulated relevant contexts

The classifier and refiner can be used in two modes:
1. Local mode: Load the models directly (for single-threaded use)
2. Service mode: Call via HTTP API (for parallel processing)
"""

import argparse
import json
import os
import sys
import requests
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union

from google import genai
from google.genai import types
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from iteretgen import generate_answer, get_prompt, evaluate_answer, evaluate_with_llm
from search import search

# Gemini client setup
client = genai.Client(api_key='')
MODEL = 'gemini-2.5-flash'

# Label mapping
LABEL2ID = {"Relevant": 0, "Irrelevant": 1, "Contain Answer": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ============================================================================
# HTTP Clients for Service Mode
# ============================================================================

class ClassifierClient:
    """
    HTTP client for the classifier service.
    Use this for parallel processing to avoid CUDA OOM issues.
    """
    
    def __init__(self, service_url: str = "http://127.0.0.1:8001"):
        self.service_url = service_url.rstrip('/')
        self._check_health()
    
    def _check_health(self):
        """Check if the classifier service is healthy"""
        try:
            resp = requests.get(f"{self.service_url}/health", timeout=5)
            resp.raise_for_status()
            health = resp.json()
            if not health.get('model_loaded'):
                raise RuntimeError("Classifier model not loaded in service")
            print(f"Connected to classifier service at {self.service_url}")
            print(f"Service device: {health.get('device')}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Cannot connect to classifier service at {self.service_url}. "
                f"Please start the service first: python classifier_service.py --checkpoint <path>\n"
                f"Error: {e}"
            )
    
    def classify(self, query: str, context: str) -> Tuple[str, Dict[str, float]]:
        """Classify a single query-context pair via HTTP API."""
        resp = requests.post(
            f"{self.service_url}/classify",
            json={"query": query, "context": context},
            timeout=30
        )
        resp.raise_for_status()
        result = resp.json()
        return result['label'], result['probabilities']
    
    def classify_batch(self, query: str, contexts: List[str]) -> List[Tuple[str, Dict[str, float]]]:
        """Classify multiple contexts for a single query via HTTP API."""
        resp = requests.post(
            f"{self.service_url}/classify_batch",
            json={"query": query, "contexts": contexts},
            timeout=60
        )
        resp.raise_for_status()
        results = resp.json()['results']
        return [(r['label'], r['probabilities']) for r in results]


class RefinerClient:
    """
    HTTP client for the query refiner service.
    Use this for parallel processing to avoid CUDA OOM issues.
    """
    
    def __init__(self, service_url: str = "http://127.0.0.1:8002"):
        self.service_url = service_url.rstrip('/')
        self._check_health()
    
    def _check_health(self):
        """Check if the refiner service is healthy"""
        try:
            resp = requests.get(f"{self.service_url}/health", timeout=5)
            resp.raise_for_status()
            health = resp.json()
            if not health.get('model_loaded'):
                raise RuntimeError("Refiner model not loaded in service")
            print(f"Connected to refiner service at {self.service_url}")
            print(f"Service device: {health.get('device')}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Cannot connect to refiner service at {self.service_url}. "
                f"Please start the service first: python refiner_service.py --model_path <path>\n"
                f"Error: {e}"
            )
    
    def refine(self, query: str, context: str) -> str:
        """Refine a single query with context via HTTP API."""
        resp = requests.post(
            f"{self.service_url}/refine",
            json={"query": query, "context": context},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()['refined_query']
    
    def refine_batch(self, query: str, contexts: List[str]) -> List[str]:
        """Refine a query with multiple contexts via HTTP API."""
        resp = requests.post(
            f"{self.service_url}/refine_batch",
            json={"query": query, "contexts": contexts},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()['refined_queries']


class BGEClassifier(nn.Module):
    """BGE-M3 with 2-layer classification head"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        hidden_size: int = 1024,
        classifier_hidden_size: int = 512,
        num_labels: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Load BGE-M3 encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 2-layer classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, classifier_hidden_size),
            nn.GELU(),
            nn.LayerNorm(classifier_hidden_size),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_labels)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_embedding)
        
        return {'logits': logits, 'embeddings': cls_embedding}


class ContextClassifier:
    """Wrapper class for context classification"""
    
    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "BAAI/bge-m3",
        hidden_size: int = 1024,
        classifier_hidden_size: int = 512,
        max_length: int = 512,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = BGEClassifier(
            model_name=model_name,
            hidden_size=hidden_size,
            classifier_hidden_size=classifier_hidden_size,
            num_labels=len(LABEL2ID),
            dropout=0.1
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Classifier loaded from {checkpoint_path}")
        if 'accuracy' in checkpoint:
            print(f"Checkpoint accuracy: {checkpoint['accuracy']:.4f}")
    
    def classify(self, query: str, context: str) -> Tuple[str, Dict[str, float]]:
        """
        Classify a single query-context pair.
        
        Returns:
            Tuple of (label, probabilities_dict)
        """
        text = f"Question: {query}\n\nContext: {context}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        
        pred_id = torch.argmax(probs).item()
        label = ID2LABEL[pred_id]
        
        probabilities = {
            ID2LABEL[i]: probs[i].item()
            for i in range(len(ID2LABEL))
        }
        
        return label, probabilities


class QueryRefiner:
    """Wrapper class for query refinement using fine-tuned Qwen model"""
    
    SYSTEM_PROMPT = """Bạn là một chuyên gia tinh chỉnh câu hỏi. Nhiệm vụ của bạn là loại bỏ các thông tin nhiễu, không liên quan trong câu hỏi đầu vào để tạo ra một câu hỏi rõ ràng, ngắn gọn và chính xác hơn.

Quy tắc:
1. Giữ nguyên ý nghĩa cốt lõi của câu hỏi
2. Loại bỏ các thông tin dư thừa, không cần thiết
3. Đảm bảo câu hỏi tinh chỉnh vẫn có thể trả lời được dựa trên ngữ cảnh
4. Không thêm thông tin mới không có trong câu hỏi gốc

Chỉ trả lời với câu hỏi đã được tinh chỉnh, không giải thích."""
    
    def __init__(
        self,
        model_path: str,
        max_seq_length: int = 8192,
        load_in_4bit: bool = True,
        max_context_length: int = 2048,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        repetition_penalty: float = 1.2
    ):
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        
        print(f"Loading query refiner from {model_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        
        # Enable faster inference
        FastLanguageModel.for_inference(self.model)
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print("Query refiner loaded successfully")
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for the query refinement task"""
        user_prompt = f"""Câu hỏi gốc: {query}

Ngữ cảnh tham khảo: {context}

Refined Query:"""

        return f"""<|im_start|>system
{self.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
    
    def _extract_refined_query(self, response: str) -> str:
        """Extract refined query from model response"""
        if "Refined Query:" in response:
            pred = response.split("Refined Query:")[-1].strip()
        elif "<|im_start|>assistant" in response:
            pred = response.split("<|im_start|>assistant")[-1].strip()
        elif "assistant" in response.lower():
            parts = response.split("assistant")
            pred = parts[-1].strip() if len(parts) > 1 else response
        else:
            pred = response
        
        # Clean up common artifacts
        pred = pred.strip()
        if pred.startswith("\n"):
            pred = pred[1:].strip()
        if pred.endswith("<|im_end|>"):
            pred = pred[:-len("<|im_end|>")].strip()
        if pred.endswith("<|endoftext|>"):
            pred = pred[:-len("<|endoftext|>")].strip()
        
        return pred
    
    def refine(self, query: str, context: str) -> str:
        """
        Refine a query based on the given context.
        
        Args:
            query: The original query
            context: The context to use for refinement
            
        Returns:
            The refined query
        """
        # Truncate context if needed
        context = context[:self.max_context_length]
        
        prompt = self._create_prompt(query=query, context=context)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=self.repetition_penalty,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_refined_query(response)


def iteretgen_with_classifier_and_refiner(
    query: str,
    classifier: Union[ContextClassifier, ClassifierClient],
    refiner: Union[QueryRefiner, RefinerClient],
    max_iterations: int = 3,
    verbose: bool = True
) -> Dict:
    """
    IterRetGen with classifier-based context filtering and query refinement.
    
    Pipeline logic for each iteration:
    1. Search for contexts using current query
    2. Classify each context as Irrelevant/Relevant/Contain Answer
    3. If "Contain Answer" found: Stop and generate final answer
    4. If "Relevant" contexts exist: Use query refiner to get refined query for next iteration
    5. If no relevant contexts: Fallback to original iteretgen (LLM answer + concat)
    6. At max iterations: Generate final answer with all accumulated relevant contexts
    
    Args:
        query: Original query
        classifier: ContextClassifier instance
        refiner: QueryRefiner instance
        max_iterations: Maximum number of iterations
        verbose: Whether to print progress
        
    Returns:
        Dictionary with final answer, all generations, and detailed history
    """
    original_query = query
    current_query = query
    all_relevant_contexts = []
    all_contain_answer_contexts = []
    all_generations = []
    all_doc_ids = set()
    all_refined_queries = []
    classification_history = []
    fallback_iterations = []
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{max_iterations}")
            print(f"{'='*60}")
            print(f"Current query: {current_query[:150]}...")
        
        # Step 1: Search for contexts
        search_results = search(current_query)
        new_contexts = [doc['content'] for doc in search_results]
        
        if verbose:
            print(f"Retrieved {len(new_contexts)} contexts")
        
        # Collect doc_ids
        for doc in search_results:
            all_doc_ids.add(doc.get('doc_id', f'doc_{len(all_doc_ids)}'))
        
        # Step 2: Classify each context
        iteration_classifications = []
        found_answer = False
        iteration_relevant_contexts = []
        
        for i, context in enumerate(new_contexts):
            label, probs = classifier.classify(original_query, context)
            
            classification_info = {
                'context': context[:200] + "..." if len(context) > 200 else context,
                'full_context': context,
                'label': label,
                'probabilities': probs,
                'doc_id': search_results[i].get('doc_id', f'doc_{i}')
            }
            iteration_classifications.append(classification_info)
            
            if verbose:
                print(f"\n  Context {i+1}: {label} (confidence: {probs[label]:.3f})")
                print(f"    Preview: {context[:100]}...")
            
            # Collect contexts based on label
            if label == "Contain Answer":
                found_answer = True
                if context not in all_contain_answer_contexts:
                    all_contain_answer_contexts.append(context)
                # Also add to relevant for this iteration
                iteration_relevant_contexts.append(context)
            elif label == "Relevant":
                iteration_relevant_contexts.append(context)
                if context not in all_relevant_contexts:
                    all_relevant_contexts.append(context)
            # Irrelevant contexts are discarded
        
        classification_history.append({
            'iteration': iteration + 1,
            'query': current_query,
            'classifications': iteration_classifications,
            'found_answer': found_answer,
            'relevant_count': len(iteration_relevant_contexts)
        })
        
        # Step 3: Check if we found an answer
        if found_answer:
            if verbose:
                print(f"\n{'*'*60}")
                print("FOUND CONTEXT WITH ANSWER! Generating final response...")
                print(f"{'*'*60}")
            
            # Combine relevant and contain-answer contexts for final generation
            final_contexts = all_relevant_contexts + all_contain_answer_contexts
            
            if verbose:
                print(f"Using {len(final_contexts)} contexts for final answer")
                print(f"  - Relevant: {len(all_relevant_contexts)}")
                print(f"  - Contain Answer: {len(all_contain_answer_contexts)}")
            
            # Generate final answer
            prompt = get_prompt(original_query, final_contexts)
            final_answer = generate_answer(prompt)
            all_generations.append(final_answer)
            
            return {
                "final_answer": final_answer,
                "all_generations": all_generations,
                "final_contexts": final_contexts,
                "relevant_contexts": all_relevant_contexts,
                "contain_answer_contexts": all_contain_answer_contexts,
                "classification_history": classification_history,
                "refined_queries": all_refined_queries,
                "fallback_iterations": fallback_iterations,
                "stopped_early": True,
                "total_iterations": iteration + 1,
                "all_doc_ids": list(all_doc_ids)
            }
        
        # Step 4: No answer found - decide next action
        if iteration_relevant_contexts:
            # We have relevant contexts - use query refiner for each context
            if verbose:
                print(f"\nUsing query refiner with {len(iteration_relevant_contexts)} relevant contexts...")
            
            # Refine query with each relevant context separately
            refined_queries_this_iteration = []
            for ctx_idx, ctx in enumerate(iteration_relevant_contexts):
                refined_q = refiner.refine(current_query, ctx)
                refined_queries_this_iteration.append(refined_q)
                if verbose:
                    print(f"  Refined query {ctx_idx + 1}: {refined_q[:100]}...")
            
            # Concatenate all refined queries to make new query
            combined_refined_query = " ".join(refined_queries_this_iteration)
            
            all_refined_queries.append({
                'iteration': iteration + 1,
                'original_query': current_query,
                'refined_queries_per_context': refined_queries_this_iteration,
                'combined_refined_query': combined_refined_query,
                'context_count': len(iteration_relevant_contexts)
            })
            
            if verbose:
                print(f"\nCombined refined query: {combined_refined_query[:200]}...")
            
            # Update query for next iteration
            current_query = combined_refined_query
            
        else:
            # No relevant contexts - fallback to original iteretgen
            if verbose:
                print(f"\n{'!'*60}")
                print("NO RELEVANT CONTEXTS! Falling back to iteretgen...")
                print(f"{'!'*60}")
            
            fallback_iterations.append(iteration + 1)
            
            # Use all contexts (even irrelevant) for LLM generation
            all_current_contexts = [doc['content'] for doc in search_results]
            
            if all_current_contexts:
                prompt = get_prompt(original_query, all_current_contexts)
                current_generation = generate_answer(prompt)
                all_generations.append(current_generation)
                
                # Update query by concatenating with answer (original iteretgen style)
                current_query = f"{original_query} {current_generation}"
                
                if verbose:
                    print(f"Fallback generation: {current_generation[:200]}...")
            else:
                if verbose:
                    print("No contexts retrieved. Continuing with original query...")
                current_query = original_query
    
    # Max iterations reached
    if verbose:
        print(f"\n{'='*60}")
        print("Max iterations reached. Generating final answer...")
        print(f"{'='*60}")
    
    final_contexts = all_relevant_contexts + all_contain_answer_contexts
    
    if final_contexts:
        if verbose:
            print(f"Using {len(final_contexts)} accumulated contexts")
        prompt = get_prompt(original_query, final_contexts)
        final_answer = generate_answer(prompt)
    else:
        if verbose:
            print("No relevant contexts found. Generating answer without context...")
        final_answer = generate_answer(get_prompt(original_query, []))
    
    return {
        "final_answer": final_answer,
        "all_generations": all_generations,
        "final_contexts": final_contexts,
        "relevant_contexts": all_relevant_contexts,
        "contain_answer_contexts": all_contain_answer_contexts,
        "classification_history": classification_history,
        "refined_queries": all_refined_queries,
        "fallback_iterations": fallback_iterations,
        "stopped_early": False,
        "total_iterations": max_iterations,
        "all_doc_ids": list(all_doc_ids)
    }


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IterRetGen with Classifier and Query Refiner")
    parser.add_argument(
        "--classifier_checkpoint",
        type=str,
        default="../bge_classifier_full/best_model/checkpoint.pt",
        help="Path to classifier checkpoint"
    )
    parser.add_argument(
        "--refiner_model",
        type=str,
        default="../qwen_lora_query_refiner",
        help="Path to query refiner model"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Thủ tục đăng ký khai sinh cho trẻ em mới sinh như thế nào?",
        help="Query to process"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="Maximum number of iterations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load refiner model in 4-bit quantization"
    )
    
    args = parser.parse_args()
    
    # Initialize classifier
    print("Loading classifier...")
    classifier = ContextClassifier(
        checkpoint_path=args.classifier_checkpoint,
        device=args.device
    )
    
    # Initialize query refiner
    print("\nLoading query refiner...")
    refiner = QueryRefiner(
        model_path=args.refiner_model,
        load_in_4bit=args.load_in_4bit
    )
    
    # Run pipeline
    print(f"\nProcessing query: {args.query}")
    result = iteretgen_with_classifier_and_refiner(
        query=args.query,
        classifier=classifier,
        refiner=refiner,
        max_iterations=args.max_iterations,
        verbose=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nQuery: {args.query}")
    print(f"Stopped early: {result['stopped_early']}")
    print(f"Total iterations: {result['total_iterations']}")
    print(f"Relevant contexts: {len(result['relevant_contexts'])}")
    print(f"Contain Answer contexts: {len(result['contain_answer_contexts'])}")
    print(f"Refined queries used: {len(result['refined_queries'])}")
    print(f"Fallback iterations: {result['fallback_iterations']}")
    
    if result['refined_queries']:
        print("\nRefined Queries:")
        for rq in result['refined_queries']:
            print(f"  Iteration {rq['iteration']}: {rq.get('combined_refined_query', '')[:100]}...")
    
    print(f"\nFinal Answer:\n{result['final_answer']}")


# ============================================================================
# Evaluation Functions
# ============================================================================

def process_single_example(
    example: Dict[str, Any],
    classifier: Union[ContextClassifier, ClassifierClient],
    refiner: Union[QueryRefiner, RefinerClient],
    max_iterations: int = 3
) -> Dict[str, Any]:
    """
    Process a single evaluation example.
    
    Args:
        example: Dictionary with 'question', 'answer' fields
        classifier: Classifier instance (local or client)
        refiner: Refiner instance (local or client)
        max_iterations: Maximum iterations
        
    Returns:
        Dictionary with prediction, ground truth, and metrics
    """
    query = example['question']
    ground_truth = example.get('answer', '')
    
    try:
        result = iteretgen_with_classifier_and_refiner(
            query=query,
            classifier=classifier,
            refiner=refiner,
            max_iterations=max_iterations,
            verbose=False
        )
        predicted_answer = result['final_answer']
    except Exception as e:
        print(f"Error processing query: {query[:50]}... - {e}")
        predicted_answer = ""
    
    # Calculate metrics
    em_score, f1_score = evaluate_answer(predicted_answer, ground_truth)
    
    return {
        'question': query,
        'ground_truth': ground_truth,
        'predicted': predicted_answer,
        'em_score': em_score,
        'f1_score': f1_score,
        'result_details': result if 'result' in dir() else None
    }


def run_evaluation(
    data: List[Dict],
    classifier: Union[ContextClassifier, ClassifierClient],
    refiner: Union[QueryRefiner, RefinerClient],
    max_iterations: int = 3,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run evaluation on a dataset (single-threaded).
    
    Args:
        data: List of examples with 'question' and 'answer' fields
        classifier: Classifier instance
        refiner: Refiner instance
        max_iterations: Maximum iterations
        output_file: Optional path to save results
        
    Returns:
        Dictionary with evaluation metrics and results
    """
    results = []
    total_em = 0.0
    total_f1 = 0.0
    
    for example in tqdm(data, desc="Evaluating"):
        result = process_single_example(example, classifier, refiner, max_iterations)
        results.append(result)
        total_em += result['em_score']
        total_f1 += result['f1_score']
    
    n = len(data)
    evaluation_results = {
        'total_examples': n,
        'avg_em': total_em / n if n > 0 else 0.0,
        'avg_f1': total_f1 / n if n > 0 else 0.0,
        'results': results
    }
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
    
    return evaluation_results


def run_evaluation_parallel(
    data: List[Dict],
    classifier_url: str,
    refiner_url: str,
    max_iterations: int = 3,
    num_workers: int = 4,
    output_file: Optional[str] = None,
    llm_accuracy: bool = False
) -> Dict[str, Any]:
    """
    Run evaluation on a dataset with parallel processing.
    Uses HTTP clients to avoid CUDA OOM issues.
    
    Args:
        data: List of examples with 'question' and 'answer' fields
        classifier_url: URL of the classifier service
        refiner_url: URL of the refiner service
        max_iterations: Maximum iterations
        num_workers: Number of parallel workers
        output_file: Optional path to save results
        llm_accuracy: Whether to evaluate using LLM (Gemini) for accuracy
        
    Returns:
        Dictionary with evaluation metrics and results
    """
    results = [None] * len(data)
    
    def process_with_index(args):
        idx, example = args
        # Each thread creates its own HTTP client instances
        classifier_client = ClassifierClient(classifier_url)
        refiner_client = RefinerClient(refiner_url)
        
        result = process_single_example(
            example, classifier_client, refiner_client, max_iterations
        )
        return idx, result
    
    print(f"Running parallel evaluation with {num_workers} workers...")
    print(f"Classifier service: {classifier_url}")
    print(f"Refiner service: {refiner_url}")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_with_index, (i, ex)) for i, ex in enumerate(data)]
        
        for future in tqdm(as_completed(futures), total=len(data), desc="Evaluating"):
            idx, result = future.result()
            results[idx] = result
    
    # Calculate aggregate metrics
    total_em = sum(r['em_score'] for r in results)
    total_f1 = sum(r['f1_score'] for r in results)
    n = len(data)
    
    evaluation_results = {
        'total_examples': n,
        'avg_em': total_em / n if n > 0 else 0.0,
        'avg_f1': total_f1 / n if n > 0 else 0.0,
        'results': results
    }
    
    # Optional LLM-based accuracy evaluation
    if llm_accuracy:
        print("\nRunning LLM-based accuracy evaluation...")
        correct_count = 0
        for i, result in enumerate(tqdm(results, desc="LLM Accuracy")):
            accuracy_result = evaluate_with_llm(
                result['question'],
                result['ground_truth'],
                result['predicted']
            )
            results[i]['llm_accuracy'] = accuracy_result
            if accuracy_result.get('is_correct', False):
                correct_count += 1
        
        evaluation_results['llm_accuracy'] = correct_count / n if n > 0 else 0.0
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
    
    return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IterRetGen with Classifier and Query Refiner")
    
    # Model/Service configuration
    parser.add_argument(
        "--classifier_checkpoint",
        type=str,
        default="../bge_classifier_full/best_model/checkpoint.pt",
        help="Path to classifier checkpoint (for local mode)"
    )
    parser.add_argument(
        "--classifier_url",
        type=str,
        default=None,
        help="URL of classifier service (e.g., http://127.0.0.1:8001). If provided, uses service mode."
    )
    parser.add_argument(
        "--refiner_model",
        type=str,
        default="../qwen_lora_query_refiner",
        help="Path to query refiner model (for local mode)"
    )
    parser.add_argument(
        "--refiner_url",
        type=str,
        default=None,
        help="URL of refiner service (e.g., http://127.0.0.1:8002). If provided, uses service mode."
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Path to evaluation data JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (only used in service mode)"
    )
    parser.add_argument(
        "--llm_accuracy",
        action="store_true",
        help="Run LLM-based accuracy evaluation"
    )
    
    # Pipeline configuration
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to process (for testing)"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="Maximum number of iterations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu) for local mode"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load refiner model in 4-bit quantization (local mode only)"
    )
    
    args = parser.parse_args()
    
    # Determine mode (service vs local)
    use_classifier_service = args.classifier_url is not None
    use_refiner_service = args.refiner_url is not None
    
    # Both must be service or both must be local for parallel processing
    if use_classifier_service != use_refiner_service:
        print("WARNING: Mixed mode detected. For parallel processing, both classifier and refiner must use service mode.")
    
    # Initialize classifier
    if use_classifier_service:
        print(f"Using classifier service at {args.classifier_url}")
        classifier = ClassifierClient(args.classifier_url)
    else:
        print("Loading classifier locally...")
        classifier = ContextClassifier(
            checkpoint_path=args.classifier_checkpoint,
            device=args.device
        )
    
    # Initialize refiner
    if use_refiner_service:
        print(f"Using refiner service at {args.refiner_url}")
        refiner = RefinerClient(args.refiner_url)
    else:
        print("\nLoading query refiner locally...")
        from unsloth import FastLanguageModel
        refiner = QueryRefiner(
            model_path=args.refiner_model,
            load_in_4bit=args.load_in_4bit
        )
    
    # Run single query or evaluation
    if args.eval_data:
        # Evaluation mode
        print(f"\nLoading evaluation data from {args.eval_data}...")
        with open(args.eval_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} examples")
        
        # Use parallel evaluation only if both services are available
        if use_classifier_service and use_refiner_service and args.num_workers > 1:
            results = run_evaluation_parallel(
                data=data,
                classifier_url=args.classifier_url,
                refiner_url=args.refiner_url,
                max_iterations=args.max_iterations,
                num_workers=args.num_workers,
                output_file=args.output_file,
                llm_accuracy=args.llm_accuracy
            )
        else:
            results = run_evaluation(
                data=data,
                classifier=classifier,
                refiner=refiner,
                max_iterations=args.max_iterations,
                output_file=args.output_file
            )
            
            # Run LLM accuracy if requested
            if args.llm_accuracy:
                print("\nRunning LLM-based accuracy evaluation...")
                correct_count = 0
                for i, r in enumerate(tqdm(results['results'], desc="LLM Accuracy")):
                    accuracy_result = evaluate_with_llm(
                        r['question'],
                        r['ground_truth'],
                        r['predicted']
                    )
                    results['results'][i]['llm_accuracy'] = accuracy_result
                    if accuracy_result.get('is_correct', False):
                        correct_count += 1
                results['llm_accuracy'] = correct_count / len(data) if len(data) > 0 else 0.0
                
                # Update output file
                if args.output_file:
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total examples: {results['total_examples']}")
        print(f"Average EM: {results['avg_em']:.4f}")
        print(f"Average F1: {results['avg_f1']:.4f}")
        if 'llm_accuracy' in results:
            print(f"LLM Accuracy: {results['llm_accuracy']:.4f}")
    
    else:
        # Single query mode
        query = args.query or "Thủ tục đăng ký khai sinh cho trẻ em mới sinh như thế nào?"
        print(f"\nProcessing query: {query}")
        
        result = iteretgen_with_classifier_and_refiner(
            query=query,
            classifier=classifier,
            refiner=refiner,
            max_iterations=args.max_iterations,
            verbose=True
        )
        
        # Print results
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"\nQuery: {query}")
        print(f"Stopped early: {result['stopped_early']}")
        print(f"Total iterations: {result['total_iterations']}")
        print(f"Relevant contexts: {len(result['relevant_contexts'])}")
        print(f"Contain Answer contexts: {len(result['contain_answer_contexts'])}")
        print(f"Refined queries used: {len(result['refined_queries'])}")
        print(f"Fallback iterations: {result['fallback_iterations']}")
        
        if result['refined_queries']:
            print("\nRefined Queries:")
            for rq in result['refined_queries']:
                print(f"  Iteration {rq['iteration']}: {rq.get('combined_refined_query', '')[:100]}...")
        
        print(f"\nFinal Answer:\n{result['final_answer']}")
