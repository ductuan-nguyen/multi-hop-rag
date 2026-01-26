"""
IterRetGen with BGE-M3 Classifier Integration.

This module extends the iteretgen approach by using a trained classifier
to filter contexts based on their relevance to the query.
- Irrelevant: Discard
- Relevant: Keep for LLM generation
- Contain Answer: Stop iteration and generate final answer

The classifier can be used in two modes:
1. Local mode: Load the model directly (for single-threaded use)
2. Service mode: Call the classifier via HTTP API (for parallel processing)
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
from typing import List, Dict, Tuple, Any, Optional, Union

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
        """
        Classify a single query-context pair via HTTP API.
        
        Args:
            query: The question/query
            context: The context/document to classify
            
        Returns:
            Tuple of (label, probabilities_dict)
        """
        resp = requests.post(
            f"{self.service_url}/classify",
            json={"query": query, "context": context},
            timeout=30
        )
        resp.raise_for_status()
        result = resp.json()
        return result['label'], result['probabilities']
    
    def classify_batch(self, query: str, contexts: List[str]) -> List[Tuple[str, Dict[str, float]]]:
        """
        Classify multiple contexts for a single query via HTTP API.
        
        Args:
            query: The question/query
            contexts: List of contexts to classify
            
        Returns:
            List of (label, probabilities_dict) tuples
        """
        resp = requests.post(
            f"{self.service_url}/classify_batch",
            json={"query": query, "contexts": contexts},
            timeout=60
        )
        resp.raise_for_status()
        results = resp.json()['results']
        return [(r['label'], r['probabilities']) for r in results]


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
        
        Args:
            query: The question/query
            context: The context/document to classify
            
        Returns:
            Tuple of (label, probabilities_dict)
        """
        # Combine question and context
        text = f"Question: {query}\n\nContext: {context}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        
        # Get prediction
        pred_id = torch.argmax(probs).item()
        label = ID2LABEL[pred_id]
        
        probabilities = {
            ID2LABEL[i]: probs[i].item()
            for i in range(len(ID2LABEL))
        }
        
        return label, probabilities
    
    def classify_batch(self, query: str, contexts: List[str]) -> List[Tuple[str, Dict[str, float]]]:
        """
        Classify multiple contexts for a single query.
        
        Args:
            query: The question/query
            contexts: List of contexts to classify
            
        Returns:
            List of (label, probabilities_dict) tuples
        """
        results = []
        for context in contexts:
            label, probs = self.classify(query, context)
            results.append((label, probs))
        return results


def iteretgen_with_classifier(
    query: str,
    classifier: Union[ContextClassifier, ClassifierClient],
    max_iterations: int = 3,
    verbose: bool = True
) -> Dict:
    """
    IterRetGen with classifier-based context filtering.
    
    For each iteration:
    1. Search for contexts
    2. Classify each context as Irrelevant/Relevant/Contain Answer
    3. If any context is "Contain Answer", stop and generate final answer
    4. Otherwise, use "Relevant" contexts to generate intermediate answer
    5. Continue to next iteration with updated query
    
    Args:
        query: Original query
        classifier: ContextClassifier or ClassifierClient instance
        max_iterations: Maximum number of iterations
        verbose: Whether to print progress
        
    Returns:
        Dictionary with final answer, all generations, and classification details
    """
    current_query = query
    all_relevant_contexts = []  # Accumulate relevant contexts across iterations
    all_contain_answer_contexts = []
    all_generations = []
    all_doc_ids = set()
    classification_history = []
    fallback_iterations = []
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{max_iterations}")
            print(f"{'='*60}")
            print(f"Current query: {current_query[:100]}...")
        
        # Step 1: Search for contexts
        search_results = search(current_query)
        new_contexts = [doc['content'] for doc in search_results]
        
        if verbose:
            print(f"Retrieved {len(new_contexts)} contexts")
        
        # Step 2: Classify each context
        iteration_classifications = []
        found_answer = False
        
        # Collect doc_ids from this iteration
        for doc in search_results:
            all_doc_ids.add(doc.get('doc_id', f'doc_{len(all_doc_ids)}'))
        
        for i, context in enumerate(new_contexts):
            label, probs = classifier.classify(query, context)  # Use original query for classification
            
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
            elif label == "Relevant":
                if context not in all_relevant_contexts:
                    all_relevant_contexts.append(context)
            # Irrelevant contexts are discarded
        
        classification_history.append({
            'iteration': iteration + 1,
            'query': current_query,
            'classifications': iteration_classifications
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
            prompt = get_prompt(query, final_contexts)
            final_answer = generate_answer(prompt)
            all_generations.append(final_answer)
            
            return {
                "final_answer": final_answer,
                "all_generations": all_generations,
                "final_contexts": final_contexts,
                "relevant_contexts": all_relevant_contexts,
                "contain_answer_contexts": all_contain_answer_contexts,
                "classification_history": classification_history,
                "fallback_iterations": fallback_iterations,
                "stopped_early": True,
                "total_iterations": iteration + 1,
                "all_doc_ids": list(all_doc_ids)
            }
        
        # Step 4: No answer found - generate intermediate answer with relevant contexts
        if all_relevant_contexts:
            if verbose:
                print(f"\nGenerating intermediate answer with {len(all_relevant_contexts)} relevant contexts...")
            
            prompt = get_prompt(query, all_relevant_contexts)
            current_generation = generate_answer(prompt)
            all_generations.append(current_generation)
            
            # Update query for next iteration
            current_query = f"{query} {current_generation}"
            
            if verbose:
                print(f"Intermediate answer: {current_generation[:200]}...")
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
                prompt = get_prompt(query, all_current_contexts)
                current_generation = generate_answer(prompt)
                all_generations.append(current_generation)
                
                # Update query by concatenating with answer (original iteretgen style)
                current_query = f"{query} {current_generation}"
                
                if verbose:
                    print(f"Fallback generation: {current_generation[:200]}...")
            else:
                if verbose:
                    print("No contexts retrieved. Continuing with original query...")
                current_query = query
    
    # Max iterations reached
    if verbose:
        print(f"\n{'='*60}")
        print("Max iterations reached. Generating final answer...")
        print(f"{'='*60}")
    
    final_contexts = all_relevant_contexts + all_contain_answer_contexts
    
    if final_contexts:
        prompt = get_prompt(query, final_contexts)
        final_answer = generate_answer(prompt)
    else:
        final_answer = "Unable to find relevant information to answer the question."
    
    return {
        "final_answer": final_answer,
        "all_generations": all_generations,
        "final_contexts": final_contexts,
        "relevant_contexts": all_relevant_contexts,
        "contain_answer_contexts": all_contain_answer_contexts,
        "classification_history": classification_history,
        "fallback_iterations": fallback_iterations,
        "stopped_early": False,
        "total_iterations": max_iterations,
        "all_doc_ids": list(all_doc_ids)
    }


def process_single_example(
    example: Dict,
    idx: int,
    classifier: Union[ContextClassifier, ClassifierClient],
    max_iterations: int = 3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Process a single example for parallel evaluation.
    
    Args:
        example: Test example with 'question' and 'answer' keys
        idx: Index of the example
        classifier: ContextClassifier or ClassifierClient instance
        max_iterations: Maximum iterations for IterRetGen
        verbose: Whether to print progress
        
    Returns:
        Dictionary with result for this example
    """
    query = example.get('question', '')
    ground_truth = example.get('answer', '')
    source_doc_id = example.get('source_doc_id', '')
    target_doc_id = example.get('target_doc_id', '')
    
    if not query:
        return None
    
    try:
        # Run IterRetGen with classifier
        result = iteretgen_with_classifier(
            query=query,
            classifier=classifier,
            max_iterations=max_iterations,
            verbose=verbose
        )
        
        # Evaluate with all three metrics (EM, F1, Accuracy)
        eval_metrics = evaluate_answer(
            result['final_answer'], 
            ground_truth,
            question=query
        )
        
        # Check if source/target docs were retrieved
        retrieved_doc_ids = result['all_doc_ids']
        source_retrieved = source_doc_id in retrieved_doc_ids if source_doc_id else None
        target_retrieved = target_doc_id in retrieved_doc_ids if target_doc_id else None
        
        # Record result
        return {
            "id": idx,
            "question": query,
            "ground_truth": ground_truth,
            "predicted_answer": result['final_answer'],
            "all_generations": result['all_generations'],
            "all_doc_ids": result['all_doc_ids'],
            "total_iterations": result['total_iterations'],
            "stopped_early": result['stopped_early'],
            "num_relevant_contexts": len(result['relevant_contexts']),
            "num_contain_answer_contexts": len(result['contain_answer_contexts']),
            "fallback_iterations": result['fallback_iterations'],
            "source_doc_id": source_doc_id,
            "target_doc_id": target_doc_id,
            "source_retrieved": source_retrieved,
            "target_retrieved": target_retrieved,
            "metrics": eval_metrics
        }
        
    except Exception as e:
        return {
            "id": idx,
            "question": query,
            "ground_truth": ground_truth,
            "error": str(e)
        }


def run_evaluation(
    test_data: List[Dict],
    classifier: Union[ContextClassifier, ClassifierClient],
    max_iterations: int = 3,
    output_path: str = None,
    verbose: bool = False,
    num_workers: int = 1
) -> Dict[str, Any]:
    """
    Run evaluation on test data with optional parallel processing.
    
    Args:
        test_data: List of test examples with 'question' and 'answer' keys
        classifier: ContextClassifier or ClassifierClient instance
        max_iterations: Maximum iterations for IterRetGen
        output_path: Path to save results (optional)
        verbose: Whether to print progress
        num_workers: Number of parallel workers (1 = sequential, >1 = parallel)
        
    Returns:
        Dictionary with evaluation results
    """
    results = []
    
    print(f"\nRunning IterRetGen with Classifier evaluation on {len(test_data)} examples")
    print(f"Max iterations: {max_iterations}")
    print(f"Num workers: {num_workers}")
    print("=" * 60)
    
    if num_workers > 1:
        # Parallel processing with ThreadPoolExecutor
        results = run_evaluation_parallel(
            test_data=test_data,
            classifier=classifier,
            max_iterations=max_iterations,
            num_workers=num_workers,
            verbose=verbose
        )
    else:
        # Sequential processing
        for idx, example in enumerate(tqdm(test_data, desc="Evaluating")):
            result = process_single_example(
                example=example,
                idx=idx,
                classifier=classifier,
                max_iterations=max_iterations,
                verbose=verbose
            )
            if result is not None:
                results.append(result)
                
                if verbose:
                    query = result.get('question', '')[:80]
                    gt = result.get('ground_truth', '')
                    pred = result.get('predicted_answer', '')[:100]
                    metrics = result.get('metrics', {})
                    print(f"\n[{idx+1}] Q: {query}...")
                    print(f"    GT: {gt}")
                    print(f"    Pred: {pred}...")
                    if metrics:
                        print(f"    EM: {metrics.get('exact_match')}, F1: {metrics.get('f1', 0):.3f}, Acc: {metrics.get('accuracy')}")
    
    # Calculate aggregate metrics
    successful_results = [r for r in results if 'error' not in r]
    num_results = len(successful_results)
    
    total_exact_match = sum(int(r['metrics']['exact_match']) for r in successful_results)
    total_f1 = sum(r['metrics']['f1'] for r in successful_results)
    total_accuracy = sum(int(r['metrics']['accuracy']) for r in successful_results)
    total_stopped_early = sum(int(r['stopped_early']) for r in successful_results)
    
    aggregate_metrics = {
        "total_examples": len(test_data),
        "successful_examples": num_results,
        "exact_match_rate": total_exact_match / num_results if num_results > 0 else 0,
        "avg_f1": total_f1 / num_results if num_results > 0 else 0,
        "accuracy_rate": total_accuracy / num_results if num_results > 0 else 0,
        "early_stopping_rate": total_stopped_early / num_results if num_results > 0 else 0,
        "source_retrieval_rate": sum(1 for r in successful_results if r.get('source_retrieved')) / num_results if num_results > 0 else 0,
        "target_retrieval_rate": sum(1 for r in successful_results if r.get('target_retrieved')) / num_results if num_results > 0 else 0,
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total examples: {aggregate_metrics['total_examples']}")
    print(f"Successful: {aggregate_metrics['successful_examples']}")
    print(f"Exact Match Rate: {aggregate_metrics['exact_match_rate']:.4f}")
    print(f"Average F1: {aggregate_metrics['avg_f1']:.4f}")
    print(f"Accuracy Rate (LLM): {aggregate_metrics['accuracy_rate']:.4f}")
    print(f"Early Stopping Rate: {aggregate_metrics['early_stopping_rate']:.4f}")
    print(f"Source Doc Retrieval Rate: {aggregate_metrics['source_retrieval_rate']:.4f}")
    print(f"Target Doc Retrieval Rate: {aggregate_metrics['target_retrieval_rate']:.4f}")
    
    # Sort results by id
    results = sorted(results, key=lambda x: x.get('id', 0))
    
    # Save results
    output = {
        "config": {
            "max_iterations": max_iterations,
            "model": MODEL,
            "pipeline": "iteretgen_with_classifier",
            "timestamp": datetime.now().isoformat()
        },
        "aggregate_metrics": aggregate_metrics,
        "results": results
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return output


def run_evaluation_parallel(
    test_data: List[Dict],
    classifier: Union[ContextClassifier, ClassifierClient],
    max_iterations: int = 3,
    num_workers: int = 4,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Run evaluation in parallel using ThreadPoolExecutor (similar to p_map from p-tqdm).
    
    Args:
        test_data: List of test examples
        classifier: ContextClassifier or ClassifierClient instance
        max_iterations: Maximum iterations for IterRetGen
        num_workers: Number of parallel workers
        verbose: Whether to print progress
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Create list of (example, idx) tuples
    indexed_data = [(example, idx) for idx, example in enumerate(test_data)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                process_single_example,
                example=example,
                idx=idx,
                classifier=classifier,
                max_iterations=max_iterations,
                verbose=False  # Disable verbose in parallel mode to avoid output mixing
            ): idx
            for example, idx in indexed_data
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(test_data), desc="Evaluating (parallel)") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        
                        # Show brief progress if verbose
                        if verbose and 'metrics' in result:
                            metrics = result['metrics']
                            em = metrics.get('exact_match', False)
                            acc = metrics.get('accuracy', False)
                            stopped = result.get('stopped_early', False)
                            tqdm.write(f"[{idx}] EM: {em}, Acc: {acc}, Early Stop: {stopped}")
                            
                except Exception as e:
                    tqdm.write(f"Error in example {idx}: {e}")
                    results.append({
                        "id": idx,
                        "question": test_data[idx].get('question', ''),
                        "ground_truth": test_data[idx].get('answer', ''),
                        "error": str(e)
                    })
                
                pbar.update(1)
    
    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="IterRetGen with Classifier: Iterative Retrieval-Generation Pipeline with Context Classification"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../bge_classifier_full/best_model/checkpoint.pt",
        help="Path to classifier checkpoint (for local mode)"
    )
    parser.add_argument(
        "--classifier_url",
        type=str,
        default=None,
        help="URL of classifier service (for service mode, e.g., http://127.0.0.1:8001). "
             "Use this for parallel processing to avoid CUDA OOM."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="/home3/vietld/master/web_mining/playground/multihop_test_data.json",
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home3/vietld/master/web_mining/playground/pipeline_results",
        help="Directory to save output results"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="Maximum number of retrieval-generation iterations"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to process (if not using test file)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test examples to process"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (1 = sequential, >1 = parallel processing). "
             "For num_workers > 1, use --classifier_url to avoid CUDA OOM."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for classifier in local mode (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Initialize classifier (service mode or local mode)
    if args.classifier_url:
        # Service mode - use HTTP client (recommended for parallel processing)
        print(f"Using classifier service at: {args.classifier_url}")
        classifier = ClassifierClient(service_url=args.classifier_url)
    else:
        # Local mode - load model directly
        if args.num_workers > 1:
            print("WARNING: Using local classifier with num_workers > 1 may cause CUDA OOM!")
            print("Consider using --classifier_url with the classifier service instead.")
            print("Start the service with: python classifier_service.py --checkpoint <path>")
            print()
        print("Loading classifier locally...")
        classifier = ContextClassifier(
            checkpoint_path=args.checkpoint,
            device=args.device
        )
    
    # Single query mode
    if args.query:
        print(f"Processing single query: {args.query}")
        result = iteretgen_with_classifier(
            query=args.query,
            classifier=classifier,
            max_iterations=args.max_iterations,
            verbose=True
        )
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Final Answer: {result['final_answer']}")
        print(f"Stopped early: {result['stopped_early']}")
        print(f"Total iterations: {result['total_iterations']}")
        print(f"Relevant contexts: {len(result['relevant_contexts'])}")
        print(f"Contain Answer contexts: {len(result['contain_answer_contexts'])}")
        print(f"Documents retrieved: {len(result['all_doc_ids'])}")
        return
    
    # Batch evaluation mode
    print(f"Loading test data from: {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if args.limit:
        test_data = test_data[:args.limit]
        print(f"Limited to {args.limit} examples")
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"iteretgen_classifier_results_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Run evaluation
    run_evaluation(
        test_data=test_data,
        classifier=classifier,
        max_iterations=args.max_iterations,
        output_path=output_path,
        verbose=args.verbose,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
