"""
IterRetGen (Iterative Retrieval-Generation) Pipeline.

This module implements the iterative retrieval-augmented generation approach
where retrieval and generation are interleaved across multiple iterations.
Each iteration:
1. Retrieves relevant documents for the current query
2. Generates an answer based on accumulated contexts
3. Updates the query by appending the generated answer

Reference: Shao et al., "Enhancing Retrieval-Augmented Large Language Models 
with Iterative Retrieval-Generation Synergy"
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from typing import Dict, List, Any, Tuple, Optional

from google import genai
from google.genai import types
from tqdm import tqdm

from search import search

# Gemini client setup
client = genai.Client(api_key='AIzaSyAG9oVHdcaKR2kv5xAihHYj8yehQqsl3ok')
MODEL = 'gemini-2.5-flash-lite'


def get_prompt(query: str, context: List[str]) -> str:
    """
    Create a prompt for the LLM with the given query and context.
    
    Args:
        query: The question to answer
        context: List of context documents
        
    Returns:
        Formatted prompt string
    """
    context_str = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context)])
    return f"""Answer the question based on the context provided. 
Reason step-by-step before providing the final answer.

Context:
{context_str}

Question: {query}

Let's think step by step. Answer in Vietnamese.
"""


def generate_answer(prompt: str) -> str:
    """
    Generate an answer using the Gemini API.
    
    Args:
        prompt: The formatted prompt
        
    Returns:
        Generated answer text
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=types.Part.from_text(text=prompt),
        config=types.GenerateContentConfig(
            temperature=0,
            top_p=0.95,
            top_k=20,
        ),
    )
    return response.text.strip()


def iteretgen(
    query: str,
    max_iterations: int = 3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run the IterRetGen pipeline for a single query.
    
    Args:
        query: The original question
        max_iterations: Maximum number of retrieval-generation iterations
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing:
            - final_answer: The last generated answer
            - all_generations: List of all generated answers
            - final_contexts: All accumulated context documents
            - all_doc_ids: List of all retrieved document IDs
            - iteration_history: Detailed history of each iteration
            - total_iterations: Number of iterations performed
    """
    current_contexts = set()
    current_query = query
    all_generations = []
    all_doc_ids = set()
    iteration_history = []
    current_generation = ""
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n  Iteration {iteration + 1}/{max_iterations}")
            print(f"  Query: {current_query[:100]}...")
        
        # Step 1: Retrieve documents
        context_documents = search(current_query)
        new_contexts = [doc['content'] for doc in context_documents]
        new_doc_ids = [doc['doc_id'] for doc in context_documents]
        
        # Accumulate contexts and doc_ids
        current_contexts.update(new_contexts)
        all_doc_ids.update(new_doc_ids)
        
        if verbose:
            print(f"  Retrieved {len(new_contexts)} documents")
        
        # Step 2: Generate answer with all accumulated contexts
        prompt = get_prompt(query, list(current_contexts))
        current_generation = generate_answer(prompt)
        
        if verbose:
            print(f"  Generated: {current_generation[:100]}...")
        
        # Record iteration history
        iteration_history.append({
            'iteration': iteration + 1,
            'query': current_query,
            'retrieved_contexts': new_contexts,
            'retrieved_doc_ids': new_doc_ids,
            'generation': current_generation
        })
        
        # Step 3: Update query for next iteration
        current_query = f"{query} {current_generation}"
        all_generations.append(current_generation)
    
    return {
        "final_answer": current_generation,
        "all_generations": all_generations,
        "final_contexts": list(current_contexts),
        "all_doc_ids": list(all_doc_ids),
        "iteration_history": iteration_history,
        "total_iterations": max_iterations,
    }


def evaluate_answer(predicted: str, ground_truth: str, question: str = None) -> Dict[str, Any]:
    """
    Evaluate predicted answer against ground truth using EM, F1, and LLM accuracy.
    
    Args:
        predicted: The predicted answer
        ground_truth: The ground truth answer
        question: The original question (used for LLM accuracy evaluation)
        
    Returns:
        Dictionary with evaluation metrics (exact_match, f1, accuracy)
    """
    pred_lower = predicted.lower()
    gt_lower = ground_truth.lower()
    
    # Exact match (case-insensitive) - checks if ground truth is contained in prediction
    exact_match = gt_lower in pred_lower
    
    # Token-level F1 score
    pred_tokens = set(pred_lower.split())
    gt_tokens = set(gt_lower.split())
    
    if len(gt_tokens) > 0:
        recall = len(pred_tokens & gt_tokens) / len(gt_tokens)
    else:
        recall = 0.0
    
    if len(pred_tokens) > 0:
        precision = len(pred_tokens & gt_tokens) / len(pred_tokens)
    else:
        precision = 0.0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    # LLM-based accuracy evaluation using Gemini 2.5 Pro
    accuracy = evaluate_with_llm(predicted, ground_truth, question)
    
    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }


def evaluate_with_llm(predicted: str, ground_truth: str, question: str = None) -> bool:
    """
    Use Gemini 2.5 Pro to evaluate if the predicted answer is correct.
    
    Args:
        predicted: The predicted answer
        ground_truth: The ground truth answer
        question: The original question (optional, for context)
        
    Returns:
        Boolean indicating if the answer is considered correct
    """
    evaluation_prompt = f"""You are an expert evaluator. Your task is to determine if the predicted answer is semantically correct compared to the ground truth answer.

Question: {question if question else "N/A"}

Ground Truth Answer: {ground_truth}

Predicted Answer: {predicted}

Instructions:
1. The predicted answer does NOT need to be an exact match.
2. The predicted answer is correct if it conveys the same meaning or contains the correct information.
3. Minor differences in wording, formatting, or additional context are acceptable.
4. The key information from the ground truth must be present in the prediction.

Respond with ONLY "CORRECT" or "INCORRECT" (no explanation needed).
"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=types.Part.from_text(text=evaluation_prompt),
            config=types.GenerateContentConfig(
                temperature=0,
                top_p=1.0,
                top_k=1,
            ),
        )
        result = response.text.strip().upper()
        return "CORRECT" in result
    except Exception as e:
        print(f"LLM evaluation error: {e}")
        # Fallback to simple string matching if LLM fails
        return ground_truth.lower() in predicted.lower()


def process_single_example(
    example: Dict,
    idx: int,
    max_iterations: int = 3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Process a single example for parallel evaluation.
    
    Args:
        example: Test example with 'question' and 'answer' keys
        idx: Index of the example
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
        # Run IterRetGen
        result = iteretgen(
            query=query,
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
            "num_contexts": len(result['final_contexts']),
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
    max_iterations: int = 3,
    output_path: str = None,
    verbose: bool = False,
    num_workers: int = 1
) -> Dict[str, Any]:
    """
    Run evaluation on test data with optional parallel processing.
    
    Args:
        test_data: List of test examples with 'question' and 'answer' keys
        max_iterations: Maximum iterations for IterRetGen
        output_path: Path to save results (optional)
        verbose: Whether to print progress
        num_workers: Number of parallel workers (1 = sequential, >1 = parallel)
        
    Returns:
        Dictionary with evaluation results
    """
    results = []
    
    print(f"\nRunning IterRetGen evaluation on {len(test_data)} examples")
    print(f"Max iterations: {max_iterations}")
    print(f"Num workers: {num_workers}")
    print("=" * 60)
    
    if num_workers > 1:
        # Parallel processing with ThreadPoolExecutor
        results = run_evaluation_parallel(
            test_data=test_data,
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
    
    aggregate_metrics = {
        "total_examples": len(test_data),
        "successful_examples": num_results,
        "exact_match_rate": total_exact_match / num_results if num_results > 0 else 0,
        "avg_f1": total_f1 / num_results if num_results > 0 else 0,
        "accuracy_rate": total_accuracy / num_results if num_results > 0 else 0,
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
    print(f"Source Doc Retrieval Rate: {aggregate_metrics['source_retrieval_rate']:.4f}")
    print(f"Target Doc Retrieval Rate: {aggregate_metrics['target_retrieval_rate']:.4f}")
    
    # Sort results by id
    results = sorted(results, key=lambda x: x.get('id', 0))
    
    # Save results
    output = {
        "config": {
            "max_iterations": max_iterations,
            "model": MODEL,
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
    max_iterations: int = 3,
    num_workers: int = 4,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Run evaluation in parallel using ThreadPoolExecutor (similar to p_map from p-tqdm).
    
    Args:
        test_data: List of test examples
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
                            tqdm.write(f"[{idx}] EM: {em}, Acc: {acc}")
                            
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
        description="IterRetGen: Iterative Retrieval-Generation Pipeline"
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
        help="Number of parallel workers (1 = sequential, >1 = parallel processing)"
    )
    
    args = parser.parse_args()
    
    # Single query mode
    if args.query:
        print(f"Processing single query: {args.query}")
        result = iteretgen(
            query=args.query,
            max_iterations=args.max_iterations,
            verbose=True
        )
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Final Answer: {result['final_answer']}")
        print(f"Iterations: {result['total_iterations']}")
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
    output_filename = f"iteretgen_results_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Run evaluation
    run_evaluation(
        test_data=test_data,
        max_iterations=args.max_iterations,
        output_path=output_path,
        verbose=args.verbose,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
