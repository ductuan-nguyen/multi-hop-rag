import os
import json
import logging
import numpy as np
import sys
from pathlib import Path
from dotenv import load_dotenv

from config import IterRetGenConfig
from generator import LLMGenerator
from evaluator import Evaluator

load_dotenv()

def load_results_from_directory(result_dir: str):
    """Load all result JSON files from directory"""
    results = []
    ground_truths = []
    
    result_path = Path(result_dir)
    if not result_path.exists():
        print(f"Directory {result_dir} does not exist!")
        return results, ground_truths
    
    # Get all result files sorted by number
    result_files = sorted(
        result_path.glob("result_*.json"),
        key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[1].isdigit() else 0
    )
    
    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
                ground_truths.append(data.get('answer', ''))
                print(f"Loaded: {result_file.name} - Answer: {data.get('answer', 'N/A')}")
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    return results, ground_truths

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Windows terminals may default to non-UTF8 encodings; avoid crashing on Vietnamese prints
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    
    # Load results
    result_dir = 'result_iter_1'
    print(f"\n{'='*60}")
    print(f"Loading results from: {result_dir}")
    print(f"{'='*60}\n")
    
    results, ground_truths = load_results_from_directory(result_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"\nLoaded {len(results)} results\n")
    
    # Initialize evaluator
    evaluator = Evaluator()

    # Eval config (Recall@K)
    eval_config = IterRetGenConfig()
    recall_k = max(1, int(getattr(eval_config, "eval_recall_k", 3) or 3))
    
    # Initialize LLM generator for Acc† metric (optional)
    api_key = os.getenv("GEMINI_API_KEY")
    llm_generator = None
    if api_key:
        config = IterRetGenConfig()
        llm_generator = LLMGenerator(config, api_key)
        print("LLM Generator initialized for Acc† metric")
    else:
        print("Warning: GEMINI_API_KEY not found. Acc† metric will be skipped.")
    
    # Calculate metrics
    print(f"\n{'='*60}")
    print("Calculating Metrics...")
    print(f"{'='*60}\n")
    
    all_em = []
    all_f1 = []
    all_latency = []
    all_acc_dagger = []
    all_accuracy = []
    all_accuracy_f1_05 = []
    all_recall_at_k = []
    
    for idx, (result, gt) in enumerate(zip(results, ground_truths)):
        pred = result.get('final_answer', '')
        em = evaluator.exact_match_score(pred, gt)
        f1 = evaluator.f1_score(pred, gt)
        latency = result.get('latency', 0)
        
        # Accuracy (EM-based) + Accuracy@F1>=0.5 (more forgiving; useful for verbose answers)
        accuracy = evaluator.accuracy(pred, gt, use_f1_threshold=False)
        accuracy_f1_05 = evaluator.accuracy(pred, gt, use_f1_threshold=True, f1_threshold=0.5)
        all_accuracy.append(accuracy)
        all_accuracy_f1_05.append(accuracy_f1_05)
        
        # Calculate Recall@K for retrieval (max over retrieval steps)
        recall_at_k = 0.0
        source_doc_id = result.get('source_doc', {}).get('id', '')
        target_doc_id = result.get('target_doc', {}).get('id', '')
        retrieval_history = result.get('retrieval_history', [])
        
        if retrieval_history and (source_doc_id or target_doc_id):
            for retrieval in retrieval_history:
                retrieved_docs = retrieval.get('retrieved_docs', [])
                retrieved_doc_ids = [
                    doc.get('doc_id', '')
                    for doc in retrieved_docs
                    if doc.get('doc_id', '')
                ]
                if not retrieved_doc_ids:
                    continue
                recall_at_k = max(
                    recall_at_k,
                    evaluator.recall_at_k(retrieved_doc_ids, source_doc_id, target_doc_id, k=recall_k),
                )
        all_recall_at_k.append(recall_at_k)
        
        all_em.append(em)
        all_f1.append(f1)
        all_latency.append(latency)
        
        # Calculate Acc† if LLM is available
        acc_dagger = 0
        if llm_generator:
            try:
                question = result.get("multi_hop_question")
                acc_dagger = evaluator.model_based_accuracy(pred, gt, llm_generator, question=question)
            except Exception as e:
                print(f"Error calculating Acc† for result {idx}: {e}")
                acc_dagger = 0
        all_acc_dagger.append(acc_dagger)
        
        print(f"Result {idx}:")
        print(f"  Question: {result.get('multi_hop_question', 'N/A')[:80]}...")
        print(f"  Ground Truth: {gt}")
        print(f"  Prediction:   {pred}")
        print(f"  EM: {em}, F1: {f1:.3f}, Accuracy(EM): {accuracy}, Accuracy@F1>=0.5: {accuracy_f1_05}, Recall@{recall_k}: {recall_at_k:.3f}, Acc†: {acc_dagger}, Latency: {latency:.2f}s")
        print()
    
    # Calculate averages
    avg_em = np.mean(all_em) if all_em else 0
    avg_f1 = np.mean(all_f1) if all_f1 else 0
    avg_latency = np.mean(all_latency) if all_latency else 0
    avg_acc_dagger = np.mean(all_acc_dagger) if all_acc_dagger else 0
    avg_accuracy = np.mean(all_accuracy) if all_accuracy else 0
    avg_accuracy_f1_05 = np.mean(all_accuracy_f1_05) if all_accuracy_f1_05 else 0
    avg_recall_at_k = np.mean(all_recall_at_k) if all_recall_at_k else 0
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}\n")
    print(f"Total Results: {len(results)}")
    print(f"\nAnswer Task Metrics:")
    print(f"  Accuracy (EM-based):  {avg_accuracy:.3f} ({sum(all_accuracy)}/{len(all_accuracy)})")
    print(f"  Accuracy@F1>=0.5:     {avg_accuracy_f1_05:.3f} ({sum(all_accuracy_f1_05)}/{len(all_accuracy_f1_05)})")
    print(f"  Exact Match (EM):     {avg_em:.3f} ({sum(all_em)}/{len(all_em)})")
    print(f"  F1 Score:             {avg_f1:.3f}")
    print(f"  Acc† (Model-based):   {avg_acc_dagger:.3f} ({sum(all_acc_dagger)}/{len(all_acc_dagger)})")
    print(f"\nRetrieval Task Metrics:")
    print(f"  Recall@{recall_k}:             {avg_recall_at_k:.3f} ({sum(all_recall_at_k)}/{len(all_recall_at_k)})")
    print(f"\nPerformance:")
    print(f"  Average Latency:      {avg_latency:.2f}s")
    print(f"\n{'='*60}\n")
    
    # Save results to file
    output_file = 'evaluation_results.json'
    output_data = {
        "total_results": len(results),
        "metrics": {
            "answer_task": {
                "accuracy": avg_accuracy,
                "accuracy_f1_0_5": avg_accuracy_f1_05,
                "exact_match": avg_em,
                "f1_score": avg_f1,
                "acc_dagger": avg_acc_dagger
            },
            "retrieval_task": {
                f"recall_at_{recall_k}": avg_recall_at_k
            },
            "performance": {
                "avg_latency": avg_latency
            }
        },
        "detailed_results": [
            {
                "result_id": idx,
                "question": result.get('multi_hop_question', ''),
                "ground_truth": gt,
                "prediction": result.get('final_answer', ''),
                "answer_metrics": {
                    "accuracy": acc,
                    "accuracy_f1_0_5": acc_f1_05,
                    "em": em,
                    "f1": f1,
                    "acc_dagger": acc_d
                },
                "retrieval_metrics": {
                    f"recall_at_{recall_k}": rec_k
                },
                "latency": lat
            }
            for idx, (result, gt, acc, acc_f1_05, em, f1, acc_d, rec_k, lat) in enumerate(
                zip(
                    results,
                    ground_truths,
                    all_accuracy,
                    all_accuracy_f1_05,
                    all_em,
                    all_f1,
                    all_acc_dagger,
                    all_recall_at_k,
                    all_latency,
                )
            )
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
