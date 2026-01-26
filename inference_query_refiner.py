"""
Inference script for fine-tuned Qwen LoRA model for query refinement.
Generates refined queries and calculates BERTScore metrics.
"""

import json
import torch
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm

from unsloth import FastLanguageModel
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'inference_query_refiner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Bạn là một chuyên gia tinh chỉnh câu hỏi. Nhiệm vụ của bạn là loại bỏ các thông tin nhiễu, không liên quan trong câu hỏi đầu vào để tạo ra một câu hỏi rõ ràng, ngắn gọn và chính xác hơn.

Quy tắc:
1. Giữ nguyên ý nghĩa cốt lõi của câu hỏi
2. Loại bỏ các thông tin dư thừa, không cần thiết
3. Đảm bảo câu hỏi tinh chỉnh vẫn có thể trả lời được dựa trên ngữ cảnh
4. Không thêm thông tin mới không có trong câu hỏi gốc

Chỉ trả lời với câu hỏi đã được tinh chỉnh, không giải thích."""


def create_prompt(query: str, context: str, label: Optional[str] = None) -> str:
    """Create prompt for the query refinement task"""
    user_prompt = f"""Câu hỏi gốc: {query}

Ngữ cảnh tham khảo: {context}

Refined Query:"""

    if label:
        return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
{label}<|im_end|>"""
    else:
        return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""


def load_model(model_path: str, max_seq_length: int = 8192, load_in_4bit: bool = True):
    """Load the fine-tuned model"""
    logger.info(f"Loading model from {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    
    # Enable faster inference
    FastLanguageModel.for_inference(model)
    
    logger.info("Model loaded successfully")
    return model, tokenizer


def load_data(data_path: str) -> List[Dict]:
    """Load evaluation data"""
    logger.info(f"Loading data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples")
    return data


def extract_refined_query(response: str) -> str:
    """Extract refined query from model response"""
    # Try different extraction strategies
    if "Refined Query:" in response:
        pred = response.split("Refined Query:")[-1].strip()
    elif "<|im_start|>assistant" in response:
        pred = response.split("<|im_start|>assistant")[-1].strip()
    elif "assistant" in response.lower():
        # Find the last occurrence of assistant marker
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


def predict_single(
    model, 
    tokenizer, 
    query: str, 
    context: str,
    max_context_length: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    repetition_penalty: float = 1.2
) -> str:
    """Predict refined query for a single sample"""
    # Truncate context
    context = context[:max_context_length]
    
    prompt = create_prompt(query=query, context=context, label=None)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_refined_query(response)


def predict_batch(
    model,
    tokenizer,
    data: List[Dict],
    batch_size: int = 4,
    max_context_length: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    repetition_penalty: float = 1.2,
    show_progress: bool = True
) -> List[str]:
    """
    Predict refined queries for a batch of samples.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        data: List of samples with 'query' and 'context' keys
        batch_size: Number of samples to process in each batch
        max_context_length: Maximum context length
        max_new_tokens: Maximum new tokens for generation
        temperature: Temperature for generation
        repetition_penalty: Penalty for repeating tokens (>1.0 reduces repetition)
        show_progress: Whether to show progress bar
    
    Returns:
        List of predicted refined queries
    """
    predictions = []
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Create batches
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    iterator = range(num_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Predicting", total=num_batches)
    
    for batch_idx in iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(data))
        batch_data = data[start_idx:end_idx]
        
        # Create prompts for the batch
        prompts = [
            create_prompt(
                query=item['query'],
                context=item['context'][:max_context_length],
                label=None
            )
            for item in batch_data
        ]
        
        # Tokenize batch with padding
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(model.device)
        
        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
            )
        
        # Decode and extract refined queries
        for output in outputs:
            response = tokenizer.decode(output, skip_special_tokens=True)
            pred = extract_refined_query(response)
            predictions.append(pred)
    
    return predictions


def calculate_bertscore(predictions: List[str], references: List[str], lang: str = "vi") -> Dict:
    """Calculate BERTScore metrics"""
    try:
        from bert_score import score as bert_score
    except ImportError:
        logger.error("bert-score not installed. Run: pip install bert-score")
        return None
    
    logger.info("Computing BERTScore...")
    P, R, F1 = bert_score(predictions, references, lang=lang, verbose=True)
    
    metrics = {
        'precision': {
            'mean': P.mean().item(),
            'std': P.std().item(),
            'min': P.min().item(),
            'max': P.max().item(),
        },
        'recall': {
            'mean': R.mean().item(),
            'std': R.std().item(),
            'min': R.min().item(),
            'max': R.max().item(),
        },
        'f1': {
            'mean': F1.mean().item(),
            'std': F1.std().item(),
            'min': F1.min().item(),
            'max': F1.max().item(),
        },
        'per_sample': {
            'precision': P.tolist(),
            'recall': R.tolist(),
            'f1': F1.tolist(),
        }
    }
    
    return metrics


def calculate_additional_metrics(predictions: List[str], references: List[str]) -> Dict:
    """Calculate additional text similarity metrics"""
    metrics = {}
    
    # Length statistics
    pred_lengths = [len(p) for p in predictions]
    ref_lengths = [len(r) for r in references]
    
    metrics['length_stats'] = {
        'prediction': {
            'mean': np.mean(pred_lengths),
            'std': np.std(pred_lengths),
            'min': min(pred_lengths),
            'max': max(pred_lengths),
        },
        'reference': {
            'mean': np.mean(ref_lengths),
            'std': np.std(ref_lengths),
            'min': min(ref_lengths),
            'max': max(ref_lengths),
        }
    }
    
    # Exact match rate
    exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    metrics['exact_match_rate'] = exact_matches / len(predictions)
    
    # Character-level similarity (simple approach)
    def char_similarity(pred, ref):
        pred_chars = set(pred)
        ref_chars = set(ref)
        intersection = pred_chars & ref_chars
        union = pred_chars | ref_chars
        return len(intersection) / len(union) if union else 0
    
    char_similarities = [char_similarity(p, r) for p, r in zip(predictions, references)]
    metrics['char_similarity'] = {
        'mean': np.mean(char_similarities),
        'std': np.std(char_similarities),
    }
    
    return metrics


def print_metrics(bertscore_metrics: Dict, additional_metrics: Dict = None):
    """Print metrics in a formatted way"""
    logger.info("=" * 60)
    logger.info("BERTSCORE RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\nPrecision:")
    logger.info(f"  Mean: {bertscore_metrics['precision']['mean']:.4f} (±{bertscore_metrics['precision']['std']:.4f})")
    logger.info(f"  Range: [{bertscore_metrics['precision']['min']:.4f}, {bertscore_metrics['precision']['max']:.4f}]")
    
    logger.info(f"\nRecall:")
    logger.info(f"  Mean: {bertscore_metrics['recall']['mean']:.4f} (±{bertscore_metrics['recall']['std']:.4f})")
    logger.info(f"  Range: [{bertscore_metrics['recall']['min']:.4f}, {bertscore_metrics['recall']['max']:.4f}]")
    
    logger.info(f"\nF1 Score:")
    logger.info(f"  Mean: {bertscore_metrics['f1']['mean']:.4f} (±{bertscore_metrics['f1']['std']:.4f})")
    logger.info(f"  Range: [{bertscore_metrics['f1']['min']:.4f}, {bertscore_metrics['f1']['max']:.4f}]")
    
    if additional_metrics:
        logger.info("\n" + "-" * 60)
        logger.info("ADDITIONAL METRICS")
        logger.info("-" * 60)
        
        logger.info(f"\nExact Match Rate: {additional_metrics['exact_match_rate']:.4f}")
        logger.info(f"Character Similarity: {additional_metrics['char_similarity']['mean']:.4f} (±{additional_metrics['char_similarity']['std']:.4f})")
        
        logger.info(f"\nLength Statistics:")
        logger.info(f"  Predictions - Mean: {additional_metrics['length_stats']['prediction']['mean']:.1f}, "
                   f"Range: [{additional_metrics['length_stats']['prediction']['min']}, {additional_metrics['length_stats']['prediction']['max']}]")
        logger.info(f"  References - Mean: {additional_metrics['length_stats']['reference']['mean']:.1f}, "
                   f"Range: [{additional_metrics['length_stats']['reference']['min']}, {additional_metrics['length_stats']['reference']['max']}]")
    
    logger.info("\n" + "=" * 60)


def save_results(
    data: List[Dict],
    predictions: List[str],
    bertscore_metrics: Dict,
    additional_metrics: Dict,
    output_dir: str
):
    """Save predictions and metrics to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed predictions
    predictions_data = []
    for i, (item, pred) in enumerate(zip(data, predictions)):
        predictions_data.append({
            'index': i,
            'original_query': item['query'],
            'context': item['context'][:500] + "..." if len(item['context']) > 500 else item['context'],
            'reference': item.get('label', 'N/A'),
            'prediction': pred,
            'bertscore_f1': bertscore_metrics['per_sample']['f1'][i] if bertscore_metrics else None,
            'bertscore_precision': bertscore_metrics['per_sample']['precision'][i] if bertscore_metrics else None,
            'bertscore_recall': bertscore_metrics['per_sample']['recall'][i] if bertscore_metrics else None,
        })
    
    predictions_file = output_path / f"predictions_{timestamp}.json"
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Predictions saved to {predictions_file}")
    
    # Save metrics summary
    metrics_summary = {
        'bertscore': {
            'precision': bertscore_metrics['precision'],
            'recall': bertscore_metrics['recall'],
            'f1': bertscore_metrics['f1'],
        } if bertscore_metrics else None,
        'additional': additional_metrics,
        'num_samples': len(predictions),
    }
    
    metrics_file = output_path / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Save detailed report as text
    report_file = output_path / f"report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("QUERY REFINEMENT EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total samples: {len(predictions)}\n\n")
        
        if bertscore_metrics:
            f.write("BERTSCORE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Precision: {bertscore_metrics['precision']['mean']:.4f} (±{bertscore_metrics['precision']['std']:.4f})\n")
            f.write(f"Recall: {bertscore_metrics['recall']['mean']:.4f} (±{bertscore_metrics['recall']['std']:.4f})\n")
            f.write(f"F1: {bertscore_metrics['f1']['mean']:.4f} (±{bertscore_metrics['f1']['std']:.4f})\n\n")
        
        if additional_metrics:
            f.write("ADDITIONAL METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Exact Match Rate: {additional_metrics['exact_match_rate']:.4f}\n")
            f.write(f"Character Similarity: {additional_metrics['char_similarity']['mean']:.4f}\n\n")
        
        f.write("\nSAMPLE PREDICTIONS\n")
        f.write("=" * 60 + "\n\n")
        
        # Sort by F1 score and show best and worst
        if bertscore_metrics:
            scores = list(enumerate(bertscore_metrics['per_sample']['f1']))
            scores.sort(key=lambda x: x[1], reverse=True)
            
            f.write("TOP 5 PREDICTIONS (by BERTScore F1)\n")
            f.write("-" * 40 + "\n")
            for idx, score in scores[:5]:
                f.write(f"\n[Sample {idx}] BERTScore F1: {score:.4f}\n")
                f.write(f"Original Query: {data[idx]['query'][:200]}...\n")
                f.write(f"Reference: {data[idx].get('label', 'N/A')}\n")
                f.write(f"Prediction: {predictions[idx]}\n")
            
            f.write("\n\nBOTTOM 5 PREDICTIONS (by BERTScore F1)\n")
            f.write("-" * 40 + "\n")
            for idx, score in scores[-5:]:
                f.write(f"\n[Sample {idx}] BERTScore F1: {score:.4f}\n")
                f.write(f"Original Query: {data[idx]['query'][:200]}...\n")
                f.write(f"Reference: {data[idx].get('label', 'N/A')}\n")
                f.write(f"Prediction: {predictions[idx]}\n")
    
    logger.info(f"Report saved to {report_file}")
    
    return predictions_file, metrics_file, report_file


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for query refinement model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./qwen_lora_query_refiner",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="query_refiner_eval.json",
        help="Path to the evaluation data JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results_query_refiner",
        help="Directory to save results"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8192,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=2048,
        help="Maximum context length (will be truncated)"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit quantization"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition penalty for generation (>1.0 reduces repetition, 1.0 = no penalty)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save results to files"
    )
    parser.add_argument(
        "--skip_bertscore",
        action="store_true",
        help="Skip BERTScore calculation (faster inference)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting Inference for Query Refinement")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Max context length: {args.max_context_length}")
    
    # Load model
    model, tokenizer = load_model(
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit
    )
    
    # Load data
    data = load_data(args.data_path)
    
    # Optionally limit number of samples
    if args.num_samples is not None:
        data = data[:args.num_samples]
        logger.info(f"Limited to {len(data)} samples")
    
    # Get ground truth labels
    references = [item['label'] for item in data]
    
    # Log some statistics
    logger.info("\nData statistics:")
    query_lengths = [len(item['query']) for item in data]
    ref_lengths = [len(item['label']) for item in data]
    logger.info(f"  Query length - avg: {np.mean(query_lengths):.1f}, max: {max(query_lengths)}")
    logger.info(f"  Reference length - avg: {np.mean(ref_lengths):.1f}, max: {max(ref_lengths)}")
    
    # Run predictions
    logger.info("\nRunning predictions...")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Repetition penalty: {args.repetition_penalty}")
    predictions = predict_batch(
        model=model,
        tokenizer=tokenizer,
        data=data,
        batch_size=args.batch_size,
        max_context_length=args.max_context_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        show_progress=True
    )
    
    # Calculate metrics
    logger.info("\nCalculating metrics...")
    
    bertscore_metrics = None
    if not args.skip_bertscore:
        bertscore_metrics = calculate_bertscore(predictions, references, lang="vi")
    
    additional_metrics = calculate_additional_metrics(predictions, references)
    
    # Print metrics
    if bertscore_metrics:
        print_metrics(bertscore_metrics, additional_metrics)
    else:
        logger.info("BERTScore skipped. Additional metrics:")
        logger.info(f"  Exact Match Rate: {additional_metrics['exact_match_rate']:.4f}")
        logger.info(f"  Character Similarity: {additional_metrics['char_similarity']['mean']:.4f}")
    
    # Show sample predictions
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE PREDICTIONS")
    logger.info("=" * 60)
    for i in range(min(5, len(predictions))):
        logger.info(f"\n--- Example {i+1} ---")
        logger.info(f"Original Query: {data[i]['query'][:150]}...")
        logger.info(f"Reference: {references[i]}")
        logger.info(f"Prediction: {predictions[i]}")
        if bertscore_metrics:
            logger.info(f"BERTScore F1: {bertscore_metrics['per_sample']['f1'][i]:.4f}")
    
    # Save results
    if not args.no_save:
        save_results(
            data=data,
            predictions=predictions,
            bertscore_metrics=bertscore_metrics,
            additional_metrics=additional_metrics,
            output_dir=args.output_dir
        )
    
    logger.info("\nInference completed successfully!")
    
    return predictions, bertscore_metrics, additional_metrics


if __name__ == "__main__":
    main()
