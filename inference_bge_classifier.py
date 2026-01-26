"""
Inference script for BGE-M3 classifier for document relevance classification.
"""

import json
import torch
import torch.nn as nn
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'inference_bge_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Label mapping
LABEL2ID = {"Relevant": 0, "Irrelevant": 1, "Contain Answer": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
VALID_LABELS = list(LABEL2ID.keys())


class RelevanceDataset(Dataset):
    """Dataset for document relevance classification"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine question and context
        text = f"Question: {item['question']}\n\nContext: {item['context']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        
        # Get label if available
        if 'label' in item:
            result['label'] = torch.tensor(LABEL2ID[item['label']], dtype=torch.long)
        
        return result


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


def load_model(
    checkpoint_path: str,
    model_name: str = "BAAI/bge-m3",
    hidden_size: int = 1024,
    classifier_hidden_size: int = 512,
    device: str = "cuda"
) -> BGEClassifier:
    """Load trained model from checkpoint"""
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Create model
    model = BGEClassifier(
        model_name=model_name,
        hidden_size=hidden_size,
        classifier_hidden_size=classifier_hidden_size,
        num_labels=len(LABEL2ID),
        dropout=0.1
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'accuracy' in checkpoint:
        logger.info(f"Loaded checkpoint with accuracy: {checkpoint['accuracy']:.4f}")
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint from epoch: {checkpoint['epoch']}")
    
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def load_data(data_path: str) -> List[Dict]:
    """Load evaluation data"""
    logger.info(f"Loading data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples")
    return data


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Dict:
    """Run predictions on dataloader"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    has_labels = False
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            
            # Get predictions
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Get labels if available
            if 'label' in batch:
                has_labels = True
                all_labels.extend(batch['label'].numpy())
    
    # Convert to label names
    pred_labels = [ID2LABEL[p] for p in all_predictions]
    
    result = {
        'predictions': pred_labels,
        'probabilities': all_probs,
    }
    
    if has_labels:
        true_labels = [ID2LABEL[l] for l in all_labels]
        result['true_labels'] = true_labels
    
    return result


def calculate_metrics(true_labels: List[str], predictions: List[str]) -> Dict:
    """Calculate classification metrics"""
    accuracy = accuracy_score(true_labels, predictions)
    
    report_dict = classification_report(
        true_labels, predictions,
        labels=VALID_LABELS,
        output_dict=True,
        zero_division=0
    )
    
    report_str = classification_report(
        true_labels, predictions,
        labels=VALID_LABELS,
        zero_division=0
    )
    
    cm = confusion_matrix(true_labels, predictions, labels=VALID_LABELS)
    
    return {
        'accuracy': accuracy,
        'macro_f1': report_dict['macro avg']['f1-score'],
        'weighted_f1': report_dict['weighted avg']['f1-score'],
        'per_class': {
            label: {
                'precision': report_dict[label]['precision'],
                'recall': report_dict[label]['recall'],
                'f1-score': report_dict[label]['f1-score'],
                'support': report_dict[label]['support']
            }
            for label in VALID_LABELS
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report_str
    }


def print_metrics(metrics: Dict):
    """Print metrics in formatted way"""
    logger.info("=" * 60)
    logger.info("CLASSIFICATION RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\nAccuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    
    logger.info("\n" + "-" * 60)
    logger.info("Per-Class Metrics:")
    logger.info("-" * 60)
    
    for label in VALID_LABELS:
        class_metrics = metrics['per_class'][label]
        logger.info(f"\n{label}:")
        logger.info(f"  Precision: {class_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {class_metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {class_metrics['f1-score']:.4f}")
        logger.info(f"  Support:   {class_metrics['support']}")
    
    logger.info("\n" + "-" * 60)
    logger.info("Confusion Matrix:")
    logger.info("-" * 60)
    
    cm = np.array(metrics['confusion_matrix'])
    cm_df = pd.DataFrame(
        cm,
        index=[f"True: {l}" for l in VALID_LABELS],
        columns=[f"Pred: {l}" for l in VALID_LABELS]
    )
    logger.info(f"\n{cm_df.to_string()}")


def save_results(
    data: List[Dict],
    predictions: List[str],
    probabilities: List,
    metrics: Dict,
    output_dir: str
):
    """Save predictions and metrics"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions
    predictions_data = []
    for i, (item, pred, probs) in enumerate(zip(data, predictions, probabilities)):
        predictions_data.append({
            'question': item['question'],
            'context': item['context'][:500] + "..." if len(item['context']) > 500 else item['context'],
            'true_label': item.get('label', 'N/A'),
            'predicted_label': pred,
            'probabilities': {
                VALID_LABELS[j]: float(probs[j])
                for j in range(len(VALID_LABELS))
            },
            'correct': item.get('label', '') == pred
        })
    
    predictions_file = output_path / f"predictions_{timestamp}.json"
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Predictions saved to {predictions_file}")
    
    # Save metrics
    if metrics:
        metrics_file = output_path / f"metrics_{timestamp}.json"
        metrics_to_save = {k: v for k, v in metrics.items() if k != 'classification_report'}
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_to_save, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Save report
        report_file = output_path / f"classification_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {metrics['weighted_f1']:.4f}\n\n")
            f.write(metrics['classification_report'])
        logger.info(f"Report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference for BGE-M3 document relevance classifier"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./bge_classifier/best_model/checkpoint.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="multihop_eval_data_v2.json",
        help="Path to evaluation data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results_bge",
        help="Directory to save results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-m3",
        help="Base model name"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1024,
        help="BGE-M3 hidden size"
    )
    parser.add_argument(
        "--classifier_hidden_size",
        type=int,
        default=512,
        help="Classifier hidden size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save results"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("BGE-M3 Classifier Inference")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Device: {args.device}")
    
    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        hidden_size=args.hidden_size,
        classifier_hidden_size=args.classifier_hidden_size,
        device=args.device
    )
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load data
    data = load_data(args.data_path)
    
    # Create dataset and dataloader
    dataset = RelevanceDataset(data, tokenizer, args.max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Log label distribution if available
    if 'label' in data[0]:
        labels = [item['label'] for item in data]
        logger.info("\nLabel distribution:")
        for label in VALID_LABELS:
            count = labels.count(label)
            logger.info(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
    
    # Run predictions
    logger.info("\nRunning predictions...")
    results = predict(model, dataloader, args.device)
    
    # Calculate metrics if we have labels
    metrics = None
    if 'true_labels' in results:
        logger.info("\nCalculating metrics...")
        metrics = calculate_metrics(results['true_labels'], results['predictions'])
        print_metrics(metrics)
    
    # Save results
    if not args.no_save:
        save_results(
            data=data,
            predictions=results['predictions'],
            probabilities=results['probabilities'],
            metrics=metrics,
            output_dir=args.output_dir
        )
    
    logger.info("\nInference completed!")
    
    return results, metrics


if __name__ == "__main__":
    main()
