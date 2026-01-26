"""
Fine-tune BGE-M3 embedding model with LoRA + classification head
for document relevance classification.
Supports both LoRA fine-tuning and full fine-tuning modes.
Labels: Relevant, Irrelevant, Contain Answer
"""

import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from transformers import AutoTokenizer, AutoModel
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import preprocess_text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_bge_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Label mapping
LABEL2ID = {"Relevant": 0, "Irrelevant": 1, "Contain Answer": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        return False, 0, 0, 1
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    return True, rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)"""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


@dataclass
class ModelConfig:
    """Configuration for model and training"""
    # Model config
    model_name: str = "BAAI/bge-m3"
    max_seq_length: int = 4096
    hidden_size: int = 1024  # BGE-M3 hidden size
    classifier_hidden_size: int = 512
    classifier_dropout: float = 0.1
    
    # Fine-tuning mode: "lora", "full", or "freeze"
    finetune_mode: str = "lora"
    
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "query", "key", "value", "dense",  # Attention layers
        "intermediate.dense", "output.dense"  # FFN layers
    ])
    lora_bias: str = "none"
    
    # Training config
    output_dir: str = "./bge_classifier_lora"
    num_train_epochs: int = 10
    train_batch_size: int = 4
    eval_batch_size: int = 16
    learning_rate: float = 2e-5  # For full fine-tuning
    lora_learning_rate: float = 1e-4  # Higher LR for LoRA
    classifier_lr: float = 1e-4  # LR for classifier head
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    seed: int = 42
    
    # Logging
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    logging_dir: str = "./tensorboard_logs_bge_lora"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Mixed precision training
    # Options: "fp32" (default), "fp16", "bf16"
    # - fp16: Fast but may need GradScaler to avoid overflow
    # - bf16: Fast, no overflow issues, recommended for A100/H100/RTX30+
    precision: str = "fp32"
    
    # Length-based batching
    sort_by_length: bool = False  # Sort samples by length and use dynamic padding
    
    # Distributed training
    use_ddp: bool = False  # Use DistributedDataParallel for multi-GPU
    local_rank: int = -1  # Local rank for distributed training (set by launcher)
    world_size: int = 1  # Number of processes (GPUs)


class RelevanceDataset(Dataset):
    """Dataset for document relevance classification"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 4096,
        precompute_lengths: bool = False
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lengths = None
        
        # Optionally precompute lengths for sorting
        if precompute_lengths:
            self._precompute_lengths()
    
    def _precompute_lengths(self):
        """Precompute token lengths for all samples (for length-based batching)"""
        logger.info("Precomputing sequence lengths for length-based batching...")
        self.lengths = []
        for item in tqdm(self.data, desc="Computing lengths"):
            text = f"Question: {item['question']}\n\nContext: {item['context']}"
            text = preprocess_text(text)
            # Use fast tokenizer length estimation
            length = len(self.tokenizer.encode(text, truncation=True, max_length=self.max_length))
            self.lengths.append(length)
        logger.info(f"Length stats: min={min(self.lengths)}, max={max(self.lengths)}, avg={sum(self.lengths)/len(self.lengths):.1f}")
    
    def get_lengths(self) -> List[int]:
        """Get precomputed lengths or compute them"""
        if self.lengths is None:
            self._precompute_lengths()
        return self.lengths
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine question and context
        text = f"Question: {item['question']}\n\nContext: {item['context']}"
        text = preprocess_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            # padding='longest',
            # padding='max_length',
            return_tensors='pt'
        )
        
        # Get label
        label = LABEL2ID[item['label']]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class LengthBasedBatchSampler(Sampler):
    """
    Batch sampler that groups samples by length for efficient padding.
    Samples within each batch have similar lengths, reducing padding waste.
    """
    
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Sort indices by length
        self.sorted_indices = np.argsort(lengths)
    
    def __iter__(self):
        # Create batches of similar-length samples
        batches = []
        
        if self.shuffle:
            # Shuffle within length buckets for some randomness
            # Divide into chunks, shuffle within chunks, then create batches
            chunk_size = self.batch_size * 10  # Larger chunks for more randomness
            indices = self.sorted_indices.copy()
            
            # Shuffle within chunks of similar length
            for i in range(0, len(indices), chunk_size):
                chunk = indices[i:i + chunk_size]
                np.random.shuffle(chunk)
                indices[i:i + chunk_size] = chunk
        else:
            indices = self.sorted_indices
        
        # Create batches
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size].tolist()
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        if self.shuffle:
            # Shuffle batch order
            np.random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def collate_fn_dynamic_padding(batch, pad_token_id: int = 0):
    """
    Collate function that pads to the longest sequence in the batch.
    Much more efficient than padding to max_length.
    
    Args:
        batch: List of samples from dataset
        pad_token_id: Token ID to use for padding
    
    Returns:
        Batched and padded tensors
    """
    # Find max length in this batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        
        # Calculate padding needed
        padding_length = max_len - len(input_ids)
        
        if padding_length > 0:
            # Pad on the right
            input_ids = torch.cat([
                input_ids,
                torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=attention_mask.dtype)
            ])
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(item['label'])
    
    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
        'label': torch.stack(labels_list)
    }


class EmbeddingDataset(Dataset):
    """
    Dataset that stores pre-computed embeddings.
    Used for freeze mode to avoid recomputing embeddings every epoch.
    """
    
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            embeddings: Pre-computed embeddings of shape (N, hidden_size)
            labels: Labels of shape (N,)
        """
        assert len(embeddings) == len(labels), "Embeddings and labels must have same length"
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'label': self.labels[idx]
        }


class ClassifierOnlyModel(nn.Module):
    """
    Classifier-only model that takes pre-computed embeddings as input.
    Used for freeze mode training with cached embeddings.
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        classifier_hidden_size: int = 512,
        num_labels: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # 2-layer classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, classifier_hidden_size),
            nn.GELU(),
            nn.LayerNorm(classifier_hidden_size),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_labels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with pre-computed embeddings.
        
        Args:
            embeddings: Pre-computed CLS embeddings of shape (batch_size, hidden_size)
            labels: Optional labels for loss computation
        """
        logits = self.classifier(embeddings)
        
        result = {'logits': logits}
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            result['loss'] = loss
        
        return result


def precompute_embeddings(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: str,
    cache_path: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute embeddings for all samples in the dataloader.
    Optionally save to disk for reuse.
    
    Args:
        encoder: The encoder model (BGE-M3)
        dataloader: DataLoader with tokenized data
        device: Device to use
        cache_path: Optional path to save/load cached embeddings
    
    Returns:
        embeddings: Tensor of shape (N, hidden_size)
        labels: Tensor of shape (N,)
    """
    # Check if cache exists
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        cached = torch.load(cache_path, map_location='cpu')
        return cached['embeddings'], cached['labels']
    
    logger.info("Pre-computing embeddings (this only happens once)...")
    encoder.eval()
    encoder.to(device)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            # Get encoder outputs
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract CLS embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            
            all_embeddings.append(cls_embeddings)
            all_labels.append(labels)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    logger.info(f"Computed {len(embeddings)} embeddings of size {embeddings.shape[1]}")
    
    # Save cache if path provided
    if cache_path:
        cache_dir = Path(cache_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'embeddings': embeddings, 'labels': labels}, cache_path)
        logger.info(f"Saved embeddings cache to {cache_path}")
    
    return embeddings, labels


class BGEClassifierWithLoRA(nn.Module):
    """
    BGE-M3 with optional LoRA and 2-layer classification head.
    
    Supports three modes:
    - "lora": Apply LoRA adapters, freeze base model
    - "full": Full fine-tuning of all parameters
    - "freeze": Freeze encoder, only train classifier
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        hidden_size: int = 1024,
        classifier_hidden_size: int = 512,
        num_labels: int = 3,
        dropout: float = 0.1,
        finetune_mode: str = "lora",
        lora_config: Optional[LoraConfig] = None,
        precision: str = "fp32"  # "fp32", "fp16", or "bf16"
    ):
        super().__init__()
        
        self.finetune_mode = finetune_mode
        self.hidden_size = hidden_size
        self.precision = precision
        
        # Set dtype based on precision
        if precision == "bf16":
            self.dtype = torch.bfloat16
        elif precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        # Load BGE-M3 encoder
        logger.info(f"Loading base model: {model_name} (precision={precision}, dtype={self.dtype})")
        self.encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
        
        # Apply LoRA if specified
        if finetune_mode == "lora" and lora_config is not None:
            logger.info("Applying LoRA adapters...")
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()
        elif finetune_mode == "freeze":
            logger.info("Freezing encoder parameters...")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            logger.info("Full fine-tuning mode - all encoder parameters trainable")
        
        # 2-layer classification head (always trainable)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, classifier_hidden_size),
            nn.GELU(),
            nn.LayerNorm(classifier_hidden_size),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_labels)
        )
        
        # Initialize classifier weights
        self._init_weights()
        
        # Convert classifier to lower precision if needed
        if precision in ("fp16", "bf16"):
            self.classifier = self.classifier.to(self.dtype)
            logger.info(f"Classifier converted to {precision.upper()}")
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_embedding)
        
        result = {'logits': logits, 'embeddings': cls_embedding}
        
        # Compute loss if labels provided
        # IMPORTANT: Cast logits to float32 for stable loss computation
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.float(), labels)
            result['loss'] = loss
        
        return result
    
    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get embeddings without classification"""
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            return outputs.last_hidden_state[:, 0, :]
    
    def merge_and_unload(self):
        """Merge LoRA weights into base model (only for LoRA mode)"""
        if self.finetune_mode == "lora":
            logger.info("Merging LoRA weights into base model...")
            self.encoder = self.encoder.merge_and_unload()
            self.finetune_mode = "merged"
        else:
            logger.warning("merge_and_unload only works in LoRA mode")
    
    def save_pretrained(self, save_path: str):
        """Save model with appropriate method based on mode"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.finetune_mode == "lora":
            # Save LoRA adapters separately
            lora_path = save_path / "lora_adapters"
            self.encoder.save_pretrained(lora_path)
            logger.info(f"Saved LoRA adapters to {lora_path}")
        
        # Always save classifier head
        classifier_path = save_path / "classifier.pt"
        torch.save(self.classifier.state_dict(), classifier_path)
        logger.info(f"Saved classifier to {classifier_path}")
        
        # Save config
        config = {
            'finetune_mode': self.finetune_mode,
            'hidden_size': self.hidden_size,
        }
        with open(save_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        base_model_name: str = "BAAI/bge-m3",
        classifier_hidden_size: int = 512,
        num_labels: int = 3,
        dropout: float = 0.1
    ):
        """Load a saved model"""
        model_path = Path(model_path)
        
        # Load config
        with open(model_path / "config.json", 'r') as f:
            config = json.load(f)
        
        # Create model
        if config['finetune_mode'] == "lora":
            # Load base model first
            encoder = AutoModel.from_pretrained(base_model_name)
            # Load LoRA adapters
            lora_path = model_path / "lora_adapters"
            encoder = PeftModel.from_pretrained(encoder, lora_path)
        else:
            encoder = AutoModel.from_pretrained(base_model_name)
        
        # Create classifier
        classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(config['hidden_size'], classifier_hidden_size),
            nn.GELU(),
            nn.LayerNorm(classifier_hidden_size),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_labels)
        )
        
        # Load classifier weights
        classifier_path = model_path / "classifier.pt"
        classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
        
        # Create wrapper
        model = cls.__new__(cls)
        model.encoder = encoder
        model.classifier = classifier
        model.finetune_mode = config['finetune_mode']
        model.hidden_size = config['hidden_size']
        
        return model


def load_data(train_path: str, eval_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load training and evaluation data"""
    logger.info(f"Loading training data from {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    logger.info(f"Loading evaluation data from {eval_path}")
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Evaluation samples: {len(eval_data)}")
    
    # Log label distribution
    for name, data in [("Training", train_data), ("Evaluation", eval_data)]:
        labels = [item['label'] for item in data]
        logger.info(f"{name} label distribution:")
        for label in LABEL2ID.keys():
            count = labels.count(label)
            logger.info(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
    
    return train_data, eval_data


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    return_predictions: bool = False,
    precision: str = "fp32"  # "fp32", "fp16", or "bf16"
) -> Dict:
    """Evaluate model on dataloader"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    # Setup autocast
    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Use autocast for mixed precision inference
            with autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = model(input_ids, attention_mask, labels)
            
            total_loss += outputs['loss'].item()
            num_batches += 1
            
            predictions = torch.argmax(outputs['logits'], dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to label names
    pred_labels = [ID2LABEL[p] for p in all_predictions]
    true_labels = [ID2LABEL[l] for l in all_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    avg_loss = total_loss / num_batches
    
    # Classification report (dict)
    report = classification_report(
        true_labels, pred_labels,
        labels=list(LABEL2ID.keys()),
        output_dict=True,
        zero_division=0
    )
    
    # Classification report (string)
    report_str = classification_report(
        true_labels, pred_labels,
        labels=list(LABEL2ID.keys()),
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(
        true_labels, pred_labels,
        labels=list(LABEL2ID.keys())
    )
    
    result = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'classification_report': report,
        'classification_report_str': report_str,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_array': cm,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score']
    }
    
    if return_predictions:
        result['predictions'] = pred_labels
        result['true_labels'] = true_labels
    
    return result


def get_optimizer_and_scheduler(
    model: nn.Module,
    config: ModelConfig,
    total_steps: int
):
    """Create optimizer with different LRs for different parameter groups"""
    
    # Separate parameters
    classifier_params = list(model.classifier.parameters())
    classifier_param_ids = set(id(p) for p in classifier_params)
    
    if config.finetune_mode == "lora":
        # For LoRA: only LoRA params + classifier are trainable
        encoder_params = [
            p for p in model.encoder.parameters() 
            if p.requires_grad and id(p) not in classifier_param_ids
        ]
        encoder_lr = config.lora_learning_rate
    elif config.finetune_mode == "full":
        # For full: all encoder params + classifier
        encoder_params = [
            p for p in model.encoder.parameters() 
            if id(p) not in classifier_param_ids
        ]
        encoder_lr = config.learning_rate
    else:
        # For freeze: only classifier
        encoder_params = []
        encoder_lr = 0.0
    
    param_groups = []
    
    if encoder_params:
        param_groups.append({
            'params': encoder_params,
            'lr': encoder_lr,
            'name': 'encoder'
        })
    
    param_groups.append({
        'params': classifier_params,
        'lr': config.classifier_lr,
        'name': 'classifier'
    })
    
    optimizer = AdamW(param_groups, weight_decay=config.weight_decay)
    
    # Get max LRs for scheduler
    max_lrs = [g['lr'] for g in param_groups]
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy='cos'
    )
    
    return optimizer, scheduler


def train_with_cached_embeddings(
    classifier: ClassifierOnlyModel,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: ModelConfig
):
    """
    Training loop for freeze mode with pre-computed embeddings.
    Much faster since we skip the encoder forward pass.
    """
    device = config.device
    classifier.to(device)
    
    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_dir = f"{config.logging_dir}/{timestamp}_freeze_cached"
    Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    logger.info(f"TensorBoard logs: {tensorboard_dir}")
    
    # Calculate total steps
    total_steps = len(train_dataloader) * config.num_train_epochs
    
    # Optimizer - only classifier parameters
    optimizer = AdamW(
        classifier.parameters(),
        lr=config.classifier_lr,
        weight_decay=config.weight_decay
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.classifier_lr,
        total_steps=total_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy='cos'
    )
    
    # Log parameters
    total_params = sum(p.numel() for p in classifier.parameters())
    logger.info(f"Classifier parameters: {total_params:,}")
    
    # Training
    global_step = 0
    best_eval_accuracy = 0.0
    best_eval_f1 = 0.0
    
    for epoch in range(config.num_train_epochs):
        classifier.train()
        epoch_loss = 0.0
        num_batches = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{config.num_train_epochs}")
        logger.info(f"{'='*60}")
        
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass (just classifier!)
            outputs = classifier(embeddings, labels)
            loss = outputs['loss']
            
            # Backward pass
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % config.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    writer.add_scalar('train/loss', loss.item() * config.gradient_accumulation_steps, global_step)
                    writer.add_scalar('train/learning_rate', current_lr, global_step)
                    logger.info(f"Step {global_step}: loss={loss.item() * config.gradient_accumulation_steps:.4f}, lr={current_lr:.2e}")
            
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1
            progress_bar.set_postfix({'loss': epoch_loss / num_batches})
        
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch + 1)
        
        # Evaluation
        logger.info(f"\nRunning evaluation after epoch {epoch + 1}...")
        eval_results = evaluate_cached(classifier, eval_dataloader, device)
        
        logger.info(f"Eval Loss: {eval_results['loss']:.4f}")
        logger.info(f"Eval Accuracy: {eval_results['accuracy']:.4f}")
        logger.info(f"Eval Macro F1: {eval_results['macro_f1']:.4f}")
        logger.info(f"Eval Weighted F1: {eval_results['weighted_f1']:.4f}")
        
        # Log classification report
        logger.info(f"\n{'-'*60}")
        logger.info(f"Classification Report - Epoch {epoch + 1}:")
        logger.info(f"{'-'*60}")
        logger.info(f"\n{eval_results['classification_report_str']}")
        
        # Log confusion matrix
        logger.info(f"\nConfusion Matrix - Epoch {epoch + 1}:")
        import pandas as pd
        cm_df = pd.DataFrame(
            eval_results['confusion_matrix_array'],
            index=[f"True: {l}" for l in LABEL2ID.keys()],
            columns=[f"Pred: {l}" for l in LABEL2ID.keys()]
        )
        logger.info(f"\n{cm_df.to_string()}")
        logger.info(f"{'-'*60}\n")
        
        # Log to TensorBoard
        writer.add_scalar('eval/loss', eval_results['loss'], epoch + 1)
        writer.add_scalar('eval/accuracy', eval_results['accuracy'], epoch + 1)
        writer.add_scalar('eval/macro_f1', eval_results['macro_f1'], epoch + 1)
        writer.add_scalar('eval/weighted_f1', eval_results['weighted_f1'], epoch + 1)
        
        for label in LABEL2ID.keys():
            if label in eval_results['classification_report']:
                writer.add_scalar(
                    f'eval/{label}/f1',
                    eval_results['classification_report'][label]['f1-score'],
                    epoch + 1
                )
        
        # Save best model
        if eval_results['accuracy'] > best_eval_accuracy:
            best_eval_accuracy = eval_results['accuracy']
            best_eval_f1 = eval_results['macro_f1']
            
            save_path = Path(config.output_dir) / "best_model"
            save_path.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_eval_accuracy,
                'macro_f1': best_eval_f1,
                'hidden_size': classifier.hidden_size,
                'finetune_mode': 'freeze',
            }, save_path / "classifier_checkpoint.pt")
            
            # Also save just the classifier for easy loading
            torch.save(classifier.classifier.state_dict(), save_path / "classifier.pt")
            
            logger.info(f"Saved best classifier with accuracy: {best_eval_accuracy:.4f}")
        
        # Save epoch checkpoint
        checkpoint_path = Path(config.output_dir) / f"checkpoint-epoch-{epoch + 1}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch + 1,
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy': eval_results['accuracy'],
            'macro_f1': eval_results['macro_f1'],
            'finetune_mode': 'freeze',
        }, checkpoint_path / "classifier_checkpoint.pt")
    
    writer.close()
    logger.info(f"\nTraining completed!")
    logger.info(f"Best accuracy: {best_eval_accuracy:.4f}")
    logger.info(f"Best macro F1: {best_eval_f1:.4f}")
    
    return classifier


def evaluate_cached(
    classifier: ClassifierOnlyModel,
    dataloader: DataLoader,
    device: str,
    return_predictions: bool = False
) -> Dict:
    """Evaluate classifier with pre-computed embeddings"""
    classifier.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            
            outputs = classifier(embeddings, labels)
            
            total_loss += outputs['loss'].item()
            num_batches += 1
            
            predictions = torch.argmax(outputs['logits'], dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to label names
    pred_labels = [ID2LABEL[p] for p in all_predictions]
    true_labels = [ID2LABEL[l] for l in all_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    avg_loss = total_loss / num_batches
    
    report = classification_report(
        true_labels, pred_labels,
        labels=list(LABEL2ID.keys()),
        output_dict=True,
        zero_division=0
    )
    
    report_str = classification_report(
        true_labels, pred_labels,
        labels=list(LABEL2ID.keys()),
        zero_division=0
    )
    
    cm = confusion_matrix(
        true_labels, pred_labels,
        labels=list(LABEL2ID.keys())
    )
    
    result = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'classification_report': report,
        'classification_report_str': report_str,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_array': cm,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score']
    }
    
    if return_predictions:
        result['predictions'] = pred_labels
        result['true_labels'] = true_labels
    
    return result


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: ModelConfig
):
    """Training loop with optional DDP support"""
    device = config.device
    
    # Move model to device
    model.to(device)
    
    # Wrap model with DDP if enabled
    if config.use_ddp:
        model = DDP(
            model,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
            find_unused_parameters=True  # Needed for LoRA
        )
        logger.info(f"Model wrapped with DDP on GPU {config.local_rank}")
    
    # Setup TensorBoard (only on main process)
    writer = None
    if is_main_process():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_suffix = f"_{config.finetune_mode}"
        tensorboard_dir = f"{config.logging_dir}/{timestamp}{mode_suffix}"
        Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)
        logger.info(f"TensorBoard logs: {tensorboard_dir}")
    
    # Calculate total steps
    total_steps = len(train_dataloader) * config.num_train_epochs
    
    # Get optimizer and scheduler
    # Use raw model for optimizer (not DDP wrapper)
    raw_model = model.module if config.use_ddp else model
    optimizer, scheduler = get_optimizer_and_scheduler(raw_model, config, total_steps)
    
    # Setup mixed precision training
    use_amp = config.precision in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
    
    # GradScaler only needed for FP16 (BF16 doesn't overflow)
    use_scaler = config.precision == "fp16"
    scaler = GradScaler(enabled=use_scaler)
    
    if config.precision == "bf16":
        logger.info("Using BF16 mixed precision training (no GradScaler needed)")
    elif config.precision == "fp16":
        logger.info("Using FP16 mixed precision training with GradScaler")
    else:
        logger.info("Using FP32 full precision training")
    
    # Log trainable parameters (only on main process)
    if is_main_process():
        raw_model = model.module if config.use_ddp else model
        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        if config.use_ddp:
            logger.info(f"Training on {config.world_size} GPUs")
    
    # Training
    global_step = 0
    best_eval_accuracy = 0.0
    best_eval_f1 = 0.0
    
    for epoch in range(config.num_train_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Set epoch for distributed sampler
        if config.use_ddp and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        
        if is_main_process():
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{config.num_train_epochs}")
            logger.info(f"{'='*60}")
        
        progress_bar = tqdm(
            train_dataloader, 
            desc=f"Training Epoch {epoch + 1}",
            disable=not is_main_process()  # Only show progress on main process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass with autocast
            with autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping...")
                optimizer.zero_grad()
                continue
            
            # Backward pass
            loss = loss / config.gradient_accumulation_steps
            
            if use_scaler:
                # FP16: Use gradient scaling
                scaler.scale(loss).backward()
            else:
                # BF16/FP32: Direct backward
                loss.backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if use_scaler:
                    # FP16: Unscale, clip, step with scaler
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # BF16/FP32: Direct clip and step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % config.logging_steps == 0 and is_main_process():
                    current_lr = scheduler.get_last_lr()[0]
                    writer.add_scalar('train/loss', loss.item() * config.gradient_accumulation_steps, global_step)
                    writer.add_scalar('train/learning_rate', current_lr, global_step)
                    if use_scaler:
                        writer.add_scalar('train/grad_scale', scaler.get_scale(), global_step)
                    logger.info(f"Step {global_step}: loss={loss.item() * config.gradient_accumulation_steps:.4f}, lr={current_lr:.2e}")
            
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1
            progress_bar.set_postfix({'loss': epoch_loss / num_batches})
        
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch + 1)
        
        # Evaluation at end of each epoch (only on main process)
        if is_main_process():
            logger.info(f"\nRunning evaluation after epoch {epoch + 1}...")
            # Use raw model for evaluation
            eval_model = model.module if config.use_ddp else model
            eval_results = evaluate(eval_model, eval_dataloader, device, precision=config.precision)
            
            logger.info(f"Eval Loss: {eval_results['loss']:.4f}")
            logger.info(f"Eval Accuracy: {eval_results['accuracy']:.4f}")
            logger.info(f"Eval Macro F1: {eval_results['macro_f1']:.4f}")
            logger.info(f"Eval Weighted F1: {eval_results['weighted_f1']:.4f}")
        
        # Log full classification report
        if is_main_process():
            logger.info(f"\n{'-'*60}")
            logger.info(f"Classification Report - Epoch {epoch + 1}:")
            logger.info(f"{'-'*60}")
            logger.info(f"\n{eval_results['classification_report_str']}")
            
            # Log confusion matrix
            logger.info(f"\nConfusion Matrix - Epoch {epoch + 1}:")
            import pandas as pd
            cm_df = pd.DataFrame(
                eval_results['confusion_matrix_array'],
                index=[f"True: {l}" for l in LABEL2ID.keys()],
                columns=[f"Pred: {l}" for l in LABEL2ID.keys()]
            )
            logger.info(f"\n{cm_df.to_string()}")
            logger.info(f"{'-'*60}\n")
            
            # Log to TensorBoard
            writer.add_scalar('eval/loss', eval_results['loss'], epoch + 1)
            writer.add_scalar('eval/accuracy', eval_results['accuracy'], epoch + 1)
            writer.add_scalar('eval/macro_f1', eval_results['macro_f1'], epoch + 1)
            writer.add_scalar('eval/weighted_f1', eval_results['weighted_f1'], epoch + 1)
            
            # Per-class metrics
            for label in LABEL2ID.keys():
                if label in eval_results['classification_report']:
                    writer.add_scalar(
                        f'eval/{label}/f1',
                        eval_results['classification_report'][label]['f1-score'],
                        epoch + 1
                    )
        
        # Save best model (only on main process)
        if is_main_process() and eval_results['accuracy'] > best_eval_accuracy:
            best_eval_accuracy = eval_results['accuracy']
            best_eval_f1 = eval_results['macro_f1']
            
            save_path = Path(config.output_dir) / "best_model"
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save using model's save method (use raw model)
            raw_model = model.module if config.use_ddp else model
            raw_model.save_pretrained(save_path)
            
            # Also save full checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_eval_accuracy,
                'macro_f1': best_eval_f1,
                'finetune_mode': config.finetune_mode,
            }, save_path / "checkpoint.pt")
            
            logger.info(f"Saved best model with accuracy: {best_eval_accuracy:.4f}")
        
        # Save epoch checkpoint (only on main process)
        if is_main_process():
            checkpoint_path = Path(config.output_dir) / f"checkpoint-epoch-{epoch + 1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            raw_model = model.module if config.use_ddp else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': eval_results['accuracy'],
                'macro_f1': eval_results['macro_f1'],
                'finetune_mode': config.finetune_mode,
            }, checkpoint_path / "checkpoint.pt")
        
        # Synchronize all processes before next epoch
        if config.use_ddp:
            dist.barrier()
    
    if is_main_process() and writer:
        writer.close()
    
    if is_main_process():
        logger.info(f"\nTraining completed!")
        logger.info(f"Best accuracy: {best_eval_accuracy:.4f}")
        logger.info(f"Best macro F1: {best_eval_f1:.4f}")
    
    # Return raw model
    return model.module if config.use_ddp else model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune BGE-M3 with LoRA or full fine-tuning")
    parser.add_argument(
        "--finetune_mode", 
        type=str, 
        default="lora",
        choices=["lora", "full", "freeze"],
        help="Fine-tuning mode: lora, full, or freeze"
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Max sequence length")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--precision", 
        type=str, 
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Training precision: fp32, fp16 (with GradScaler), or bf16 (recommended)"
    )
    parser.add_argument("--sort_by_length", action="store_true", help="Sort samples by length and use dynamic padding")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (set by torchrun)")
    
    args = parser.parse_args()
    
    # Setup distributed training
    use_ddp, rank, local_rank, world_size = setup_distributed()
    
    config = ModelConfig()
    config.use_ddp = use_ddp
    config.local_rank = local_rank
    config.world_size = world_size
    
    # Set device based on distributed training
    if use_ddp:
        config.device = f"cuda:{local_rank}"
    
    config.finetune_mode = args.finetune_mode
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.num_train_epochs = args.epochs
    config.train_batch_size = args.batch_size
    config.max_seq_length = args.max_seq_length
    config.precision = args.precision
    config.sort_by_length = args.sort_by_length
    
    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        config.output_dir = f"./bge_classifier_{config.finetune_mode}"
    
    # Set seed (add rank to make each process different)
    seed = config.seed + (rank if use_ddp else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Only log on main process
    if is_main_process():
        logger.info("=" * 60)
        logger.info(f"BGE-M3 Classifier Training ({config.finetune_mode.upper()} mode)")
        logger.info("=" * 60)
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Fine-tuning mode: {config.finetune_mode}")
        logger.info(f"Max sequence length: {config.max_seq_length}")
        logger.info(f"Classifier hidden size: {config.classifier_hidden_size}")
        
        if config.finetune_mode == "lora":
            logger.info(f"LoRA rank: {config.lora_r}")
            logger.info(f"LoRA alpha: {config.lora_alpha}")
            logger.info(f"LoRA dropout: {config.lora_dropout}")
            logger.info(f"LoRA learning rate: {config.lora_learning_rate}")
        elif config.finetune_mode == "full":
            logger.info(f"Encoder learning rate: {config.learning_rate}")
        
        logger.info(f"Precision: {config.precision.upper()}")
        logger.info(f"Classifier learning rate: {config.classifier_lr}")
        logger.info(f"Epochs: {config.num_train_epochs}")
        logger.info(f"Device: {config.device}")
        if use_ddp:
            logger.info(f"Distributed training: {world_size} GPUs")
    
    # Load data
    train_data, eval_data = load_data(
        train_path="multihop_train_data_v2.json",
        eval_path="multihop_eval_data_v2.json"
    )
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = RelevanceDataset(
        train_data, tokenizer, config.max_seq_length,
        precompute_lengths=config.sort_by_length
    )
    eval_dataset = RelevanceDataset(
        eval_data, tokenizer, config.max_seq_length,
        precompute_lengths=False  # Eval doesn't need length sorting
    )
    
    # Create dataloaders
    # Get pad token id for collate function
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate_fn = lambda batch: collate_fn_dynamic_padding(batch, pad_token_id)
    
    # Create train sampler based on configuration
    train_sampler = None
    if config.use_ddp:
        # Use DistributedSampler for DDP
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=rank,
            shuffle=True
        )
        if is_main_process():
            logger.info(f"Using DistributedSampler for {config.world_size} GPUs")
    
    if config.sort_by_length and not config.use_ddp:
        # Length-based batching (not compatible with DDP currently)
        if is_main_process():
            logger.info("Using length-based batch sampling with dynamic padding")
        train_batch_sampler = LengthBasedBatchSampler(
            lengths=train_dataset.get_lengths(),
            batch_size=config.train_batch_size,
            shuffle=True,
            drop_last=False
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
    elif config.use_ddp:
        # DDP with DistributedSampler
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True  # Recommended for DDP to avoid uneven batches
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    # Eval dataloader always uses dynamic padding collate but no length sorting
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    if is_main_process():
        logger.info(f"Train batches: {len(train_dataloader)}")
        logger.info(f"Eval batches: {len(eval_dataloader)}")
    
    # Handle freeze mode with embedding caching
    if config.finetune_mode == "freeze":
        logger.info("\n" + "="*60)
        logger.info("FREEZE MODE: Using embedding caching for efficiency")
        logger.info("="*60)
        
        # Load just the encoder for embedding computation
        logger.info("Loading BGE-M3 encoder for embedding computation...")
        encoder = AutoModel.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        
        # Define cache paths
        cache_dir = Path(config.output_dir) / "embedding_cache"
        train_cache_path = cache_dir / "train_embeddings.pt"
        eval_cache_path = cache_dir / "eval_embeddings.pt"
        
        # Pre-compute embeddings (with caching)
        # For train, we need sequential dataloader for caching
        train_dataloader_sequential = DataLoader(
            train_dataset,
            batch_size=1,  # Can use larger batch since no gradients
            shuffle=False,  # Must be False for caching
            num_workers=4,
            pin_memory=True
        )
        
        eval_dataloader_sequential = DataLoader(
            eval_dataset,
            batch_size=1,  # Can use larger batch since no gradients
            shuffle=False,  # Must be False for caching
            num_workers=4,
            pin_memory=True
        )
        
        train_embeddings, train_labels = precompute_embeddings(
            encoder, train_dataloader_sequential, config.device, str(train_cache_path)
        )
        eval_embeddings, eval_labels = precompute_embeddings(
            encoder, eval_dataloader_sequential, config.device, str(eval_cache_path)
        )
        
        # Free encoder memory
        del encoder
        torch.cuda.empty_cache()
        logger.info("Encoder removed from memory - using cached embeddings")
        
        # Create embedding datasets
        train_emb_dataset = EmbeddingDataset(train_embeddings, train_labels)
        eval_emb_dataset = EmbeddingDataset(eval_embeddings, eval_labels)
        
        # Create dataloaders for embeddings (can use much larger batch size!)
        train_emb_dataloader = DataLoader(
            train_emb_dataset,
            batch_size=config.train_batch_size * 8,  # Much larger batch size possible
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        eval_emb_dataloader = DataLoader(
            eval_emb_dataset,
            batch_size=config.eval_batch_size * 8,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Embedding train batches: {len(train_emb_dataloader)} (vs {len(train_dataloader)} original)")
        logger.info(f"Embedding eval batches: {len(eval_emb_dataloader)} (vs {len(eval_dataloader)} original)")
        
        # Create classifier-only model
        classifier = ClassifierOnlyModel(
            hidden_size=config.hidden_size,
            classifier_hidden_size=config.classifier_hidden_size,
            num_labels=NUM_LABELS,
            dropout=config.classifier_dropout
        )
        
        # Train with cached embeddings
        classifier = train_with_cached_embeddings(
            classifier, train_emb_dataloader, eval_emb_dataloader, config
        )
        
        # Final evaluation
        logger.info("\n" + "=" * 60)
        logger.info("Final Evaluation")
        logger.info("=" * 60)
        
        classifier.eval()
        final_results = evaluate_cached(
            classifier, eval_emb_dataloader, config.device, return_predictions=True
        )
        
        model = classifier  # For consistency with the rest of the code
        
    else:
        # Create LoRA config if needed
        lora_config = None
        if config.finetune_mode == "lora":
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias=config.lora_bias,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
        
        # Create model
        if is_main_process():
            logger.info("Loading BGE-M3 model...")
        model = BGEClassifierWithLoRA(
            model_name=config.model_name,
            hidden_size=config.hidden_size,
            classifier_hidden_size=config.classifier_hidden_size,
            num_labels=NUM_LABELS,
            dropout=config.classifier_dropout,
            finetune_mode=config.finetune_mode,
            lora_config=lora_config,
            precision=config.precision
        )
        
        # Train
        model = train(model, train_dataloader, eval_dataloader, config)
        
        # Final evaluation (only on main process)
        if is_main_process():
            logger.info("\n" + "=" * 60)
            logger.info("Final Evaluation")
            logger.info("=" * 60)
            
            model.eval()
            final_results = evaluate(model, eval_dataloader, config.device, return_predictions=True, precision=config.precision)
    
    # Final logging and saving (only on main process)
    if is_main_process():
        logger.info(f"Final Accuracy: {final_results['accuracy']:.4f}")
        logger.info(f"Final Macro F1: {final_results['macro_f1']:.4f}")
        logger.info(f"Final Weighted F1: {final_results['weighted_f1']:.4f}")
        
        # Print classification report
        logger.info("\nClassification Report:")
        report_str = classification_report(
            final_results['true_labels'],
            final_results['predictions'],
            labels=list(LABEL2ID.keys()),
            zero_division=0
        )
        logger.info(f"\n{report_str}")
        
        # Print confusion matrix
        logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(
            final_results['true_labels'],
            final_results['predictions'],
            labels=list(LABEL2ID.keys())
        )
        logger.info(f"\n{cm}")
        
        # Save final results
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "final_results.json", 'w') as f:
            json.dump({
                'accuracy': final_results['accuracy'],
                'macro_f1': final_results['macro_f1'],
                'weighted_f1': final_results['weighted_f1'],
                'confusion_matrix': final_results['confusion_matrix'],
                'finetune_mode': config.finetune_mode,
                'config': {
                    'model_name': config.model_name,
                    'max_seq_length': config.max_seq_length,
                    'classifier_hidden_size': config.classifier_hidden_size,
                    'lora_r': config.lora_r if config.finetune_mode == "lora" else None,
                    'lora_alpha': config.lora_alpha if config.finetune_mode == "lora" else None,
                    'epochs': config.num_train_epochs,
                }
            }, f, indent=2)
        
        logger.info(f"\nResults saved to {output_path}")
        
        # Optionally merge LoRA weights
        if config.finetune_mode == "lora":
            logger.info("\nTo merge LoRA weights into base model, call:")
            logger.info("  model.merge_and_unload()")
    
    # Cleanup distributed training
    cleanup_distributed()
    
    return model


if __name__ == "__main__":
    main()
