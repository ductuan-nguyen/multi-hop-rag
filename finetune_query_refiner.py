"""
Fine-tune Qwen 0.6B with LoRA using Unsloth for document relevance classification.
Labels: Relevant, Irrelevant, Contain Answer
"""

import json
import torch
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from datasets import Dataset
from transformers import TrainingArguments, TrainerCallback
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from trl import SFTTrainer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np

# Setup logging - must be done explicitly since unsloth configures logging on import
def setup_logging():
    """Setup logging with both file and console handlers"""
    log_filename = Path(__file__).parent / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Get our logger
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    _logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)
    
    # Prevent propagation to root logger (which unsloth may have configured)
    _logger.propagate = False
    
    _logger.info(f"Logging to file: {log_filename}")
    
    return _logger

logger = setup_logging()


@dataclass
class ModelConfig:
    """Configuration for model and training"""
    model_name: str = "unsloth/Qwen2.5-0.5B-Instruct"  # Qwen 0.5B (closest to 0.6B)
    max_seq_length: int = 8192
    load_in_4bit: bool = True
    
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training config
    output_dir: str = "./qwen_lora_relevance"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 100
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    seed: int = 42
    
    # TensorBoard config
    logging_dir: str = "./tensorboard_logs"
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint directory to resume from


class EpochLoggingCallback(TrainerCallback):
    """Custom callback for logging at each epoch and running evaluation"""
    
    def __init__(self, logger, eval_data=None, tokenizer=None, create_prompt_fn=None):
        self.logger = logger
        self.epoch_metrics = []
        self.eval_data = eval_data
        self.tokenizer = tokenizer
        self.create_prompt_fn = create_prompt_fn
        self.valid_labels = ["Relevant", "Irrelevant", "Contain Answer"]
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.logger.info(f"=" * 50)
        self.logger.info(f"Starting Epoch {state.epoch + 1}/{args.num_train_epochs}")
        self.logger.info(f"=" * 50)
    
    def _compute_eval_metrics(self, model):
        """Compute accuracy and F1 on eval data"""
        if self.eval_data is None or self.tokenizer is None or self.create_prompt_fn is None:
            return None
        
        model.eval()
        predictions = []
        labels = []
        
        # Sample subset for faster evaluation during training (use first 100 samples)
        eval_subset = self.eval_data[:100] if len(self.eval_data) > 100 else self.eval_data
        
        for item in eval_subset:
            prompt = self.create_prompt_fn(
                tokenizer=self.tokenizer,
                question=item['question'],
                context=item['context'],
                label=None
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = response.split("Classification:")[-1].strip()
            
            # Map to valid label
            pred_label = None
            for valid_label in self.valid_labels:
                if valid_label.lower() in pred.lower():
                    pred_label = valid_label
                    break
            
            if pred_label is None:
                pred_label = "Irrelevant"
            
            predictions.append(pred_label)
            labels.append(item['label'])
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        weighted_f1 = f1_score(labels, predictions, labels=self.valid_labels, average='weighted')
        
        return {'accuracy': accuracy, 'weighted_f1': weighted_f1}
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self.logger.info(f"Completed Epoch {state.epoch:.0f}")
        
        latest_metrics = {}
        
        # Get loss metrics from log history
        if state.log_history:
            for log in reversed(state.log_history):
                if 'loss' in log and 'loss' not in latest_metrics:
                    latest_metrics['loss'] = log['loss']
                if 'eval_loss' in log and 'eval_loss' not in latest_metrics:
                    latest_metrics['eval_loss'] = log['eval_loss']
                if len(latest_metrics) >= 2:
                    break
        
        # Compute eval accuracy and F1
        # if model is not None:
        #     try:
        #         eval_metrics = self._compute_eval_metrics(model)
        #         if eval_metrics:
        #             latest_metrics['eval_accuracy'] = eval_metrics['accuracy']
        #             latest_metrics['eval_weighted_f1'] = eval_metrics['weighted_f1']
        #             self.logger.info(f"Epoch {state.epoch:.0f} - Eval Accuracy: {eval_metrics['accuracy']:.4f}")
        #             self.logger.info(f"Epoch {state.epoch:.0f} - Eval Weighted F1: {eval_metrics['weighted_f1']:.4f}")
        #     except Exception as e:
        #         self.logger.warning(f"Failed to compute eval metrics: {e}")
        
        self.logger.info(f"Epoch {state.epoch:.0f} Metrics: {latest_metrics}")
        self.epoch_metrics.append({
            'epoch': state.epoch,
            'metrics': latest_metrics
        })
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info(f"Step {state.global_step}: {logs}")


SYSTEM_PROMPT = """Bạn là một chuyên gia tinh chỉnh câu hỏi. Nhiệm vụ của bạn là loại bỏ các thông tin nhiễu, không liên quan trong câu hỏi đầu vào để tạo ra một câu hỏi rõ ràng, ngắn gọn và chính xác hơn.

Quy tắc:
1. Giữ nguyên ý nghĩa cốt lõi của câu hỏi
2. Loại bỏ các thông tin dư thừa, không cần thiết
3. Đảm bảo câu hỏi tinh chỉnh vẫn có thể trả lời được dựa trên ngữ cảnh
4. Không thêm thông tin mới không có trong câu hỏi gốc

Chỉ trả lời với câu hỏi đã được tinh chỉnh, không giải thích."""


def create_prompt(tokenizer, query: str, context: str, label: Optional[str] = None) -> str:
    """Create prompt for the query refinement task using tokenizer.apply_chat_template"""
    user_prompt = f"""Câu hỏi gốc: {query}

Ngữ cảnh tham khảo: {context}

Câu hỏi tinh chỉnh:"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    if label:
        messages.append({"role": "assistant", "content": label})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
def load_data(train_path: str, eval_path: str):
    """Load training and evaluation data"""
    logger.info(f"Loading training data from {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    logger.info(f"Loading evaluation data from {eval_path}")
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Evaluation samples: {len(eval_data)}")
    
    # Log some statistics
    train_query_lengths = [len(item['query']) for item in train_data]
    train_label_lengths = [len(item['label']) for item in train_data]
    
    logger.info(f"Training query length - avg: {sum(train_query_lengths)/len(train_query_lengths):.1f}, "
                f"max: {max(train_query_lengths)}, min: {min(train_query_lengths)}")
    logger.info(f"Training label length - avg: {sum(train_label_lengths)/len(train_label_lengths):.1f}, "
                f"max: {max(train_label_lengths)}, min: {min(train_label_lengths)}")
    
    return train_data, eval_data

def prepare_dataset(data: list, tokenizer, max_context_length: int = 4096) -> Dataset:
    """Prepare dataset with formatted prompts"""
    formatted_data = []
    for item in data:
        # Truncate context to avoid very long sequences
        context = item['context'][:max_context_length]
        
        text = create_prompt(
            tokenizer=tokenizer,
            query=item['query'],
            context=context,
            label=item['label']
        )
        formatted_data.append({
            'text': text,
            'query': item['query'],
            'context': context,
            'label': item['label']
        })
    
    return Dataset.from_list(formatted_data)


def evaluate_model(model, tokenizer, eval_dataset, config: ModelConfig):
    """Run evaluation and compute metrics"""
    logger.info("Running evaluation...")
    
    model.eval()
    predictions = []
    labels = []
    
    valid_labels = ["Relevant", "Irrelevant", "Contain Answer"]
    
    for i, item in enumerate(eval_dataset):
        # Create prompt without label
        prompt = create_prompt(
            tokenizer=tokenizer,
            question=item['question'],
            context=item['context'],
            label=None
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the prediction (last part after "Classification:")
        pred = response.split("Classification:")[-1].strip()
        
        # Map to valid label
        pred_label = None
        for valid_label in valid_labels:
            if valid_label.lower() in pred.lower():
                pred_label = valid_label
                break
        
        if pred_label is None:
            pred_label = "Irrelevant"  # Default fallback
        
        predictions.append(pred_label)
        labels.append(item['label'])
        
        if (i + 1) % 50 == 0:
            logger.info(f"Evaluated {i + 1}/{len(eval_dataset)} samples")
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    weighted_f1 = f1_score(labels, predictions, labels=valid_labels, average='weighted')
    weighted_precision = precision_score(labels, predictions, labels=valid_labels, average='weighted')
    weighted_recall = recall_score(labels, predictions, labels=valid_labels, average='weighted')
    
    logger.info(f"\n{'=' * 50}")
    logger.info(f"EVALUATION RESULTS")
    logger.info(f"{'=' * 50}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    logger.info(f"Weighted Precision: {weighted_precision:.4f}")
    logger.info(f"Weighted Recall: {weighted_recall:.4f}")
    
    logger.info("\nClassification Report:")
    report = classification_report(labels, predictions, labels=valid_labels)
    report_dict = classification_report(labels, predictions, labels=valid_labels, output_dict=True)
    logger.info(f"\n{report}")
    
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(labels, predictions, labels=valid_labels)
    logger.info(f"\n{cm}")
    
    return {
        'accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'classification_report': report_dict,
        'predictions': predictions,
        'labels': labels
    }


def main(resume_from_checkpoint: Optional[str] = None, epochs: Optional[int] = None):
    """Main training function
    
    Args:
        resume_from_checkpoint: Path to checkpoint directory to resume training from.
                               e.g., "./qwen_lora_relevance/checkpoint-3390"
        epochs: Total number of epochs to train. When resuming, this is the NEW total
                (e.g., if trained for 3 epochs and want to continue to 10, set epochs=10)
    """
    config = ModelConfig()
    
    # Override resume_from_checkpoint if provided
    if resume_from_checkpoint:
        config.resume_from_checkpoint = resume_from_checkpoint
    
    # Override epochs if provided
    if epochs is not None:
        config.num_train_epochs = epochs
    
    logger.info("=" * 60)
    if config.resume_from_checkpoint:
        logger.info("Resuming LoRA Fine-tuning from checkpoint")
        logger.info(f"Checkpoint: {config.resume_from_checkpoint}")
        logger.info(f"Will train to total epochs: {config.num_train_epochs}")
    else:
        logger.info("Starting LoRA Fine-tuning for Document Relevance Classification")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Max sequence length: {config.max_seq_length}")
    logger.info(f"LoRA rank: {config.lora_r}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.num_train_epochs}")
    
    # Load data
    # train_data, eval_data = load_data(
    #     train_path="multihop_train_data_v2.json",
    #     eval_path="multihop_eval_data_v2.json"
    # )
    train_data, eval_data = load_data(
        train_path="query_refiner_train.json",
        eval_path="query_refiner_eval.json"
    )
    
    # Load model with Unsloth
    if config.resume_from_checkpoint:
        # Resume from checkpoint - load the checkpoint directly
        logger.info(f"Loading model from checkpoint: {config.resume_from_checkpoint}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.resume_from_checkpoint,
            max_seq_length=config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=config.load_in_4bit,
        )
    else:
        # Fresh training - load base model
        logger.info("Loading model with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=config.load_in_4bit,
        )
    
    # Prepare datasets (after loading tokenizer)
    train_dataset = prepare_dataset(train_data, tokenizer)
    eval_dataset = prepare_dataset(eval_data, tokenizer)
    
    logger.info(f"Prepared {len(train_dataset)} training samples")
    logger.info(f"Prepared {len(eval_dataset)} evaluation samples")
    
    # Add LoRA adapters (only for fresh training, checkpoint already has them)
    if not config.resume_from_checkpoint:
        logger.info("Adding LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
            use_rslora=False,
            loftq_config=None,
        )
    else:
        logger.info("Using LoRA adapters from checkpoint")
        # Enable gradient checkpointing for continued training
        model.gradient_checkpointing_enable()
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Create TensorBoard log directory with timestamp
    tensorboard_log_dir = f"{config.logging_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        eval_strategy=config.evaluation_strategy,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        seed=config.seed,
        report_to="tensorboard",  # Enable TensorBoard logging
        logging_dir=tensorboard_log_dir,  # TensorBoard log directory
        logging_first_step=True,
    )
    
    # Create custom callback
    epoch_callback = EpochLoggingCallback(
        logger=logger,
        eval_data=eval_data,
        tokenizer=tokenizer,
        create_prompt_fn=create_prompt
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
        callbacks=[epoch_callback],
    )
    
    # Train
    logger.info("Starting training...")
    # Note: We don't use trainer.train(resume_from_checkpoint=...) because:
    # 1. We already loaded the model weights from the checkpoint
    # 2. Using resume_from_checkpoint would restore the old epoch counter,
    #    preventing training for additional epochs beyond the original num_train_epochs
    # Instead, we just train with the loaded weights as a "warm start"
    train_result = trainer.train()
    
    # Log training results
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 50)
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    logger.info(f"Total steps: {train_result.global_step}")
    
    # Save model
    logger.info(f"Saving model to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Save LoRA adapters separately (only for fresh training, skip for resumed training to avoid merge issues)
    if not config.resume_from_checkpoint:
        try:
            lora_dir = f"{config.output_dir}/lora_adapters"
            logger.info(f"Saving LoRA adapters to {lora_dir}")
            model.save_pretrained_merged(lora_dir, tokenizer, save_method="lora")
        except Exception as e:
            logger.warning(f"Failed to save merged LoRA adapters: {e}")
            logger.info("LoRA adapters are still saved in the main output directory")
    else:
        logger.info("Skipping save_pretrained_merged for resumed training (adapters saved in main output dir)")
    
    # Final evaluation
    logger.info("\nRunning final evaluation on best model...")
    FastLanguageModel.for_inference(model)  # Enable faster inference
    eval_results = evaluate_model(model, tokenizer, eval_data, config)
    
    # Save evaluation results
    eval_output_path = f"{config.output_dir}/evaluation_results.json"
    with open(eval_output_path, 'w') as f:
        json.dump({
            'accuracy': eval_results['accuracy'],
            'weighted_f1': eval_results['weighted_f1'],
            'weighted_precision': eval_results['weighted_precision'],
            'weighted_recall': eval_results['weighted_recall'],
            'classification_report': eval_results['classification_report'],
            'config': {
                'model_name': config.model_name,
                'lora_r': config.lora_r,
                'lora_alpha': config.lora_alpha,
                'learning_rate': config.learning_rate,
                'epochs': config.num_train_epochs,
            }
        }, f, indent=2)
    
    logger.info(f"Evaluation results saved to {eval_output_path}")
    logger.info("Training pipeline completed successfully!")
    
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with LoRA for document relevance classification")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from (e.g., ./qwen_lora_relevance/checkpoint-3390)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total number of training epochs. When resuming from epoch 3 to train until epoch 10, use --epochs 10"
    )
    args = parser.parse_args()
    
    main(resume_from_checkpoint=args.resume_from_checkpoint, epochs=args.epochs)
