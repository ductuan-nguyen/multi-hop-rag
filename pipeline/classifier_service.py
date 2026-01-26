"""
Classifier Service - FastAPI server for BGE-M3 context classification.

This service hosts the classifier model and exposes an HTTP API for classification.
Multiple clients can call this service concurrently without CUDA OOM issues.

Usage:
    python classifier_service.py --checkpoint path/to/checkpoint.pt --port 8000

API Endpoints:
    POST /classify - Classify a single query-context pair
    POST /classify_batch - Classify multiple contexts for a single query
    GET /health - Health check
"""

import argparse
import torch
import torch.nn as nn
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from transformers import AutoTokenizer, AutoModel


# Label mapping
LABEL2ID = {"Relevant": 0, "Irrelevant": 1, "Contain Answer": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


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


class ClassifierService:
    """Singleton service for classifier inference with thread-safe queue"""
    
    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "BAAI/bge-m3",
        hidden_size: int = 1024,
        classifier_hidden_size: int = 512,
        max_length: int = 512,
        device: str = None,
        max_batch_size: int = 32
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.max_batch_size = max_batch_size
        
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
        
        # Thread pool for CPU-bound tokenization
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Lock for thread-safe GPU access
        self._lock = asyncio.Lock()
        
        print(f"Classifier loaded from {checkpoint_path}")
        print(f"Device: {self.device}")
        if 'accuracy' in checkpoint:
            print(f"Checkpoint accuracy: {checkpoint['accuracy']:.4f}")
    
    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize texts (CPU operation)"""
        encoding = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors='pt'
        )
        return encoding
    
    @torch.no_grad()
    def _inference(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run model inference (GPU operation)"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        outputs = self.model(input_ids, attention_mask)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)
        
        return probs.cpu()
    
    async def classify(self, query: str, context: str) -> Dict:
        """
        Classify a single query-context pair.
        Thread-safe with async lock.
        """
        text = f"Question: {query}\n\nContext: {context}"
        
        # Tokenize (can run in thread pool)
        loop = asyncio.get_event_loop()
        encoding = await loop.run_in_executor(
            self.executor, 
            self._tokenize, 
            [text]
        )
        
        # Inference with lock to prevent concurrent GPU access
        async with self._lock:
            probs = self._inference(
                encoding['input_ids'],
                encoding['attention_mask']
            )
        
        probs = probs.squeeze(0)
        pred_id = torch.argmax(probs).item()
        label = ID2LABEL[pred_id]
        
        probabilities = {
            ID2LABEL[i]: probs[i].item()
            for i in range(len(ID2LABEL))
        }
        
        return {
            "label": label,
            "probabilities": probabilities,
            "confidence": probabilities[label]
        }
    
    async def classify_batch(self, query: str, contexts: List[str]) -> List[Dict]:
        """
        Classify multiple contexts for a single query.
        Processes in batches to avoid OOM.
        """
        texts = [f"Question: {query}\n\nContext: {ctx}" for ctx in contexts]
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.max_batch_size):
            batch_texts = texts[i:i + self.max_batch_size]
            
            # Tokenize
            loop = asyncio.get_event_loop()
            encoding = await loop.run_in_executor(
                self.executor,
                self._tokenize,
                batch_texts
            )
            
            # Inference with lock
            async with self._lock:
                probs = self._inference(
                    encoding['input_ids'],
                    encoding['attention_mask']
                )
            
            # Process results
            for j in range(probs.shape[0]):
                prob = probs[j]
                pred_id = torch.argmax(prob).item()
                label = ID2LABEL[pred_id]
                
                probabilities = {
                    ID2LABEL[k]: prob[k].item()
                    for k in range(len(ID2LABEL))
                }
                
                results.append({
                    "label": label,
                    "probabilities": probabilities,
                    "confidence": probabilities[label]
                })
        
        return results


# Global service instance
classifier_service: Optional[ClassifierService] = None


def init_classifier_service(
    checkpoint_path: str,
    model_name: str = "BAAI/bge-m3",
    device: str = None,
    max_batch_size: int = 32
):
    """Initialize the global classifier service"""
    global classifier_service
    print("Loading classifier...")
    classifier_service = ClassifierService(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=device,
        max_batch_size=max_batch_size
    )
    return classifier_service


# Request/Response models
class ClassifyRequest(BaseModel):
    query: str
    context: str


class ClassifyBatchRequest(BaseModel):
    query: str
    contexts: List[str]


class ClassifyResponse(BaseModel):
    label: str
    probabilities: Dict[str, float]
    confidence: float


class ClassifyBatchResponse(BaseModel):
    results: List[ClassifyResponse]


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Classifier service ready!")
    yield
    # Shutdown
    if classifier_service:
        classifier_service.executor.shutdown(wait=True)
    print("Classifier service shutdown complete.")


app = FastAPI(
    title="BGE-M3 Context Classifier Service",
    description="API for classifying query-context pairs as Relevant/Irrelevant/Contain Answer",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": classifier_service is not None,
        "device": classifier_service.device if classifier_service else None
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    """Classify a single query-context pair"""
    if classifier_service is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    result = await classifier_service.classify(request.query, request.context)
    return result


@app.post("/classify_batch", response_model=ClassifyBatchResponse)
async def classify_batch(request: ClassifyBatchRequest):
    """Classify multiple contexts for a single query"""
    if classifier_service is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    results = await classifier_service.classify_batch(request.query, request.contexts)
    return {"results": results}


def main():
    global classifier_service
    
    parser = argparse.ArgumentParser(description="Classifier Service")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to classifier checkpoint"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-m3",
        help="Base model name"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=32,
        help="Maximum batch size for inference"
    )
    
    args = parser.parse_args()
    
    # Initialize classifier service
    init_classifier_service(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=args.device,
        max_batch_size=args.max_batch_size
    )
    
    # Run server - use app directly without reload to preserve global state
    print(f"\nStarting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
