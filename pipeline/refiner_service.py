"""
Query Refiner Service - FastAPI server for query refinement using fine-tuned Qwen model.

This service hosts the query refiner model and exposes an HTTP API for refinement.
Multiple clients can call this service concurrently without CUDA OOM issues.

Usage:
    python refiner_service.py --model_path path/to/model --port 8002

API Endpoints:
    POST /refine - Refine a single query with context
    POST /refine_batch - Refine multiple query-context pairs
    GET /health - Health check
"""

import argparse
import torch
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from unsloth import FastLanguageModel


class QueryRefinerService:
    """Service for query refinement with thread-safe inference"""
    
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
        
        # Thread pool for CPU-bound tokenization
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Lock for thread-safe GPU access
        self._lock = asyncio.Lock()
        
        self.device = next(self.model.parameters()).device
        print(f"Query refiner loaded successfully on device: {self.device}")
    
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
    
    def _tokenize(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Tokenize prompt (CPU operation)"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        return inputs
    
    @torch.no_grad()
    def _inference(self, inputs: Dict[str, torch.Tensor]) -> str:
        """Run model inference (GPU operation)"""
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
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
        return response
    
    async def refine(self, query: str, context: str) -> str:
        """
        Refine a query based on the given context.
        Thread-safe with async lock.
        
        Args:
            query: The original query
            context: The context to use for refinement
            
        Returns:
            The refined query
        """
        # Truncate context if needed
        context = context[:self.max_context_length]
        
        prompt = self._create_prompt(query=query, context=context)
        
        # Tokenize (can run in thread pool)
        loop = asyncio.get_event_loop()
        inputs = await loop.run_in_executor(
            self.executor,
            self._tokenize,
            prompt
        )
        
        # Inference with lock to prevent concurrent GPU access
        async with self._lock:
            response = self._inference(inputs)
        
        return self._extract_refined_query(response)
    
    async def refine_batch(self, query: str, contexts: List[str]) -> List[str]:
        """
        Refine a query with multiple contexts.
        
        Args:
            query: The original query
            contexts: List of contexts to use for refinement
            
        Returns:
            List of refined queries
        """
        results = []
        for context in contexts:
            refined = await self.refine(query, context)
            results.append(refined)
        return results


# Global service instance
refiner_service: Optional[QueryRefinerService] = None


def init_refiner_service(
    model_path: str,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    max_context_length: int = 2048,
    max_new_tokens: int = 256
):
    """Initialize the global refiner service"""
    global refiner_service
    refiner_service = QueryRefinerService(
        model_path=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        max_context_length=max_context_length,
        max_new_tokens=max_new_tokens
    )
    return refiner_service


# Request/Response models
class RefineRequest(BaseModel):
    query: str
    context: str


class RefineBatchRequest(BaseModel):
    query: str
    contexts: List[str]


class RefineResponse(BaseModel):
    refined_query: str


class RefineBatchResponse(BaseModel):
    refined_queries: List[str]


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Query refiner service ready!")
    yield
    # Shutdown
    if refiner_service:
        refiner_service.executor.shutdown(wait=True)
    print("Query refiner service shutdown complete.")


app = FastAPI(
    title="Query Refiner Service",
    description="API for refining queries using fine-tuned Qwen model",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": refiner_service is not None,
        "device": str(refiner_service.device) if refiner_service else None
    }


@app.post("/refine", response_model=RefineResponse)
async def refine(request: RefineRequest):
    """Refine a single query with context"""
    if refiner_service is None:
        raise HTTPException(status_code=503, detail="Refiner not initialized")
    
    refined_query = await refiner_service.refine(request.query, request.context)
    return {"refined_query": refined_query}


@app.post("/refine_batch", response_model=RefineBatchResponse)
async def refine_batch(request: RefineBatchRequest):
    """Refine a query with multiple contexts"""
    if refiner_service is None:
        raise HTTPException(status_code=503, detail="Refiner not initialized")
    
    refined_queries = await refiner_service.refine_batch(request.query, request.contexts)
    return {"refined_queries": refined_queries}


def main():
    global refiner_service
    
    parser = argparse.ArgumentParser(description="Query Refiner Service")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to query refiner model"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8002,
        help="Port to bind"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit quantization"
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
        help="Maximum context length for truncation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Initialize refiner service
    init_refiner_service(
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        max_context_length=args.max_context_length,
        max_new_tokens=args.max_new_tokens
    )
    
    # Run server
    print(f"\nStarting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
