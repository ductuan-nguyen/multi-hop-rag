import os
import json
import logging
from dotenv import load_dotenv

from config import IterRetGenConfig
from self_ask import SelfAsk

load_dotenv()

# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    document_file_path = 'self_ask\datasets\db_web_mining.json'
    my_documents = []
    doc_metadata = []  # Store id and title for each document
    
    with open(document_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)   # ⬅️ đọc TOÀN BỘ array
    
    for obj in data:
        my_documents.append(obj["contents"])
        # Store metadata (id, title) for retrieval logging
        doc_metadata.append({
            "id": obj.get("doc_id", obj.get("id", "")),
            "title": obj.get("title", "")
        })
    
    print(len(my_documents))

    

    # Set index path based on document file (saves to same directory)
    # index_path = os.path.join(os.path.dirname(document_file_path), 'faiss_index')
    index_path = 'faiss/filted/faiss_index'
    my_config = IterRetGenConfig(index_path=index_path)
    
    # Initialize
    # Replace with your actual API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in your .env file.")
    
    # Use SelfAsk instead of IterRetGen
    self_ask = SelfAsk(my_config, my_documents, api_key=api_key, doc_metadata=doc_metadata)
    
    # Run - Test với 1 item đầu tiên thôi
    with open(
        'self_ask\datasets\multi_hop_ihanoi_v1.0.json',
        'r',
        encoding='utf-8'
    ) as f:
        data = json.load(f)

    # Chỉ test với item đầu tiên
    
    # Uncomment để chạy tất cả items:
    # comment full
    for i,item in enumerate(data[:30]):
        query = item['multi_hop_question']
        result = self_ask.solve(query)
        print(f"Final Answer: {result['final_answer']}")
        print(f"Time Taken: {result['latency']:.2f}s")
        item['final_answer'] = result['final_answer']
        item['latency'] = result['latency']
        item['followup_questions'] = result['followup_questions']
        item['intermediate_answers'] = result['intermediate_answers']
        item['prompt_history'] = result['prompt_history']  # Lưu lịch sử prompts
        item['retrieval_history'] = result['retrieval_history']
        os.makedirs('result_iter', exist_ok=True)
        with open(f'result_iter/result_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(item, f, indent=4, ensure_ascii=False)
