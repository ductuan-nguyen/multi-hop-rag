import time
import logging
import re
from typing import List, Dict, Any, Optional

from config import IterRetGenConfig
from retriever import VectorDatabase
from generator import LLMGenerator

# --- Core Logic: SELF-ASK Class ---
class SelfAsk:
    def __init__(self, config: IterRetGenConfig, documents: List[str], api_key: str, doc_metadata: Optional[List[Dict[str, str]]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.logger.info(f"Initializing SelfAsk with {len(documents)} documents")
        self.retriever = VectorDatabase(config, documents)
        self.llm = LLMGenerator(config, api_key)
        # Store document metadata (id, title) for retrieval logging
        self.doc_metadata = doc_metadata if doc_metadata else []
        
        # Self-Ask prompt template (few-shot examples)
        self.prompt_template = ['''Question: Ai sống lâu hơn, Muhammad Ali hay Alan Turing?
Are follow up questions needed here: Yes.
Follow up: Muhammad Ali mất khi bao nhiêu tuổi?
Intermediate answer: Muhammad Ali mất khi 74 tuổi.
Follow up: Alan Turing mất khi bao nhiêu tuổi?
Intermediate answer: Alan Turing mất khi 41 tuổi.
So the final answer is: Muhammad Ali

Question: Người sáng lập craigslist sinh năm nào?
Are follow up questions needed here: Yes.
Follow up: Ai là người sáng lập craigslist?
Intermediate answer: Craigslist được sáng lập bởi Craig Newmark.
Follow up: Craig Newmark sinh năm nào?
Intermediate answer: Craig Newmark sinh ngày 6 tháng 12 năm 1952.
So the final answer is: 6 tháng 12 năm 1952

Question: Ai là ông ngoại của George Washington?
Are follow up questions needed here: Yes.
Follow up: Mẹ của George Washington là ai?
Intermediate answer: Mẹ của George Washington là Mary Ball Washington.
Follow up: Cha của Mary Ball Washington là ai?
Intermediate answer: Cha của Mary Ball Washington là Joseph Ball.
So the final answer is: Joseph Ball

Question: ''', 
'''
Are follow up questions needed here:''', ]
        
        self.intermediate = "\nIntermediate answer:"
        self.followup = "Follow up:"
        self.finalans = '\nSo the final answer is:'
        
        self.logger.info("SelfAsk initialized successfully")

    def _get_answer_from_search(self, question: str):
        k = max(1, int(getattr(self.config, "top_k", 3) or 3))
        docs, ids, scores = self.retriever.search(question, k=k, return_ids=True)

        print("\n================ RETRIEVAL DEBUG ================")
        print("QUERY:", question)
        # avoid spam
        for i, (doc_id, score, doc) in enumerate(zip(ids, scores, docs[: min(3, len(docs))])):
            print(f"\n--- DOC {i} (ID: {doc_id}, Score: {score:.4f}) ---")
            print(doc[:800])  # cắt để khỏi spam
        print("=================================================\n")

        # Return the first document and retrieval info
        first_doc = docs[0] if docs else ""
        retrieval_info = {
            "query": question,
            "k": k,
            "retrieved_docs": []
        }
        
        # Build retrieval info with id and title from metadata
        for doc_id, score, doc in zip(ids, scores, docs):
            doc_index = int(doc_id)
            doc_info = {
                "doc_index": doc_index,
                "score": float(score),
                "content_preview": doc[:500] + "..." if len(doc) > 500 else doc
            }
            
            # Add id and title from metadata if available
            if doc_index < len(self.doc_metadata):
                metadata = self.doc_metadata[doc_index]
                doc_info["doc_id"] = metadata.get("id", "")
                doc_info["doc_title"] = metadata.get("title", "")
            
            retrieval_info["retrieved_docs"].append(doc_info)
        
        return first_doc, retrieval_info

    def _extract_question(self, text: str) -> Optional[str]:
        lines = text.splitlines()
        for line in reversed(lines):
            if line.strip().startswith("Follow up:"):
                return line.split("Follow up:", 1)[1].strip()
        return None

    def _get_last_line(self, text: str) -> str:
        """Get the last line of text"""
        if '\n' not in text:
            return text
        else:
            return text.split('\n')[-1]

    def solve(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.info(f"Starting SelfAsk solve for query: '{query}'")

        # ---- Build initial prompt ----
        cur_prompt = self.prompt_template[0] + query + self.prompt_template[1]

        prompt_history = []
        all_followups = []
        all_intermediate_answers = []
        retrieval_history = []

        # ---- First LLM call ----
        ret_text = self.llm.generate(cur_prompt)

        prompt_history.append({
            "step": 0,
            "prompt": cur_prompt,
            "response": ret_text,
            "type": "initial"
        })

        iteration = 0
        max_iterations = self.config.max_iterations

        # ---- Main Self-Ask loop ----
        while iteration < max_iterations:
            iteration += 1

            # 1. Extract follow-up question
            followup_question = self._extract_question(ret_text)
            if followup_question is None:
                self.logger.info("No more follow-up questions found.")
                break

            if followup_question in all_followups:
                self.logger.warning("Repeated follow-up detected. Stopping loop.")
                break

            all_followups.append(followup_question)
            self.logger.info(f"[Iter {iteration}] Follow-up: {followup_question}")

            # 2. Retrieve answer from vector DB
            external_answer, retrieval_info = self._get_answer_from_search(followup_question)
            # Save retrieval info
            retrieval_history.append(retrieval_info)
            
            if external_answer is None or external_answer == "":
                external_answer = ""
                self.logger.warning("No retrieval answer found.")
            else:
                # Increase limit to 1000 chars to preserve more context
                if len(external_answer) > 1000:
                    external_answer = external_answer[:1000] + "..."

            all_intermediate_answers.append(external_answer)

            # 3. Append to prompt
            cur_prompt += f"\nFollow up: {followup_question}"
            cur_prompt += f"{self.intermediate} {external_answer}"
            
            # Add instruction to keep answers concise
            if iteration == 1:  # Only add once
                cur_prompt += "\n(Lưu ý: Khi trả lời, chỉ cần đưa ra thông tin ngắn gọn từ tài liệu, không cần giải thích dài dòng)"

            # 4. Generate next step
            ret_text = self.llm.generate(cur_prompt)

            prompt_history.append({
                "step": iteration,
                "prompt": cur_prompt,
                "response": ret_text,
                "type": "followup",
                "followup_question": followup_question,
                "intermediate_answer": external_answer
            })

            # 5. Stop if final answer is produced
            if self.finalans in ret_text:
                self.logger.info("Final answer detected. Stopping Self-Ask loop.")
                break

        # ---- Ensure final answer exists ----
        if self.finalans not in ret_text:
            # Add instruction to ensure short answer format
            cur_prompt += self.finalans + " "
            # Add instruction to return only the answer, not explanation
            cur_prompt += " (Chỉ trả về câu trả lời ngắn gọn, không giải thích thêm) "
            ret_text = self.llm.generate(cur_prompt)

            prompt_history.append({
                "step": iteration + 1,
                "prompt": cur_prompt,
                "response": ret_text,
                "type": "final"
            })

        # Extract final answer from the last response, not from the full prompt
        final_answer = self._extract_final_answer(ret_text)

        latency = time.time() - start_time
        self.logger.info(
            f"SelfAsk completed in {latency:.2f}s | "
            f"Follow-ups: {len(all_followups)}"
        )

        return {
            "final_answer": final_answer,
            "full_response": cur_prompt,
            "followup_questions": all_followups,
            "intermediate_answers": all_intermediate_answers,
            "retrieval_history": retrieval_history,
            "prompt_history": prompt_history,
            "latency": latency
        }


    def _extract_final_answer(self, response_text: str) -> str:
        """Extract the final answer from the LLM response"""
        import re
        
        if not response_text or not response_text.strip():
            return ""
        
        # Look for "So the final answer is:" in the response
        if self.finalans in response_text:
            parts = response_text.split(self.finalans)
            if len(parts) > 1:
                answer = parts[-1].strip()
                # Remove any trailing newlines and clean up
                answer = answer.split('\n')[0].strip()
                # Remove instruction text if present
                if "(Chỉ trả về" in answer:
                    answer = answer.split("(Chỉ trả về")[0].strip()
                # Remove trailing period if present
                if answer.endswith('.'):
                    answer = answer[:-1].strip()
                # Normalize leading zeros in numbers (05 -> 5, 07 -> 7)
                answer = re.sub(r'\b0+(\d+)\b', r'\1', answer)
                return answer
        
        # If no "So the final answer is:" found, try to find a short answer
        # Look for patterns like "X ngày", "X đồng", numbers, etc.
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        # First, try to find a short line (likely the answer)
        for line in reversed(lines):
            # Skip very long lines (likely explanations)
            if len(line) > 150:
                continue
            # Look for common answer patterns
            if any(keyword in line.lower() for keyword in ['ngày', 'đồng', 'tháng', 'năm', 'giờ', 'phút']):
                # Clean up the answer
                answer = line.strip()
                # Remove instruction text if present
                if "(Chỉ trả về" in answer or "(Lưu ý:" in answer:
                    answer = answer.split("(Chỉ trả về")[0].split("(Lưu ý:")[0].strip()
                if answer.endswith('.'):
                    answer = answer[:-1].strip()
                # Normalize leading zeros
                answer = re.sub(r'\b0+(\d+)\b', r'\1', answer)
                return answer
        
        # If still no answer, try to extract from the last short line
        for line in reversed(lines):
            if len(line) <= 100:  # Short line might be the answer
                answer = line.strip()
                # Remove instruction text if present
                if "(Chỉ trả về" in answer or "(Lưu ý:" in answer:
                    answer = answer.split("(Chỉ trả về")[0].split("(Lưu ý:")[0].strip()
                if answer.endswith('.'):
                    answer = answer[:-1].strip()
                # Normalize leading zeros
                answer = re.sub(r'\b0+(\d+)\b', r'\1', answer)
                return answer
        
        # Last resort: return the last line
        if lines:
            return lines[-1]
        
        return ""
