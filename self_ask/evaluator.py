import numpy as np
import logging
import string
import re
from collections import Counter
from typing import List, Dict, Any, Optional

from generator import LLMGenerator

# --- Evaluation Metrics ---
class Evaluator:
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        import re
        def remove_articles(text):
            return " ".join([word for word in text.split() if word not in ["a", "an", "the"]])
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        def normalize_numbers(text):
            # Normalize leading zeros: "05" -> "5", "07" -> "7", etc.
            # But keep numbers like "100.000" as is (they have dots/commas)
            # Match standalone numbers with leading zeros
            def replace_leading_zero(match):
                num_str = match.group(0)
                # Only normalize simple integers, not numbers with dots/commas
                if '.' not in num_str and ',' not in num_str:
                    try:
                        return str(int(num_str))
                    except:
                        return num_str
                return num_str
            # Match numbers at word boundaries (standalone numbers)
            text = re.sub(r'\b0+(\d+)\b', replace_leading_zero, text)
            return text
        
        normalized = white_space_fix(remove_articles(remove_punc(lower(s))))
        normalized = normalize_numbers(normalized)
        return normalized

    def exact_match_score(self, prediction, ground_truth):
        return (self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0: return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _parse_binary_judgment(self, text: str) -> Optional[int]:
        """
        Parse an LLM judge output into {1, 0} when possible.

        We accept common variants because the judge may answer in Vietnamese ("Có/Không")
        or add extra text after the first token.
        """
        if not text:
            return None

        t = text.strip().lower()
        if not t:
            return None

        # Prefer the first non-empty line for stricter parsing.
        first_line = ""
        for line in t.splitlines():
            line = line.strip()
            if line:
                first_line = line
                break
        if not first_line:
            first_line = t

        # Normalize punctuation-like wrappers around the first token.
        first_line = re.sub(r'^[\s"“”\'`]+', "", first_line)
        first_line = re.sub(r'[\s"“”\'`]+$', "", first_line)

        def _begins_with_token(s: str, tok: str) -> bool:
            # Avoid relying on \b for Vietnamese diacritics; use explicit separators.
            if s == tok:
                return True
            for sep in (" ", "\t", "\r", "\n", ".", "!", "?", ",", ":", ";"):
                if s.startswith(tok + sep):
                    return True
            return False

        # Strict match at the beginning (English + Vietnamese).
        if _begins_with_token(first_line, "yes") or _begins_with_token(first_line, "true") or _begins_with_token(first_line, "correct"):
            return 1
        if _begins_with_token(first_line, "no") or _begins_with_token(first_line, "false") or _begins_with_token(first_line, "incorrect"):
            return 0

        if _begins_with_token(first_line, "có") or _begins_with_token(first_line, "co") or _begins_with_token(first_line, "đúng") or _begins_with_token(first_line, "dung"):
            return 1
        if _begins_with_token(first_line, "không") or _begins_with_token(first_line, "khong") or _begins_with_token(first_line, "sai"):
            return 0

        # Fallback: look anywhere for an unambiguous token.
        yes_hit = re.search(r"(^|[^a-z0-9_])(yes|true|correct|có|co|đúng|dung)($|[^a-z0-9_])", t)
        no_hit = re.search(r"(^|[^a-z0-9_])(no|false|incorrect|không|khong|sai)($|[^a-z0-9_])", t)
        if yes_hit and not no_hit:
            return 1
        if no_hit and not yes_hit:
            return 0

        return None

    def model_based_accuracy(self, prediction: str, ground_truth: str, llm_generator: LLMGenerator, question: Optional[str] = None) -> int:
        # Implementation of Acc† (Section 4.2 in paper)
        q = question or "(not provided)"
        prompt = f"""You are an impartial grader.

Question: {q}
Prediction: {prediction}
Ground-truth Answer: {ground_truth}

Does the Prediction imply the Ground-truth Answer for this Question?
Answer with exactly one token: YES or NO.
"""

        judgment = llm_generator.generate(prompt)
        parsed = self._parse_binary_judgment(judgment)
        if parsed is None:
            # Backwards-compatible fallback (very permissive).
            return 1 if "yes" in (judgment or "").lower() else 0
        return parsed

    def recall_at_k(self, retrieved_doc_ids: List[str], source_doc_id: str, target_doc_id: str, k: int = 3) -> float:
        """
        Calculate Recall@K for retrieval task.
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs (top K)
            source_doc_id: Ground truth source document ID
            target_doc_id: Ground truth target document ID
            k: Number of top documents to consider
        
        Returns:
            Recall@K score (1.0 if either source or target is in top K, 0.0 otherwise)
        """
        if not retrieved_doc_ids:
            return 0.0
        
        # Take top K retrieved documents
        top_k_ids = retrieved_doc_ids[:k]
        
        # Check if source_doc_id or target_doc_id is in top K
        if source_doc_id in top_k_ids or target_doc_id in top_k_ids:
            return 1.0
        return 0.0

    def accuracy(self, prediction: str, ground_truth: str, use_f1_threshold: bool = False, f1_threshold: float = 0.5) -> float:
        """
        Calculate accuracy for answer task.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            use_f1_threshold: If True, use F1 score threshold instead of exact match
            f1_threshold: F1 score threshold for accuracy (default 0.5)
        
        Returns:
            Accuracy score (1.0 if correct, 0.0 otherwise)
        """
        if use_f1_threshold:
            f1 = self.f1_score(prediction, ground_truth)
            return 1.0 if f1 >= f1_threshold else 0.0
        else:
            # Use exact match
            return 1.0 if self.exact_match_score(prediction, ground_truth) else 0.0

    def evaluate(self, results, ground_truths, llm_generator):
        logger = logging.getLogger(__name__)
        logger.info(f"Starting evaluation on {len(results)} results")
        metrics = {"EM": [], "F1": [], "Acc_dagger": [], "Latency": []}
        
        for idx, (res, gt) in enumerate(zip(results, ground_truths)):
            pred = res['final_answer']
            em = self.exact_match_score(pred, gt)
            f1 = self.f1_score(pred, gt)
            metrics["EM"].append(em)
            metrics["F1"].append(f1)
            # Acc† requires LLM call
            logger.debug(f"Evaluating result {idx+1}/{len(results)}: EM={em}, F1={f1:.3f}")
            acc_dagger = 0
            question = res.get("multi_hop_question")
            acc_dagger = self.model_based_accuracy(pred, gt, llm_generator, question=question)
            metrics["Acc_dagger"].append(acc_dagger)
            metrics["Latency"].append(res['latency'])
            
        final_metrics = {k: np.mean(v) for k, v in metrics.items()}
        logger.info(f"Evaluation completed. Final metrics: {final_metrics}")
        return final_metrics
