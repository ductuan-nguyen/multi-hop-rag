import json
from collections import Counter
from typing import Dict, List

def analyze_evaluation_results(file_path: str = 'evaluation_results.json'):
    """Phân tích chi tiết kết quả evaluation"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metrics = data['metrics']
    results = data['detailed_results']
    
    print("="*80)
    print("PHÂN TÍCH KẾT QUẢ EVALUATION")
    print("="*80)
    
    # 1. Tổng quan metrics
    print("\n📊 TỔNG QUAN METRICS:")
    print("-" * 80)
    print(f"Tổng số câu hỏi: {data['total_results']}")
    print(f"\n📝 Answer Task:")
    print(f"  • Accuracy:        {metrics['answer_task']['accuracy']:.1%} ({metrics['answer_task']['accuracy']*data['total_results']:.0f}/{data['total_results']})")
    print(f"  • Exact Match:     {metrics['answer_task']['exact_match']:.1%}")
    print(f"  • F1 Score:        {metrics['answer_task']['f1_score']:.3f}")
    print(f"  • Acc† (Model):    {metrics['answer_task']['acc_dagger']:.1%}")
    print(f"\n🔍 Retrieval Task:")
    print(f"  • Recall@3:        {metrics['retrieval_task']['recall_at_3']:.1%} ({metrics['retrieval_task']['recall_at_3']*data['total_results']:.0f}/{data['total_results']})")
    print(f"\n⚡ Performance:")
    print(f"  • Avg Latency:     {metrics['performance']['avg_latency']:.2f}s")
    
    # 2. Phân tích các loại lỗi
    print("\n" + "="*80)
    print("🔍 PHÂN TÍCH CÁC LOẠI LỖI")
    print("="*80)
    
    correct_answers = []
    wrong_answers = []
    retrieval_failures = []
    answer_format_issues = []
    partial_correct = []
    
    for r in results:
        if r['answer_metrics']['accuracy'] == 1.0:
            correct_answers.append(r)
        else:
            wrong_answers.append(r)
            
            # Phân loại lỗi
            if r['retrieval_metrics']['recall_at_3'] == 0.0:
                retrieval_failures.append(r)
            
            if r['answer_metrics']['f1'] > 0.5 and r['answer_metrics']['f1'] < 1.0:
                partial_correct.append(r)
            
            # Kiểm tra format issues
            pred = r['prediction'].lower()
            if any(phrase in pred for phrase in ['không có thông tin', 'does not contain', 'the provided text']):
                answer_format_issues.append(r)
    
    print(f"\n✅ Câu trả lời đúng: {len(correct_answers)}/{data['total_results']} ({len(correct_answers)/data['total_results']:.1%})")
    print(f"❌ Câu trả lời sai: {len(wrong_answers)}/{data['total_results']} ({len(wrong_answers)/data['total_results']:.1%})")
    print(f"\n📊 Phân loại lỗi:")
    print(f"  • Retrieval thất bại (Recall@3=0): {len(retrieval_failures)} ({len(retrieval_failures)/len(wrong_answers):.1%} của các lỗi)")
    print(f"  • Partial correct (F1 > 0.5):      {len(partial_correct)} ({len(partial_correct)/len(wrong_answers):.1%} của các lỗi)")
    print(f"  • Format issues (không có thông tin): {len(answer_format_issues)} ({len(answer_format_issues)/len(wrong_answers):.1%} của các lỗi)")
    
    # 3. Phân tích mối quan hệ giữa Retrieval và Answer
    print("\n" + "="*80)
    print("🔗 MỐI QUAN HỆ GIỮA RETRIEVAL VÀ ANSWER")
    print("="*80)
    
    retrieval_success_answer_correct = 0
    retrieval_success_answer_wrong = 0
    retrieval_fail_answer_correct = 0
    retrieval_fail_answer_wrong = 0
    
    for r in results:
        recall = r['retrieval_metrics']['recall_at_3']
        accuracy = r['answer_metrics']['accuracy']
        
        if recall == 1.0 and accuracy == 1.0:
            retrieval_success_answer_correct += 1
        elif recall == 1.0 and accuracy == 0.0:
            retrieval_success_answer_wrong += 1
        elif recall == 0.0 and accuracy == 1.0:
            retrieval_fail_answer_correct += 1
        elif recall == 0.0 and accuracy == 0.0:
            retrieval_fail_answer_wrong += 1
    
    print(f"\nRetrieval thành công + Answer đúng:  {retrieval_success_answer_correct} ({retrieval_success_answer_correct/data['total_results']:.1%})")
    print(f"Retrieval thành công + Answer sai:   {retrieval_success_answer_wrong} ({retrieval_success_answer_wrong/data['total_results']:.1%})")
    print(f"Retrieval thất bại + Answer đúng:    {retrieval_fail_answer_correct} ({retrieval_fail_answer_correct/data['total_results']:.1%})")
    print(f"Retrieval thất bại + Answer sai:     {retrieval_fail_answer_wrong} ({retrieval_fail_answer_wrong/data['total_results']:.1%})")
    
    # 4. Phân tích các vấn đề cụ thể
    print("\n" + "="*80)
    print("⚠️  CÁC VẤN ĐỀ CỤ THỂ")
    print("="*80)
    
    # Format issues
    print(f"\n1. Format Issues ({len(answer_format_issues)} cases):")
    for i, r in enumerate(answer_format_issues[:5], 1):
        print(f"\n   Case {i}:")
        print(f"   Q: {r['question'][:100]}...")
        print(f"   GT: {r['ground_truth']}")
        print(f"   Pred: {r['prediction'][:100]}...")
        print(f"   Retrieval: Recall@3 = {r['retrieval_metrics']['recall_at_3']}")
    
    # Partial matches
    print(f"\n2. Partial Matches - Gần đúng nhưng format khác ({len(partial_correct)} cases):")
    for i, r in enumerate(partial_correct[:5], 1):
        print(f"\n   Case {i}:")
        print(f"   Q: {r['question'][:80]}...")
        print(f"   GT: {r['ground_truth']}")
        print(f"   Pred: {r['prediction']}")
        print(f"   F1: {r['answer_metrics']['f1']:.3f}, Acc†: {r['answer_metrics']['acc_dagger']}")
    
    # Retrieval failures
    print(f"\n3. Retrieval Failures ({len(retrieval_failures)} cases):")
    print(f"   - Khi retrieval thất bại, accuracy: {sum(1 for r in retrieval_failures if r['answer_metrics']['accuracy']==1.0)/len(retrieval_failures) if retrieval_failures else 0:.1%}")
    
    # 5. Phân tích theo F1 score ranges
    print("\n" + "="*80)
    print("📈 PHÂN TÍCH THEO F1 SCORE")
    print("="*80)
    
    f1_ranges = {
        "Perfect (F1=1.0)": [],
        "Good (0.7 <= F1 < 1.0)": [],
        "Fair (0.4 <= F1 < 0.7)": [],
        "Poor (0 < F1 < 0.4)": [],
        "Zero (F1=0)": []
    }
    
    for r in results:
        f1 = r['answer_metrics']['f1']
        if f1 == 1.0:
            f1_ranges["Perfect (F1=1.0)"].append(r)
        elif f1 >= 0.7:
            f1_ranges["Good (0.7 <= F1 < 1.0)"].append(r)
        elif f1 >= 0.4:
            f1_ranges["Fair (0.4 <= F1 < 0.7)"].append(r)
        elif f1 > 0:
            f1_ranges["Poor (0 < F1 < 0.4)"].append(r)
        else:
            f1_ranges["Zero (F1=0)"].append(r)
    
    for range_name, cases in f1_ranges.items():
        print(f"\n{range_name}: {len(cases)} cases ({len(cases)/data['total_results']:.1%})")
        if cases:
            avg_recall = sum(c['retrieval_metrics']['recall_at_3'] for c in cases) / len(cases)
            print(f"  • Avg Recall@3: {avg_recall:.1%}")
    
    # 6. Recommendations
    print("\n" + "="*80)
    print("💡 KHUYẾN NGHỊ")
    print("="*80)
    
    print("\n1. Retrieval Task:")
    recall_rate = metrics['retrieval_task']['recall_at_3']
    if recall_rate < 0.5:
        print(f"   ⚠️  Recall@3 thấp ({recall_rate:.1%}) - Cần cải thiện:")
        print(f"      • Tăng số lượng retrieved docs (k > 3)")
        print(f"      • Cải thiện embedding model hoặc fine-tuning")
        print(f"      • Thử re-ranking")
    else:
        print(f"   ✅ Recall@3 ở mức chấp nhận được ({recall_rate:.1%})")
    
    print("\n2. Answer Task:")
    accuracy = metrics['answer_task']['accuracy']
    if accuracy < 0.3:
        print(f"   ⚠️  Accuracy rất thấp ({accuracy:.1%}) - Các vấn đề chính:")
        print(f"      • {len(answer_format_issues)} cases: LLM trả về 'không có thông tin'")
        print(f"      • {len(partial_correct)} cases: Gần đúng nhưng format khác (F1 > 0.5)")
        print(f"      • Cần cải thiện prompt để LLM trả về đúng format")
        print(f"      • Cần cải thiện hàm extract_final_answer để xử lý các trường hợp edge case")
    
    print("\n3. Cải thiện cụ thể:")
    if len(answer_format_issues) > 0:
        print(f"   • Xử lý {len(answer_format_issues)} cases LLM nói 'không có thông tin':")
        print(f"     - Cải thiện prompt để buộc LLM trả về câu trả lời")
        print(f"     - Tăng độ dài intermediate_answer (hiện tại 1000 chars)")
        print(f"     - Thử retrieve nhiều docs hơn và combine")
    
    if len(partial_correct) > 0:
        print(f"   • Xử lý {len(partial_correct)} cases format khác:")
        print(f"     - Normalize answer extraction (xử lý '05' vs '5', 'Ngày' vs 'ngày')")
        print(f"     - Cải thiện hàm normalize_answer trong evaluator")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_evaluation_results()
