import re
import json
import random
import os
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser
import openai
from openai import OpenAI
import time
import concurrent.futures
from tqdm import tqdm

def get_args():
    parser = ArgumentParser(
        description="(全功能版) 使用GPT-4o并行评测Pass@k，具备断点续传、智能重试和可配置API端点。"
    )
    parser.add_argument('--prediction_file', type=str, required=True, help="推理脚本生成的包含多次尝试的结果文件")
    parser.add_argument('--ground_truth_file', type=str, required=True, help="原始数据文件作为标准答案")
    parser.add_argument('--output_file', type=str, required=True, help="保存详细评测结果的JSON文件")
    parser.add_argument(
        '--api_key', 
        type=str, 
        default=os.getenv("OPENAI_API_KEY"),
        help="API Key。默认从环境变量 OPENAI_API_KEY 读取。"
    )
    parser.add_argument(
        '--base_url', 
        type=str, 
        default=os.getenv("OPENAI_BASE_URL", "https://aihubmix.com/v1"),
        help="API Base URL。默认从环境变量 OPENAI_BASE_URL 读取，若无则使用硬编码值。"
    )
    parser.add_argument('--model_name', type=str, default='gpt-4o-2024-05-13', help="用于评测的模型名称")
    parser.add_argument('--parallel_size', type=int, default=32, help="最大并发API请求数")
    parser.add_argument('--resume', action='store_true', help='从上次中断的地方继续评测')
    args = parser.parse_args()
    return args


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"警告: 无法解析行: {line.strip()}")
    return data


class RobustPassKEvaluator:
    
    def __init__(self, api_key: str, model_name: str, base_url: str, parallel_size: int):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.parallel_size = parallel_size
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        self.evaluation_prompt = """
        你是一个专业的LaTeX公式评估专家。请评估预测的LaTeX公式与标准答案的匹配程度。

        评估标准：
        1. 精确匹配：预测公式与标准答案在数学意义上完全相同。
        2. 部分匹配：预测公式与标准答案在数学意义上相似，但可能有细微差异。
        3. 不匹配：预测公式与标准答案在数学意义上不同。

        标准答案：{ground_truth}
        预测结果：{prediction}

        请严格按照以下JSON格式返回评估结果，不要添加任何额外说明：
        {{
            "exact_match": true/false,
            "partial_match": true/false,
            "explanation": "评估说明"
        }}
    """

    def _validate_response(self, response: Any) -> bool:
        try:
            if not hasattr(response, 'choices') or not response.choices: return False
            message = response.choices[0].message
            if not hasattr(message, 'content') or not message.content.strip(): return False
            json.loads(message.content)
            return True
        except (AttributeError, json.JSONDecodeError, Exception):
            return False

    def call_gpt4o(self, prompt: str, max_retries: int = 10) -> Dict[str, Any]:
        for attempt in range(max_retries):
            try:
                time.sleep(random.uniform(0.1, 0.5))
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.0,
                    seed=1234,
                    response_format={"type": "json_object"}
                )
                
                if self._validate_response(response):
                    result_text = response.choices[0].message.content
                    result = json.loads(result_text)
                    if 'exact_match' in result and 'partial_match' in result:
                        return result
                else:
                    print(f"\n❌ Invalid response format on try {attempt + 1}/{max_retries}. Response: {str(response)[:200]}...")

            except openai.RateLimitError as e:
                wait_time = 60
                print(f"\n⏳ Rate limit hit on try {attempt + 1}. Waiting for {wait_time}s... Error: {e}")
                time.sleep(wait_time)
            except openai.APIStatusError as e:
                print(f"\n🚫 API Status Error on try {attempt + 1} (e.g., 500 server error). Retrying... Error: {e}")
                time.sleep(5)
            except Exception as e:
                print(f"\n💥 Unexpected API error on try {attempt + 1}: {type(e).__name__}: {e}. Retrying with exponential backoff...")
                time.sleep(2 ** attempt)
        
        print(f"\n❌ API call failed after all {max_retries} retries.")
        return {'exact_match': False, 'partial_match': False, 'explanation': f'API调用失败超过{max_retries}次'}

    def evaluate_single_pair(self, prediction: str, ground_truth: str) -> Dict[str, Any]:
        if not prediction or not ground_truth:
            return {'exact_match': False, 'partial_match': False, 'explanation': '预测或真值为空'}
        
        # 关键：对反斜杠进行转义以确保JSON payload有效
        safe_prediction = prediction.replace('\\', '\\\\')
        safe_ground_truth = ground_truth.replace('\\', '\\\\')
            
        prompt = self.evaluation_prompt.format(ground_truth=safe_ground_truth, prediction=safe_prediction)
        return self.call_gpt4o(prompt)

    def evaluate_sample(self, prediction_item: Dict, ground_truth_item: Dict) -> Dict:
        sample_id = ground_truth_item.get('id', 'unknown_id')
        gt_answers = ground_truth_item.get('answers', [])
        pred_extracts = prediction_item.get('extract_answers', [])
        
        if not gt_answers:
            return {'id': sample_id, 'correct_attempts': 0, 'exact_match_attempts': 0, 'total_attempts': len(pred_extracts), 'per_attempt_details': []}
        
        first_gt_answer = gt_answers[0].get('content', '')
        
        combined_correct_count = 0
        exact_match_count = 0
        attempt_details = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_size) as executor:
            future_to_extract = {
                executor.submit(self.evaluate_single_pair, extract.get('formula', ''), first_gt_answer): extract
                for extract in pred_extracts
            }
            
            for future in concurrent.futures.as_completed(future_to_extract):
                pred_extract = future_to_extract[future]
                try:
                    gpt_result = future.result()
                    
                    is_exact_match = gpt_result.get('exact_match', False)
                    if is_exact_match:
                        exact_match_count += 1
                    
                    is_combined_correct = is_exact_match or gpt_result.get('partial_match', False)
                    if is_combined_correct:
                        combined_correct_count += 1

                    attempt_details.append({
                        "attempted_formula": pred_extract.get('formula', ''),
                        "is_correct_combined": is_combined_correct,
                        "is_correct_exact": is_exact_match,
                        "gpt_evaluation": gpt_result
                    })
                except Exception as e:
                    attempt_details.append({"is_correct_combined": False, "is_correct_exact": False, "gpt_evaluation": {"error": str(e)}})

        return {
            'id': sample_id,
            'correct_attempts': combined_correct_count,
            'exact_match_attempts': exact_match_count,
            'total_attempts': len(pred_extracts),
            'ground_truth': first_gt_answer,
            'per_attempt_details': attempt_details
        }

    def batch_evaluate(self, predictions: List[Dict], ground_truths: List[Dict], intermediate_file_path: str) -> List[Dict]:
        if len(predictions) != len(ground_truths):
            raise ValueError("预测和真值文件行数不匹配!")

        all_sample_results = []
        
        with tqdm(total=len(predictions), desc="Evaluating Samples") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_size) as executor:
                future_to_sample_idx = {
                    executor.submit(self.evaluate_sample, pred, gt): idx
                    for idx, (pred, gt) in enumerate(zip(predictions, ground_truths))
                }
                
                for future in concurrent.futures.as_completed(future_to_sample_idx):
                    idx = future_to_sample_idx[future]
                    try:
                        sample_result = future.result()
                        sample_result['line_number'] = idx + 1
                        all_sample_results.append(sample_result)
                        
                        with open(intermediate_file_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(sample_result, ensure_ascii=False) + '\n')
                    except Exception as e:
                        print(f"\n[!] Failed to process sample index {idx}: {e}")
                    pbar.update(1)

        return all_sample_results


def calculate_and_report_metrics(sample_results: List[Dict]):
    total_samples = len(sample_results)
    if total_samples == 0:
        print("没有可供分析的结果。"); return {}

    combined_correct_list = [res.get('correct_attempts', 0) for res in sample_results]
    exact_match_list = [res.get('exact_match_attempts', 0) for res in sample_results]
    
    pass_at_k_thresholds = [8, 16]
    
    pass_counts_combined = {k: sum(1 for c in combined_correct_list if c >= k) for k in pass_at_k_thresholds}
    pass_rates_combined = {k: (pass_counts_combined[k] / total_samples) * 100 for k in pass_at_k_thresholds}

    pass_counts_exact = {k: sum(1 for c in exact_match_list if c >= k) for k in pass_at_k_thresholds}
    pass_rates_exact = {k: (pass_counts_exact[k] / total_samples) * 100 for k in pass_at_k_thresholds}
    
    descriptive_stats = {'mean': np.mean(combined_correct_list), 'median': np.median(combined_correct_list), 'std_dev': np.std(combined_correct_list), 'min': int(np.min(combined_correct_list)), 'max': int(np.max(combined_correct_list))}
    
    samples_grouped_by_count = defaultdict(list)
    for res in sample_results:
        identifier = f"{res.get('id', 'N/A')} (line {res.get('line_number', 'N/A')})"
        samples_grouped_by_count[res.get('correct_attempts', 0)].append(identifier)
            
    metrics = {
        'total_samples': total_samples,
        'pass_k_rates_combined': {f'pass_at_{k}_rate': rate for k, rate in pass_rates_combined.items()},
        'pass_k_counts_combined': {f'pass_at_{k}_count': count for k, count in pass_counts_combined.items()},
        'pass_k_rates_exact_only': {f'pass_at_{k}_rate': rate for k, rate in pass_rates_exact.items()},
        'pass_k_counts_exact_only': {f'pass_at_{k}_count': count for k, count in pass_counts_exact.items()},
        'descriptive_statistics_of_correct_attempts': descriptive_stats,
        'samples_grouped_by_correct_count': {str(k): v for k, v in sorted(samples_grouped_by_count.items(), key=lambda item: item[0], reverse=True)}
    }

    report = "\n" + "="*70 + "\n"
    report += "Pass@k 评估报告 (基于GPT-4o)\n"
    report += "="*70 + "\n\n"
    report += f"评测配置:\n  - 总样本数: {total_samples}\n  - 每个样本的尝试次数: 32 (假设)\n\n"
    
    report += "Pass@k 核心指标 (正确 = 精确匹配 或 部分匹配):\n"
    for k in pass_at_k_thresholds:
        report += f"  - Pass@{k} 通过率: {pass_rates_combined[k]:.2f}% ({pass_counts_combined[k]}/{total_samples} 个样本通过)\n"

    report += "\n新增: Pass@k 核心指标 (正确 = 仅精确匹配):\n"
    for k in pass_at_k_thresholds:
        report += f"  - Pass@{k} 通过率 (精确): {pass_rates_exact[k]:.2f}% ({pass_counts_exact[k]}/{total_samples} 个样本通过)\n"
    
    report += "\n成功尝试次数的描述性统计 (基于组合正确率):\n"
    report += f"  - 平均成功次数: {descriptive_stats['mean']:.2f}\n"
    report += f"  - 成功次数中位数: {descriptive_stats['median']}\n"
    report += f"  - 标准差: {descriptive_stats['std_dev']:.2f}\n"
    report += f"  - 最少成功次数: {descriptive_stats['min']}\n"
    report += f"  - 最多成功次数: {descriptive_stats['max']}\n"
    
    report += "\n" + "-"*40 + "\n"
    report += "按成功次数分组的问题列表 (基于组合正确率):\n"
    report += "(格式: <成功次数>: <ID (line 行号)> ...)\n"
    report += "-"*40 + "\n"
    for count, identifiers in sorted(samples_grouped_by_count.items(), key=lambda item: item[0], reverse=True):
        report += f"  - {count}: {' '.join(identifiers)}\n"
    
    report += "="*70
    print(report)
    return metrics


def main():
    args = get_args()
    
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(args.output_file).replace('.json', '')
    intermediate_file_path = os.path.join(output_dir, f"{base_name}_intermediate.jsonl")

    all_ground_truths = read_jsonl(args.ground_truth_file)
    all_predictions = read_jsonl(args.prediction_file)

    existing_results = []
    if args.resume:
        print(f"🔄 恢复模式已启动，正在读取: {intermediate_file_path}")
        existing_results = read_jsonl(intermediate_file_path)
        if existing_results:
            processed_ids = {res['id'] for res in existing_results}
            print(f"    已找到 {len(processed_ids)} 个已处理的样本。")
            unprocessed_pairs = [(p, g) for p, g in zip(all_predictions, all_ground_truths) if p.get('id') not in processed_ids]
            if not unprocessed_pairs:
                predictions_to_run, ground_truths_to_run = [], []
                print("✅ 所有样本均已处理完毕。")
            else:
                predictions_to_run, ground_truths_to_run = zip(*unprocessed_pairs)
        else:
            print("    未找到中间文件或文件为空，将从头开始处理。")
            predictions_to_run, ground_truths_to_run = all_predictions, all_ground_truths
    else:
        if os.path.exists(intermediate_file_path):
            print(f"🧹 清理旧的中间文件: {intermediate_file_path}")
            os.remove(intermediate_file_path)
        predictions_to_run, ground_truths_to_run = all_predictions, all_ground_truths

    if predictions_to_run:
        evaluator = RobustPassKEvaluator(
            api_key=args.api_key,
            model_name=args.model_name,
            base_url=args.base_url,
            parallel_size=args.parallel_size
        )
        print(f"\n🚀 开始使用 {args.model_name} (at {args.base_url}) 并行评估 Pass@k...")
        print(f"  - 待处理样本数: {len(predictions_to_run)}")

        new_results = evaluator.batch_evaluate(list(predictions_to_run), list(ground_truths_to_run), intermediate_file_path)
        all_results = existing_results + new_results
    else:
        all_results = existing_results

    final_metrics = calculate_and_report_metrics(all_results)
    
    output_data = {
        'metrics': final_metrics,
        'detailed_evaluations_by_sample': sorted(all_results, key=lambda x: x.get('line_number', 0)),
        'evaluation_summary': {
            'model_used_for_evaluation': args.model_name,
            'parallel_size': args.parallel_size,
            'total_samples_evaluated': len(all_results)
        }
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 详细评测结果已保存到: {args.output_file}")

if __name__ == "__main__":
    main()
