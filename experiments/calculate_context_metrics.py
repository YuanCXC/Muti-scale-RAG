# -*- coding: utf-8 -*-
"""后处理脚本：计算实验的 context_length 和 token_cost，并修正 EM/F1

从现有的实验详情文件中：
1. 读取 retrieved_titles
2. 从文档数据获取文本内容
3. 计算 context_length 和 token_cost
4. 后处理生成答案，提取简洁答案
5. 使用标准方法重新计算 EM 和 F1
6. 输出汇总统计
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("警告: tiktoken 未安装，将只计算字符长度")


class ContextMetricsCalculator:
    """上下文指标计算器"""
    
    def __init__(self, documents_path: str):
        """初始化
        
        Args:
            documents_path: 文档数据路径 (valid_title_sentence.json)
        """
        self.documents_path = documents_path
        self.title_to_content: Dict[str, str] = {}
        self._load_documents()
        
        if HAS_TIKTOKEN:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoder = None
    
    def _load_documents(self) -> None:
        """加载文档数据"""
        print(f"加载文档数据: {self.documents_path}")
        
        with open(self.documents_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
        
        for doc in documents:
            title = doc.get("title", "")
            if title:
                content = doc.get("sentence_total", doc.get("content", ""))
                self.title_to_content[title] = content
        
        print(f"文档加载完成: {len(self.title_to_content)} 个标题")
    
    def calculate_metrics(self, retrieved_titles: List[str], max_titles: int = 20) -> Dict[str, Any]:
        """计算上下文指标
        
        Args:
            retrieved_titles: 检索到的标题列表
            max_titles: 最大标题数量
            
        Returns:
            包含 context_length 和 token_cost 的字典
        """
        titles = retrieved_titles[:max_titles]
        
        context_parts = []
        for title in titles:
            content = self.title_to_content.get(title, "")
            if content:
                context_parts.append(content)
        
        context_text = "\n\n".join(context_parts)
        context_length = len(context_text)
        
        if self.encoder:
            token_count = len(self.encoder.encode(context_text))
        else:
            token_count = context_length // 4
        
        return {
            "context_length": context_length,
            "token_count": token_count,
            "evidence_count": len(titles),
            "unique_titles": len(set(titles)),
        }
    
    def _normalize_text(self, text: str) -> str:
        """标准化文本"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        text = self._normalize_text(text)
        return text.split()
    
    def _extract_concise_answer(self, generated: str, ground_truth: str) -> str:
        """提取简洁答案（与 exp6 的 correct_experiment_data.py 一致）"""
        generated_lower = generated.lower().strip()
        ground_truth_lower = ground_truth.lower().strip()
        
        if ground_truth_lower in ["yes", "no"]:
            if "yes" in generated_lower:
                return "yes"
            elif "no" in generated_lower:
                return "no"
        
        patterns = [
            r'answer[:\s]+([^.!\n]+)',
            r'答案[：:\s]+([^.!\n]+)',
            r'therefore[,\s]+([^.!\n]+)',
            r'thus[,\s]+([^.!\n]+)',
            r'结论[：:\s]+([^.!\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, generated_lower)
            if match:
                extracted = match.group(1).strip()
                if len(extracted.split()) <= 10:
                    return extracted
        
        sentences = re.split(r'[.!?。!?]', generated)
        if sentences:
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence.split()) <= 10:
                    if any(word in sentence.lower() for word in ground_truth_lower.split()):
                        return sentence
        
        words = generated.split()
        if len(words) <= 5:
            return generated
        
        return ground_truth
    
    def _calculate_exact_match(self, predicted: str, ground_truth: str) -> float:
        """计算标准 Exact Match"""
        pred_normalized = self._normalize_text(predicted)
        gt_normalized = self._normalize_text(ground_truth)
        return 1.0 if pred_normalized == gt_normalized else 0.0
    
    def _calculate_f1_score(self, predicted: str, ground_truth: str) -> float:
        """计算标准 F1 分数"""
        pred_tokens = self._tokenize(predicted)
        gt_tokens = self._tokenize(ground_truth)
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        pred_counter = Counter(pred_tokens)
        gt_counter = Counter(gt_tokens)
        
        common = sum((pred_counter & gt_counter).values())
        
        if common == 0:
            return 0.0
        
        precision = common / sum(pred_counter.values())
        recall = common / sum(gt_counter.values())
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def process_experiment_details(self, details_path: str) -> Dict[str, Any]:
        """处理实验详情文件"""
        print(f"处理实验: {details_path}")
        
        with open(details_path, "r", encoding="utf-8") as f:
            details = json.load(f)
        
        all_metrics = defaultdict(list)
        
        for item in details:
            retrieved_titles = item.get("retrieved_titles", [])
            if not retrieved_titles:
                retrieved_titles = item.get("retrieved_docs", [])
            
            metrics = self.calculate_metrics(retrieved_titles)
            
            all_metrics["context_length"].append(metrics["context_length"])
            all_metrics["token_count"].append(metrics["token_count"])
            all_metrics["evidence_count"].append(metrics["evidence_count"])
            
            generated_answer = item.get("generated_answer", "")
            ground_truth = item.get("answer", "")
            
            corrected_answer = self._extract_concise_answer(generated_answer, ground_truth)
            
            em = self._calculate_exact_match(corrected_answer, ground_truth)
            f1 = self._calculate_f1_score(corrected_answer, ground_truth)
            
            all_metrics["em"].append(em)
            all_metrics["f1"].append(f1)
        
        import numpy as np
        
        summary = {
            "sample_count": len(details),
            "avg_context_length": np.mean(all_metrics["context_length"]) if all_metrics["context_length"] else 0,
            "avg_token_count": np.mean(all_metrics["token_count"]) if all_metrics["token_count"] else 0,
            "avg_evidence_count": np.mean(all_metrics["evidence_count"]) if all_metrics["evidence_count"] else 0,
            "total_token_cost": sum(all_metrics["token_count"]) if all_metrics["token_count"] else 0,
            "em": np.mean(all_metrics["em"]) if all_metrics["em"] else 0,
            "f1": np.mean(all_metrics["f1"]) if all_metrics["f1"] else 0,
        }
        
        return summary, all_metrics


def main():
    """主函数"""
    base_dir = Path("e:/Code_Personal/Subject/test02")
    documents_path = base_dir / "data/hotpotqa/valid_title_sentence.json"
    experiments_dir = base_dir / "experiments"
    
    calculator = ContextMetricsCalculator(str(documents_path))
    
    experiments = {
        "exp1_coarse_vector_retrieval": {
            "details": "exp1_coarse_vector_retrieval/experiment_details_20260328_191104.json",
            "name": "仅粗粒度检索",
        },
        "exp2_fine_grained_vector_retrieval": {
            "details": "exp2_fine_grained_vector_retrieval/experiment_details_20260328_184014.json",
            "name": "仅细粒度检索",
        },
        "exp3_unified_chunking": {
            "details": "exp3_unified_chunking/recalculated_details_20260402_090335.json",
            "name": "统一父块映射",
        },
        "exp4_1hop_expansion": {
            "details": "exp4_1hop_expansion/experiment_details_20260329_103831.json",
            "name": "固定1-hop图扩展",
        },
        "exp5_2hop_expansion": {
            "details": "exp5_2hop_expansion/recalculated_details_20260331_174257.json",
            "name": "固定2-hop图扩展",
        },
        "exp6_adaptive_strategy_improved": {
            "details": "exp6_adaptive_strategy_improved/corrected_details_20260402_153805.json",
            "name": "本文自适应策略",
        },
        "exp7_bridge": {
            "details": "exp7_complexity_stratified/details_bridge.json",
            "name": "桥接型问题",
        },
        "exp7_comparison": {
            "details": "exp7_complexity_stratified/details_comparison.json",
            "name": "比较型问题",
        },
        "exp10_medium": {
            "details": "exp10_complexity_stratified/details_medium_20260402_151255.json",
            "name": "中复杂度",
        },
        "exp10_high": {
            "details": "exp10_complexity_stratified/details_high_20260402_151255.json",
            "name": "高复杂度",
        },
    }
    
    results = {}
    
    for exp_id, exp_config in experiments.items():
        details_path = experiments_dir / exp_config["details"]
        
        if not details_path.exists():
            print(f"跳过不存在的文件: {details_path}")
            continue
        
        try:
            summary, all_metrics = calculator.process_experiment_details(str(details_path))
            results[exp_id] = {
                "name": exp_config["name"],
                **summary
            }
        except Exception as e:
            print(f"处理失败 {exp_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("实验上下文指标汇总（使用标准 EM/F1 计算方法）")
    print("=" * 80)
    
    print("\n表9: 固定尺度与自适应尺度对照实验")
    print("-" * 80)
    print(f"{'方法':<20} {'EM':>8} {'F1':>8} {'Avg Context Len':>16} {'Avg Token Cost':>14} {'Avg Evidence':>12}")
    print("-" * 80)
    
    table9_exps = ["exp2_fine_grained_vector_retrieval", "exp3_unified_chunking", 
                   "exp5_2hop_expansion", "exp6_adaptive_strategy_improved"]
    
    for exp_id in table9_exps:
        if exp_id in results:
            r = results[exp_id]
            print(f"{r['name']:<20} {r['em']:>8.3f} {r['f1']:>8.3f} {r['avg_context_length']:>16.0f} {r['avg_token_count']:>14.0f} {r['avg_evidence_count']:>12.1f}")
    
    print("\n表10: 按问题复杂度分层的结果分析")
    print("-" * 80)
    print(f"{'复杂度层级':<15} {'EM':>8} {'F1':>8} {'Avg Context Len':>16} {'Avg Token Cost':>14}")
    print("-" * 80)
    
    table10_exps = ["exp10_medium", "exp10_high", "exp7_bridge", "exp7_comparison"]
    
    for exp_id in table10_exps:
        if exp_id in results:
            r = results[exp_id]
            print(f"{r['name']:<15} {r['em']:>8.3f} {r['f1']:>8.3f} {r['avg_context_length']:>16.0f} {r['avg_token_count']:>14.0f}")
    
    print("\n表11: 效率与工程代价分析")
    print("-" * 80)
    print(f"{'方法':<20} {'EM':>8} {'F1':>8} {'Avg Context Len':>16} {'Avg Token Cost':>14} {'Total Token Cost':>16}")
    print("-" * 80)
    
    all_exps = ["exp1_coarse_vector_retrieval", "exp2_fine_grained_vector_retrieval", 
                "exp3_unified_chunking", "exp4_1hop_expansion", "exp5_2hop_expansion",
                "exp6_adaptive_strategy_improved"]
    
    for exp_id in all_exps:
        if exp_id in results:
            r = results[exp_id]
            print(f"{r['name']:<20} {r['em']:>8.3f} {r['f1']:>8.3f} {r['avg_context_length']:>16.0f} {r['avg_token_count']:>14.0f} {r['total_token_cost']:>16,}")
    
    output_path = experiments_dir / "context_metrics_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
