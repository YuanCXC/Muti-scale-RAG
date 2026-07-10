# -*- coding: utf-8 -*-
"""计算实验2-5的语义评估指标

ACC (Answer Correctness): 答案事实正确性
FTH (Faithfulness): 答案是否由证据支撑
CtxRel (Context Relevance): 上下文是否相关
"""
from __future__ import annotations
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class SemanticMetrics:
    correctness: float = 0.0
    faithfulness: float = 0.0
    context_relevance: float = 0.0


class SemanticEvaluator:
    """语义评估器"""
    
    def __init__(self, max_workers: int = 50):
        self.llm_client: Optional[Any] = None
        self.documents: List[Dict[str, Any]] = []
        self.title_to_content: Dict[str, str] = {}
        self.max_workers = max_workers
    
    def load_resources(self, documents_path: str):
        """加载资源"""
        print("加载资源...")
        
        with open(documents_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        
        self.title_to_content = {
            doc.get("title", ""): doc.get("sentence_total", doc.get("content", ""))
            for doc in self.documents
            if doc.get("title")
        }
        print(f"  文档: {len(self.title_to_content)} 个标题")
        
        try:
            from src.llms.deepseek_client import DeepSeekClient
            self.llm_client = DeepSeekClient()
            print("  LLM客户端: DeepSeek")
        except Exception as e:
            print(f"  LLM客户端: 加载失败 ({e})")
            raise
    
    def get_context(self, retrieved_titles: List[str], max_docs: int = 5) -> str:
        """获取上下文"""
        contexts = []
        for title in retrieved_titles[:max_docs]:
            content = self.title_to_content.get(title, "")
            if content:
                contexts.append(f"【{title}】\n{content[:500]}")
        return "\n\n".join(contexts)
    
    def evaluate_single(
        self,
        question: str,
        ground_truth: str,
        generated_answer: str,
        retrieved_titles: List[str],
    ) -> SemanticMetrics:
        """评估单个样本"""
        if not self.llm_client:
            return SemanticMetrics()
        
        context = self.get_context(retrieved_titles)
        
        if not context:
            return SemanticMetrics()
        
        prompt = f"""请评估以下问答系统的输出质量。返回JSON格式的评分（0-1）。

问题: {question}

标准答案: {ground_truth}

系统生成的答案: {generated_answer}

检索到的上下文:
{context}

请评估以下三个指标（每项0-1分）：

1. **correctness (答案正确性)**: 生成的答案是否在事实层面上正确？与标准答案相比是否准确？
   - 1.0: 完全正确，与标准答案一致
   - 0.7-0.9: 基本正确，有小错误或不完整
   - 0.4-0.6: 部分正确，有重要错误
   - 0.0-0.3: 完全错误或无关

2. **faithfulness (忠实度)**: 生成的答案是否完全基于上下文信息？是否包含上下文中不存在的信息（幻觉）？
   - 1.0: 完全基于上下文，无幻觉
   - 0.7-0.9: 主要基于上下文，少量推断
   - 0.4-0.6: 部分基于上下文，有明显推断
   - 0.0-0.3: 大量幻觉或与上下文矛盾

3. **context_relevance (上下文相关性)**: 检索到的上下文是否与问题相关？是否包含回答问题所需的关键信息？
   - 1.0: 高度相关，包含所有关键信息
   - 0.7-0.9: 相关，包含大部分关键信息
   - 0.4-0.6: 部分相关，缺少关键信息
   - 0.0-0.3: 不相关或无有用信息

返回JSON格式: {{"correctness": 0.X, "faithfulness": 0.X, "context_relevance": 0.X}}
只返回JSON，不要其他内容。"""
        
        try:
            from src.llms.base_client import Message
            response = self.llm_client.generate([Message(role="user", content=prompt)], max_tokens=200)
            
            content = response.content.strip()
            
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return SemanticMetrics(
                    correctness=float(data.get("correctness", 0.5)),
                    faithfulness=float(data.get("faithfulness", 0.5)),
                    context_relevance=float(data.get("context_relevance", 0.5)),
                )
        except Exception as e:
            print(f"评估失败: {e}")
        
        return SemanticMetrics()
    
    def generate_answer(self, question: str, retrieved_titles: List[str]) -> str:
        """生成答案"""
        if not self.llm_client:
            return ""
        
        context = self.get_context(retrieved_titles)
        
        if not context:
            return ""
        
        prompt = f"""基于以下上下文回答问题。如果上下文中没有相关信息，请说明。

上下文:
{context}

问题: {question}

答案:"""
        
        try:
            from src.llms.base_client import Message
            response = self.llm_client.generate([Message(role="user", content=prompt)], max_tokens=150)
            return response.content.strip()
        except Exception as e:
            print(f"生成答案失败: {e}")
            return ""
    
    def evaluate_experiment(
        self,
        details_path: str,
        output_path: str,
        sample_size: Optional[int] = None,
        random_seed: int = 42,
        skip_generation: bool = False,
    ) -> Dict[str, Any]:
        """评估实验结果"""
        print(f"\n加载实验结果: {details_path}")
        
        with open(details_path, "r", encoding="utf-8") as f:
            details = json.load(f)
        
        if sample_size and sample_size < len(details):
            import random
            random.seed(random_seed)
            details = random.sample(details, sample_size)
        
        print(f"  样本数: {len(details)}")
        
        results = []
        results_lock = threading.Lock()
        
        def process_one(item: Dict) -> Dict:
            question = item.get("question", "")
            ground_truth = item.get("answer", "")
            retrieved_titles = item.get("retrieved_titles", [])
            
            generated_answer = item.get("generated_answer", "")
            if not generated_answer and not skip_generation:
                generated_answer = self.generate_answer(question, retrieved_titles)
            
            metrics = self.evaluate_single(question, ground_truth, generated_answer, retrieved_titles)
            
            return {
                **item,
                "generated_answer": generated_answer,
                "semantic_metrics": {
                    "correctness": metrics.correctness,
                    "faithfulness": metrics.faithfulness,
                    "context_relevance": metrics.context_relevance,
                },
            }
        
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(process_one, item): i for i, item in enumerate(details)}
                
                pbar = tqdm(total=len(details), desc="评估进度")
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        with results_lock:
                            results.append(result)
                    except Exception as e:
                        print(f"处理失败: {e}")
                    pbar.update(1)
                pbar.close()
        else:
            for item in tqdm(details, desc="评估进度"):
                try:
                    result = process_one(item)
                    results.append(result)
                except Exception as e:
                    print(f"处理失败: {e}")
        
        correctness_scores = [r["semantic_metrics"]["correctness"] for r in results]
        faithfulness_scores = [r["semantic_metrics"]["faithfulness"] for r in results]
        context_relevance_scores = [r["semantic_metrics"]["context_relevance"] for r in results]
        
        summary = {
            "total_samples": len(results),
            "avg_correctness": sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0,
            "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
            "avg_context_relevance": sum(context_relevance_scores) / len(context_relevance_scores) if context_relevance_scores else 0,
        }
        
        print(f"\n评估结果:")
        print(f"  ACC (答案正确性): {summary['avg_correctness']:.4f}")
        print(f"  FTH (忠实度): {summary['avg_faithfulness']:.4f}")
        print(f"  CtxRel (上下文相关性): {summary['avg_context_relevance']:.4f}")
        
        output_data = {
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存: {output_path}")
        
        return output_data


def main():
    parser = argparse.ArgumentParser(description="计算实验2-5的语义评估指标")
    parser.add_argument("--exp", type=str, required=True, choices=["2", "3", "4", "5"], help="实验编号")
    parser.add_argument("--sample-size", type=int, default=1000, help="样本数量")
    parser.add_argument("--random-seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-workers", type=int, default=50, help="并发数")
    parser.add_argument("--skip-generation", action="store_true", help="跳过答案生成")
    args = parser.parse_args()
    
    exp_dir = Path(f"e:/Code_Personal/Subject/test02/experiments/exp{args.exp}_{'fine_grained_vector_retrieval' if args.exp == '2' else 'unified_chunking' if args.exp == '3' else '1hop_expansion' if args.exp == '4' else '2hop_expansion'}")
    
    details_files = list(exp_dir.glob("*details*.json"))
    if not details_files:
        print(f"未找到实验结果文件: {exp_dir}")
        return 1
    
    details_path = str(details_files[0])
    output_path = str(exp_dir / f"semantic_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    documents_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/valid_title_sentence.json"
    
    evaluator = SemanticEvaluator(max_workers=args.max_workers)
    evaluator.load_resources(documents_path)
    
    evaluator.evaluate_experiment(
        details_path=details_path,
        output_path=output_path,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        skip_generation=args.skip_generation,
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
