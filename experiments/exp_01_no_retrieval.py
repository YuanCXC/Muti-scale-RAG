# -*- coding: utf-8 -*-
"""实验01: 无检索基线 (No Retrieval Baseline)

测试 LLM 自身知识储备，作为最低基准线。
不使用任何检索，直接让 LLM 回答问题。
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

from experiment_base import (
    ExperimentConfig,
    load_test_dataset,
    save_results,
    create_llm_client,
)


def run_no_retrieval_experiment(
    config: ExperimentConfig,
    max_samples: int = 10,
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"实验01: 无检索基线")
    print(f"描述: 测试 LLM 自身知识储备 (Closed-Book QA)")
    print(f"{'='*60}\n")
    
    llm_client = create_llm_client()
    
    test_data = load_test_dataset("rag_300_multihop.json")
    if max_samples > 0:
        test_data = test_data[:max_samples]
    
    print(f"测试样本数: {len(test_data)}")
    
    results = []
    success_count = 0
    failed_count = 0
    total_latency = 0.0
    
    for i, item in enumerate(tqdm(test_data, desc="处理中")):
        query = item.get("question", item.get("query", ""))
        ground_truth = item.get("ground_truth", item.get("answer", ""))
        
        result = {
            "id": item.get("id", str(i)),
            "question": query,
            "ground_truth": ground_truth,
            "predicted_answer": "",
            "success": False,
            "latency": 0.0,
            "error": None,
        }
        
        try:
            start_time = time.time()
            
            from src.llms.base_client import Message
            messages = [
                Message(role="system", content="你是一个有帮助的AI助手。请直接回答用户问题。"),
                Message(role="user", content=query)
            ]
            
            response = llm_client.generate(messages)
            
            result["predicted_answer"] = response.content
            result["success"] = True
            result["latency"] = time.time() - start_time
            
            success_count += 1
            total_latency += result["latency"]
            
        except Exception as e:
            result["error"] = str(e)
            result["latency"] = time.time() - start_time
            failed_count += 1
            print(f"\n样本 {i} 错误: {e}")
        
        results.append(result)
    
    avg_latency = total_latency / len(results) if results else 0
    
    report = {
        "experiment_name": "无检索基线",
        "experiment_id": config.experiment_id,
        "description": "不使用任何检索，直接让 LLM 回答问题",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(results),
        "successful_samples": success_count,
        "failed_samples": failed_count,
        "avg_latency": avg_latency,
        "config": {
            "max_samples": max_samples,
            "llm_model": config.llm_model,
        },
    }
    
    save_results(results, config.get_results_path())
    save_results(report, config.get_metrics_path())
    
    print(f"\n{'='*60}")
    print(f"实验完成!")
    print(f"总样本: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"平均延迟: {avg_latency:.2f}s")
    print(f"{'='*60}\n")
    
    return report


def main():
    config = ExperimentConfig(
        experiment_name="无检索基线",
        experiment_id="exp_01_no_retrieval",
        description="不使用任何检索，直接让 LLM 回答问题",
        test_samples=10,
    )
    
    report = run_no_retrieval_experiment(config, max_samples=10)
    
    print("结果已保存到:")
    print(f"  - {config.get_results_path()}")
    print(f"  - {config.get_metrics_path()}")


if __name__ == "__main__":
    main()
