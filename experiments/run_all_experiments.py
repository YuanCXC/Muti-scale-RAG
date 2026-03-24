# -*- coding: utf-8 -*-
"""运行所有实验的主脚本

使用方法:
    python run_all_experiments.py --exp 01    # 运行单个实验
    python run_all_experiments.py --all       # 运行所有实验
    python run_all_experiments.py --list      # 列出所有实验
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS = {
    "01": {
        "name": "无检索基线",
        "module": "experiments.exp_01_no_retrieval",
        "function": "run_no_retrieval_experiment",
        "description": "测试 LLM 自身知识储备 (Closed-Book QA)",
    },
    "02": {
        "name": "纯向量检索 RAG",
        "module": "experiments.exp_02_naive_rag",
        "function": "run_naive_rag_experiment",
        "description": "使用向量检索获取 Top-K 文档",
    },
    "03": {
        "name": "混合检索 RAG",
        "module": "experiments.exp_03_hybrid_rag",
        "function": "run_hybrid_rag_experiment",
        "description": "BM25 + 向量检索 + RRF 融合",
    },
    "04": {
        "name": "图谱增强 RAG",
        "module": "experiments.exp_04_graph_rag",
        "function": "run_graph_rag_experiment",
        "description": "向量检索 + 知识图谱实体链接",
    },
    "05": {
        "name": "动态路由 RAG",
        "module": "experiments.exp_05_live_rag",
        "function": "run_live_rag_experiment",
        "description": "BM25/Vector 双路检索 + 置信度路由",
    },
    "06": {
        "name": "完整系统",
        "module": "experiments.exp_06_full_system",
        "function": "run_full_system_experiment",
        "description": "问题分类 + 差异化检索 + Rerank",
    },
    "07": {
        "name": "HotpotQA 多跳问答",
        "module": "experiments.exp_07_hotpotqa",
        "function": "run_hotpotqa_experiment",
        "description": "HotpotQA 数据集多跳推理",
    },
}


def list_experiments():
    print("\n可用实验列表:")
    print("=" * 60)
    for exp_id, exp_info in EXPERIMENTS.items():
        print(f"  [{exp_id}] {exp_info['name']}")
        print(f"      描述: {exp_info['description']}")
        print()
    print("=" * 60)


def run_experiment(exp_id: str, max_samples: int = 10):
    if exp_id not in EXPERIMENTS:
        print(f"错误: 未找到实验 {exp_id}")
        return
    
    exp_info = EXPERIMENTS[exp_id]
    print(f"\n运行实验 {exp_id}: {exp_info['name']}")
    
    try:
        module = __import__(exp_info["module"], fromlist=[exp_info["function"]])
        func = getattr(module, exp_info["function"])
        
        from experiments.experiment_base import ExperimentConfig
        
        config = ExperimentConfig(
            experiment_name=exp_info["name"],
            experiment_id=f"exp_{exp_id}_{exp_info['name'].lower().replace(' ', '_')}",
            description=exp_info["description"],
            test_samples=max_samples,
        )
        
        report = func(config, max_samples=max_samples)
        
        print(f"\n实验 {exp_id} 完成!")
        return report
        
    except Exception as e:
        print(f"运行实验 {exp_id} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_experiments(max_samples: int = 10):
    print("\n" + "=" * 60)
    print("运行所有实验")
    print("=" * 60)
    
    results = {}
    
    for exp_id in EXPERIMENTS:
        result = run_experiment(exp_id, max_samples)
        results[exp_id] = result
    
    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RAG 实验运行器")
    parser.add_argument(
        "--exp", "-e",
        type=str,
        help="要运行的实验编号 (01-07)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="运行所有实验"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有可用实验"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10,
        help="每个实验的样本数量 (默认: 10)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
    elif args.all:
        run_all_experiments(args.samples)
    elif args.exp:
        run_experiment(args.exp, args.samples)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python run_all_experiments.py --list")
        print("  python run_all_experiments.py --exp 01 --samples 5")
        print("  python run_all_experiments.py --all --samples 10")


if __name__ == "__main__":
    main()
