# -*- coding: utf-8 -*-
"""运行所有实验"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from new_experiments.all.exp1_1_semantic_rag import SemanticRAGExperiment
from new_experiments.all.exp1_2_rerank_rag import RerankRAGExperiment
from new_experiments.all.exp1_3_graphrag import GraphRAGExperiment
from new_experiments.all.exp1_4_kg_rag import KGRAGExperiment
from new_experiments.all.exp1_5_macrag import MacRAGExperiment
from new_experiments.all.exp1_6_proposed import ProposedExperiment
from new_experiments.all.exp2_1_fine_only import FineOnlyExperiment
from new_experiments.all.exp2_2_uniform_parent import UniformParentExperiment
from new_experiments.all.exp2_3_fixed_1hop import Fixed1HopExperiment
from new_experiments.all.exp2_4_fixed_2hop import Fixed2HopExperiment
from new_experiments.all.exp3_1_no_graph_expansion import NoGraphExpansionExperiment
from new_experiments.all.exp3_2_no_selective_parent import NoSelectiveParentExperiment
from new_experiments.all.exp3_3_no_summary_evidence import NoSummaryEvidenceExperiment
from new_experiments.all.exp1_method_comparison import RetrievalConfig


EXPERIMENTS = {
    "1.1": ("Semantic RAG", SemanticRAGExperiment),
    "1.2": ("+Rerank", RerankRAGExperiment),
    "1.3": ("GraphRAG", GraphRAGExperiment),
    "1.4": ("KG-RAG", KGRAGExperiment),
    "1.5": ("MacRAG", MacRAGExperiment),
    "1.6": ("Proposed", ProposedExperiment),
    "2.1": ("Fine only", FineOnlyExperiment),
    "2.2": ("Uniform parent", UniformParentExperiment),
    "2.3": ("Fixed 1-hop", Fixed1HopExperiment),
    "2.4": ("Fixed 2-hop", Fixed2HopExperiment),
    "3.1": ("No graph expansion", NoGraphExpansionExperiment),
    "3.2": ("No selective parent", NoSelectiveParentExperiment),
    "3.3": ("No summary evidence", NoSummaryEvidenceExperiment),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="运行所有实验")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=250)
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--exp", type=str, default="all", help=f"运行哪个实验: all, 1.1-3.3")
    args = parser.parse_args()
    
    config = RetrievalConfig(
        sample_size=args.sample_size,
        max_workers=args.max_workers,
        run_generation=not args.no_generation,
        run_judge=not args.no_judge,
    )
    
    results = {}
    
    exps_to_run = list(EXPERIMENTS.keys()) if args.exp == "all" else [args.exp]
    
    for exp_id in exps_to_run:
        if exp_id not in EXPERIMENTS:
            print(f"未知实验: {exp_id}")
            continue
        
        exp_name, exp_class = EXPERIMENTS[exp_id]
        print("\n" + "=" * 60)
        print(f"实验 {exp_id}: {exp_name}")
        print("=" * 60)
        results[exp_id] = exp_class(config).run()
    
    print("\n" + "=" * 60)
    print("所有实验完成")
    print("=" * 60)
    for exp_id, result in results.items():
        exp_name = EXPERIMENTS[exp_id][0]
        print(f"实验 {exp_id} ({exp_name}): {result['run_dir']}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
