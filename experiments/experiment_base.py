# -*- coding: utf-8 -*-
"""实验基础配置模块

提供所有实验共用的配置、工具函数和数据加载器。
复用 src.utils.config 中的全局配置。
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_config, Config


@dataclass
class ExperimentConfig:
    experiment_name: str
    experiment_id: str
    description: str = ""
    test_samples: int = 0
    max_workers: int = 4
    checkpoint_enabled: bool = True
    save_every: int = 10
    
    _config: Config = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        self._config = get_config()
        
        self.results_dir = RESULTS_DIR / self.experiment_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "raw").mkdir(exist_ok=True)
        (self.results_dir / "processed").mkdir(exist_ok=True)
        (self.results_dir / "analysis").mkdir(exist_ok=True)
    
    @property
    def top_k(self) -> int:
        return self._config.top_k
    
    @property
    def rerank_top_k(self) -> int:
        return self._config.rerank_top_k
    
    @property
    def llm_model(self) -> str:
        return self._config.llm_model
    
    @property
    def embedding_model(self) -> str:
        return self._config.embedding_model
    
    @property
    def vector_dim(self) -> int:
        return self._config.vector_dim
    
    @property
    def similarity_threshold(self) -> float:
        return self._config.similarity_threshold
    
    @property
    def chunk_size(self) -> int:
        return self._config.chunk_size
    
    @property
    def chunk_overlap(self) -> int:
        return self._config.chunk_overlap
    
    @property
    def llm_temperature(self) -> float:
        return self._config.llm_temperature
    
    @property
    def llm_max_tokens(self) -> int:
        return self._config.llm_max_tokens
    
    def get_checkpoint_path(self) -> Path:
        return self.results_dir / "raw" / "checkpoint.json"
    
    def get_results_path(self) -> Path:
        return self.results_dir / "raw" / "predictions.json"
    
    def get_metrics_path(self) -> Path:
        return self.results_dir / "processed" / "metrics.json"
    
    def get_report_path(self) -> Path:
        return self.results_dir / "analysis" / "report.md"
    
    def get_global_config(self) -> Config:
        return self._config


def load_knowledge_base(kb_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    if kb_path is None:
        kb_path = DATA_DIR / "agent" / "processed_md.json"
    
    with open(kb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "chunks" in data:
        return data["chunks"]
    elif isinstance(data, list):
        return data
    else:
        return [data]


def load_test_dataset(dataset_name: str = "rag_300_multihop.json") -> List[Dict[str, Any]]:
    dataset_path = DATA_DIR / "agent" / dataset_name
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "samples" in data:
        return data["samples"]
    else:
        return [data]


def load_knowledge_graph(kg_path: Optional[Path] = None) -> Dict[str, Any]:
    if kg_path is None:
        kg_path = DATA_DIR / "agent" / "real_kg_from_json.json"
    
    with open(kg_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def save_report(
    report: Dict[str, Any],
    output_path: Path,
    format: str = "json",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    elif format == "markdown":
        md_content = generate_markdown_report(report)
        with open(output_path.with_suffix('.md'), 'w', encoding='utf-8') as f:
            f.write(md_content)


def generate_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        f"# {report.get('experiment_name', '实验报告')}",
        "",
        f"**实验时间**: {report.get('timestamp', 'N/A')}",
        f"**实验描述**: {report.get('description', 'N/A')}",
        "",
        "## 概览",
        "",
        f"- 总样本数: {report.get('total_samples', 0)}",
        f"- 成功样本数: {report.get('successful_samples', 0)}",
        f"- 失败样本数: {report.get('failed_samples', 0)}",
        f"- 平均延迟: {report.get('avg_latency', 0):.2f}s",
        "",
    ]
    
    if "avg_generation_metrics" in report and report["avg_generation_metrics"]:
        lines.extend([
            "## 生成指标",
            "",
            "| 指标 | 分数 |",
            "|------|------|",
        ])
        
        gm = report["avg_generation_metrics"]
        for key, value in gm.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.4f} |")
        
        lines.append("")
    
    return "\n".join(lines)


def create_llm_client():
    from src.llms import create_client
    return create_client()


def create_embedding_client():
    from src.llms import create_embedding_client
    return create_embedding_client()
