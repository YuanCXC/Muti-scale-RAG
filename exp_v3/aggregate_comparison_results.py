# -*- coding: utf-8 -*-
"""汇总 v3 五个对比基线与 Proposed 的离线结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results"

EXPECTED_METHODS = (
    "Semantic RAG",
    "Rerank RAG",
    "GraphRAG",
    "KG-RAG",
    "MacRAG",
    "Proposed",
)

RESULT_PREFIXES = {
    "Semantic RAG": "semantic_rag_",
    "Rerank RAG": "rerank_rag_",
    "GraphRAG": "graphrag_",
    "KG-RAG": "kg_rag_",
    "MacRAG": "macrag_",
    "Proposed": "adaptive_multiscale_rag_",
}

REQUIRED_FILES = {
    "result_summary.csv",
    "details.json",
    "config.json",
}

SUMMARY_COLUMNS = (
    "Method",
    "Samples",
    "Generation Success Samples",
    "Generation Success Rate",
    "Valid Judge Samples",
    "Refusal Rate",
    "Recall",
    "Precision",
    "F1",
    "MRR",
    "NDCG",
    "MAP",
    "Avg Len",
    "Time/ms (diagnostic)",
    "Expanded Nodes",
    "correctness",
    "faithfulness",
    "answer_relevance",
    "context_relevance",
)

TYPE_COLUMNS = (
    "Method",
    "Type",
    "Samples",
    "Recall",
    "Precision",
    "F1",
    "Generation Success Rate",
    "Correctness",
    "Faithfulness",
    "Answer Relevance",
    "Context Relevance",
    "Avg Len",
    "Time/ms (diagnostic)",
    "Expanded Nodes",
    "Refusal Rate",
)


@dataclass
class LoadedRun:
    """一个方法的已加载实验结果。"""

    method: str
    run_dir: Path
    summary_row: Dict[str, Any]
    sample_rows: List[Dict[str, Any]]
    config: Dict[str, Any]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _sample_id(row: Mapping[str, Any]) -> str:
    value = row.get("id")
    if value is None:
        return ""
    return str(value).strip()


def _find_method_rows(details: Mapping[str, Any], method: str) -> List[Dict[str, Any]]:
    rows_by_method = details.get("rows_by_method")
    if not isinstance(rows_by_method, Mapping):
        raise ValueError("details.json 缺少 rows_by_method")
    rows = rows_by_method.get(method)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"details.json 中不存在 {method} 的逐样本结果")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{method} 的逐样本结果格式错误")
    return rows


def _validate_sample_rows(method: str, rows: Sequence[Mapping[str, Any]]) -> None:
    identifiers = [_sample_id(row) for row in rows]
    if any(not identifier for identifier in identifiers):
        raise ValueError(f"{method} 存在空样本 ID")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{method} 存在重复样本 ID")

    for row in rows:
        retrieval = row.get("retrieval_metrics")
        if not isinstance(retrieval, Mapping):
            raise ValueError(f"{method} 的样本 {_sample_id(row)} 缺少 retrieval_metrics")
        for key in (
            "recall",
            "precision",
            "mrr",
            "ndcg",
            "map_score",
            "avg_len",
            "time_ms",
            "expanded_nodes",
        ):
            if key not in retrieval:
                raise ValueError(
                    f"{method} 的样本 {_sample_id(row)} 缺少检索指标 {key}"
                )
            value = retrieval.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(
                    f"{method} 的样本 {_sample_id(row)} 指标 {key} 不是有限数值"
                )
            if key in {"recall", "precision", "mrr", "ndcg", "map_score"}:
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError(
                        f"{method} 的样本 {_sample_id(row)} 指标 {key} 超出 [0,1]"
                    )

        semantic = row.get("semantic_metrics")
        if isinstance(semantic, Mapping) and semantic:
            score_keys = (
                "correctness",
                "faithfulness",
                "answer_relevance",
                "context_relevance",
            )
            present_scores = [key for key in score_keys if key in semantic]
            if present_scores and len(present_scores) != len(score_keys):
                raise ValueError(
                    f"{method} 的样本 {_sample_id(row)} 语义指标不完整"
                )
            score_values = [semantic.get(key) for key in present_scores]
            if score_values and all(value is None for value in score_values):
                continue
            if any(value is None for value in score_values):
                raise ValueError(
                    f"{method} 的样本 {_sample_id(row)} 语义指标部分缺失"
                )
            for key in present_scores:
                value = semantic.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(
                        f"{method} 的样本 {_sample_id(row)} 语义指标 {key} 非法"
                    )
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError(
                        f"{method} 的样本 {_sample_id(row)} 语义指标 {key} 超出 [0,1]"
                    )


def load_run(run_dir: Path, method: str) -> LoadedRun:
    """加载并验证单个结果目录。"""
    run_dir = run_dir.resolve()
    missing = [
        filename
        for filename in sorted(REQUIRED_FILES)
        if not (run_dir / filename).is_file()
    ]
    if missing:
        raise ValueError(f"{run_dir} 缺少文件：{', '.join(missing)}")

    summary_rows = _read_csv(run_dir / "result_summary.csv")
    matching = [row for row in summary_rows if row.get("Method") == method]
    if len(matching) != 1:
        raise ValueError(
            f"{run_dir / 'result_summary.csv'} 中 {method} 方法行数量不是 1"
        )

    details = _read_json(run_dir / "details.json")
    if not isinstance(details, Mapping):
        raise ValueError(f"{run_dir / 'details.json'} 顶层必须是对象")
    sample_rows = _find_method_rows(details, method)
    _validate_sample_rows(method, sample_rows)

    config = _read_json(run_dir / "config.json")
    if not isinstance(config, dict):
        raise ValueError(f"{run_dir / 'config.json'} 顶层必须是对象")

    return LoadedRun(
        method=method,
        run_dir=run_dir,
        summary_row=dict(matching[0]),
        sample_rows=list(sample_rows),
        config=config,
    )


def discover_latest_run(results_root: Path, method: str) -> Path:
    """按目录名倒序选择最新且可完整解析的结果。"""
    results_root = results_root.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"结果根目录不存在：{results_root}")

    prefix = RESULT_PREFIXES[method]
    candidates = sorted(
        (
            path
            for path in results_root.iterdir()
            if path.is_dir()
            and path.name.startswith(prefix)
            and all((path / filename).is_file() for filename in REQUIRED_FILES)
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    errors: List[str] = []
    for candidate in candidates:
        try:
            load_run(candidate, method)
            return candidate
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{candidate.name}: {exc}")

    detail = f"；候选错误：{' | '.join(errors)}" if errors else ""
    raise FileNotFoundError(f"未找到 {method} 的有效结果目录{detail}")


def _ids(run: LoadedRun) -> Set[str]:
    return {_sample_id(row) for row in run.sample_rows}


def validate_runs(
    runs: Mapping[str, LoadedRun],
    allow_intersection: bool,
) -> tuple[Set[str], List[str]]:
    """检查方法集合、随机种子和样本集合。"""
    if tuple(runs.keys()) != EXPECTED_METHODS:
        raise ValueError("输入方法不完整或顺序错误")

    reference = runs["Proposed"]
    reference_ids = _ids(reference)
    common_ids = set(reference_ids)
    warnings: List[str] = []

    reference_seed = reference.config.get("random_seed")
    reference_modes = (
        reference.config.get("run_generation"),
        reference.config.get("run_judge"),
    )
    for method in EXPECTED_METHODS:
        run = runs[method]
        sample_ids = _ids(run)
        common_ids &= sample_ids

        if len(sample_ids) != len(run.sample_rows):
            raise ValueError(f"{method} 样本 ID 数量异常")
        if sample_ids != reference_ids:
            if not allow_intersection:
                missing = len(reference_ids - sample_ids)
                extra = len(sample_ids - reference_ids)
                raise ValueError(
                    f"{method} 与 Proposed 的样本 ID 不一致："
                    f"缺少 {missing}，额外 {extra}"
                )
            warnings.append(
                f"{method} 与 Proposed 的样本 ID 不一致，汇总仅保留共同样本"
            )

        seed = run.config.get("random_seed")
        if seed != reference_seed:
            raise ValueError(
                f"{method} 的 random_seed={seed!r}，"
                f"与 Proposed 的 {reference_seed!r} 不一致"
            )
        modes = (
            run.config.get("run_generation"),
            run.config.get("run_judge"),
        )
        if modes != reference_modes:
            raise ValueError(
                f"{method} 的生成/评判模式 {modes!r}，"
                f"与 Proposed 的 {reference_modes!r} 不一致"
            )

    if not common_ids:
        raise ValueError("所有方法之间没有共同样本")
    return common_ids, warnings


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return None if math.isnan(number) else number


def _mean(values: Iterable[Any], denominator: Optional[int] = None) -> Optional[float]:
    numbers = [number for value in values if (number := _number(value)) is not None]
    if denominator is not None:
        return sum(numbers) / denominator if denominator else None
    return sum(numbers) / len(numbers) if numbers else None


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 2)


def _metric(row: Mapping[str, Any], key: str) -> Any:
    retrieval = row.get("retrieval_metrics")
    return retrieval.get(key) if isinstance(retrieval, Mapping) else None


def _semantic(row: Mapping[str, Any], key: str) -> Any:
    semantic = row.get("semantic_metrics")
    return semantic.get(key) if isinstance(semantic, Mapping) else None


def _generation_success(row: Mapping[str, Any]) -> bool:
    if "generation_success" in row:
        return bool(row.get("generation_success"))
    return bool(str(row.get("generated_answer") or "").strip())


def _judge_valid(row: Mapping[str, Any]) -> bool:
    semantic = row.get("semantic_metrics")
    if not isinstance(semantic, Mapping):
        return False
    if "_judge_valid" in semantic:
        return bool(semantic.get("_judge_valid"))
    return all(
        _number(semantic.get(key)) is not None
        for key in (
            "correctness",
            "faithfulness",
            "answer_relevance",
            "context_relevance",
        )
    )


def _is_refusal(row: Mapping[str, Any]) -> bool:
    semantic = row.get("semantic_metrics")
    if isinstance(semantic, Mapping) and "_is_refusal" in semantic:
        return bool(semantic.get("_is_refusal"))
    answer = str(row.get("generated_answer") or "").strip().lower()
    if not answer:
        return True
    refusal_markers = (
        "insufficient information",
        "not enough information",
        "cannot determine",
        "cannot be determined",
        "can not be determined",
        "can't determine",
        "unable to determine",
        "insufficient evidence",
        "not enough evidence",
        "unknown",
        "无法确定",
        "信息不足",
    )
    return any(marker in answer for marker in refusal_markers)


def _f1(row: Mapping[str, Any]) -> float:
    value = _number(_metric(row, "f1"))
    if value is not None:
        return value
    recall = _number(_metric(row, "recall")) or 0.0
    precision = _number(_metric(row, "precision")) or 0.0
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def aggregate_rows(method: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """从逐样本明细重新计算统一方法摘要。"""
    total = len(rows)
    return {
        "Method": method,
        "Samples": total,
        "Generation Success Samples": sum(_generation_success(row) for row in rows),
        "Generation Success Rate": _round(
            sum(_generation_success(row) for row in rows) / total
            if total
            else 0.0
        ),
        "Valid Judge Samples": sum(_judge_valid(row) for row in rows),
        "Refusal Rate": _round(sum(_is_refusal(row) for row in rows) / total if total else 0.0),
        "Recall": _round(_mean(_metric(row, "recall") for row in rows)),
        "Precision": _round(_mean(_metric(row, "precision") for row in rows)),
        "F1": _round(_mean(_f1(row) for row in rows)),
        "MRR": _round(_mean(_metric(row, "mrr") for row in rows)),
        "NDCG": _round(_mean(_metric(row, "ndcg") for row in rows)),
        "MAP": _round(_mean(_metric(row, "map_score") for row in rows)),
        "Avg Len": _round(_mean(_metric(row, "avg_len") for row in rows)),
        "Time/ms (diagnostic)": _round(_mean(_metric(row, "time_ms") for row in rows)),
        "Expanded Nodes": _round(_mean(_metric(row, "expanded_nodes") for row in rows)),
        "correctness": _round(_mean((_semantic(row, "correctness") for row in rows), total)),
        "faithfulness": _round(_mean((_semantic(row, "faithfulness") for row in rows), total)),
        "answer_relevance": _round(
            _mean((_semantic(row, "answer_relevance") for row in rows), total)
        ),
        "context_relevance": _round(
            _mean((_semantic(row, "context_relevance") for row in rows), total)
        ),
    }


def aggregate_by_type(
    method: str,
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """从逐样本结果重新计算问题类型分层表。"""
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("type") or "unknown"), []).append(row)

    output: List[Dict[str, Any]] = []
    for question_type, group in sorted(grouped.items()):
        total = len(group)
        output.append(
            {
                "Method": method,
                "Type": question_type,
                "Samples": total,
                "Recall": _round(_mean(_metric(row, "recall") for row in group)),
                "Precision": _round(_mean(_metric(row, "precision") for row in group)),
                "F1": _round(_mean(_f1(row) for row in group)),
                "Generation Success Rate": _round(
                    sum(_generation_success(row) for row in group) / total
                    if total
                    else 0.0
                ),
                "Correctness": _round(
                    _mean((_semantic(row, "correctness") for row in group), total)
                ),
                "Faithfulness": _round(
                    _mean((_semantic(row, "faithfulness") for row in group), total)
                ),
                "Answer Relevance": _round(
                    _mean((_semantic(row, "answer_relevance") for row in group), total)
                ),
                "Context Relevance": _round(
                    _mean((_semantic(row, "context_relevance") for row in group), total)
                ),
                "Avg Len": _round(_mean(_metric(row, "avg_len") for row in group)),
                "Time/ms (diagnostic)": _round(
                    _mean(_metric(row, "time_ms") for row in group)
                ),
                "Expanded Nodes": _round(
                    _mean(_metric(row, "expanded_nodes") for row in group)
                ),
                "Refusal Rate": _round(
                    sum(_is_refusal(row) for row in group) / total if total else 0.0
                ),
            }
        )
    return output


def _resolve_run_dir(
    explicit: Optional[Path],
    results_root: Path,
    method: str,
) -> Path:
    return explicit.resolve() if explicit is not None else discover_latest_run(results_root, method)


def aggregate(args: argparse.Namespace) -> Dict[str, Any]:
    """加载、校验并写出统一对比结果。"""
    results_root = args.results_root.resolve()
    explicit_dirs = {
        "Semantic RAG": args.semantic_rag_dir,
        "Rerank RAG": args.rerank_rag_dir,
        "GraphRAG": args.graphrag_dir,
        "KG-RAG": args.kg_rag_dir,
        "MacRAG": args.macrag_dir,
        "Proposed": args.proposed_dir,
    }

    runs: Dict[str, LoadedRun] = {}
    for method in EXPECTED_METHODS:
        run_dir = _resolve_run_dir(explicit_dirs[method], results_root, method)
        runs[method] = load_run(run_dir, method)

    common_ids, warnings = validate_runs(runs, args.allow_intersection)
    filtered_rows: Dict[str, List[Dict[str, Any]]] = {}
    for method in EXPECTED_METHODS:
        filtered_rows[method] = sorted(
            (
                row
                for row in runs[method].sample_rows
                if _sample_id(row) in common_ids
            ),
            key=_sample_id,
        )

    summary = [
        aggregate_rows(method, filtered_rows[method])
        for method in EXPECTED_METHODS
    ]
    by_type: List[Dict[str, Any]] = []
    for method in EXPECTED_METHODS:
        by_type.extend(aggregate_by_type(method, filtered_rows[method]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir.resolve() if args.output_dir else results_root
    run_dir = output_root / f"comparison_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    _write_csv(run_dir / "comparison_summary.csv", summary, SUMMARY_COLUMNS)
    _write_csv(run_dir / "comparison_by_type.csv", by_type, TYPE_COLUMNS)
    _write_json(
        run_dir / "comparison_details.json",
        {
            "rows_by_method": filtered_rows,
            "summary": summary,
            "comparison_by_type": by_type,
        },
    )
    manifest = {
        "created_at": datetime.now().isoformat(),
        "strict_mode": not args.allow_intersection,
        "methods": {
            method: str(runs[method].run_dir)
            for method in EXPECTED_METHODS
        },
        "sample_count": len(common_ids),
        "random_seed": runs["Proposed"].config.get("random_seed"),
        "warnings": warnings,
        "arguments": {
            "results_root": str(results_root),
            "output_dir": str(output_root),
            "allow_intersection": bool(args.allow_intersection),
            "explicit_run_dirs": {
                method: str(path.resolve()) if path is not None else None
                for method, path in explicit_dirs.items()
            },
        },
    }
    _write_json(run_dir / "manifest.json", manifest)

    return {
        "run_dir": str(run_dir),
        "sample_count": len(common_ids),
        "methods": list(EXPECTED_METHODS),
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="汇总 v3 五个基线与 Proposed 的结果"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="默认结果根目录",
    )
    parser.add_argument("--semantic-rag-dir", type=Path, default=None)
    parser.add_argument("--rerank-rag-dir", type=Path, default=None)
    parser.add_argument("--graphrag-dir", type=Path, default=None)
    parser.add_argument("--kg-rag-dir", type=Path, default=None)
    parser.add_argument("--macrag-dir", type=Path, default=None)
    parser.add_argument("--proposed-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="汇总目录的父目录，默认使用 results-root",
    )
    parser.add_argument(
        "--allow-intersection",
        action="store_true",
        help="样本不一致时仅汇总所有方法的共同样本",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = aggregate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
