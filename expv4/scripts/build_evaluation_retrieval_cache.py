from __future__ import annotations

import argparse
import json
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig
from src.pipeline import AdaptiveRecoveryPipeline


def _save(path: Path, cache: dict) -> None:
    with path.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Evaluation Retrieval Cache（构建评估集检索缓存）"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--requests-per-second", type=float, default=6.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    split_path = config.work_data_dir / "calibration_split_v4.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    evaluation_ids = split["evaluation_ids"]
    if args.limit is not None:
        evaluation_ids = evaluation_ids[: args.limit]
    frame = pd.read_parquet(config.validation_file)
    rows_by_id = {str(row["id"]): row for _, row in frame.iterrows()}
    output_path = (
        Path(args.output)
        if args.output
        else config.work_data_dir / "evaluation_retrieval_cache_v4.pkl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with output_path.open("rb") as handle:
            cache = pickle.load(handle)
    else:
        cache = {}

    cache = {
        example_id: item
        for example_id, item in cache.items()
        if example_id in evaluation_ids
    }

    def has_rerank_error(item: dict) -> bool:
        return any(
            unit.metadata.get("rerank_error")
            for unit in item["initial_evidence"]
        )

    pending = [
        example_id
        for example_id in evaluation_ids
        if example_id not in cache or has_rerank_error(cache[example_id])
    ]
    pipeline = AdaptiveRecoveryPipeline(config, enable_generation=False)
    started_at = time.perf_counter()
    rate_lock = threading.Lock()
    next_request_at = 0.0

    for chunk_start in range(0, len(pending), args.chunk_size):
        chunk_ids = pending[chunk_start : chunk_start + args.chunk_size]
        questions = [str(rows_by_id[example_id]["question"]) for example_id in chunk_ids]
        vectors = [
            cache[example_id]["query_vector"] if example_id in cache else None
            for example_id in chunk_ids
        ]
        missing_positions = [
            position for position, vector in enumerate(vectors) if vector is None
        ]
        for batch_start in range(0, len(missing_positions), config.embedding_batch_size):
            positions = missing_positions[
                batch_start : batch_start + config.embedding_batch_size
            ]
            embedded = pipeline.embedder.embed([questions[position] for position in positions])
            for position, vector in zip(positions, embedded):
                vectors[position] = vector

        def retrieve(position: int) -> tuple[str, dict]:
            nonlocal next_request_at
            example_id = chunk_ids[position]
            query = questions[position]
            query_vector = vectors[position]
            if args.requests_per_second > 0:
                with rate_lock:
                    wait_seconds = max(0.0, next_request_at - time.perf_counter())
                    next_request_at = max(next_request_at, time.perf_counter()) + (
                        1.0 / args.requests_per_second
                    )
                if wait_seconds:
                    time.sleep(wait_seconds)
            retrieval_started = time.perf_counter()
            initial, _ = pipeline.retriever.retrieve_with_vector(query, query_vector)
            return example_id, {
                "question": query,
                "initial_evidence": initial,
                "query_vector": query_vector,
                "retrieval_time_ms": (
                    time.perf_counter() - retrieval_started
                )
                * 1000,
            }

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(retrieve, index) for index in range(len(chunk_ids))]
            for future in as_completed(futures):
                example_id, item = future.result()
                cache[example_id] = item
        _save(output_path, cache)
        completed = sum(
            example_id in cache and not has_rerank_error(cache[example_id])
            for example_id in evaluation_ids
        )
        print(
            f"Completed {completed}/{len(evaluation_ids)} examples"
            "（已完成评估集检索缓存）",
            flush=True,
        )

    rerank_error_examples = sum(
        any(unit.metadata.get("rerank_error") for unit in item["initial_evidence"])
        for item in cache.values()
    )
    report = {
        "dataset_version": "HotpotQA v4（HotpotQA 第四版实验数据）",
        "artifact": "Evaluation Retrieval Cache（评估集检索缓存）",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "examples": len(cache),
        "expected_examples": len(evaluation_ids),
        "cache_complete": (
            len(cache) == len(evaluation_ids) and rerank_error_examples == 0
        ),
        "embedding_model": config.embedding_model,
        "rerank_model": config.rerank_model,
        "embedding_batch_size": config.embedding_batch_size,
        "rerank_workers": args.workers,
        "rerank_requests_per_second": args.requests_per_second,
        "rerank_error_examples": rerank_error_examples,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "cache_file": str(output_path),
        "split_file": str(split_path),
    }
    output_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
