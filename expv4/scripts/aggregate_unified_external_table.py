from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "expv4" / "results" / "final_2000_v4" / "unified_external"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    retrieval_path = OUTPUT_DIR / "table2_unified_external_retrieval_v4.json"
    generation_path = OUTPUT_DIR / "table3_unified_external_generation_v4.json"
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    generation = json.loads(generation_path.read_text(encoding="utf-8"))

    generation["new_generated_answers"] = 12000 - generation["reused_identical_answers"]
    generation["new_semantic_evaluations"] = 12000 - generation["reused_identical_evaluations"]
    generation["provider_request_attempts"] = "not_reported_due_to_resumable_retries"
    generation.pop("new_generation_requests", None)
    generation.pop("new_evaluation_requests", None)
    write_json(generation_path, generation)

    retrieval_by_method = {row["method"]: row for row in retrieval["table"]}
    generation_by_method = {row["method"]: row for row in generation["table"]}
    method_order = [row["method"] for row in retrieval["table"]]
    table = []
    for method in method_order:
        ret = retrieval_by_method[method]
        gen = generation_by_method[method]
        table.append(
            {
                "method": method,
                "label": ret["label"],
                "examples": ret["examples"],
                "title_recall": ret["title_recall"],
                "title_precision": ret["title_precision"],
                "title_mrr": ret["title_mrr"],
                "title_ndcg": ret["title_ndcg"],
                "title_map": ret["title_average_precision"],
                "avg_context_tokens": ret["avg_token"],
                "refusal_rate": gen["refusal_rate"],
                "answer_em": gen["answer_em"],
                "answer_f1": gen["answer_f1"],
                "accuracy": gen["accuracy"],
                "faithfulness": gen["faithfulness"],
                "answer_relevance": gen["answer_relevance"],
                "context_relevance": gen["context_relevance"],
            }
        )

    payload = {
        "dataset": retrieval["dataset"],
        "comparison_protocol": retrieval["comparison_protocol"],
        "shared_corpus_documents": retrieval["shared_corpus_documents"],
        "shared_validated_relation_edges": retrieval["shared_validated_relation_edges"],
        "context_budget_tokens": retrieval["context_budget_tokens"],
        "max_context_units": retrieval["max_context_units"],
        "generation_model": generation["generation_model"],
        "thinking_mode": generation["thinking_mode"],
        "temperature": generation["temperature"],
        "evaluation_protocol": generation["evaluation_protocol"],
        "baseline_scope": (
            "Core online mechanisms reproduced over identical V4 resources; "
            "not original-paper default configurations or copied paper scores."
        ),
        "table": table,
    }
    json_path = OUTPUT_DIR / "table2_unified_external_full_v4.json"
    csv_path = OUTPUT_DIR / "table2_unified_external_full_v4.csv"
    write_json(json_path, payload)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    progress_path = OUTPUT_DIR / "generation_progress_v4.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.pop("new_generation_requests", None)
    progress.pop("new_evaluation_requests", None)
    progress.update(
        {
            "new_generated_answers": generation["new_generated_answers"],
            "new_semantic_evaluations": generation["new_semantic_evaluations"],
            "provider_request_attempts": "not_reported_due_to_resumable_retries",
        }
    )
    write_json(progress_path, progress)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
