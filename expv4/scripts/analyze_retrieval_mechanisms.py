from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import pandas as pd


EXPV4_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPV4_ROOT.parent
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig
from src.indexing.bridge_index import load_bridge_index
from src.metrics.paper_metrics import title_ranking_scores, unique_titles


ERROR_LABELS = {
    "seed_retrieval_failure": "A. Seed Retrieval Failure（种子检索失败）",
    "bridge_index_coverage_failure": "B. Bridge Index Coverage Failure（桥接索引覆盖失败）",
    "bridge_ranking_failure": "C. Bridge Ranking/Gating Failure（桥接排序或门控失败）",
    "target_localization_failure": "D. Target/Supporting Sentence Localization Failure（目标或支持句定位失败）",
    "evidence_selection_drop": "E. Evidence Selection Drop（证据选择丢失）",
    "retrieval_metric_penalty": "F. Retrieval Metric Penalty（恢复成功但排序不佳）",
    "complete_success": "Complete Success（完整成功）",
    "other_incomplete": "Other Incomplete（其他未完整情况）",
}


def facts(values: list[list | tuple]) -> set[tuple[str, int]]:
    return {(str(title), int(sent_id)) for title, sent_id in values}


def titles(values: set[tuple[str, int]]) -> set[str]:
    return {title for title, _ in values}


def ranking(context: list[list | tuple], gold_facts: set[tuple[str, int]]) -> dict:
    predicted_titles = unique_titles([tuple(value) for value in context])
    return title_ranking_scores(predicted_titles, sorted(titles(gold_facts)))


def load_ablation_rows(
    path: Path, selected_ids: set[str], variants: set[str]
) -> dict[str, dict[str, dict]]:
    rows = {variant: {} for variant in variants}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            example_id = str(item["id"])
            variant = str(item["variant"])
            if example_id in selected_ids and variant in rows:
                rows[variant][example_id] = item
    return rows


def bridge_error_analysis(
    bridge_ids: list[str],
    full_rows: dict[str, dict],
    retrieval_cache: dict,
    bridge_index: dict,
) -> tuple[list[dict], dict]:
    cases = []
    primary_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()

    for example_id in bridge_ids:
        row = full_rows[example_id]
        gold = facts(row["gold_supporting_facts"])
        gold_titles = titles(gold)
        initial = {unit.key for unit in retrieval_cache[example_id]["initial_evidence"]}
        initial_titles = titles(initial)
        candidate = facts(row["candidate_evidence"])
        candidate_titles = titles(candidate)
        selected = facts(row["context_evidence"])
        selected_titles = titles(selected)
        missing_initial_titles = gold_titles - initial_titles

        reachable_targets: set[str] = set()
        correct_links = 0
        for seed_title in initial_titles & gold_titles:
            for link in bridge_index.get(seed_title, []):
                if link.target_title in missing_initial_titles:
                    reachable_targets.add(link.target_title)
                    correct_links += 1

        initial_gold_documents = len(initial_titles & gold_titles)
        seed_failure = initial_gold_documents == 0
        coverage_failure = bool(
            initial_gold_documents > 0
            and missing_initial_titles
            and (missing_initial_titles - reachable_targets)
        )
        ranking_failure = bool(
            (missing_initial_titles & reachable_targets) - candidate_titles
        )
        missing_localized_gold_facts = {
            item
            for item in gold
            if item[0] in candidate_titles and item not in candidate
        }
        localization_failure = bool(missing_localized_gold_facts)
        selection_drop = bool((gold & candidate) - selected)
        scores = ranking(row["context_evidence"], gold)
        all_gold_titles_selected = gold_titles <= selected_titles
        retrieval_penalty = bool(
            all_gold_titles_selected
            and (
                scores["title_mrr"] < 1.0
                or scores["title_average_precision"] < 1.0
            )
        )
        all_gold_facts_selected = gold <= selected

        flags = {
            "seed_retrieval_failure": seed_failure,
            "bridge_index_coverage_failure": coverage_failure,
            "bridge_ranking_failure": ranking_failure,
            "target_localization_failure": localization_failure,
            "evidence_selection_drop": selection_drop,
            "retrieval_metric_penalty": retrieval_penalty,
        }
        for name, active in flags.items():
            if active:
                flag_counts[name] += 1

        if seed_failure:
            primary = "seed_retrieval_failure"
        elif coverage_failure:
            primary = "bridge_index_coverage_failure"
        elif ranking_failure:
            primary = "bridge_ranking_failure"
        elif localization_failure:
            primary = "target_localization_failure"
        elif selection_drop:
            primary = "evidence_selection_drop"
        elif retrieval_penalty:
            primary = "retrieval_metric_penalty"
        elif all_gold_facts_selected:
            primary = "complete_success"
        else:
            primary = "other_incomplete"
        primary_counts[primary] += 1

        cases.append(
            {
                "id": example_id,
                "question": row["question"],
                "gold_titles": sorted(gold_titles),
                "initial_titles": sorted(initial_titles),
                "candidate_titles": sorted(candidate_titles),
                "selected_titles": sorted(selected_titles),
                "missing_initial_gold_titles": sorted(missing_initial_titles),
                "reachable_missing_gold_titles": sorted(reachable_targets),
                "missing_gold_facts_after_localization": sorted(missing_localized_gold_facts),
                "correct_bridge_links": correct_links,
                "candidate_gold_facts": len(gold & candidate),
                "selected_gold_facts": len(gold & selected),
                "gold_facts": len(gold),
                "all_gold_titles_selected": all_gold_titles_selected,
                "all_gold_facts_selected": all_gold_facts_selected,
                "title_recall": scores["title_recall"],
                "title_mrr": scores["title_mrr"],
                "title_map": scores["title_average_precision"],
                "flags": flags,
                "primary_category": primary,
                "primary_label": ERROR_LABELS[primary],
            }
        )

    total = len(cases)
    summary = {
        "analysis": "Bridge Error Analysis（桥接型错误分析）",
        "examples": total,
        "classification_note": (
            "Flags are multi-label; primary category follows A→F pipeline order. "
            "Complete success and other incomplete are reported separately."
        ),
        "multi_label_counts": {
            name: {
                "label": ERROR_LABELS[name],
                "count": flag_counts[name],
                "rate": flag_counts[name] / total,
            }
            for name in list(ERROR_LABELS)[:6]
        },
        "primary_counts": {
            name: {
                "label": ERROR_LABELS[name],
                "count": primary_counts[name],
                "rate": primary_counts[name] / total,
            }
            for name in ERROR_LABELS
        },
        "complete_gold_title_chain_rate": mean(
            float(case["all_gold_titles_selected"]) for case in cases
        ),
        "complete_gold_fact_chain_rate": mean(
            float(case["all_gold_facts_selected"]) for case in cases
        ),
        "external_model_calls": 0,
    }
    return cases, summary


def utility_category(
    full_gold_count: int,
    one_gold_count: int,
    added_facts: set[tuple[str, int]],
    one_facts: set[tuple[str, int]],
) -> str:
    if full_gold_count > one_gold_count:
        return "helpful"
    if full_gold_count < one_gold_count:
        return "harmful"
    if not added_facts or titles(added_facts) <= titles(one_facts):
        return "redundant"
    return "retrieval_neutral"


def second_hop_analysis(
    example_ids: list[str],
    type_by_id: dict[str, str],
    full_rows: dict[str, dict],
    one_hop_rows: dict[str, dict],
) -> tuple[list[dict], dict]:
    cases = []
    for example_id in example_ids:
        full = full_rows[example_id]
        if not full["stats"]["second_bridge_hop"]:
            continue
        one = one_hop_rows[example_id]
        gold = facts(full["gold_supporting_facts"])
        gold_titles = titles(gold)
        full_selected = facts(full["context_evidence"])
        one_selected = facts(one["context_evidence"])
        added = full_selected - one_selected
        removed = one_selected - full_selected
        full_gold_titles = len(titles(full_selected) & gold_titles)
        one_gold_titles = len(titles(one_selected) & gold_titles)
        full_gold_facts = len(full_selected & gold)
        one_gold_facts = len(one_selected & gold)
        title_category = utility_category(
            full_gold_titles, one_gold_titles, added, one_selected
        )
        fact_category = utility_category(
            full_gold_facts, one_gold_facts, added, one_selected
        )
        full_scores = ranking(full["context_evidence"], gold)
        one_scores = ranking(one["context_evidence"], gold)
        cases.append(
            {
                "id": example_id,
                "question": full["question"],
                "question_type": type_by_id[example_id],
                "title_utility": title_category,
                "fact_utility": fact_category,
                "one_hop_complete_title_chain": gold_titles <= titles(one_selected),
                "full_complete_title_chain": gold_titles <= titles(full_selected),
                "one_hop_complete_fact_chain": gold <= one_selected,
                "full_complete_fact_chain": gold <= full_selected,
                "one_hop_gold_titles": one_gold_titles,
                "full_gold_titles": full_gold_titles,
                "one_hop_gold_facts": one_gold_facts,
                "full_gold_facts": full_gold_facts,
                "added_facts": sorted(added),
                "removed_facts": sorted(removed),
                "one_hop_title_recall": one_scores["title_recall"],
                "full_title_recall": full_scores["title_recall"],
                "one_hop_map": one_scores["title_average_precision"],
                "full_map": full_scores["title_average_precision"],
            }
        )

    def subset_summary(values: list[dict]) -> dict:
        title_counts = Counter(case["title_utility"] for case in values)
        fact_counts = Counter(case["fact_utility"] for case in values)
        total = len(values)
        categories = ["helpful", "harmful", "retrieval_neutral", "redundant"]
        if total == 0:
            return {
                "examples": 0,
                "title_level": {
                    name: {"count": 0, "rate": 0.0} for name in categories
                },
                "fact_level": {
                    name: {"count": 0, "rate": 0.0} for name in categories
                },
                "one_hop_complete_title_chain_rate": None,
                "full_complete_title_chain_rate": None,
                "one_hop_mean_title_recall": None,
                "full_mean_title_recall": None,
                "one_hop_mean_map": None,
                "full_mean_map": None,
            }
        return {
            "examples": total,
            "title_level": {
                name: {"count": title_counts[name], "rate": title_counts[name] / total}
                for name in categories
            },
            "fact_level": {
                name: {"count": fact_counts[name], "rate": fact_counts[name] / total}
                for name in categories
            },
            "one_hop_complete_title_chain_rate": mean(
                float(case["one_hop_complete_title_chain"]) for case in values
            ),
            "full_complete_title_chain_rate": mean(
                float(case["full_complete_title_chain"]) for case in values
            ),
            "one_hop_mean_title_recall": mean(
                case["one_hop_title_recall"] for case in values
            ),
            "full_mean_title_recall": mean(
                case["full_title_recall"] for case in values
            ),
            "one_hop_mean_map": mean(case["one_hop_map"] for case in values),
            "full_mean_map": mean(case["full_map"] for case in values),
        }

    bridge_cases = [case for case in cases if case["question_type"] == "bridge"]
    summary = {
        "analysis": "Second-hop Utility Analysis（第二跳效用分析）",
        "triggered_examples": len(cases),
        "all_triggered": subset_summary(cases),
        "bridge_questions_only": subset_summary(bridge_cases),
        "external_model_calls": 0,
    }
    return cases, summary


def structural_analysis(
    example_ids: list[str],
    type_by_id: dict[str, str],
    full_rows: dict[str, dict],
    without_rows: dict[str, dict],
) -> tuple[list[dict], dict]:
    cases = []
    for example_id in example_ids:
        full = full_rows[example_id]
        if not full["stats"]["structural_activated"]:
            continue
        without = without_rows[example_id]
        gold = facts(full["gold_supporting_facts"])
        full_candidate = facts(full["candidate_evidence"])
        full_selected = facts(full["context_evidence"])
        without_selected = facts(without["context_evidence"])
        full_scores = ranking(full["context_evidence"], gold)
        without_scores = ranking(without["context_evidence"], gold)
        displaced_gold = (without_selected & gold) - full_selected
        displaced_gold_in_full_candidates = displaced_gold & full_candidate
        recovered_gold = (full_selected & gold) - without_selected

        if (
            full_scores["title_recall"] > without_scores["title_recall"]
            or (
                full_scores["title_recall"] == without_scores["title_recall"]
                and full_scores["title_average_precision"]
                > without_scores["title_average_precision"]
            )
        ):
            category = "useful_structural_refinement"
        elif (
            full_scores["title_recall"] < without_scores["title_recall"]
            or full_scores["title_average_precision"]
            < without_scores["title_average_precision"]
        ) and displaced_gold_in_full_candidates:
            category = "budget_displacement"
        elif (
            full_scores["title_recall"] < without_scores["title_recall"]
            or full_scores["title_average_precision"]
            < without_scores["title_average_precision"]
        ):
            category = "other_interaction_loss"
        else:
            category = "redundant_structural_expansion"

        cases.append(
            {
                "id": example_id,
                "question": full["question"],
                "question_type": type_by_id[example_id],
                "category": category,
                "full_title_recall": full_scores["title_recall"],
                "without_structural_title_recall": without_scores["title_recall"],
                "full_map": full_scores["title_average_precision"],
                "without_structural_map": without_scores[
                    "title_average_precision"
                ],
                "full_selected_gold_facts": len(full_selected & gold),
                "without_structural_selected_gold_facts": len(
                    without_selected & gold
                ),
                "recovered_gold_facts": sorted(recovered_gold),
                "displaced_gold_facts": sorted(displaced_gold),
                "displaced_gold_present_in_full_candidates": sorted(
                    displaced_gold_in_full_candidates
                ),
                "structural_added_units": full["stats"]["structural_added_units"],
                "full_context_tokens": full["stats"]["selected_context_tokens"],
                "without_structural_context_tokens": without["stats"][
                    "selected_context_tokens"
                ],
            }
        )

    def subset_summary(values: list[dict]) -> dict:
        counts = Counter(case["category"] for case in values)
        total = len(values)
        categories = [
            "useful_structural_refinement",
            "redundant_structural_expansion",
            "budget_displacement",
            "other_interaction_loss",
        ]
        return {
            "examples": total,
            "categories": {
                name: {"count": counts[name], "rate": counts[name] / total}
                for name in categories
            },
            "full_mean_title_recall": mean(
                case["full_title_recall"] for case in values
            ),
            "without_structural_mean_title_recall": mean(
                case["without_structural_title_recall"] for case in values
            ),
            "full_mean_map": mean(case["full_map"] for case in values),
            "without_structural_mean_map": mean(
                case["without_structural_map"] for case in values
            ),
            "full_mean_context_tokens": mean(
                case["full_context_tokens"] for case in values
            ),
            "without_structural_mean_context_tokens": mean(
                case["without_structural_context_tokens"] for case in values
            ),
        }

    bridge_cases = [case for case in cases if case["question_type"] == "bridge"]
    summary = {
        "analysis": "Structural Recovery Utility Analysis（结构恢复效用分析）",
        "activated_examples": len(cases),
        "all_activated": subset_summary(cases),
        "bridge_questions_only": subset_summary(bridge_cases),
        "external_model_calls": 0,
    }
    return cases, summary


def write_outputs(
    output_dir: Path,
    stem: str,
    cases: list[dict],
    summary: dict,
    category_rows: list[dict],
) -> None:
    cases_path = output_dir / f"{stem}.cases.jsonl"
    summary_path = output_dir / f"{stem}.summary.json"
    csv_path = output_dir / f"{stem}.summary.csv"
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    summary["cases_file"] = str(cases_path)
    summary["csv_file"] = str(csv_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(category_rows[0]))
        writer.writeheader()
        writer.writerows(category_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval Mechanism Error Analysis（检索机制误差分析）"
    )
    parser.add_argument("--split-file")
    parser.add_argument(
        "--split-part",
        choices=("calibration_ids", "evaluation_ids"),
        default="evaluation_ids",
    )
    parser.add_argument("--retrieval-cache")
    parser.add_argument("--ablation-file")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = ExperimentConfig.load(EXPV4_ROOT / "configs" / "proposed.json")
    split_path = (
        Path(args.split_file)
        if args.split_file
        else PROJECT_ROOT / "data" / "v4" / "final_evaluation_split_2000_v4.json"
    )
    cache_path = (
        Path(args.retrieval_cache)
        if args.retrieval_cache
        else PROJECT_ROOT / "data" / "v4" / "final_evaluation_retrieval_cache_2000_v4.pkl"
    )
    ablation_path = (
        Path(args.ablation_file)
        if args.ablation_file
        else EXPV4_ROOT / "results" / "frozen_ablation_study_v4.jsonl"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else EXPV4_ROOT / "results" / "final_2000_v4" / "retrieval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    example_ids = [str(value) for value in split[args.split_part]]
    selected_ids = set(example_ids)
    if split.get("examples"):
        type_by_id = {
            str(item["id"]): str(item["type"]) for item in split["examples"]
        }
    else:
        frame = pd.read_parquet(config.validation_file, columns=["id", "type"])
        type_by_id = {
            str(row["id"]): str(row["type"])
            for _, row in frame.iterrows()
            if str(row["id"]) in selected_ids
        }
    bridge_ids = [
        example_id for example_id in example_ids if type_by_id[example_id] == "bridge"
    ]
    with cache_path.open("rb") as handle:
        retrieval_cache = pickle.load(handle)
    variants = {"full_adaptive", "one_hop_only", "without_structural"}
    rows = load_ablation_rows(
        ablation_path,
        selected_ids,
        variants,
    )
    bridge_index = load_bridge_index(config.bridge_index_file)

    bridge_cases, bridge_summary = bridge_error_analysis(
        bridge_ids, rows["full_adaptive"], retrieval_cache, bridge_index
    )
    bridge_csv = []
    for classification, values in (
        ("multi_label", bridge_summary["multi_label_counts"]),
        ("primary", bridge_summary["primary_counts"]),
    ):
        for category, item in values.items():
            bridge_csv.append(
                {
                    "classification": classification,
                    "category": category,
                    "label": item["label"],
                    "count": item["count"],
                    "rate": item["rate"],
                }
            )
    write_outputs(
        output_dir,
        "bridge_error_analysis_v4",
        bridge_cases,
        bridge_summary,
        bridge_csv,
    )

    second_cases, second_summary = second_hop_analysis(
        example_ids,
        type_by_id,
        rows["full_adaptive"],
        rows["one_hop_only"],
    )
    second_csv = []
    for subset in ("all_triggered", "bridge_questions_only"):
        for level in ("title_level", "fact_level"):
            for category, item in second_summary[subset][level].items():
                second_csv.append(
                    {
                        "subset": subset,
                        "level": level,
                        "category": category,
                        "count": item["count"],
                        "rate": item["rate"],
                    }
                )
    write_outputs(
        output_dir,
        "second_hop_utility_analysis_v4",
        second_cases,
        second_summary,
        second_csv,
    )

    structural_cases, structural_summary = structural_analysis(
        example_ids,
        type_by_id,
        rows["full_adaptive"],
        rows["without_structural"],
    )
    structural_csv = []
    for subset in ("all_activated", "bridge_questions_only"):
        for category, item in structural_summary[subset]["categories"].items():
            structural_csv.append(
                {
                    "subset": subset,
                    "category": category,
                    "count": item["count"],
                    "rate": item["rate"],
                }
            )
    write_outputs(
        output_dir,
        "structural_utility_analysis_v4",
        structural_cases,
        structural_summary,
        structural_csv,
    )

    report = {
        "analysis": "Retrieval Mechanism Error Analysis（检索机制误差分析）",
        "split_part": args.split_part,
        "split_file": str(split_path),
        "retrieval_cache": str(cache_path),
        "ablation_file": str(ablation_path),
        "bridge_examples": len(bridge_cases),
        "second_hop_triggered_examples": len(second_cases),
        "structural_activated_examples": len(structural_cases),
        "external_model_calls": 0,
        "large_language_model_calls": 0,
        "output_dir": str(output_dir),
    }
    (output_dir / "retrieval_mechanism_analysis_report_v4.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
