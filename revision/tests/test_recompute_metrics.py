import unittest

from revision.recompute_metrics import flatten_v1_row, flatten_v2_row, mean_record


class RecomputeMetricsTest(unittest.TestCase):
    def test_flatten_v1_row_extracts_nested_metrics(self):
        row = {
            "metrics": {
                "recall_at_k": {"10": 0.5},
                "precision_at_k": {"10": 0.2},
                "mrr": 1.0,
            },
            "generation_metrics": {"semantic_similarity": 0.7},
            "semantic_metrics": {"correctness": 0.8},
            "title_recall": 0.6,
        }

        flat = flatten_v1_row(row)

        self.assertEqual(flat["Recall@10"], 0.5)
        self.assertEqual(flat["Precision@10"], 0.2)
        self.assertEqual(flat["MRR"], 1.0)
        self.assertEqual(flat["Semantic Similarity"], 0.7)
        self.assertEqual(flat["correctness"], 0.8)
        self.assertEqual(flat["Title Recall"], 0.6)

    def test_flatten_v2_row_extracts_retrieval_and_semantic_metrics(self):
        row = {
            "retrieval_metrics": {"recall": 0.5, "precision": 0.25, "map_score": 0.4},
            "semantic_metrics": {"faithfulness": 0.9},
            "stats": {"route": "graph"},
        }

        flat = flatten_v2_row(row)

        self.assertEqual(flat["Recall"], 0.5)
        self.assertEqual(flat["Precision"], 0.25)
        self.assertEqual(flat["MAP"], 0.4)
        self.assertEqual(flat["faithfulness"], 0.9)

    def test_mean_record_ignores_none_values(self):
        record = mean_record([{"A": 1.0, "B": None}, {"A": 3.0, "B": 2.0}])

        self.assertEqual(record["A"], 2.0)
        self.assertEqual(record["B"], 2.0)


if __name__ == "__main__":
    unittest.main()
