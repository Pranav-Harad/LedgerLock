"""Evaluation Harness for LedgerLock Reconciliation Pipeline.

Compares reconciliation outputs against ground_truth.json to compute:
- Match rates and tier breakdown per injected failure category
- Exception detection Precision, Recall, and F1-Score
- False-Match Count (honesty metric ensuring true exceptions are never falsely matched)
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Any, Optional
from tabulate import tabulate

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class ReconciliationEvalHarness:
    """Quantitative evaluation against ground truth."""

    def __init__(self, ground_truth_path: Optional[str] = None):
        self.gt_path = ground_truth_path or os.path.join(DATA_DIR, "ground_truth.json")
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> Dict[str, Dict[str, Any]]:
        with open(self.gt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(
        self,
        tier1_results: Dict[str, Any],
        tier2_results: Dict[str, Any],
        tds_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate full pipeline reconciliation output against ground truth."""
        t1_matched_ids = set()
        for _, row in tier1_results["all_matched"].iterrows():
            if "transaction_id_ledger" in row and pd_not_na(row["transaction_id_ledger"]):
                t1_matched_ids.add(str(row["transaction_id_ledger"]))
            if "transaction_id_settlement" in row and pd_not_na(row["transaction_id_settlement"]):
                t1_matched_ids.add(str(row["transaction_id_settlement"]))

        t2_matched_ids = set()
        for m in tier2_results.get("tier2_matches", []):
            for tid in m.get("transaction_ids", []):
                t2_matched_ids.add(str(tid))

        all_matched_ids = t1_matched_ids | t2_matched_ids

        flagged_exc_ids = set()
        for exc in tier2_results.get("exceptions_flagged", []):
            flagged_exc_ids.add(str(exc.get("transaction_id")))

        tds_violation_ids = set(v.get("transaction_id") for v in tds_results.get("violations", []))

        category_breakdown: Dict[str, Dict[str, Any]] = {}
        false_matches: List[str] = []
        true_positives_exc = 0
        false_positives_exc = 0
        false_negatives_exc = 0
        true_negatives_exc = 0

        for tx_id, gt_info in self.ground_truth.items():
            category = gt_info["category"]
            expected_res = gt_info["expected_resolution"]
            master_id = gt_info.get("master_id")

            if category not in category_breakdown:
                category_breakdown[category] = {
                    "total": 0,
                    "matched_t1": 0,
                    "matched_t2": 0,
                    "flagged_exception": 0,
                    "tds_flagged": 0,
                }

            cat_stat = category_breakdown[category]
            cat_stat["total"] += 1

            is_in_t1 = tx_id in t1_matched_ids or (master_id and master_id in t1_matched_ids)
            is_in_t2 = tx_id in t2_matched_ids or (master_id and master_id in t2_matched_ids)
            is_flagged = tx_id in flagged_exc_ids or (master_id and master_id in flagged_exc_ids)

            if is_in_t1:
                cat_stat["matched_t1"] += 1
            if is_in_t2:
                cat_stat["matched_t2"] += 1
            if is_flagged:
                cat_stat["flagged_exception"] += 1
            if tx_id in tds_violation_ids:
                cat_stat["tds_flagged"] += 1

            # Check for False Matches on genuine exceptions
            if expected_res == "exception":
                if is_in_t1 or is_in_t2:
                    false_matches.append(tx_id)
                if is_flagged:
                    true_positives_exc += 1
                else:
                    false_negatives_exc += 1
            else:  # expected matched
                if is_flagged and not (is_in_t1 or is_in_t2):
                    false_positives_exc += 1
                else:
                    true_negatives_exc += 1

        precision = (
            true_positives_exc / (true_positives_exc + false_positives_exc)
            if (true_positives_exc + false_positives_exc) > 0
            else 1.0
        )
        recall = (
            true_positives_exc / (true_positives_exc + false_negatives_exc)
            if (true_positives_exc + false_negatives_exc) > 0
            else 1.0
        )
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 1.0

        eval_summary = {
            "total_ground_truth_records": len(self.ground_truth),
            "false_match_count": len(false_matches),
            "false_matches": false_matches,
            "exception_detection": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "true_positives": true_positives_exc,
                "false_positives": false_positives_exc,
                "false_negatives": false_negatives_exc,
            },
            "category_breakdown": category_breakdown,
        }

        return eval_summary


def pd_not_na(val: Any) -> bool:
    return val is not None and str(val) != "nan" and str(val) != ""


def print_eval_report(eval_summary: Dict[str, Any]) -> None:
    """Print formatted ASCII table report of evaluation metrics."""
    print("\n" + "=" * 75)
    print("           LEDGERLOCK — QUANTITATIVE EVALUATION HARNESS REPORT")
    print("=" * 75)

    exc_metrics = eval_summary["exception_detection"]
    print(f"Total Ground Truth Records Evaluated: {eval_summary['total_ground_truth_records']}")
    print(f"False-Match Count (Honesty Metric):   {eval_summary['false_match_count']} (Target: 0)")
    print(f"Exception Detection Precision:         {exc_metrics['precision']*100:.1f}%")
    print(f"Exception Detection Recall:            {exc_metrics['recall']*100:.1f}%")
    print(f"Exception Detection F1-Score:          {exc_metrics['f1_score']:.4f}")
    print("-" * 75)

    table_data = []
    for cat, data in eval_summary["category_breakdown"].items():
        total = data["total"]
        resolved = data["matched_t1"] + data["matched_t2"]
        res_rate = (resolved / total * 100) if total > 0 else 0.0
        table_data.append([
            cat,
            total,
            data["matched_t1"],
            data["matched_t2"],
            data["flagged_exception"],
            data["tds_flagged"],
            f"{res_rate:.1f}%",
        ])

    headers = ["Category", "Total", "Tier 1 Matched", "Tier 2 Matched", "Flagged Exc", "TDS Flagged", "Resolution %"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("=" * 75 + "\n")
