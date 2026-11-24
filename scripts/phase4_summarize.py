"""Summarize ML Pipeline Monitor Phase 3 improvements results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent.parent / "experiment_results" / "mlpm_phase3_improvements.json"


def main() -> int:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        r = json.load(f)

    print("Datasets:", list(r.get("datasets", {}).keys()))
    for ds_name, ds_data in r.get("datasets", {}).items():
        print(f"  {ds_name} ({ds_data.get('task')}): {ds_data.get('type')}")
        for model_name, model_data in ds_data.get("models", {}).items():
            if ds_data.get("task") == "classification":
                print(
                    f"    {model_name}: Acc={model_data.get('accuracy', 0):.4f} "
                    f"F1={model_data.get('f1_score', 0):.4f} "
                    f"ROC={model_data.get('roc_auc', 0)}"
                )
            else:
                print(
                    f"    {model_name}: R2={model_data.get('r2', 0):.4f} "
                    f"RMSE={model_data.get('rmse', 0):.4f}"
                )
    print()
    print("Drift methods:")
    for method_name, method_data in r.get("drift", {}).get("methods", {}).items():
        print(
            f"  {method_name}: F1={method_data.get('f1', 0):.4f} "
            f"Precision={method_data.get('precision', 0):.4f} "
            f"Recall={method_data.get('recall', 0):.4f} "
            f"FPR={method_data.get('fpr', 0):.4f}"
        )
    print()
    print("Nested CV:")
    for model_name, nc_data in r.get("nested_cv", {}).items():
        print(f"  {model_name}: {nc_data}")

    return 0


if __name__ == "__main__":
    sys.exit(main())