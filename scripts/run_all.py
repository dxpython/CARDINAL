"""Sequential run of the full reproducibility pipeline.

Usage:  python scripts/run_all.py    (from the code/ directory)

Runs each experiment in order, finishing with figure/table export and an
environment report.  Individual failures are recorded and reported but do not
stop the pipeline, so missing data or dependencies are surfaced honestly.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import get_config
from src.utils import device_info, save_json

RUN_ORDER = [
    "run_clinical_main",
    "run_small_sample",
    "run_uncertainty",
    "run_graph_analysis",
    "run_ablation",
    "run_label_exclusion",
    "run_public_benchmarks",
]


def main():
    cfg = get_config()
    status = {"device": device_info(), "steps": []}

    import run_clinical_main, run_small_sample, run_uncertainty, run_graph_analysis
    import run_ablation, run_label_exclusion, run_public_benchmarks

    fn = {
        "run_clinical_main": run_clinical_main.main,
        "run_small_sample": run_small_sample.main,
        "run_uncertainty": run_uncertainty.main,
        "run_graph_analysis": run_graph_analysis.main,
        "run_ablation": run_ablation.main,
        "run_label_exclusion": run_label_exclusion.main,
        "run_public_benchmarks": run_public_benchmarks.main,
    }

    for name in RUN_ORDER:
        print(f"\n=== {name} ===")
        try:
            fn[name]()
            status["steps"].append({"step": name, "status": "ok"})
        except Exception as e:  # noqa: BLE001
            status["steps"].append({"step": name, "status": "failed",
                                    "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()

    # figure and table export
    try:
        import make_all_figures, make_all_tables
        make_all_figures.main()
        make_all_tables.main()
        status["steps"].append({"step": "make_all_figures", "status": "ok"})
        status["steps"].append({"step": "make_all_tables", "status": "ok"})
    except Exception as e:  # noqa: BLE001
        status["steps"].append({"step": "export", "status": "failed",
                                "error": f"{type(e).__name__}: {e}"})

    save_json(status, cfg.results_dir() / "RUN_STATUS.json")
    n_fail = sum(1 for s in status["steps"] if s["status"] == "failed")
    print(f"\n[run_all] finished. {len(status['steps']) - n_fail}/{len(status['steps'])} "
          f"steps OK. Details in {cfg.results_dir() / 'RUN_STATUS.json'}")


if __name__ == "__main__":
    main()
