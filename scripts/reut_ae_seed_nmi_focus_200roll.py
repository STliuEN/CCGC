from __future__ import annotations

import argparse
import runpy
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuters-only AE seed roll with an NMI-focused score. This thin "
            "wrapper keeps the same AE build/evaluation contract as "
            "ae_graph_seed_sweep.py while ranking assets more strongly by NMI."
        )
    )
    parser.add_argument("--attempts", type=int, default=200)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume-jsonl", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--update-summary-every", type=int, default=1)
    parser.add_argument("--force-ae", action="store_true")
    parser.add_argument("--train-only-existing-ae", action="store_true")
    parser.add_argument(
        "--stop-on-main-table-pass",
        action="store_true",
        help="Stop when the inherited main-table pass condition is reached. Default OFF for full screening.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT / "experiment_output" / "ae_graph_seed_sweep" / f"{stamp}_reut_nmi_focus_200"
    else:
        run_dir = args.run_dir
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir

    sweep_args = [
        "ae_graph_seed_sweep.py",
        "--dataset",
        "reut",
        "--random-seed-attempts",
        str(args.attempts),
        "--runs",
        str(args.runs),
        "--seed-start",
        str(args.seed_start),
        "--device",
        args.device,
        "--run-dir",
        str(run_dir),
        "--update-summary-every",
        str(args.update_summary_every),
        # Reuters target: push NMI first, while keeping ACC/ARI/F1 in the
        # selection score so a high-NMI but unstable asset does not dominate.
        "--score-acc-weight",
        "0.55",
        "--score-nmi-weight",
        "0.90",
        "--score-ari-weight",
        "0.55",
        "--score-f1-weight",
        "0.25",
        "--score-acc-std-penalty",
        "0.20",
        "--score-nmi-std-penalty",
        "0.25",
        "--score-ari-std-penalty",
        "0.20",
        "--score-f1-std-penalty",
        "0.15",
    ]
    if args.python is not None:
        sweep_args.extend(["--python", str(args.python)])
    if args.timeout:
        sweep_args.extend(["--timeout", str(args.timeout)])
    if args.resume_jsonl is not None:
        sweep_args.extend(["--resume-jsonl", str(args.resume_jsonl)])
    if args.force_ae:
        sweep_args.append("--force-ae")
    if args.train_only_existing_ae:
        sweep_args.append("--train-only-existing-ae")
    if args.stop_on_main_table_pass:
        sweep_args.append("--stop-on-main-table-pass")

    sys.argv = sweep_args
    runpy.run_path(str(ROOT / "scripts" / "ae_graph_seed_sweep.py"), run_name="__main__")


if __name__ == "__main__":
    main()
