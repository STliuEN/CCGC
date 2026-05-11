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
            "USPS-only AE seed roll with a cluster-focused score. This is a thin "
            "wrapper around ae_graph_seed_sweep.py so the AE build/evaluation "
            "contract stays identical while NMI/ARI receive higher selection weight."
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
        help="Keep the inherited stop condition available, but default OFF for full 200-roll screening.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT / "experiment_output" / "ae_graph_seed_sweep" / f"{stamp}_usps_cluster_focus_200"
    else:
        run_dir = args.run_dir
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir

    sweep_args = [
        "ae_graph_seed_sweep.py",
        "--dataset",
        "usps",
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
        # Cluster-focused objective: keep ACC/F1 in the score, but rank AE
        # assets more aggressively by NMI/ARI and penalize their volatility.
        "--score-acc-weight",
        "0.70",
        "--score-nmi-weight",
        "0.60",
        "--score-ari-weight",
        "0.60",
        "--score-f1-weight",
        "0.30",
        "--score-acc-std-penalty",
        "0.20",
        "--score-nmi-std-penalty",
        "0.15",
        "--score-ari-std-penalty",
        "0.15",
        "--score-f1-std-penalty",
        "0.20",
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

