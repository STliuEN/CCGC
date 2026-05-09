from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
if not DEFAULT_PYTHON.exists():
    DEFAULT_PYTHON = Path(sys.executable)


JOBS = [
    {
        "tag": "reut",
        "dataset": "reut",
        "cluster": 4,
        "extra": [
            "--fusion_hidden", "64",
            "--fusion_temp", "1.6",
            "--fusion_balance", "0.25",
            "--lambda_inst", "0.06",
            "--lambda_clu", "0.02",
            "--warmup_epochs", "35",
            "--fusion_min_weight", "0.15",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.3",
            "--disable_dcgl_neg_reliability_gate",
        ],
    },
    {
        "tag": "uat",
        "dataset": "uat",
        "cluster": 4,
        "extra": [
            "--t", "6",
            "--epochs", "500",
            "--lr", "0.00012",
            "--alpha", "0.45",
            "--fusion_hidden", "64",
            "--fusion_temp", "1.0",
            "--fusion_balance", "0.35",
            "--lambda_inst", "0.09",
            "--lambda_clu", "0.09",
            "--warmup_epochs", "35",
            "--fusion_min_weight", "0.20",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
            "--disable_dcgl_neg_reliability_gate",
        ],
    },
    {
        "tag": "amap",
        "dataset": "amap",
        "cluster": 8,
        "extra": [
            "--fusion_hidden", "64",
            "--fusion_temp", "1.0",
            "--fusion_balance", "0.08",
            "--lambda_inst", "0.0",
            "--lambda_clu", "0.035",
            "--warmup_epochs", "35",
            "--fusion_min_weight", "0.05",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
            "--disable_dcgl_neg_reliability_gate",
        ],
    },
    {
        "tag": "usps",
        "dataset": "usps",
        "cluster": 10,
        "extra": [
            "--t", "6",
            "--fusion_hidden", "64",
            "--fusion_temp", "1.0",
            "--fusion_balance", "0.3",
            "--lambda_inst", "0.09",
            "--lambda_clu", "0.09",
            "--warmup_epochs", "35",
            "--fusion_min_weight", "0.20",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
            "--disable_dcgl_neg_reliability_gate",
        ],
    },
    {
        "tag": "cora_free",
        "dataset": "cora",
        "cluster": 7,
        "extra": [
            "--fusion_hidden", "64",
            "--fusion_temp", "1.0",
            "--fusion_balance", "0.0",
            "--lambda_inst", "0.03",
            "--lambda_clu", "0.02",
            "--warmup_epochs", "70",
            "--fusion_min_weight", "0.0",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
        ],
    },
    {
        "tag": "cora_rawbias",
        "dataset": "cora",
        "cluster": 7,
        "extra": [
            "--fusion_hidden", "64",
            "--fusion_temp", "1.0",
            "--fusion_balance", "0.0",
            "--lambda_inst", "0.03",
            "--lambda_clu", "0.02",
            "--warmup_epochs", "70",
            "--fusion_min_weight", "0.0",
            "--enable_branch_bias_fusion",
            "--branch_bias_target", "raw",
            "--branch_bias_cap", "0.10",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
        ],
    },
    {
        "tag": "cite_free",
        "dataset": "cite",
        "cluster": 6,
        "extra": [
            "--fusion_hidden", "64",
            "--fusion_temp", "1.0",
            "--fusion_balance", "0.15",
            "--lambda_inst", "0.045",
            "--lambda_clu", "0.02",
            "--warmup_epochs", "55",
            "--fusion_min_weight", "0.10",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
            "--disable_dcgl_neg_reliability_gate",
        ],
    },
    {
        "tag": "cite_rawbias",
        "dataset": "cite",
        "cluster": 6,
        "extra": [
            "--fusion_hidden", "64",
            "--fusion_temp", "1.0",
            "--fusion_balance", "0.15",
            "--lambda_inst", "0.045",
            "--lambda_clu", "0.02",
            "--warmup_epochs", "55",
            "--fusion_min_weight", "0.10",
            "--enable_branch_bias_fusion",
            "--branch_bias_target", "raw",
            "--branch_bias_cap", "0.15",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
            "--disable_dcgl_neg_reliability_gate",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run no-adaptive-bias baseline checks before mechanism changes.")
    parser.add_argument("--jobs", default="all", help="Comma-separated tags or 'all'.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def pick_jobs(raw: str) -> list[dict[str, object]]:
    if raw.strip().lower() == "all":
        return JOBS
    wanted = {token.strip().lower() for token in raw.split(",") if token.strip()}
    jobs = [job for job in JOBS if str(job["tag"]).lower() in wanted]
    missing = wanted - {str(job["tag"]).lower() for job in jobs}
    if missing:
        raise ValueError(f"Unknown job tags: {', '.join(sorted(missing))}")
    return jobs


def metric_lines(stdout: str) -> list[str]:
    return [
        line
        for line in stdout.splitlines()
        if re.match(r"^(ACC|NMI|ARI|F1)\s+\|", line)
    ]


def run_details(stdout: str) -> list[str]:
    return [
        line
        for line in stdout.splitlines()
        if re.match(r"^Run \d+ Done", line)
    ]


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "experiment_output" / "adaptive_branch_bias_prechange_baseline" / stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.md"
    jobs = pick_jobs(args.jobs)

    lines = [
        "# Adaptive Branch Bias Pre-change Baseline",
        "",
        f"- Started: `{datetime.now().isoformat()}`",
        f"- Runs: `{args.runs}`",
        f"- Seeds: `{args.seed_start}..{args.seed_start + args.runs - 1}`",
        "- Note: `--enable_adaptive_branch_bias` is intentionally not passed.",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    for job in jobs:
        tag = str(job["tag"])
        dataset = str(job["dataset"])
        ae_graph = ROOT / "data" / "ae_graph" / f"{dataset}_ae_graph.txt"
        cmd = [
            str(args.python),
            "train.py",
            "--dataset", dataset,
            "--cluster_num", str(job["cluster"]),
            "--graph_mode", "dual",
            "--ae_graph_path", str(ae_graph),
            "--fusion_mode", "attn",
            "--knn_k", "5",
            "--runs", str(args.runs),
            "--seed_start", str(args.seed_start),
            "--dims", "500",
            "--linlayers", "1",
            "--epochs", "400",
            "--lr", "0.0001",
            "--device", args.device,
            "--threshold", "0.4",
            "--alpha", "0.5",
            "--enable_dcgl_negative_loss",
            *list(job["extra"]),
        ]

        print(f"[RUN] {tag}", flush=True)
        print(" ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        log_path = run_dir / f"{tag}.log"
        log_path.write_text(
            "\n".join(["[COMMAND]", " ".join(cmd), "", "[STDOUT]", proc.stdout or "", "", "[STDERR]", proc.stderr or ""]),
            encoding="utf-8",
        )

        add = ["", f"## {tag}", "", "```powershell", " ".join(cmd), "```", "", f"- Return code: `{proc.returncode}`"]
        add.extend(f"- {line}" for line in metric_lines(proc.stdout))
        details = run_details(proc.stdout)
        if details:
            add.extend(["", "Run details:"])
            add.extend(f"- {line}" for line in details)
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(add) + "\n")

        if proc.returncode != 0:
            print(proc.stdout, flush=True)
            print(proc.stderr, flush=True)
            raise RuntimeError(f"{tag} failed with return code {proc.returncode}; see {log_path}")

    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n- Finished: `{datetime.now().isoformat()}`\n")
    print(f"[DONE] summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
