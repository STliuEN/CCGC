from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "feedback_safe_tuning"
METRICS = ("ACC", "NMI", "ARI", "F1")
DEFAULT_PYTHON = Path("/root/miniconda3/envs/SCGC_2/bin/python")

CURRENT_OURS_TARGET = {
    "reut": {"ACC": 83.20, "NMI": 59.82, "ARI": 66.01, "F1": 70.57},
    "uat": {"ACC": 56.24, "NMI": 27.31, "ARI": 21.98, "F1": 56.44},
    "amap": {"ACC": 77.39, "NMI": 67.22, "ARI": 58.25, "F1": 71.69},
    "usps": {"ACC": 82.40, "NMI": 73.29, "ARI": 68.39, "F1": 82.16},
    "eat": {"ACC": 54.76, "NMI": 31.97, "ARI": 24.32, "F1": 52.98},
    "cora": {"ACC": 73.49, "NMI": 55.55, "ARI": 50.14, "F1": 71.17},
    "cite": {"ACC": 70.74, "NMI": 45.28, "ARI": 45.18, "F1": 61.56},
}

MAIN_TABLE_BEST = {
    "reut": {"ACC": 83.20, "NMI": 59.90, "ARI": 66.01, "F1": 70.57},
    "uat": {"ACC": 56.58, "NMI": 28.15, "ARI": 25.52, "F1": 56.44},
    "amap": {"ACC": 77.48, "NMI": 67.67, "ARI": 58.48, "F1": 72.22},
    "usps": {"ACC": 84.91, "NMI": 84.16, "ARI": 79.50, "F1": 82.16},
    "eat": {"ACC": 57.94, "NMI": 33.91, "ARI": 27.71, "F1": 57.96},
    "cora": {"ACC": 73.88, "NMI": 57.58, "ARI": 52.51, "F1": 71.17},
    "cite": {"ACC": 73.29, "NMI": 46.92, "ARI": 50.21, "F1": 64.80},
}


@dataclass(frozen=True)
class Candidate:
    name: str
    base_id: str
    changed_params: dict[str, Any]
    note: str


INITIAL_CANDIDATES: dict[str, list[Candidate]] = {
    "reut": [
        Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck."),
        Candidate("probe_dcgl_neg_weight_0p4", "base_current", {"dcgl_neg_weight": 0.4}, "Lower negative strength first."),
        Candidate("probe_dcgl_neg_tau_0p75", "base_current", {"dcgl_neg_tau": 0.75}, "Softer negative temperature."),
        Candidate("probe_fusion_min_weight_0p20", "base_current", {"fusion_min_weight": 0.20}, "Check whether a slightly higher floor helps F1."),
    ],
    "uat": [
        Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck."),
        Candidate("probe_dcgl_neg_weight_0p4", "base_current", {"dcgl_neg_weight": 0.4}, "Lower negative strength first."),
        Candidate("probe_dcgl_neg_tau_0p75", "base_current", {"dcgl_neg_tau": 0.75}, "Softer negative temperature."),
        Candidate("probe_fusion_balance_0p45", "base_current", {"fusion_balance": 0.45}, "Higher balance pressure without changing other axes."),
    ],
    "amap": [
        Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck."),
        Candidate("probe_fusion_balance_0p05", "base_current", {"fusion_balance": 0.05}, "Best first one-parameter direction from the optional grid memo."),
        Candidate("probe_fusion_min_weight_0p00", "base_current", {"fusion_min_weight": 0.0}, "Test whether removing the branch floor helps the attention branch."),
        Candidate("probe_dcgl_neg_tau_1p00", "base_current", {"dcgl_neg_tau": 1.0}, "Sharper vs softer negative behavior probe."),
        Candidate("probe_dcgl_neg_weight_0p80", "base_current", {"dcgl_neg_weight": 0.8}, "Slightly stronger negative guidance."),
        Candidate("probe_lambda_inst_0p03", "base_current", {"lambda_inst": 0.03}, "Lower instance regularization from the current base."),
        Candidate(
            "memo_joint_fa18e3c0",
            "base_current",
            {
                "fusion_balance": 0.05,
                "fusion_min_weight": 0.0,
                "lambda_inst": 0.03,
                "dcgl_neg_tau": 1.0,
                "dcgl_neg_weight": 0.8,
            },
            "Optional memo confirmation candidate after first single-axis probes.",
        ),
    ],
    "usps": [
        Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck."),
        Candidate("probe_dcgl_neg_weight_0p4", "base_current", {"dcgl_neg_weight": 0.4}, "Lower negative strength per handoff."),
        Candidate("probe_fusion_temp_2p0", "base_current", {"fusion_temp": 2.0}, "Softer attention temperature."),
        Candidate("probe_fusion_min_weight_0p15", "base_current", {"fusion_min_weight": 0.15}, "Lower branch floor."),
    ],
    "eat": [
        Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck."),
        Candidate("probe_threshold_0p45", "base_current", {"threshold": 0.45}, "Confidence threshold probe."),
        Candidate("probe_dcgl_neg_weight_0p4", "base_current", {"dcgl_neg_weight": 0.4}, "Lower negative strength for stability."),
        Candidate("probe_dcgl_neg_tau_0p75", "base_current", {"dcgl_neg_tau": 0.75}, "Softer negative temperature."),
    ],
    "cora": [
        Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck."),
        Candidate("probe_threshold_0p35", "base_current", {"threshold": 0.35}, "Slightly more selective confidence cutoff."),
        Candidate("probe_branch_bias_cap_0p12", "base_current", {"branch_bias_cap": 0.12}, "Raw-branch bias cap probe."),
        Candidate("probe_dcgl_neg_weight_0p4", "base_current", {"dcgl_neg_weight": 0.4}, "Lower negative strength."),
    ],
    "cite": [
        Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck."),
        Candidate("probe_fusion_balance_0p25", "base_current", {"fusion_balance": 0.25}, "Stronger balance pressure."),
        Candidate("probe_fusion_min_weight_0p15", "base_current", {"fusion_min_weight": 0.15}, "Slightly higher branch floor."),
        Candidate("probe_branch_bias_cap_0p18", "base_current", {"branch_bias_cap": 0.18}, "Raw-branch bias cap probe."),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run safe feedback-tuning probes under the DSAFC handoff contract.")
    parser.add_argument("--dataset", choices=tuple(INITIAL_CANDIDATES.keys()), default="amap")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=0, help="Per-command timeout in seconds. 0 disables timeout.")
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means all candidates in the selected first batch.")
    parser.add_argument(
        "--candidate-names",
        default="",
        help="Comma-separated candidate names to run. Empty means the dataset first batch.",
    )
    parser.add_argument("--force-ae", action="store_true", help="Regenerate the fixed AE graph even if it already exists.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_experiment_config() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("ccgc_experiment_config", ROOT / "experiment.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, dict):
        raise RuntimeError("experiment.py does not expose a CONFIG dict")
    return config


def dict_to_cli(values: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        cli.extend([flag, str(value)])
    return cli


def merge_args(*dicts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in dicts:
        if item:
            merged.update(item)
    return merged


def parse_metrics(text: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(ACC|NMI|ARI|F1)\s*\|(.+)$", line)
        if not match:
            continue
        nums = re.findall(r"\d+(?:\.\d+)?", match.group(2))
        if len(nums) >= 2:
            metrics[match.group(1)] = {"mean": float(nums[0]), "std": float(nums[1])}
    return metrics


def score_metrics(metrics: dict[str, dict[str, float]]) -> float | None:
    if not all(metric in metrics for metric in METRICS):
        return None
    return (
        metrics["ACC"]["mean"]
        + 0.4 * metrics["F1"]["mean"]
        + 0.2 * metrics["NMI"]["mean"]
        + 0.2 * metrics["ARI"]["mean"]
        - 0.25 * (metrics["ACC"]["std"] + metrics["F1"]["std"])
    )


def metric_gaps(metrics: dict[str, dict[str, float]], target: dict[str, float]) -> dict[str, float]:
    return {
        metric: metrics[metric]["mean"] - target[metric]
        for metric in METRICS
        if metric in metrics
    }


def decision_label(ours_gaps: dict[str, float], best_gaps: dict[str, float]) -> str:
    ours_wins = sum(1 for metric in METRICS if ours_gaps.get(metric, float("-inf")) > 0)
    best_wins = sum(1 for metric in METRICS if best_gaps.get(metric, float("-inf")) > 0)
    if best_wins == 4:
        return "HEADLINE_ACCEPT"
    if ours_wins == 4:
        return "STRONG_ACCEPT"
    if ours_wins >= 3:
        return "ACCEPT"
    if ours_wins >= 1:
        return "SIDE_NOTE"
    return "REJECT"


def run_subprocess(cmd: list[str], log_path: Path, timeout: int) -> tuple[int, float, bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.now()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout if timeout > 0 else None,
        )
        output = proc.stdout or ""
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        output += f"\n[TIMEOUT] Command exceeded {timeout} seconds.\n"
        returncode = 124
        timed_out = True

    elapsed = (datetime.now() - start).total_seconds()
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(f"COMMAND: {' '.join(cmd)}\n")
        handle.write(f"RETURN_CODE: {returncode}\n")
        handle.write(f"ELAPSED_SEC: {elapsed:.2f}\n")
        handle.write(f"TIMED_OUT: {'YES' if timed_out else 'NO'}\n")
        handle.write("=" * 80 + "\n")
        handle.write(output)
    return returncode, elapsed, timed_out, output


def select_candidates(dataset: str, candidate_names: str, max_candidates: int) -> list[Candidate]:
    candidates = INITIAL_CANDIDATES[dataset]
    if candidate_names.strip():
        wanted = {name.strip() for name in candidate_names.split(",") if name.strip()}
        candidates = [candidate for candidate in candidates if candidate.name in wanted]
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    if not candidates:
        raise ValueError("No candidates selected.")
    return candidates


def build_ae_command(config: dict[str, Any], dataset: str, python_path: Path, device: str) -> tuple[list[str], Path]:
    profile = config["dataset_profiles"][dataset]
    ae_args = merge_args(config.get("ae_args", {}), profile.get("ae_args", {}))
    graph_path = ROOT / "data" / "ae_graph" / f"{dataset}_ae_graph.txt"
    model_path = OUTPUT_ROOT / dataset / "fixed_ae" / f"{dataset}_ae_pretrain_seed42_graph42.pkl"
    ae_args.update(
        {
            "dataset": dataset,
            "cluster_num": profile["cluster_num"],
            "device": device,
            "out_graph_path": str(graph_path.relative_to(ROOT)),
            "model_save_path": str(model_path.relative_to(ROOT)),
            "pretrain_seed": 42,
            "graph_seed": 42,
        }
    )
    cmd = [str(python_path), "data/pretrain_optimize_A_graph.py"] + dict_to_cli(ae_args)
    return cmd, graph_path


def build_train_command(
    config: dict[str, Any],
    dataset: str,
    candidate: Candidate,
    python_path: Path,
    device: str,
    runs: int,
    seed_start: int,
    graph_path: Path,
) -> list[str]:
    profile = config["dataset_profiles"][dataset]
    train_args = merge_args(
        config.get("train_common_args", {}),
        profile.get("train_args", {}),
        config.get("baseline_args", {}),
        config.get("dual_attn_args", {}),
        profile.get("dual_attn_args", {}),
        config.get("dcgl_negative_args", {}),
    )
    train_args.update(candidate.changed_params)
    train_args.update(
        {
            "device": device,
            "runs": runs,
            "seed_start": seed_start,
            "ae_graph_path": str(graph_path.relative_to(ROOT)),
            "enable_dcgl_negative_loss": True,
        }
    )

    cmd = [
        str(python_path),
        "train.py",
        "--dataset",
        dataset,
        "--cluster_num",
        str(profile["cluster_num"]),
        "--graph_mode",
        "dual",
        "--fusion_mode",
        "attn",
    ] + dict_to_cli(train_args)
    return cmd


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def write_summary(dataset_dir: Path, dataset: str, results: list[dict[str, Any]]) -> None:
    summary_path = dataset_dir / "summary.md"
    ordered_results = sorted(
        results,
        key=lambda row: (
            float("-inf") if row.get("score") is None else float(row["score"]),
            row.get("timestamp", ""),
        ),
        reverse=True,
    )
    lines = [
        "# Safe Feedback Tuning",
        "",
        f"- Dataset: {dataset}",
        f"- Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Current Ours target: {CURRENT_OURS_TARGET[dataset]}",
        f"- Main-table best target: {MAIN_TABLE_BEST[dataset]}",
        "",
        "| Candidate | Score | Label | Ours wins | Best wins | ACC | NMI | ARI | F1 | Log |",
        "| --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]

    for row in ordered_results:
        metrics = row.get("metrics", {})
        ours_gaps = row.get("ours_gaps", {})
        best_gaps = row.get("best_gaps", {})
        ours_wins = sum(1 for metric in METRICS if ours_gaps.get(metric, float("-inf")) > 0)
        best_wins = sum(1 for metric in METRICS if best_gaps.get(metric, float("-inf")) > 0)

        def metric_text(metric: str) -> str:
            value = metrics.get(metric)
            if not value:
                return "-"
            return f"{value['mean']:.2f}±{value['std']:.2f}"

        score = row.get("score")
        score_text = "-" if score is None else f"{score:.2f}"
        lines.append(
            f"| {row['candidate']} | {score_text} | {row['label']} | {ours_wins} | {best_wins} | "
            f"{metric_text('ACC')} | {metric_text('NMI')} | {metric_text('ARI')} | {metric_text('F1')} | "
            f"{row['log_path']} |"
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_experiment_config()
    dataset = args.dataset
    dataset_dir = OUTPUT_ROOT / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": dataset,
        "python": str(args.python),
        "device": args.device,
        "runs": int(args.runs),
        "seed_start": int(args.seed_start),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "hard_contract": {
            "graph_mode": "dual",
            "fusion_mode": "attn",
            "enable_dcgl_negative_loss": True,
            "disable_dynamic_threshold": True,
            "disable_ema_prototypes": True,
            "disable_dcgl_cluster_level": True,
            "disable_gcn_backbone": True,
            "pretrain_seed": 42,
            "graph_seed": 42,
        },
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    candidates = select_candidates(dataset, args.candidate_names, args.max_candidates)
    ae_cmd, graph_path = build_ae_command(config, dataset, args.python, args.device)
    ae_log = dataset_dir / "logs" / "fixed_ae_seed42_graph42.txt"
    results_jsonl = dataset_dir / "results.jsonl"
    results = load_existing_results(results_jsonl)

    if args.force_ae or not graph_path.exists():
        print(f"[AE] Preparing fixed AE graph for {dataset}: {graph_path}", flush=True)
        if args.dry_run:
            print(" ".join(ae_cmd), flush=True)
        else:
            returncode, elapsed, timed_out, _output = run_subprocess(ae_cmd, ae_log, args.timeout)
            if returncode != 0:
                print(f"[AE] Failed for {dataset}. See {ae_log}", flush=True)
                return returncode
            print(f"[AE] Done in {elapsed:.2f}s", flush=True)
    else:
        print(f"[AE] Reusing existing fixed AE graph: {graph_path}", flush=True)

    for index, candidate in enumerate(candidates, start=1):
        print(f"[RUN {index}/{len(candidates)}] {dataset} | {candidate.name}", flush=True)
        train_cmd = build_train_command(
            config=config,
            dataset=dataset,
            candidate=candidate,
            python_path=args.python,
            device=args.device,
            runs=args.runs,
            seed_start=args.seed_start,
            graph_path=graph_path,
        )
        log_path = dataset_dir / "logs" / f"{candidate.name}.txt"
        if args.dry_run:
            print(" ".join(train_cmd), flush=True)
            continue

        returncode, elapsed, timed_out, output = run_subprocess(train_cmd, log_path, args.timeout)
        metrics = parse_metrics(output)
        ours_gaps = metric_gaps(metrics, CURRENT_OURS_TARGET[dataset]) if metrics else {}
        best_gaps = metric_gaps(metrics, MAIN_TABLE_BEST[dataset]) if metrics else {}
        score = score_metrics(metrics)
        label = "FAILED" if returncode != 0 else decision_label(ours_gaps, best_gaps)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": dataset,
            "candidate": candidate.name,
            "base_id": candidate.base_id,
            "changed_params": candidate.changed_params,
            "note": candidate.note,
            "cmd": train_cmd,
            "returncode": returncode,
            "elapsed_sec": elapsed,
            "timed_out": timed_out,
            "metrics": metrics,
            "score": score,
            "ours_gaps": ours_gaps,
            "best_gaps": best_gaps,
            "label": label,
            "log_path": str(log_path.relative_to(ROOT)),
            "ae_graph_path": str(graph_path.relative_to(ROOT)),
        }
        append_jsonl(results_jsonl, row)
        results.append(row)
        write_summary(dataset_dir, dataset, results)

        if returncode != 0:
            print(f"[RUN] Failed for {candidate.name}. See {log_path}", flush=True)
            return returncode

        if metrics:
            acc = metrics["ACC"]["mean"]
            f1 = metrics["F1"]["mean"]
            score_text = "nan" if score is None else f"{score:.2f}"
            print(
                f"[DONE] {candidate.name} | ACC {acc:.2f} | F1 {f1:.2f} | score {score_text}",
                flush=True,
            )

    write_summary(dataset_dir, dataset, results)
    print(f"[SUMMARY] Wrote {dataset_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
