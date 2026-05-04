from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "feedback_safe_tuning"
DEFAULT_PYTHON = Path("/root/miniconda3/envs/SCGC_2/bin/python")
DATASET_ORDER = ("amap", "reut", "uat", "cora", "cite", "eat", "usps")
METRICS = ("ACC", "NMI", "ARI", "F1")
SUMMARY_LOCK = threading.Lock()

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
    phase: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full 7-dataset safe feedback tuning campaign.")
    parser.add_argument("--datasets", default=",".join(DATASET_ORDER), help="Comma-separated dataset order.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--per-dataset-hours", type=float, default=6.0)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--force-ae", action="store_true")
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


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
    changed_params: dict[str, Any],
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
    train_args.update(changed_params)
    train_args.update(
        {
            "device": device,
            "runs": runs,
            "seed_start": seed_start,
            "ae_graph_path": str(graph_path.relative_to(ROOT)),
            "enable_dcgl_negative_loss": True,
        }
    )
    return [
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


def build_dataset_candidates(dataset: str) -> list[Candidate]:
    if dataset == "amap":
        return [
            Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck.", "phase1"),
            Candidate("probe_fusion_balance_0p05", "base_current", {"fusion_balance": 0.05}, "Best first one-parameter direction from prior local evidence.", "phase1"),
            Candidate("probe_fusion_min_weight_0p00", "base_current", {"fusion_min_weight": 0.0}, "Remove branch floor.", "phase1"),
            Candidate("probe_dcgl_neg_tau_1p00", "base_current", {"dcgl_neg_tau": 1.0}, "Softer negative temperature.", "phase1"),
            Candidate("probe_dcgl_neg_weight_0p80", "base_current", {"dcgl_neg_weight": 0.8}, "Stronger negative weight.", "phase1"),
            Candidate("probe_lambda_inst_0p03", "base_current", {"lambda_inst": 0.03}, "Lower instance regularization.", "phase1"),
            Candidate("pair_balance0p05_lambda0p03", "probe_fusion_balance_0p05", {"fusion_balance": 0.05, "lambda_inst": 0.03}, "Pairwise follow-up around the strongest single-axis hint.", "phase2"),
            Candidate("pair_balance0p05_negw0p80", "probe_fusion_balance_0p05", {"fusion_balance": 0.05, "dcgl_neg_weight": 0.8}, "Pairwise low-balance plus stronger DCGL-negative.", "phase2"),
            Candidate("pair_balance0p05_tau1p00", "probe_fusion_balance_0p05", {"fusion_balance": 0.05, "dcgl_neg_tau": 1.0}, "Pairwise low-balance plus higher negative temperature.", "phase2"),
            Candidate("memo_joint_fa18e3c0", "probe_fusion_balance_0p05", {"fusion_balance": 0.05, "fusion_min_weight": 0.0, "lambda_inst": 0.03, "dcgl_neg_tau": 1.0, "dcgl_neg_weight": 0.8}, "Memo-guided joint confirmation candidate.", "phase3"),
        ]
    if dataset == "reut":
        return [
            Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck.", "phase1"),
            Candidate("probe_dcgl_neg_weight_0p4", "base_current", {"dcgl_neg_weight": 0.4}, "Lower negative strength first.", "phase1"),
            Candidate("probe_dcgl_neg_weight_0p3", "base_current", {"dcgl_neg_weight": 0.3}, "Further lower negative strength.", "phase1"),
            Candidate("probe_dcgl_neg_tau_0p75", "base_current", {"dcgl_neg_tau": 0.75}, "Softer negative temperature.", "phase1"),
            Candidate("probe_dcgl_neg_tau_1p00", "base_current", {"dcgl_neg_tau": 1.0}, "Further soften negative temperature.", "phase1"),
            Candidate("probe_fusion_min_weight_0p20", "base_current", {"fusion_min_weight": 0.20}, "Slightly higher branch floor.", "phase1"),
            Candidate("probe_fusion_balance_0p20", "base_current", {"fusion_balance": 0.20}, "Slightly lower balance pressure.", "phase1"),
            Candidate("pair_negw0p4_tau0p75", "base_current", {"dcgl_neg_weight": 0.4, "dcgl_neg_tau": 0.75}, "Pairwise negative-strength refinement.", "phase2"),
            Candidate("pair_floor0p20_tau0p75", "base_current", {"fusion_min_weight": 0.20, "dcgl_neg_tau": 0.75}, "F1-oriented pairwise test.", "phase2"),
            Candidate("pair_floor0p20_negw0p4", "base_current", {"fusion_min_weight": 0.20, "dcgl_neg_weight": 0.4}, "Floor plus lower negative weight.", "phase2"),
        ]
    if dataset == "uat":
        return [
            Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck.", "phase1"),
            Candidate("probe_dcgl_neg_weight_0p5", "base_current", {"dcgl_neg_weight": 0.5}, "Slightly lower negative strength.", "phase1"),
            Candidate("probe_dcgl_neg_weight_0p4", "base_current", {"dcgl_neg_weight": 0.4}, "Lower negative strength.", "phase1"),
            Candidate("probe_dcgl_neg_weight_0p3", "base_current", {"dcgl_neg_weight": 0.3}, "Further lower negative strength.", "phase1"),
            Candidate("probe_dcgl_neg_tau_0p75", "base_current", {"dcgl_neg_tau": 0.75}, "Softer negative temperature.", "phase1"),
            Candidate("probe_dcgl_neg_tau_1p00", "base_current", {"dcgl_neg_tau": 1.0}, "Higher negative temperature.", "phase1"),
            Candidate("probe_fusion_balance_0p40", "base_current", {"fusion_balance": 0.40}, "Intermediate balance pressure.", "phase1"),
            Candidate("probe_lambda_clu_0p075", "base_current", {"lambda_clu": 0.075}, "Slightly higher cluster consistency.", "phase1"),
            Candidate("pair_negw0p4_tau0p75", "base_current", {"dcgl_neg_weight": 0.4, "dcgl_neg_tau": 0.75}, "Pairwise negative-strength refinement.", "phase2"),
            Candidate("pair_balance0p40_negw0p4", "base_current", {"fusion_balance": 0.40, "dcgl_neg_weight": 0.4}, "Balance plus lower negative strength.", "phase2"),
        ]
    if dataset == "cora":
        return [
            Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck.", "phase1"),
            Candidate("probe_threshold_0p35", "base_current", {"threshold": 0.35}, "More selective confidence threshold.", "phase1"),
            Candidate("probe_branch_bias_cap_0p08", "base_current", {"branch_bias_cap": 0.08}, "Lower raw-branch bias cap.", "phase1"),
            Candidate("probe_branch_bias_cap_0p12", "base_current", {"branch_bias_cap": 0.12}, "Higher raw-branch bias cap.", "phase1"),
            Candidate("probe_dcgl_neg_weight_0p4", "base_current", {"dcgl_neg_weight": 0.4}, "Lower negative strength.", "phase1"),
            Candidate("probe_dcgl_neg_tau_0p75", "base_current", {"dcgl_neg_tau": 0.75}, "Softer negative temperature.", "phase1"),
            Candidate("probe_lambda_inst_0p04", "base_current", {"lambda_inst": 0.04}, "Slightly stronger instance consistency.", "phase1"),
            Candidate("probe_warmup_55", "base_current", {"warmup_epochs": 55}, "Shorter warmup.", "phase1"),
            Candidate("pair_threshold0p35_negw0p4", "base_current", {"threshold": 0.35, "dcgl_neg_weight": 0.4}, "Threshold plus lower negative weight.", "phase2"),
            Candidate("pair_cap0p12_negw0p4", "base_current", {"branch_bias_cap": 0.12, "dcgl_neg_weight": 0.4}, "Higher branch cap plus lower negative weight.", "phase2"),
        ]
    if dataset == "cite":
        return [
            Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck.", "phase1"),
            Candidate("probe_fusion_balance_0p25", "base_current", {"fusion_balance": 0.25}, "Higher balance pressure.", "phase1"),
            Candidate("probe_fusion_balance_0p35", "base_current", {"fusion_balance": 0.35}, "Even higher balance pressure.", "phase1"),
            Candidate("probe_fusion_min_weight_0p15", "base_current", {"fusion_min_weight": 0.15}, "Slightly higher branch floor.", "phase1"),
            Candidate("probe_branch_bias_cap_0p12", "base_current", {"branch_bias_cap": 0.12}, "Lower raw-branch bias cap.", "phase1"),
            Candidate("probe_branch_bias_cap_0p18", "base_current", {"branch_bias_cap": 0.18}, "Higher raw-branch bias cap.", "phase1"),
            Candidate("probe_dcgl_neg_weight_0p4", "base_current", {"dcgl_neg_weight": 0.4}, "Lower negative strength.", "phase1"),
            Candidate("probe_dcgl_neg_tau_0p75", "base_current", {"dcgl_neg_tau": 0.75}, "Higher negative temperature.", "phase1"),
            Candidate("pair_cap0p18_floor0p15", "base_current", {"branch_bias_cap": 0.18, "fusion_min_weight": 0.15}, "Citation-specific pairwise test.", "phase2"),
            Candidate("pair_balance0p25_negw0p4", "base_current", {"fusion_balance": 0.25, "dcgl_neg_weight": 0.4}, "Balance plus lower negative strength.", "phase2"),
        ]
    if dataset == "eat":
        return [
            Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck.", "phase1"),
            Candidate("probe_threshold_0p45", "base_current", {"threshold": 0.45}, "Confidence threshold probe.", "phase1"),
            Candidate("probe_threshold_0p50", "base_current", {"threshold": 0.50}, "Higher confidence threshold.", "phase1"),
            Candidate("probe_dcgl_neg_weight_0p4", "base_current", {"dcgl_neg_weight": 0.4}, "Lower negative strength.", "phase1"),
            Candidate("probe_dcgl_neg_weight_0p3", "base_current", {"dcgl_neg_weight": 0.3}, "Further lower negative strength.", "phase1"),
            Candidate("probe_dcgl_neg_tau_0p75", "base_current", {"dcgl_neg_tau": 0.75}, "Softer negative temperature.", "phase1"),
            Candidate("probe_fusion_temp_1p8", "base_current", {"fusion_temp": 1.8}, "Lower attention temperature.", "phase1"),
            Candidate("probe_lambda_inst_0p10", "base_current", {"lambda_inst": 0.10}, "Slightly stronger instance consistency.", "phase1"),
            Candidate("pair_threshold0p45_negw0p4", "base_current", {"threshold": 0.45, "dcgl_neg_weight": 0.4}, "Threshold plus lower negative weight.", "phase2"),
            Candidate("pair_temp1p8_negw0p4", "base_current", {"fusion_temp": 1.8, "dcgl_neg_weight": 0.4}, "Lower temperature plus lower negative weight.", "phase2"),
        ]
    if dataset == "usps":
        return [
            Candidate("base_current", "base_current", {}, "Fixed-AE 10-run base recheck.", "phase1"),
            Candidate("probe_dcgl_neg_weight_0p5", "base_current", {"dcgl_neg_weight": 0.5}, "Slightly lower negative strength.", "phase1"),
            Candidate("probe_dcgl_neg_weight_0p4", "base_current", {"dcgl_neg_weight": 0.4}, "Lower negative strength.", "phase1"),
            Candidate("probe_dcgl_neg_weight_0p3", "base_current", {"dcgl_neg_weight": 0.3}, "Further lower negative strength.", "phase1"),
            Candidate("probe_fusion_temp_2p0", "base_current", {"fusion_temp": 2.0}, "Softer attention temperature.", "phase1"),
            Candidate("probe_fusion_temp_2p2", "base_current", {"fusion_temp": 2.2}, "Even softer attention temperature.", "phase1"),
            Candidate("probe_fusion_min_weight_0p15", "base_current", {"fusion_min_weight": 0.15}, "Lower branch floor.", "phase1"),
            Candidate("probe_dcgl_neg_tau_0p75", "base_current", {"dcgl_neg_tau": 0.75}, "Higher negative temperature.", "phase1"),
            Candidate("pair_temp2p0_negw0p4", "base_current", {"fusion_temp": 2.0, "dcgl_neg_weight": 0.4}, "Temperature plus lower negative weight.", "phase2"),
            Candidate("pair_floor0p15_negw0p4", "base_current", {"fusion_min_weight": 0.15, "dcgl_neg_weight": 0.4}, "Lower floor plus lower negative weight.", "phase2"),
        ]
    raise ValueError(f"Unsupported dataset: {dataset}")


def metric_text(metrics: dict[str, dict[str, float]], metric: str) -> str:
    value = metrics.get(metric)
    if not value:
        return "-"
    return f"{value['mean']:.2f}"


def std_text(metrics: dict[str, dict[str, float]]) -> str:
    if not metrics:
        return "-"
    return "/".join(f"{metrics[m]['std']:.2f}" for m in METRICS if m in metrics)


def write_feedback_log(dataset_dir: Path, results: list[dict[str, Any]]) -> None:
    path = dataset_dir / "feedback_log.md"
    ordered = sorted(results, key=lambda row: row.get("timestamp", ""))
    lines = [
        "# Feedback Log",
        "",
        "| Step | Base | Changed | ACC | NMI | ARI | F1 | Std ACC/NMI/ARI/F1 | Decision | Note | Log |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(ordered, start=1):
        metrics = row.get("metrics", {})
        changed = row.get("changed_params") or {}
        changed_text = ", ".join(f"{k}={v}" for k, v in changed.items()) if changed else "base"
        lines.append(
            f"| {idx} | {row.get('base_id', '-')} | {changed_text} | "
            f"{metric_text(metrics, 'ACC')} | {metric_text(metrics, 'NMI')} | {metric_text(metrics, 'ARI')} | {metric_text(metrics, 'F1')} | "
            f"{std_text(metrics)} | {row.get('label', '-')} | {row.get('note', '-')} | {row.get('log_path', '-')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_global_summary(dataset_names: list[str]) -> None:
    path = OUTPUT_ROOT / "summary.md"
    lines = [
        "# Safe Feedback Tuning Campaign",
        "",
        f"- Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Dataset | Best Candidate | Score | Label | ACC | NMI | ARI | F1 | Feedback Log | Summary |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for dataset in dataset_names:
        results = load_existing_results(OUTPUT_ROOT / dataset / "results.jsonl")
        if not results:
            lines.append(f"| {dataset} | - | - | - | - | - | - | - | - | - |")
            continue
        best = max(
            results,
            key=lambda row: (
                float("-inf") if row.get("score") is None else float(row["score"]),
                row.get("timestamp", ""),
            ),
        )
        metrics = best.get("metrics", {})
        score = best.get("score")
        score_text = "-" if score is None else f"{score:.2f}"
        lines.append(
            f"| {dataset} | {best.get('candidate', '-')} | {score_text} | {best.get('label', '-')} | "
            f"{metric_text(metrics, 'ACC')} | {metric_text(metrics, 'NMI')} | {metric_text(metrics, 'ARI')} | {metric_text(metrics, 'F1')} | "
            f"experiment_output/feedback_safe_tuning/{dataset}/feedback_log.md | "
            f"experiment_output/feedback_safe_tuning/{dataset}/summary.md |"
        )
    with SUMMARY_LOCK:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_with_std(metrics: dict[str, dict[str, float]], metric: str) -> str:
    value = metrics.get(metric)
    if not value:
        return "-"
    return f"{value['mean']:.2f}±{value['std']:.2f}"


def write_dataset_summary(dataset_dir: Path, dataset: str, results: list[dict[str, Any]]) -> None:
    summary_path = dataset_dir / "summary.md"
    ordered = sorted(
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
        "| Candidate | Phase | Score | Label | Ours wins | Best wins | ACC | NMI | ARI | F1 | Log |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in ordered:
        metrics = row.get("metrics", {})
        ours_gaps = row.get("ours_gaps", {})
        best_gaps = row.get("best_gaps", {})
        ours_wins = sum(1 for metric in METRICS if ours_gaps.get(metric, float("-inf")) > 0)
        best_wins = sum(1 for metric in METRICS if best_gaps.get(metric, float("-inf")) > 0)
        score = row.get("score")
        score_text = "-" if score is None else f"{score:.2f}"
        lines.append(
            f"| {row['candidate']} | {row.get('phase', '-')} | {score_text} | {row.get('label', '-')} | {ours_wins} | {best_wins} | "
            f"{metric_with_std(metrics, 'ACC')} | {metric_with_std(metrics, 'NMI')} | {metric_with_std(metrics, 'ARI')} | {metric_with_std(metrics, 'F1')} | "
            f"{row.get('log_path', '-')} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_fixed_ae(
    config: dict[str, Any],
    dataset: str,
    python_path: Path,
    device: str,
    timeout: int,
    force_ae: bool,
    dry_run: bool,
) -> tuple[bool, Path]:
    dataset_dir = OUTPUT_ROOT / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ae_cmd, graph_path = build_ae_command(config, dataset, python_path, device)
    ae_log = dataset_dir / "logs" / "fixed_ae_seed42_graph42.txt"
    if force_ae or not graph_path.exists():
        print(f"[{dataset}] AE prepare: {graph_path}", flush=True)
        if dry_run:
            print(" ".join(ae_cmd), flush=True)
            return True, graph_path
        returncode, elapsed, _timed_out, _output = run_subprocess(ae_cmd, ae_log, timeout)
        if returncode != 0:
            print(f"[{dataset}] AE failed. See {ae_log}", flush=True)
            return False, graph_path
        print(f"[{dataset}] AE done in {elapsed:.2f}s", flush=True)
    else:
        print(f"[{dataset}] Reusing fixed AE graph {graph_path}", flush=True)
    return True, graph_path


def run_dataset_campaign(
    dataset: str,
    config: dict[str, Any],
    python_path: Path,
    device: str,
    runs: int,
    seed_start: int,
    timeout: int,
    force_ae: bool,
    dry_run: bool,
    budget_hours: float,
    all_datasets: list[str],
) -> dict[str, Any]:
    start_time = time.time()
    dataset_dir = OUTPUT_ROOT / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = dataset_dir / "results.jsonl"
    results = load_existing_results(results_jsonl)
    manifest = {
        "dataset": dataset,
        "python": str(python_path),
        "device": device,
        "runs": runs,
        "seed_start": seed_start,
        "budget_hours": budget_hours,
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

    ok, graph_path = prepare_fixed_ae(config, dataset, python_path, device, timeout, force_ae, dry_run)
    if not ok:
        return {"dataset": dataset, "status": "ae_failed"}

    candidates = build_dataset_candidates(dataset)
    existing_names = {row.get("candidate") for row in results}
    budget_seconds = max(0.0, float(budget_hours) * 3600.0)

    for idx, candidate in enumerate(candidates, start=1):
        if budget_seconds > 0 and (time.time() - start_time) >= budget_seconds:
            print(f"[{dataset}] Budget reached after {idx-1} candidates.", flush=True)
            break
        if candidate.name in existing_names:
            continue

        print(f"[{dataset}] [{idx}/{len(candidates)}] {candidate.name}", flush=True)
        cmd = build_train_command(config, dataset, candidate.changed_params, python_path, device, runs, seed_start, graph_path)
        log_path = dataset_dir / "logs" / f"{candidate.name}.txt"
        if dry_run:
            print(" ".join(cmd), flush=True)
            continue

        returncode, elapsed, timed_out, output = run_subprocess(cmd, log_path, timeout)
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
            "phase": candidate.phase,
            "cmd": cmd,
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
        write_feedback_log(dataset_dir, results)
        write_dataset_summary(dataset_dir, dataset, results)
        write_global_summary(all_datasets)
        existing_names.add(candidate.name)

        if returncode != 0:
            print(f"[{dataset}] Candidate failed: {candidate.name}", flush=True)
            break

        if metrics:
            score_text = "nan" if score is None else f"{score:.2f}"
            print(
                f"[{dataset}] done {candidate.name} | ACC {metrics['ACC']['mean']:.2f} | F1 {metrics['F1']['mean']:.2f} | score {score_text}",
                flush=True,
            )

    write_feedback_log(dataset_dir, results)
    write_dataset_summary(dataset_dir, dataset, results)
    write_global_summary(all_datasets)
    best_row = None
    if results:
        best_row = max(
            results,
            key=lambda row: (
                float("-inf") if row.get("score") is None else float(row["score"]),
                row.get("timestamp", ""),
            ),
        )
    return {
        "dataset": dataset,
        "status": "completed",
        "num_results": len(results),
        "best_candidate": None if best_row is None else best_row.get("candidate"),
        "best_score": None if best_row is None else best_row.get("score"),
    }


def normalize_dataset_list(raw: str) -> list[str]:
    names = []
    for token in raw.split(","):
        name = token.strip().lower()
        if not name:
            continue
        aliases = {"citeseer": "cite", "reuters": "reut"}
        name = aliases.get(name, name)
        if name not in DATASET_ORDER:
            raise ValueError(f"Unsupported dataset: {token}")
        if name not in names:
            names.append(name)
    return names or list(DATASET_ORDER)


def main() -> int:
    args = parse_args()
    datasets = normalize_dataset_list(args.datasets)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config()

    write_global_summary(datasets)
    if args.dry_run:
        for dataset in datasets:
            run_dataset_campaign(
                dataset=dataset,
                config=config,
                python_path=args.python,
                device=args.device,
                runs=args.runs,
                seed_start=args.seed_start,
                timeout=args.timeout,
                force_ae=args.force_ae,
                dry_run=True,
                budget_hours=args.per_dataset_hours,
                all_datasets=datasets,
            )
        return 0

    max_workers = max(1, min(int(args.max_workers), 6, len(datasets)))
    print(f"[campaign] datasets={datasets} max_workers={max_workers} budget_hours={args.per_dataset_hours}", flush=True)

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                run_dataset_campaign,
                dataset,
                config,
                args.python,
                args.device,
                int(args.runs),
                int(args.seed_start),
                int(args.timeout),
                bool(args.force_ae),
                False,
                float(args.per_dataset_hours),
                datasets,
            ): dataset
            for dataset in datasets
        }
        for future in concurrent.futures.as_completed(future_map):
            dataset = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"dataset": dataset, "status": "exception", "error": repr(exc)}
                print(f"[campaign] {dataset} exception: {exc}", flush=True)
            results.append(result)
            write_global_summary(datasets)

    campaign_path = OUTPUT_ROOT / "campaign_results.json"
    campaign_path.write_text(json.dumps(results, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[campaign] wrote {campaign_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
