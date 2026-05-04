from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
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
OUTPUT_ROOT = ROOT / "experiment_output" / "feedback_safe_tuning_adaptive_best_ae"
DEFAULT_PYTHON = Path("/root/miniconda3/envs/SCGC_2/bin/python")
DATASET_ORDER = ("amap", "reut", "uat", "cora", "cite", "eat", "usps")
METRICS = ("ACC", "NMI", "ARI", "F1")
SEARCH_AXIS_ORDER = (
    "dcgl_neg_weight",
    "dcgl_neg_tau",
    "fusion_balance",
    "fusion_min_weight",
    "lambda_inst",
    "lambda_clu",
    "threshold",
    "branch_bias_cap",
    "warmup_epochs",
    "fusion_temp",
)
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

BASE_POINTS: dict[str, dict[str, Any]] = {
    "reut": {
        "fusion_temp": 1.6,
        "fusion_balance": 0.25,
        "lambda_inst": 0.08,
        "lambda_clu": 0.06,
        "warmup_epochs": 35,
        "fusion_min_weight": 0.15,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    },
    "uat": {
        "fusion_temp": 1.9,
        "fusion_balance": 0.35,
        "lambda_inst": 0.08,
        "lambda_clu": 0.07,
        "warmup_epochs": 35,
        "fusion_min_weight": 0.20,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    },
    "amap": {
        "fusion_temp": 1.25,
        "fusion_balance": 0.08,
        "lambda_inst": 0.07,
        "lambda_clu": 0.035,
        "warmup_epochs": 35,
        "fusion_min_weight": 0.05,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    },
    "usps": {
        "fusion_temp": 1.8,
        "fusion_balance": 0.35,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "warmup_epochs": 35,
        "fusion_min_weight": 0.20,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    },
    "eat": {
        "fusion_temp": 2.0,
        "fusion_balance": 0.35,
        "lambda_inst": 0.08,
        "lambda_clu": 0.08,
        "warmup_epochs": 35,
        "fusion_min_weight": 0.20,
        "threshold": 0.4,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    },
    "cora": {
        "fusion_temp": 1.3,
        "fusion_balance": 0.0,
        "lambda_inst": 0.03,
        "lambda_clu": 0.01,
        "warmup_epochs": 70,
        "fusion_min_weight": 0.0,
        "enable_branch_bias_fusion": True,
        "branch_bias_target": "raw",
        "branch_bias_cap": 0.10,
        "threshold": 0.4,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    },
    "cite": {
        "fusion_temp": 1.8,
        "fusion_balance": 0.15,
        "lambda_inst": 0.045,
        "lambda_clu": 0.02,
        "warmup_epochs": 55,
        "fusion_min_weight": 0.10,
        "enable_branch_bias_fusion": True,
        "branch_bias_target": "raw",
        "branch_bias_cap": 0.15,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    },
}

SAFE_AXES: dict[str, dict[str, list[Any]]] = {
    "reut": {
        "fusion_min_weight": [0.10, 0.15, 0.20],
        "dcgl_neg_tau": [0.5, 0.75, 1.0],
        "dcgl_neg_weight": [0.3, 0.4, 0.5, 0.6],
        "fusion_balance": [0.20, 0.25, 0.30],
        "lambda_inst": [0.06, 0.08, 0.10],
        "lambda_clu": [0.04, 0.06, 0.08],
    },
    "uat": {
        "fusion_temp": [1.8, 1.9, 2.0, 2.1],
        "fusion_balance": [0.30, 0.35, 0.40, 0.45],
        "fusion_min_weight": [0.18, 0.20, 0.22, 0.25],
        "lambda_clu": [0.06, 0.07, 0.075, 0.08],
        "dcgl_neg_tau": [0.5, 0.75, 1.0],
        "dcgl_neg_weight": [0.3, 0.4, 0.5, 0.6],
    },
    "amap": {
        "fusion_balance": [0.05, 0.08, 0.10],
        "lambda_inst": [0.0, 0.03, 0.07],
        "fusion_min_weight": [0.0, 0.05, 0.10],
        "dcgl_neg_tau": [0.5, 1.0],
        "dcgl_neg_weight": [0.6, 0.8, 1.0],
        "lambda_clu": [0.02, 0.035, 0.05],
    },
    "usps": {
        "fusion_temp": [1.8, 2.0, 2.2],
        "fusion_balance": [0.30, 0.35, 0.45],
        "fusion_min_weight": [0.15, 0.20, 0.25],
        "dcgl_neg_tau": [0.5, 0.75, 1.0],
        "dcgl_neg_weight": [0.3, 0.4, 0.5, 0.6, 0.8],
        "lambda_inst": [0.07, 0.09, 0.11],
        "lambda_clu": [0.07, 0.09, 0.11],
    },
    "eat": {
        "threshold": [0.35, 0.4, 0.45, 0.5],
        "fusion_temp": [1.8, 2.0, 2.2],
        "fusion_balance": [0.25, 0.35],
        "fusion_min_weight": [0.15, 0.20],
        "dcgl_neg_tau": [0.35, 0.5, 0.75],
        "dcgl_neg_weight": [0.3, 0.4, 0.5, 0.6, 0.8],
        "lambda_inst": [0.06, 0.08, 0.10],
        "lambda_clu": [0.06, 0.08, 0.10],
    },
    "cora": {
        "threshold": [0.35, 0.4],
        "branch_bias_cap": [0.08, 0.10, 0.12],
        "dcgl_neg_tau": [0.35, 0.5, 0.75],
        "dcgl_neg_weight": [0.3, 0.4, 0.6, 0.8, 1.0],
        "lambda_inst": [0.02, 0.03, 0.04],
        "lambda_clu": [0.005, 0.01, 0.02],
        "warmup_epochs": [55, 70, 85],
    },
    "cite": {
        "fusion_balance": [0.10, 0.15, 0.25, 0.35],
        "fusion_min_weight": [0.05, 0.10, 0.15, 0.20],
        "branch_bias_cap": [0.12, 0.15, 0.18],
        "dcgl_neg_tau": [0.35, 0.5, 0.75, 1.0],
        "dcgl_neg_weight": [0.3, 0.4, 0.6, 0.8, 1.0],
        "lambda_inst": [0.03, 0.045, 0.06],
        "lambda_clu": [0.01, 0.02, 0.03],
        "warmup_epochs": [45, 55, 65],
    },
}

PAIRINGS: dict[str, list[tuple[str, str]]] = {
    "reut": [("dcgl_neg_weight", "dcgl_neg_tau"), ("fusion_min_weight", "fusion_balance"), ("lambda_inst", "lambda_clu")],
    "uat": [("dcgl_neg_weight", "dcgl_neg_tau"), ("fusion_min_weight", "fusion_balance"), ("lambda_clu", "dcgl_neg_tau")],
    "amap": [("fusion_balance", "fusion_min_weight"), ("fusion_balance", "dcgl_neg_weight"), ("lambda_inst", "lambda_clu")],
    "usps": [("dcgl_neg_weight", "dcgl_neg_tau"), ("fusion_min_weight", "fusion_temp"), ("lambda_inst", "lambda_clu")],
    "eat": [("threshold", "dcgl_neg_weight"), ("fusion_temp", "dcgl_neg_weight"), ("lambda_inst", "lambda_clu")],
    "cora": [("threshold", "dcgl_neg_weight"), ("branch_bias_cap", "fusion_min_weight"), ("lambda_inst", "lambda_clu")],
    "cite": [("branch_bias_cap", "fusion_min_weight"), ("fusion_balance", "dcgl_neg_weight"), ("lambda_inst", "lambda_clu")],
}

AXIS_PRIORITY: dict[str, list[str]] = {
    "reut": ["dcgl_neg_weight", "dcgl_neg_tau", "fusion_min_weight", "fusion_balance", "lambda_inst", "lambda_clu"],
    "uat": ["dcgl_neg_weight", "dcgl_neg_tau", "fusion_balance", "fusion_temp", "fusion_min_weight", "lambda_clu"],
    "amap": ["fusion_balance", "fusion_min_weight", "dcgl_neg_weight", "lambda_inst", "dcgl_neg_tau", "lambda_clu"],
    "usps": ["dcgl_neg_weight", "dcgl_neg_tau", "fusion_temp", "fusion_min_weight", "fusion_balance", "lambda_inst", "lambda_clu"],
    "eat": ["dcgl_neg_weight", "dcgl_neg_tau", "threshold", "fusion_temp", "lambda_inst", "lambda_clu", "fusion_balance", "fusion_min_weight"],
    "cora": ["threshold", "branch_bias_cap", "dcgl_neg_weight", "dcgl_neg_tau", "lambda_inst", "lambda_clu", "warmup_epochs"],
    "cite": ["branch_bias_cap", "fusion_balance", "fusion_min_weight", "dcgl_neg_tau", "dcgl_neg_weight", "lambda_inst", "lambda_clu", "warmup_epochs"],
}


@dataclass
class Candidate:
    name: str
    base_id: str
    changed_params: dict[str, Any]
    note: str
    phase: str
    axes: tuple[str, ...]
    is_confirmation: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive safe feedback tuning with fixed best-AE assets.")
    parser.add_argument("--datasets", default=",".join(DATASET_ORDER), help="Comma-separated dataset order.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--per-dataset-hours", type=float, default=6.0)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Print the first adaptive candidate plan without running train.py.")
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


def dedupe_preserve(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def sort_axis_values(values: list[Any]) -> list[Any]:
    if not values:
        return values
    if all(isinstance(value, (int, float)) for value in values):
        return sorted(values, key=float)
    return values


def hydrate_search_space_from_config(config: dict[str, Any]) -> None:
    for dataset in DATASET_ORDER:
        profile = config.get("dataset_profiles", {}).get(dataset, {})
        if not profile:
            continue

        merged_train_args = merge_args(config.get("train_common_args", {}), profile.get("train_args", {}))
        merged_dual_attn_args = merge_args(config.get("dual_attn_args", {}), profile.get("dual_attn_args", {}))
        merged_dcgl_negative_args = merge_args(config.get("dcgl_negative_args", {}), profile.get("dcgl_negative_args", {}))

        base_params: dict[str, Any] = {}
        for axis in SEARCH_AXIS_ORDER:
            if axis in merged_dcgl_negative_args:
                base_params[axis] = merged_dcgl_negative_args[axis]
            elif axis in merged_dual_attn_args:
                base_params[axis] = merged_dual_attn_args[axis]
            elif axis in merged_train_args:
                base_params[axis] = merged_train_args[axis]

        if "branch_bias_cap" in merged_dual_attn_args:
            if merged_dual_attn_args.get("enable_branch_bias_fusion"):
                base_params["enable_branch_bias_fusion"] = True
            if "branch_bias_target" in merged_dual_attn_args:
                base_params["branch_bias_target"] = merged_dual_attn_args["branch_bias_target"]

        if base_params:
            BASE_POINTS[dataset] = base_params

        safe_grid = profile.get("safe_tuning_grid", {})
        safe_axes: dict[str, list[Any]] = {}
        for section_name in ("dcgl_negative_args", "dual_attn_args", "train_args"):
            section = safe_grid.get(section_name, {})
            if not isinstance(section, dict):
                continue
            for axis, raw_values in section.items():
                if not isinstance(raw_values, list):
                    continue
                values = dedupe_preserve(list(raw_values))
                base_value = base_params.get(axis)
                if base_value is not None and all(value != base_value for value in values):
                    values.append(base_value)
                safe_axes[axis] = sort_axis_values(dedupe_preserve(values))

        if safe_axes:
            SAFE_AXES[dataset] = safe_axes
            AXIS_PRIORITY[dataset] = [axis for axis in SEARCH_AXIS_ORDER if axis in safe_axes] + [
                axis for axis in safe_axes if axis not in SEARCH_AXIS_ORDER
            ]


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


def decision_label(ours_gaps: dict[str, float], best_gaps: dict[str, float], metrics: dict[str, dict[str, float]], base_metrics: dict[str, dict[str, float]] | None) -> str:
    ours_wins = sum(1 for metric in METRICS if ours_gaps.get(metric, float("-inf")) > 0)
    best_wins = sum(1 for metric in METRICS if best_gaps.get(metric, float("-inf")) > 0)
    if best_wins == 4:
        return "HEADLINE_ACCEPT"
    if ours_wins == 4:
        return "STRONG_ACCEPT"
    if ours_wins >= 3:
        return "ACCEPT"
    if base_metrics:
        base_score = score_metrics(base_metrics)
        score = score_metrics(metrics)
        if score is not None and base_score is not None and score >= base_score - 0.05:
            base_std = base_metrics["ACC"]["std"] + base_metrics["F1"]["std"]
            cur_std = metrics["ACC"]["std"] + metrics["F1"]["std"]
            if cur_std + 0.2 < base_std:
                return "STABILITY_KEEP"
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
        profile.get("dcgl_negative_args", {}),
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


def metric_text(metrics: dict[str, dict[str, float]], metric: str) -> str:
    value = metrics.get(metric)
    if not value:
        return "-"
    return f"{value['mean']:.2f}"


def std_text(metrics: dict[str, dict[str, float]]) -> str:
    if not metrics:
        return "-"
    return "/".join(f"{metrics[m]['std']:.2f}" for m in METRICS if m in metrics)


def metric_with_std(metrics: dict[str, dict[str, float]], metric: str) -> str:
    value = metrics.get(metric)
    if not value:
        return "-"
    return f"{value['mean']:.2f}±{value['std']:.2f}"


def write_feedback_log(dataset_dir: Path, results: list[dict[str, Any]]) -> None:
    path = dataset_dir / "feedback_log.md"
    ordered = sorted(results, key=lambda row: row.get("timestamp", ""))
    lines = [
        "# Adaptive Feedback Log",
        "",
        "| Step | Candidate | Base | Changed | Phase | ACC | NMI | ARI | F1 | Std ACC/NMI/ARI/F1 | Decision | Note | Log |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(ordered, start=1):
        metrics = row.get("metrics", {})
        changed = row.get("changed_params") or {}
        changed_text = ", ".join(f"{k}={v}" for k, v in changed.items()) if changed else "base"
        lines.append(
            f"| {idx} | {row.get('candidate', '-')} | {row.get('base_id', '-')} | {changed_text} | {row.get('phase', '-')} | "
            f"{metric_text(metrics, 'ACC')} | {metric_text(metrics, 'NMI')} | {metric_text(metrics, 'ARI')} | {metric_text(metrics, 'F1')} | "
            f"{std_text(metrics)} | {row.get('label', '-')} | {row.get('note', '-')} | {row.get('log_path', '-')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_row_for_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    accepted = [row for row in rows if row.get("label") in {"HEADLINE_ACCEPT", "STRONG_ACCEPT", "ACCEPT", "STABILITY_KEEP"}]
    pool = accepted if accepted else rows
    return max(
        pool,
        key=lambda row: (
            label_rank(str(row.get("label", ""))),
            float("-inf") if row.get("score") is None else float(row["score"]),
            row.get("timestamp", ""),
        ),
    )


def label_rank(label: str) -> int:
    order = {
        "HEADLINE_ACCEPT": 6,
        "STRONG_ACCEPT": 5,
        "ACCEPT": 4,
        "STABILITY_KEEP": 3,
        "SIDE_NOTE": 2,
        "REJECT": 1,
        "FAILED": 0,
    }
    return order.get(label, -1)


def write_global_summary(output_root: Path, dataset_names: list[str]) -> None:
    path = output_root / "summary.md"
    lines = [
        "# Adaptive Safe Feedback Tuning Campaign",
        "",
        f"- Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Dataset | Best Candidate | Score | Label | ACC | NMI | ARI | F1 | Feedback Log | Summary |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for dataset in dataset_names:
        results = load_existing_results(output_root / dataset / "results.jsonl")
        best = best_row_for_summary(results)
        if not best:
            lines.append(f"| {dataset} | - | - | - | - | - | - | - | - | - |")
            continue
        metrics = best.get("metrics", {})
        score = best.get("score")
        score_text = "-" if score is None else f"{score:.2f}"
        lines.append(
            f"| {dataset} | {best.get('candidate', '-')} | {score_text} | {best.get('label', '-')} | "
            f"{metric_text(metrics, 'ACC')} | {metric_text(metrics, 'NMI')} | {metric_text(metrics, 'ARI')} | {metric_text(metrics, 'F1')} | "
            f"{output_root.relative_to(ROOT)}/{dataset}/feedback_log.md | {output_root.relative_to(ROOT)}/{dataset}/summary.md |"
        )
    with SUMMARY_LOCK:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_summary(output_root: Path, dataset_dir: Path, dataset: str, results: list[dict[str, Any]], current_base_name: str) -> None:
    summary_path = dataset_dir / "summary.md"
    ordered = sorted(
        results,
        key=lambda row: (
            label_rank(str(row.get("label", ""))),
            float("-inf") if row.get("score") is None else float(row["score"]),
            row.get("timestamp", ""),
        ),
        reverse=True,
    )
    lines = [
        "# Adaptive Safe Feedback Tuning",
        "",
        f"- Dataset: {dataset}",
        f"- Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Current Ours target: {CURRENT_OURS_TARGET[dataset]}",
        f"- Main-table best target: {MAIN_TABLE_BEST[dataset]}",
        f"- Current adaptive base: {current_base_name}",
        "",
        "| Candidate | Phase | Score | Label | Ours wins | Best wins | Confirmed | ACC | NMI | ARI | F1 | Log |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ordered:
        metrics = row.get("metrics", {})
        ours_gaps = row.get("ours_gaps", {})
        best_gaps = row.get("best_gaps", {})
        ours_wins = sum(1 for metric in METRICS if ours_gaps.get(metric, float("-inf")) > 0)
        best_wins = sum(1 for metric in METRICS if best_gaps.get(metric, float("-inf")) > 0)
        score = row.get("score")
        score_text = "-" if score is None else f"{score:.2f}"
        confirmed = "yes" if row.get("is_confirmation") else "no"
        lines.append(
            f"| {row['candidate']} | {row.get('phase', '-')} | {score_text} | {row.get('label', '-')} | {ours_wins} | {best_wins} | {confirmed} | "
            f"{metric_with_std(metrics, 'ACC')} | {metric_with_std(metrics, 'NMI')} | {metric_with_std(metrics, 'ARI')} | {metric_with_std(metrics, 'F1')} | "
            f"{row.get('log_path', '-')} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_global_summary(output_root, list(DATASET_ORDER))


def stable_improvement(base_metrics: dict[str, dict[str, float]] | None, metrics: dict[str, dict[str, float]]) -> bool:
    if not base_metrics:
        return False
    base_score = score_metrics(base_metrics)
    cur_score = score_metrics(metrics)
    if base_score is None or cur_score is None:
        return False
    if cur_score <= base_score + 0.02:
        return False
    wins = 0
    for metric in METRICS:
        if metrics[metric]["mean"] > base_metrics[metric]["mean"]:
            wins += 1
    cur_std = metrics["ACC"]["std"] + metrics["F1"]["std"]
    base_std = base_metrics["ACC"]["std"] + base_metrics["F1"]["std"]
    return wins >= 2 and cur_std <= base_std + 0.8


def better_axis_value(base_metrics: dict[str, dict[str, float]], candidate_metrics: dict[str, dict[str, float]], dataset: str) -> bool:
    base_score = score_metrics(base_metrics)
    cur_score = score_metrics(candidate_metrics)
    if base_score is None or cur_score is None:
        return False
    margin = 0.02
    if dataset == "eat":
        base_proxy = base_metrics["ACC"]["mean"] + 0.4 * base_metrics["F1"]["mean"] - 0.5 * (base_metrics["ACC"]["std"] + base_metrics["F1"]["std"])
        cur_proxy = candidate_metrics["ACC"]["mean"] + 0.4 * candidate_metrics["F1"]["mean"] - 0.5 * (candidate_metrics["ACC"]["std"] + candidate_metrics["F1"]["std"])
        return cur_proxy > base_proxy + 0.02
    if cur_score > base_score + margin:
        return True
    cur_std = candidate_metrics["ACC"]["std"] + candidate_metrics["F1"]["std"]
    base_std = base_metrics["ACC"]["std"] + base_metrics["F1"]["std"]
    return cur_score >= base_score - 0.02 and cur_std + 0.2 < base_std


def candidate_name(prefix: str, changed: dict[str, Any]) -> str:
    if not changed:
        return prefix
    parts = []
    for key, value in changed.items():
        text = str(value).replace(".", "p").replace("-", "m")
        parts.append(f"{key}_{text}")
    return f"{prefix}_{'__'.join(parts)}"


def build_axis_candidates(dataset: str, base_id: str, base_params: dict[str, Any], tested_names: set[str]) -> list[Candidate]:
    per_axis: list[list[Candidate]] = []
    for axis in AXIS_PRIORITY[dataset]:
        values = SAFE_AXES[dataset].get(axis, [])
        base_value = base_params.get(axis)
        axis_candidates: list[Candidate] = []
        for value in values:
            if value == base_value:
                continue
            changed = {axis: value}
            name = candidate_name("probe", changed)
            if name in tested_names:
                continue
            axis_candidates.append(
                Candidate(
                    name=name,
                    base_id=base_id,
                    changed_params=changed,
                    note=f"One-parameter probe on {axis} from adaptive base.",
                    phase="phase1",
                    axes=(axis,),
                )
            )
        if axis_candidates:
            per_axis.append(axis_candidates)

    candidates: list[Candidate] = []
    max_len = max((len(items) for items in per_axis), default=0)
    for idx in range(max_len):
        for items in per_axis:
            if idx < len(items):
                candidates.append(items[idx])
    return candidates


def build_directional_candidates(dataset: str, base_id: str, base_params: dict[str, Any], winning_axes: dict[str, Any], tested_names: set[str]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for axis in AXIS_PRIORITY[dataset]:
        if axis not in winning_axes:
            continue
        values = SAFE_AXES[dataset].get(axis, [])
        current_value = base_params.get(axis)
        if current_value not in values:
            continue
        idx = values.index(current_value)
        neighbors = []
        if idx > 0:
            neighbors.append(values[idx - 1])
        if idx + 1 < len(values):
            neighbors.append(values[idx + 1])
        for value in neighbors:
            if value == winning_axes[axis]:
                continue
            changed = {axis: value}
            name = candidate_name("extend", changed)
            if name in tested_names:
                continue
            candidates.append(
                Candidate(
                    name=name,
                    base_id=base_id,
                    changed_params=changed,
                    note=f"Directional extension on {axis} after positive probe.",
                    phase="phase2_extend",
                    axes=(axis,),
                )
            )
    return candidates


def build_pairwise_candidates(dataset: str, base_id: str, base_params: dict[str, Any], winning_axes: dict[str, Any], tested_names: set[str]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for axis_a, axis_b in PAIRINGS.get(dataset, []):
        if axis_a not in winning_axes or axis_b not in winning_axes:
            continue
        changed = {
            axis_a: winning_axes[axis_a],
            axis_b: winning_axes[axis_b],
        }
        name = candidate_name("pair", changed)
        if name in tested_names:
            continue
        candidates.append(
            Candidate(
                name=name,
                base_id=base_id,
                changed_params=changed,
                note=f"Pairwise coupling on {axis_a} x {axis_b}.",
                phase="phase3_pair",
                axes=(axis_a, axis_b),
            )
        )
    return candidates


def build_confirmation_candidate(best_name: str, base_id: str, params: dict[str, Any], tested_names: set[str]) -> Candidate | None:
    name = f"confirm_{best_name}"
    if name in tested_names:
        return None
    return Candidate(
        name=name,
        base_id=base_id,
        changed_params=dict(params),
        note=f"Exact 10-run confirmation for {best_name}.",
        phase="phase4_confirm",
        axes=tuple(sorted(params.keys())),
        is_confirmation=True,
    )


def run_candidate(
    output_root: Path,
    dataset: str,
    dataset_dir: Path,
    config: dict[str, Any],
    graph_path: Path,
    python_path: Path,
    device: str,
    runs: int,
    seed_start: int,
    timeout: int,
    candidate: Candidate,
    results_jsonl: Path,
    results: list[dict[str, Any]],
    base_metrics: dict[str, dict[str, float]] | None,
    dry_run: bool,
) -> dict[str, Any]:
    cmd = build_train_command(config, dataset, candidate.changed_params, python_path, device, runs, seed_start, graph_path)
    log_path = dataset_dir / "logs" / f"{candidate.name}.txt"
    if dry_run:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": dataset,
            "candidate": candidate.name,
            "base_id": candidate.base_id,
            "changed_params": candidate.changed_params,
            "note": candidate.note,
            "phase": candidate.phase,
            "axes": list(candidate.axes),
            "is_confirmation": candidate.is_confirmation,
            "cmd": cmd,
            "returncode": 0,
            "elapsed_sec": 0.0,
            "timed_out": False,
            "metrics": {},
            "score": None,
            "ours_gaps": {},
            "best_gaps": {},
            "label": "PLANNED",
            "log_path": str(log_path.relative_to(ROOT)),
            "ae_graph_path": str(graph_path.relative_to(ROOT)),
        }
        return row
    returncode, elapsed, timed_out, output = run_subprocess(cmd, log_path, timeout)
    metrics = parse_metrics(output)
    ours_gaps = metric_gaps(metrics, CURRENT_OURS_TARGET[dataset]) if metrics else {}
    best_gaps = metric_gaps(metrics, MAIN_TABLE_BEST[dataset]) if metrics else {}
    score = score_metrics(metrics)
    label = "FAILED" if returncode != 0 else decision_label(ours_gaps, best_gaps, metrics, base_metrics)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "candidate": candidate.name,
        "base_id": candidate.base_id,
        "changed_params": candidate.changed_params,
        "note": candidate.note,
        "phase": candidate.phase,
        "axes": list(candidate.axes),
        "is_confirmation": candidate.is_confirmation,
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
    return row


def resolve_fixed_assets(dataset: str) -> tuple[Path, Path]:
    graph_path = ROOT / "data" / "ae_graph" / f"{dataset}_ae_graph.txt"
    model_path = ROOT / "pretrain_graph" / f"{dataset}_ae_pretrain.pkl"
    return graph_path, model_path


def run_dataset_campaign(
    dataset: str,
    config: dict[str, Any],
    python_path: Path,
    device: str,
    runs: int,
    seed_start: int,
    timeout: int,
    budget_hours: float,
    output_root: Path,
    all_datasets: list[str],
    dry_run: bool,
    plan_only: bool,
) -> dict[str, Any]:
    start_time = time.time()
    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = dataset_dir / "results.jsonl"
    results: list[dict[str, Any]] = []
    graph_path, model_path = resolve_fixed_assets(dataset)
    if not graph_path.exists():
        return {"dataset": dataset, "status": "missing_ae_graph", "graph_path": str(graph_path)}
    if not model_path.exists():
        return {"dataset": dataset, "status": "missing_ae_model", "model_path": str(model_path)}

    manifest = {
        "dataset": dataset,
        "python": str(python_path),
        "device": device,
        "runs": runs,
        "seed_start": seed_start,
        "budget_hours": budget_hours,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root.relative_to(ROOT)),
        "fixed_ae_graph": str(graph_path.relative_to(ROOT)),
        "fixed_ae_model": str(model_path.relative_to(ROOT)),
        "hard_contract": {
            "graph_mode": "dual",
            "fusion_mode": "attn",
            "enable_dcgl_negative_loss": True,
            "disable_dynamic_threshold": True,
            "disable_ema_prototypes": True,
            "disable_dcgl_cluster_level": True,
            "disable_gcn_backbone": True,
        },
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    current_base_name = "base_current"
    current_base_params = dict(BASE_POINTS[dataset])
    tested_names: set[str] = set()
    winning_axes: dict[str, Any] = {}
    budget_seconds = max(0.0, float(budget_hours) * 3600.0)

    def within_budget() -> bool:
        return (time.time() - start_time) < budget_seconds if budget_seconds > 0 else True

    base_candidate = Candidate(
        name="base_current",
        base_id="base_current",
        changed_params=dict(current_base_params),
        note="Best-AE 10-run base recheck.",
        phase="phase0_base",
        axes=tuple(sorted(current_base_params.keys())),
    )

    base_row = run_candidate(
        output_root,
        dataset,
        dataset_dir,
        config,
        graph_path,
        python_path,
        device,
        runs,
        seed_start,
        timeout,
        base_candidate,
        results_jsonl,
        results,
        None,
        dry_run,
    )
    tested_names.add(base_candidate.name)
    if dry_run:
        return {
            "dataset": dataset,
            "status": "planned",
            "base_candidate": base_row,
        }

    base_metrics = base_row.get("metrics", {})
    write_feedback_log(dataset_dir, results)
    write_dataset_summary(output_root, dataset_dir, dataset, results, current_base_name)

    rounds = 0
    while within_budget() and rounds < 6:
        rounds += 1
        round_rows: list[dict[str, Any]] = []
        axis_candidates = build_axis_candidates(dataset, current_base_name, current_base_params, tested_names)
        if rounds >= 2:
            axis_candidates.extend(build_directional_candidates(dataset, current_base_name, current_base_params, winning_axes, tested_names))
        if rounds >= 3:
            axis_candidates.extend(build_pairwise_candidates(dataset, current_base_name, current_base_params, winning_axes, tested_names))

        if not axis_candidates:
            break

        if rounds == 1:
            round_limit = min(len(axis_candidates), max(12, len(AXIS_PRIORITY[dataset])))
        elif rounds == 2:
            round_limit = min(len(axis_candidates), 10)
        else:
            round_limit = min(len(axis_candidates), 8)
        for candidate in axis_candidates[:round_limit]:
            if not within_budget():
                break
            row = run_candidate(
                output_root,
                dataset,
                dataset_dir,
                config,
                graph_path,
                python_path,
                device,
                runs,
                seed_start,
                timeout,
                candidate,
                results_jsonl,
                results,
                base_metrics,
                dry_run,
            )
            tested_names.add(candidate.name)
            round_rows.append(row)
            write_feedback_log(dataset_dir, results)
            write_dataset_summary(output_root, dataset_dir, dataset, results, current_base_name)
            if row["returncode"] != 0:
                return {"dataset": dataset, "status": "candidate_failed", "candidate": candidate.name}

        if not round_rows:
            break

        improved_rows = [row for row in round_rows if row.get("metrics") and better_axis_value(base_metrics, row["metrics"], dataset)]
        if improved_rows:
            best_probe = max(
                improved_rows,
                key=lambda row: (
                    label_rank(str(row.get("label", ""))),
                    float("-inf") if row.get("score") is None else float(row["score"]),
                ),
            )
            confirm = build_confirmation_candidate(best_probe["candidate"], current_base_name, best_probe["changed_params"], tested_names)
            if confirm and within_budget():
                confirm_row = run_candidate(
                    output_root,
                    dataset,
                    dataset_dir,
                    config,
                    graph_path,
                    python_path,
                    device,
                    runs,
                    seed_start,
                    timeout,
                    confirm,
                    results_jsonl,
                    results,
                    base_metrics,
                    dry_run,
                )
                tested_names.add(confirm.name)
                write_feedback_log(dataset_dir, results)
                write_dataset_summary(output_root, dataset_dir, dataset, results, current_base_name)
                if confirm_row["returncode"] != 0:
                    return {"dataset": dataset, "status": "confirmation_failed", "candidate": confirm.name}
                if stable_improvement(base_metrics, confirm_row["metrics"]):
                    current_base_name = confirm_row["candidate"]
                    current_base_params.update(confirm_row["changed_params"])
                    base_metrics = confirm_row["metrics"]
                    for axis in confirm.axes:
                        if axis in confirm.changed_params:
                            winning_axes[axis] = confirm.changed_params[axis]
                else:
                    for axis in best_probe.get("changed_params", {}):
                        winning_axes[axis] = best_probe["changed_params"][axis]

    write_feedback_log(dataset_dir, results)
    write_dataset_summary(output_root, dataset_dir, dataset, results, current_base_name)
    best_row = best_row_for_summary(results)
    return {
        "dataset": dataset,
        "status": "completed",
        "num_results": len(results),
        "best_candidate": None if best_row is None else best_row.get("candidate"),
        "best_score": None if best_row is None else best_row.get("score"),
        "adaptive_base": current_base_name,
    }


def main() -> int:
    args = parse_args()
    datasets = normalize_dataset_list(args.datasets)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config()
    hydrate_search_space_from_config(config)
    write_global_summary(output_root, datasets)

    if args.dry_run or args.plan_only:
        for dataset in datasets:
            result = run_dataset_campaign(
                dataset=dataset,
                config=config,
                python_path=args.python,
                device=args.device,
                runs=args.runs,
                seed_start=args.seed_start,
                timeout=args.timeout,
                budget_hours=args.per_dataset_hours,
                output_root=output_root,
                all_datasets=datasets,
                dry_run=True,
                plan_only=bool(args.plan_only),
            )
            print(json.dumps(result, ensure_ascii=True))
        return 0

    max_workers = max(1, min(int(args.max_workers), 6, len(datasets)))
    print(f"[adaptive-campaign] output={output_root.relative_to(ROOT)} datasets={datasets} max_workers={max_workers} budget_hours={args.per_dataset_hours}", flush=True)
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
                float(args.per_dataset_hours),
                output_root,
                datasets,
                False,
                False,
            ): dataset
            for dataset in datasets
        }
        for future in concurrent.futures.as_completed(future_map):
            dataset = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"dataset": dataset, "status": "exception", "error": repr(exc)}
                print(f"[adaptive-campaign] {dataset} exception: {exc}", flush=True)
            results.append(result)
            write_global_summary(output_root, datasets)

    campaign_path = output_root / "campaign_results.json"
    campaign_path.write_text(json.dumps(results, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[adaptive-campaign] wrote {campaign_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
