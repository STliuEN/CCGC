from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "fusion_mechanism_iteration"
METRICS = ("ACC", "NMI", "ARI", "F1")


@dataclass(frozen=True)
class Candidate:
    dataset: str
    name: str
    params: dict[str, Any]
    note: str


COUNTERGRAD_BASE = {
    "enable_fusion_reliability_feedback_loss": True,
    "adaptive_bias_mode": "cap",
    "fusion_feedback_loss_type": "countergrad",
    "fusion_feedback_weight": 1.0,
    "fusion_feedback_temp": 0.5,
    "fusion_feedback_prior_blend": 0.85,
    "fusion_feedback_countergrad_curvature": 1.0,
    "fusion_feedback_countergrad_base": 0.05,
    "fusion_feedback_warmup_ramp_epochs": 20,
    "fusion_feedback_min_weak_weight": 0.02,
}

SELECTIVE_COUNTERGRAD_BASE = {
    **COUNTERGRAD_BASE,
    "fusion_feedback_loss_type": "selective_countergrad",
}


GRID = {
    "cora": [
        Candidate("cora", "ps14_curv10_base005", {"fusion_feedback_prior_strength": 1.4}, "current best local countergrad center"),
        Candidate("cora", "ps12_curv10_base005", {"fusion_feedback_prior_strength": 1.2}, "weaker structure-prior target"),
        Candidate("cora", "ps14_curv075_base005", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_curvature": 0.75}, "weaker counter curvature"),
        Candidate("cora", "ps14_curv125_base005", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_curvature": 1.25}, "stronger counter curvature"),
        Candidate("cora", "ps14_curv10_base002", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.02}, "lower always-on reliability anchor"),
        Candidate("cora", "ps14_curv10_base010", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.10}, "higher always-on reliability anchor"),
        Candidate("cora", "ps14_curv10_base008", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.08}, "interpolate the Cora base anchor below the current best"),
        Candidate("cora", "ps14_curv10_base012", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.12}, "interpolate the Cora base anchor above the current best"),
        Candidate("cora", "ps14_curv10_base015", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.15}, "test a stronger always-on anchor for seed stability"),
        Candidate("cora", "ps13_curv10_base010", {"fusion_feedback_prior_strength": 1.3, "fusion_feedback_countergrad_base": 0.10}, "slightly softer prior target with the better anchor"),
        Candidate("cora", "ps15_curv10_base010", {"fusion_feedback_prior_strength": 1.5, "fusion_feedback_countergrad_base": 0.10}, "slightly stronger prior target with the better anchor"),
        Candidate("cora", "ps14_curv075_base010", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_curvature": 0.75, "fusion_feedback_countergrad_base": 0.10}, "weaker curvature with the better anchor"),
        Candidate("cora", "ps14_curv125_base010", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_curvature": 1.25, "fusion_feedback_countergrad_base": 0.10}, "stronger curvature with the better anchor"),
        Candidate("cora", "ps14_curv10_base010_w075", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.10, "fusion_feedback_weight": 0.75}, "lower feedback weight to reduce over-constraint"),
        Candidate("cora", "ps14_curv10_base010_w125", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.10, "fusion_feedback_weight": 1.25}, "higher feedback weight to test stability"),
    ],
    "cite": [
        Candidate("cite", "ps14_curv10_base005", {"fusion_feedback_prior_strength": 1.4}, "current best local countergrad center"),
        Candidate("cite", "ps12_curv10_base005", {"fusion_feedback_prior_strength": 1.2}, "weaker structure-prior target"),
        Candidate("cite", "ps16_curv10_base005", {"fusion_feedback_prior_strength": 1.6}, "stronger structure-prior target"),
        Candidate("cite", "ps14_curv075_base005", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_curvature": 0.75}, "weaker counter curvature"),
        Candidate("cite", "ps14_curv125_base005", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_curvature": 1.25}, "stronger counter curvature"),
    ],
}

SELECTIVE_GRID = {
    "cora": [
        Candidate("cora", "sel_oldtemp_gateon_ps14_curv10_base000", {"fusion_temp": 1.3, "disable_dcgl_neg_reliability_gate": False, "fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.0}, "selective feedback at the old high-score Cora temperature and DCGL gate state"),
        Candidate("cora", "sel_oldtemp_gateon_ps14_curv10_base005", {"fusion_temp": 1.3, "disable_dcgl_neg_reliability_gate": False, "fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.05}, "selective feedback at the old Cora center with weak anchor"),
        Candidate("cora", "sel_oldtemp_gateon_ps14_curv075_base005", {"fusion_temp": 1.3, "disable_dcgl_neg_reliability_gate": False, "fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_curvature": 0.75, "fusion_feedback_countergrad_base": 0.05}, "softer selective feedback at the old Cora center"),
        Candidate("cora", "sel_ps14_curv10_base000", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.0}, "selective feedback, no fixed target anchor"),
        Candidate("cora", "sel_ps14_curv10_base002", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.02}, "selective feedback, very weak target anchor"),
        Candidate("cora", "sel_ps14_curv10_base005", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.05}, "selective feedback, weak target anchor"),
        Candidate("cora", "sel_ps14_curv075_base002", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_curvature": 0.75, "fusion_feedback_countergrad_base": 0.02}, "selective feedback with softer boundary curvature"),
        Candidate("cora", "sel_ps14_curv125_base002", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_curvature": 1.25, "fusion_feedback_countergrad_base": 0.02}, "selective feedback with stronger boundary curvature"),
    ],
    "cite": [
        Candidate("cite", "sel_ps14_curv10_base000", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.0}, "selective feedback, no fixed target anchor"),
        Candidate("cite", "sel_ps14_curv10_base002", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.02}, "selective feedback, very weak target anchor"),
        Candidate("cite", "sel_ps14_curv10_base005", {"fusion_feedback_prior_strength": 1.4, "fusion_feedback_countergrad_base": 0.05}, "selective feedback, weak target anchor"),
        Candidate("cite", "sel_ps16_curv10_base002", {"fusion_feedback_prior_strength": 1.6, "fusion_feedback_countergrad_base": 0.02}, "selective feedback with stronger structure evidence"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe loss-level countergrad feedback for attention fusion.")
    parser.add_argument("--dataset", default="all", choices=("all", "cora", "cite"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--candidate", default="", help="Comma-separated candidate names. Empty means all.")
    parser.add_argument("--mode", default="countergrad", choices=("countergrad", "selective"),
                        help="candidate family to evaluate")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("ccgc_experiment_config", ROOT / "experiment.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, dict):
        raise RuntimeError("experiment.py does not expose CONFIG")
    return config


def merge_args(*items: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if item:
            merged.update(item)
    return merged


def dict_to_cli(values: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        out.extend([flag, str(value)])
    return out


def apply_candidate_overrides(target: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(target)
    for key, value in overrides.items():
        if value is False and key in merged:
            merged.pop(key, None)
        elif key in merged:
            merged[key] = value
    return merged


def build_cmd(config: dict[str, Any], candidate: Candidate, python_path: Path, runs: int, seed_start: int) -> list[str]:
    dataset = candidate.dataset
    profile = config["dataset_profiles"][dataset]
    cluster_num = int(profile["cluster_num"])
    train_args = merge_args(config.get("train_common_args", {}), profile.get("train_args", {}))
    dual_args = merge_args(config.get("dual_args", {}), config.get("dual_attn_args", {}), profile.get("dual_attn_args", {}))
    improved_args: dict[str, Any] = {}
    if config.get("enable_dcgl_negative_module", False):
        improved_args["enable_dcgl_negative_loss"] = True
        improved_args.update(config.get("dcgl_negative_args", {}))
        improved_args.update(profile.get("dcgl_negative_args", {}))
    candidate_params = dict(candidate.params)
    feedback_base = SELECTIVE_COUNTERGRAD_BASE if candidate.name.startswith("sel_") else COUNTERGRAD_BASE
    feedback_keys = set(feedback_base) | {"fusion_feedback_prior_strength"}
    feedback_overrides = {
        key: candidate_params.pop(key)
        for key in list(candidate_params)
        if key in feedback_keys
    }
    train_args = apply_candidate_overrides(train_args, candidate_params)
    dual_args = apply_candidate_overrides(dual_args, candidate_params)
    improved_args = apply_candidate_overrides(improved_args, candidate_params)
    feedback_args = merge_args(feedback_base, feedback_overrides)
    ae_graph = ROOT / "data" / "ae_graph" / f"{dataset}_ae_graph.txt"
    cmd = [
        str(python_path),
        "train.py",
        "--dataset",
        dataset,
        "--cluster_num",
        str(cluster_num),
        "--graph_mode",
        "dual",
        "--ae_graph_path",
        str(ae_graph),
        "--fusion_mode",
        "attn",
        "--runs",
        str(runs),
        "--seed_start",
        str(seed_start),
    ]
    cmd += dict_to_cli(config.get("baseline_args", {}))
    cmd += dict_to_cli(train_args)
    cmd += dict_to_cli(dual_args)
    cmd += dict_to_cli(improved_args)
    cmd += dict_to_cli(feedback_args)
    return cmd


def parse_metrics(text: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    pattern = re.compile(r"^(ACC|NMI|ARI|F1)\s+\|\s+([+-]?\d+(?:\.\d+)?)\s+.+?\s+([+-]?\d+(?:\.\d+)?)$", re.MULTILINE)
    for metric, mean, std in pattern.findall(text):
        metrics[metric] = {"mean": float(mean), "std": float(std)}
    return metrics


def parse_fusion_paths(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.startswith("FusionPath ")]


def parse_prior(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("ADAPTIVE_STRUCTURE_PRIOR"):
            return line.strip()
    return "-"


def score(metrics: dict[str, dict[str, float]]) -> float:
    if not all(metric in metrics for metric in METRICS):
        return float("-inf")
    return (
        metrics["ACC"]["mean"]
        + 0.4 * metrics["F1"]["mean"]
        + 0.2 * metrics["NMI"]["mean"]
        + 0.2 * metrics["ARI"]["mean"]
        - 0.25 * (metrics["ACC"]["std"] + metrics["F1"]["std"])
    )


def run_cmd(cmd: list[str], log_path: Path, timeout: int) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        output += f"\n[TIMEOUT] exceeded {timeout} seconds\n"
        rc = 124
    log_path.write_text("COMMAND: " + " ".join(cmd) + "\n" + output, encoding="utf-8", errors="replace")
    return rc, output


def fmt_metric(metrics: dict[str, dict[str, float]], metric: str) -> str:
    item = metrics.get(metric)
    if not item:
        return "-"
    return f"{item['mean']:.2f}+-{item['std']:.2f}"


def main() -> None:
    args = parse_args()
    config = load_config()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (OUTPUT_ROOT / f"{stamp}_countergrad_probe")
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    datasets = ("cora", "cite") if args.dataset == "all" else (args.dataset,)
    wanted = {name.strip() for name in args.candidate.split(",") if name.strip()}
    grid = SELECTIVE_GRID if args.mode == "selective" else GRID
    candidates = [
        candidate
        for dataset in datasets
        for candidate in grid[dataset]
        if not wanted or candidate.name in wanted
    ]
    rows = []
    for idx, candidate in enumerate(candidates, start=1):
        cmd = build_cmd(config, candidate, args.python, args.runs, args.seed_start)
        log_path = run_dir / candidate.dataset / f"{idx:02d}_{candidate.name}.log"
        print(f"[{idx}/{len(candidates)}] {candidate.dataset}/{candidate.name}", flush=True)
        if args.dry_run:
            print(" ".join(cmd), flush=True)
            continue
        rc, output = run_cmd(cmd, log_path, int(args.timeout))
        metrics = parse_metrics(output)
        paths = parse_fusion_paths(output)
        row = {
            "dataset": candidate.dataset,
            "name": candidate.name,
            "rc": rc,
            "metrics": metrics,
            "score": score(metrics),
            "prior": parse_prior(output),
            "paths": paths,
            "log": log_path,
            "note": candidate.note,
        }
        rows.append(row)
        print(
            f"  rc={rc} ACC={fmt_metric(metrics, 'ACC')} F1={fmt_metric(metrics, 'F1')} "
            f"score={row['score']:.2f}",
            flush=True,
        )

    if args.dry_run:
        return

    rows_sorted = sorted(rows, key=lambda item: (item["dataset"], -float(item["score"])))
    lines = [
        "# Fusion Feedback Probe",
        "",
        f"- Started: {stamp}",
        f"- Mode: {args.mode}",
        f"- Runs per candidate: {args.runs}",
        f"- Seeds: {args.seed_start}..{args.seed_start + args.runs - 1}",
        "",
        "## Results",
        "",
        "| Dataset | Candidate | Score | ACC | NMI | ARI | F1 | rc | Log |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows_sorted:
        rel_log = row["log"].relative_to(ROOT).as_posix()
        lines.append(
            f"| {row['dataset']} | {row['name']} | {row['score']:.2f} | "
            f"{fmt_metric(row['metrics'], 'ACC')} | {fmt_metric(row['metrics'], 'NMI')} | "
            f"{fmt_metric(row['metrics'], 'ARI')} | {fmt_metric(row['metrics'], 'F1')} | "
            f"{row['rc']} | `{rel_log}` |"
        )
    lines.extend(["", "## Fusion Path Readout", ""])
    for row in rows_sorted:
        lines.append(f"### {row['dataset']} / {row['name']}")
        lines.append("")
        lines.append(f"- Prior: `{row['prior']}`")
        lines.append(f"- Note: {row['note']}")
        if row["paths"]:
            for path in row["paths"]:
                lines.append(f"- `{path}`")
        else:
            lines.append("- No FusionPath lines parsed.")
        lines.append("")

    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
