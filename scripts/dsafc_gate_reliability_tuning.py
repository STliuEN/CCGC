from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "dsafc_gate_reliability_tuning"
SCGC_ENV_PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
DEFAULT_PYTHON = SCGC_ENV_PYTHON if SCGC_ENV_PYTHON.exists() else Path(sys.executable)
DATASETS = ("reut", "uat", "amap", "usps", "cora", "cite")
METRICS = ("ACC", "NMI", "ARI", "F1")

MAIN_TABLE_FLOOR = {
    "reut": {"ACC": 83.95, "NMI": 60.35, "ARI": 66.32, "F1": 72.36},
    "uat": {"ACC": 55.66, "NMI": 26.98, "ARI": 22.90, "F1": 54.89},
    "amap": {"ACC": 77.49, "NMI": 67.64, "ARI": 58.41, "F1": 72.16},
    "usps": {"ACC": 81.44, "NMI": 71.88, "ARI": 66.49, "F1": 81.02},
    "cora": {"ACC": 72.78, "NMI": 55.02, "ARI": 50.16, "F1": 69.62},
    "cite": {"ACC": 71.59, "NMI": 45.06, "ARI": 46.86, "F1": 62.03},
}


@dataclass(frozen=True)
class Candidate:
    name: str
    args: dict[str, Any]
    rationale: str
    train_args: dict[str, Any] = field(default_factory=dict)
    dual_attn_args: dict[str, Any] = field(default_factory=dict)
    baseline_name: str = "a_dsf_baseline"
    enable_dcgl: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential 10-run tuning for DSAFC reliability-gated DCGL negative. "
            "Each candidate is a complete 10-run evaluation."
        )
    )
    parser.add_argument("--dataset", default="all", help="Comma-separated dataset keys or all.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--max-candidates-per-dataset", type=int, default=8)
    parser.add_argument("--min-candidates-after-pass", type=int, default=2)
    parser.add_argument("--main-floor-tolerance", type=float, default=0.05)
    parser.add_argument("--ablation-tolerance", type=float, default=0.00)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("dsafc_experiment", ROOT / "experiment.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def parse_dataset_list(raw: str) -> tuple[str, ...]:
    aliases = {"reuters": "reut", "citeseer": "cite"}
    if str(raw).strip().lower() == "all":
        return DATASETS
    out: list[str] = []
    for token in str(raw).replace(";", ",").split(","):
        name = aliases.get(token.strip().lower(), token.strip().lower())
        if not name:
            continue
        if name not in DATASETS:
            raise ValueError(f"Unsupported dataset: {token}")
        out.append(name)
    return tuple(dict.fromkeys(out))


def merge_args(*arg_dicts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for arg_dict in arg_dicts:
        if arg_dict:
            merged.update(arg_dict)
    return merged


def dict_to_cli(args: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key, value in args.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        cli.extend([flag, str(value)])
    return cli


def score(metrics: dict[str, dict[str, float]]) -> float:
    if not all(metric in metrics for metric in METRICS):
        return float("-inf")
    return (
        metrics["ACC"]["mean"]
        + 0.4 * metrics["F1"]["mean"]
        + 0.2 * metrics["NMI"]["mean"]
        + 0.2 * metrics["ARI"]["mean"]
    )


def parse_final_metrics(stdout: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    pattern = re.compile(r"^(ACC|NMI|ARI|F1)\s+\|\s+([+-]?\d+(?:\.\d+)?)\s+.+?\s+([+-]?\d+(?:\.\d+)?)$", re.MULTILINE)
    for match in pattern.finditer(stdout):
        metrics[match.group(1)] = {"mean": float(match.group(2)), "std": float(match.group(3))}
    return metrics


def parse_seed_rows(stdout: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    pattern = re.compile(
        r"Run\s+(\d+) Done \| Seed: (\d+) \| ACC: ([0-9.]+) \| NMI: ([0-9.]+) \| ARI: ([0-9.]+) \| F1: ([0-9.]+)"
    )
    for match in pattern.finditer(stdout):
        rows.append(
            {
                "run": int(match.group(1)),
                "seed": int(match.group(2)),
                "ACC": float(match.group(3)),
                "NMI": float(match.group(4)),
                "ARI": float(match.group(5)),
                "F1": float(match.group(6)),
            }
        )
    return rows


def build_base_train_args(config: dict[str, Any], dataset: str) -> tuple[int, dict[str, Any], dict[str, Any]]:
    profile = config["dataset_profiles"][dataset]
    cluster_num = int(profile["cluster_num"])
    train_args = merge_args(
        config.get("baseline_args", {}),
        config.get("train_common_args", {}),
        profile.get("train_args", {}),
    )
    dual_attn_args = merge_args(
        config.get("dual_attn_args", {}),
        profile.get("dual_attn_args", {}),
    )
    return cluster_num, train_args, dual_attn_args


def build_command(
    config: dict[str, Any],
    dataset: str,
    args: argparse.Namespace,
    *,
    candidate: Candidate | None,
    save_fusion_path: Path,
) -> list[str]:
    cluster_num, train_args, dual_attn_args = build_base_train_args(config, dataset)
    cmd = [
        str(args.python),
        "train.py",
        "--dataset",
        dataset,
        "--cluster_num",
        str(cluster_num),
        "--graph_mode",
        "dual",
        "--runs",
        str(args.runs),
        "--seed_start",
        str(args.seed_start),
        "--device",
        args.device,
    ]
    cmd.extend(dict_to_cli(train_args))
    if candidate is not None:
        cmd.extend(dict_to_cli(candidate.train_args))
    cmd.extend(["--ae_graph_path", str(ROOT / "data" / "ae_graph" / f"{dataset}_ae_graph.txt")])
    cmd.extend(["--fusion_mode", "attn"])
    cmd.extend(dict_to_cli(dual_attn_args))
    if candidate is not None:
        cmd.extend(dict_to_cli(candidate.dual_attn_args))
    cmd.extend(["--save_fusion_weights_path", str(save_fusion_path)])
    if candidate is not None and candidate.enable_dcgl:
        cmd.append("--enable_dcgl_negative_loss")
        dcgl_args = merge_args(config.get("dcgl_negative_args", {}), config["dataset_profiles"][dataset].get("dcgl_negative_args", {}))
        dcgl_args.update(candidate.args)
        cmd.extend(dict_to_cli(dcgl_args))
    return cmd


def run_command(cmd: list[str], *, cwd: Path, timeout: int) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=None if timeout <= 0 else timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_command_to_files(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: int,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, str, str, bool]:
    timed_out = False
    try:
        with stdout_path.open("w", encoding="utf-8", errors="replace") as out_handle, stderr_path.open("w", encoding="utf-8", errors="replace") as err_handle:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=out_handle,
                stderr=err_handle,
                text=True,
                timeout=None if timeout <= 0 else timeout,
            )
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = -9
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    return returncode, stdout, stderr, timed_out


def base_dcgl_args(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    return merge_args(config.get("dcgl_negative_args", {}), config["dataset_profiles"][dataset].get("dcgl_negative_args", {}))


def candidates_for_dataset(config: dict[str, Any], dataset: str) -> list[Candidate]:
    base = base_dcgl_args(config, dataset)
    tau = float(base.get("dcgl_neg_tau", 0.5))
    weight = float(base.get("dcgl_neg_weight", 0.6))

    common: list[Candidate] = [
        Candidate(
            "current_gate",
            {
                "dcgl_neg_tau": tau,
                "dcgl_neg_weight": weight,
                "dcgl_neg_gate_threshold": float(base.get("dcgl_neg_gate_threshold", 0.55)),
                "dcgl_neg_gate_power": float(base.get("dcgl_neg_gate_power", 2.0)),
                "dcgl_neg_gate_min": float(base.get("dcgl_neg_gate_min", 0.0)),
            },
            "Current reliability-gated setting from experiment.py.",
        ),
        Candidate(
            "legacy_conservative",
            {
                "dcgl_neg_tau": tau,
                "dcgl_neg_weight": weight,
                "disable_dcgl_neg_reliability_gate": True,
            },
            "Previous conservative row-weighted DCGL negative; useful for checking whether the new gate introduced regressions.",
        ),
        Candidate(
            "mild_w04_t065_p2",
            {"dcgl_neg_tau": tau, "dcgl_neg_weight": 0.4, "dcgl_neg_gate_threshold": 0.65, "dcgl_neg_gate_power": 2.0, "dcgl_neg_gate_min": 0.0},
            "More conservative gate and lower negative strength.",
        ),
        Candidate(
            "mild_w03_t065_p2",
            {"dcgl_neg_tau": tau, "dcgl_neg_weight": 0.3, "dcgl_neg_gate_threshold": 0.65, "dcgl_neg_gate_power": 2.0, "dcgl_neg_gate_min": 0.0},
            "Low-strength DCGL negative for datasets where DSAFC trails A-DSF.",
        ),
        Candidate(
            "late_w04_t075_p3",
            {"dcgl_neg_tau": tau, "dcgl_neg_weight": 0.4, "dcgl_neg_gate_threshold": 0.75, "dcgl_neg_gate_power": 3.0, "dcgl_neg_gate_min": 0.0},
            "Late activation to avoid unreliable pseudo-cluster separation.",
        ),
        Candidate(
            "late_w03_t075_p3",
            {"dcgl_neg_tau": tau, "dcgl_neg_weight": 0.3, "dcgl_neg_gate_threshold": 0.75, "dcgl_neg_gate_power": 3.0, "dcgl_neg_gate_min": 0.0},
            "Very conservative negative separation.",
        ),
        Candidate(
            "near_off_w02_t080_p3",
            {"dcgl_neg_tau": tau, "dcgl_neg_weight": 0.2, "dcgl_neg_gate_threshold": 0.80, "dcgl_neg_gate_power": 3.0, "dcgl_neg_gate_min": 0.0},
            "Near-off diagnostic; should approximate A-DSF while preserving the module path.",
        ),
    ]

    if dataset in {"reut", "amap", "cora"}:
        common.extend(
            [
                Candidate(
                    "active_w06_t045_p15",
                    {"dcgl_neg_tau": tau, "dcgl_neg_weight": 0.6, "dcgl_neg_gate_threshold": 0.45, "dcgl_neg_gate_power": 1.5, "dcgl_neg_gate_min": 0.0},
                    "Earlier activation for datasets where negative separation can help.",
                ),
                Candidate(
                    "active_w08_t055_p2",
                    {"dcgl_neg_tau": tau, "dcgl_neg_weight": 0.8, "dcgl_neg_gate_threshold": 0.55, "dcgl_neg_gate_power": 2.0, "dcgl_neg_gate_min": 0.0},
                    "Higher strength check on datasets with positive DSAFC signal.",
                ),
            ]
        )
    if dataset in {"uat", "cite", "usps"}:
        common.extend(
            [
                Candidate(
                    "ultra_late_w02_t085_p4",
                    {"dcgl_neg_tau": tau, "dcgl_neg_weight": 0.2, "dcgl_neg_gate_threshold": 0.85, "dcgl_neg_gate_power": 4.0, "dcgl_neg_gate_min": 0.0},
                    "Almost disabled unless cluster reliability is very high.",
                ),
                Candidate(
                    "floor_w03_t075_p3_min005",
                    {"dcgl_neg_tau": tau, "dcgl_neg_weight": 0.3, "dcgl_neg_gate_threshold": 0.75, "dcgl_neg_gate_power": 3.0, "dcgl_neg_gate_min": 0.05},
                    "Small nonzero floor to avoid abrupt off/on behavior.",
                ),
            ]
        )
    if dataset == "reut":
        common.extend(
            [
                Candidate(
                    "legacy_w025_tau05",
                    {"dcgl_neg_tau": 0.5, "dcgl_neg_weight": 0.25, "disable_dcgl_neg_reliability_gate": True},
                    "Reuters near-legacy check: slightly softer negative to recover F1 without changing the training center.",
                ),
                Candidate(
                    "legacy_w035_tau05",
                    {"dcgl_neg_tau": 0.5, "dcgl_neg_weight": 0.35, "disable_dcgl_neg_reliability_gate": True},
                    "Reuters near-legacy check: slightly stronger than the selected historical 0.30 weight.",
                ),
                Candidate(
                    "legacy_w040_tau075",
                    {"dcgl_neg_tau": 0.75, "dcgl_neg_weight": 0.4, "disable_dcgl_neg_reliability_gate": True},
                    "Reuters smoother-temperature legacy check for the small remaining F1 gap.",
                ),
            ]
        )
    if dataset == "uat":
        # These candidates stay within the paper-facing module contract:
        # attention fusion + DCGL negative only, with dynamic threshold / GCN /
        # EMA / branch bias left off. Each one gets its own same-center A-DSF
        # baseline via `baseline_name`.
        uat_centers = [
            (
                "uat_center_default_strong_ari",
                {"threshold": 0.4, "t": 5, "epochs": 500, "lr": 1.2e-4, "alpha": 0.45},
                {"fusion_temp": 1.8, "fusion_balance": 0.35, "lambda_inst": 0.09, "lambda_clu": 0.09, "warmup_epochs": 35, "fusion_min_weight": 0.20},
                "Current UAT center from experiment.py; preserves the best ARI-facing settings.",
            ),
            (
                "uat_center_hist_f1",
                {"threshold": 0.4, "t": 4, "epochs": 400, "lr": 1e-4, "alpha": 0.5},
                {"fusion_temp": 1.9, "fusion_balance": 0.35, "lambda_inst": 0.08, "lambda_clu": 0.07, "warmup_epochs": 35, "fusion_min_weight": 0.20},
                "Historical strict DCGL-only 10-run center with very stable F1, but weaker ARI.",
            ),
            (
                "uat_center_apex",
                {"threshold": 0.4, "t": 5, "epochs": 500, "lr": 1.2e-4, "alpha": 0.45},
                {"fusion_temp": 1.8, "fusion_balance": 0.35, "lambda_inst": 0.09, "lambda_clu": 0.09, "warmup_epochs": 35, "fusion_min_weight": 0.20},
                "Apex strict DCGL-only center from the May UAT push; repeated here with same-center A-DSF.",
            ),
            (
                "uat_center_low_balance",
                {"threshold": 0.38, "t": 5, "epochs": 500, "lr": 1.2e-4, "alpha": 0.45},
                {"fusion_temp": 1.6, "fusion_balance": 0.25, "lambda_inst": 0.08, "lambda_clu": 0.08, "warmup_epochs": 35, "fusion_min_weight": 0.15},
                "Lower raw/AE bias and slightly lower threshold to reduce the seed-7 style drop.",
            ),
            (
                "uat_center_high_conf",
                {"threshold": 0.45, "t": 5, "epochs": 500, "lr": 1.2e-4, "alpha": 0.45},
                {"fusion_temp": 1.8, "fusion_balance": 0.35, "lambda_inst": 0.08, "lambda_clu": 0.075, "warmup_epochs": 45, "fusion_min_weight": 0.22},
                "Higher-confidence warmup variant aimed at improving ARI before activating the negative term.",
            ),
        ]
        dcgl_variants = [
            ("gate_w03_t065", {"dcgl_neg_tau": 0.5, "dcgl_neg_weight": 0.3, "dcgl_neg_gate_threshold": 0.65, "dcgl_neg_gate_power": 2.0, "dcgl_neg_gate_min": 0.0}),
            ("legacy_w03", {"dcgl_neg_tau": 0.5, "dcgl_neg_weight": 0.3, "disable_dcgl_neg_reliability_gate": True}),
            ("legacy_w04", {"dcgl_neg_tau": 0.5, "dcgl_neg_weight": 0.4, "disable_dcgl_neg_reliability_gate": True}),
            ("gate_w02_t080", {"dcgl_neg_tau": 0.5, "dcgl_neg_weight": 0.2, "dcgl_neg_gate_threshold": 0.80, "dcgl_neg_gate_power": 3.0, "dcgl_neg_gate_min": 0.0}),
        ]
        extra: list[Candidate] = []
        for center_name, train_overrides, dual_overrides, center_note in uat_centers:
            extra.append(
                Candidate(
                    f"{center_name}_adsf",
                    {},
                    f"Same-center A-DSF baseline for {center_name}. {center_note}",
                    train_args=train_overrides,
                    dual_attn_args=dual_overrides,
                    baseline_name=f"{center_name}_adsf",
                    enable_dcgl=False,
                )
            )
            for suffix, dcgl_overrides in dcgl_variants:
                extra.append(
                    Candidate(
                        f"{center_name}_{suffix}",
                        dcgl_overrides,
                        center_note,
                        train_args=train_overrides,
                        dual_attn_args=dual_overrides,
                        baseline_name=f"{center_name}_adsf",
                    )
                )
        common.extend(extra)
    return common


def pass_main_floor(dataset: str, metrics: dict[str, dict[str, float]], tolerance: float) -> bool:
    floor = MAIN_TABLE_FLOOR[dataset]
    return all(metrics.get(metric, {}).get("mean", -1e9) + tolerance >= value for metric, value in floor.items())


def pass_ablation(candidate_metrics: dict[str, dict[str, float]], adsf_metrics: dict[str, dict[str, float]], tolerance: float) -> bool:
    if not candidate_metrics or not adsf_metrics:
        return False
    if score(candidate_metrics) + tolerance < score(adsf_metrics):
        return False
    return candidate_metrics["ACC"]["mean"] + tolerance >= adsf_metrics["ACC"]["mean"]


def safe_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    return obj


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_json(row), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") in {"dry_run", "pending", "running"}:
            continue
        row.setdefault("baseline_name", "a_dsf_baseline")
        row.setdefault("is_baseline", row.get("candidate") == "a_dsf_baseline" or not row.get("args"))
        rows[(row["dataset"], row["candidate"])] = row
    return rows


def metric_text(metrics: dict[str, dict[str, float]]) -> str:
    if not metrics:
        return "--"
    return " / ".join(f"{metrics[m]['mean']:.2f}+-{metrics[m]['std']:.2f}" for m in METRICS)


def write_summary(run_dir: Path, rows: list[dict[str, Any]], datasets: tuple[str, ...], args: argparse.Namespace) -> None:
    lines = [
        "# DSAFC Reliability-Gated Negative Tuning",
        "",
        f"- Generated at: `{datetime.now().isoformat()}`",
        f"- Datasets: `{', '.join(datasets)}`",
        f"- Runs per candidate: `{args.runs}`",
        f"- Seeds: `{args.seed_start}..{args.seed_start + args.runs - 1}`",
        f"- Main floor tolerance: `{args.main_floor_tolerance}`",
        f"- Ablation tolerance: `{args.ablation_tolerance}`",
        "",
    ]
    for dataset in datasets:
        ds_rows = [row for row in rows if row["dataset"] == dataset]
        if not ds_rows:
            continue
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- Main-table floor: `{MAIN_TABLE_FLOOR[dataset]}`")
        best_pass = next((row for row in ds_rows if row.get("selected_best")), None)
        if best_pass:
            lines.append(f"- Selected: `{best_pass['candidate']}` score `{best_pass['score']:.2f}` metrics `{metric_text(best_pass['metrics'])}`")
        else:
            lines.append("- Selected: `None yet`")
        lines.append("")
        lines.append("| Candidate | Baseline | ACC | NMI | ARI | F1 | Score | Main floor | A-DSF pass | Notes |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")
        for row in ds_rows:
            metrics = row.get("metrics", {})
            lines.append(
                f"| {row['candidate']} | "
                f"{row.get('baseline_name', '')} | "
                f"{metrics.get('ACC', {}).get('mean', float('nan')):.2f}+-{metrics.get('ACC', {}).get('std', float('nan')):.2f} | "
                f"{metrics.get('NMI', {}).get('mean', float('nan')):.2f}+-{metrics.get('NMI', {}).get('std', float('nan')):.2f} | "
                f"{metrics.get('ARI', {}).get('mean', float('nan')):.2f}+-{metrics.get('ARI', {}).get('std', float('nan')):.2f} | "
                f"{metrics.get('F1', {}).get('mean', float('nan')):.2f}+-{metrics.get('F1', {}).get('std', float('nan')):.2f} | "
                f"{row.get('score', float('nan')):.2f} | {row.get('pass_main_floor')} | {row.get('pass_ablation')} | {row.get('rationale', '')} |"
            )
        lines.append("")
        bad_seed_notes = [row for row in ds_rows if row.get("bad_seed_note")]
        for row in bad_seed_notes:
            lines.append(f"- `{row['candidate']}` bad-seed note: {row['bad_seed_note']}")
        if bad_seed_notes:
            lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def bad_seed_note(seed_rows: list[dict[str, float]], metrics: dict[str, dict[str, float]]) -> str:
    if not seed_rows or "ACC" not in metrics:
        return ""
    accs = [row["ACC"] for row in seed_rows]
    mean_acc = metrics["ACC"]["mean"]
    std_acc = max(metrics["ACC"]["std"], 1e-6)
    bad = [row for row in seed_rows if row["ACC"] < mean_acc - 2.0 * std_acc]
    if not bad:
        return ""
    return "; ".join(f"seed {int(row['seed'])} ACC={row['ACC']:.2f}" for row in bad)


def main() -> int:
    args = parse_args()
    config = load_config()
    datasets = parse_dataset_list(args.dataset)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (args.run_dir or OUTPUT_ROOT / f"{stamp}_{'_'.join(datasets)}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    resume_rows = load_jsonl(results_path) if args.resume else {}
    all_rows: list[dict[str, Any]] = list(resume_rows.values())

    for dataset in datasets:
        print(f"[DATASET] {dataset}", flush=True)
        adfs_candidate = Candidate("a_dsf_baseline", {}, "A-DSF baseline for same-run ablation comparison.", enable_dcgl=False)
        planned = [adfs_candidate] + candidates_for_dataset(config, dataset)
        evaluated = 0
        pass_seen = 0

        for candidate in planned:
            key = (dataset, candidate.name)
            if key in resume_rows:
                row = resume_rows[key]
                print(f"[SKIP] {dataset}/{candidate.name} cached score={row.get('score')}", flush=True)
            else:
                variant_dir = run_dir / dataset / candidate.name
                variant_dir.mkdir(parents=True, exist_ok=True)
                fusion_path = variant_dir / "fusion_diag_best.npz"
                cmd = build_command(
                    config,
                    dataset,
                    args,
                    candidate=None if candidate.name == "a_dsf_baseline" else candidate,
                    save_fusion_path=fusion_path,
                )
                row = {
                    "dataset": dataset,
                    "candidate": candidate.name,
                    "args": candidate.args,
                    "train_args": candidate.train_args,
                    "dual_attn_args": candidate.dual_attn_args,
                    "baseline_name": candidate.baseline_name,
                    "is_baseline": not candidate.enable_dcgl,
                    "rationale": candidate.rationale,
                    "cmd": cmd,
                    "log_path": str(variant_dir / "train.log"),
                    "status": "dry_run" if args.dry_run else "pending",
                }
                if args.dry_run:
                    append_jsonl(results_path, row)
                    all_rows.append(row)
                    continue

                print(f"[RUN] {dataset}/{candidate.name}", flush=True)
                row["status"] = "running"
                row["started_at"] = datetime.now().isoformat()
                append_jsonl(results_path, row)
                stdout_path = variant_dir / "stdout.raw.log"
                stderr_path = variant_dir / "stderr.raw.log"
                rc, stdout, stderr, timed_out = run_command_to_files(
                    cmd,
                    cwd=ROOT,
                    timeout=args.timeout,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                (variant_dir / "train.log").write_text(
                    "\n".join(
                        [
                            "=" * 80,
                            "[COMMAND]",
                            " ".join(cmd),
                            "",
                            "=" * 80,
                            "[STDOUT]",
                            stdout or "",
                            "",
                            "=" * 80,
                            "[STDERR]",
                            stderr or "",
                        ]
                    ),
                    encoding="utf-8",
                )
                metrics = parse_final_metrics(stdout)
                seed_rows = parse_seed_rows(stdout)
                row.update(
                    {
                        "status": "ok" if rc == 0 and metrics else "failed",
                        "returncode": rc,
                        "timed_out": timed_out,
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                        "metrics": metrics,
                        "seed_rows": seed_rows,
                        "score": score(metrics),
                        "bad_seed_note": bad_seed_note(seed_rows, metrics),
                    }
                )
                append_jsonl(results_path, row)
                all_rows.append(row)
                print(f"[{row['status'].upper()}] {dataset}/{candidate.name} score={row.get('score'):.2f}", flush=True)

            if not candidate.enable_dcgl:
                write_summary(run_dir, all_rows, datasets, args)
                continue

            evaluated += 1
            adsf = next((r for r in all_rows if r["dataset"] == dataset and r["candidate"] == candidate.baseline_name), None)
            if row.get("status") == "ok" and adsf and adsf.get("metrics"):
                row["pass_main_floor"] = pass_main_floor(dataset, row["metrics"], args.main_floor_tolerance)
                row["pass_ablation"] = pass_ablation(row["metrics"], adsf["metrics"], args.ablation_tolerance)
                if row["pass_main_floor"] and row["pass_ablation"]:
                    pass_seen += 1
            write_summary(run_dir, all_rows, datasets, args)

            if evaluated >= args.max_candidates_per_dataset:
                break
            if pass_seen > 0 and pass_seen >= args.min_candidates_after_pass:
                break

        ds_rows = [row for row in all_rows if row["dataset"] == dataset and not row.get("is_baseline") and row.get("status") == "ok"]
        for row in ds_rows:
            adsf = next((r for r in all_rows if r["dataset"] == dataset and r["candidate"] == row.get("baseline_name", "a_dsf_baseline")), None)
            row["pass_main_floor"] = pass_main_floor(dataset, row["metrics"], args.main_floor_tolerance)
            row["pass_ablation"] = bool(adsf and pass_ablation(row["metrics"], adsf["metrics"], args.ablation_tolerance))
            row["selected_best"] = False
        pass_rows = [row for row in ds_rows if row["pass_main_floor"] and row["pass_ablation"]]
        if pass_rows:
            best = max(pass_rows, key=lambda r: (float(r["score"]), float(r["metrics"]["ACC"]["mean"])))
            best["selected_best"] = True
        write_summary(run_dir, all_rows, datasets, args)

    print(f"[DONE] summary={run_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
