from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "fixed_fusion_weight_ablation"
RAW_TABLES = ROOT / "docs" / "DSAFC_raw_tables.md"
SCGC_ENV_PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
DEFAULT_PYTHON = SCGC_ENV_PYTHON if SCGC_ENV_PYTHON.exists() else Path(sys.executable)
METRICS = ("ACC", "NMI", "ARI", "F1")
FUSION_DOMINANCE_TOL = 0.05
DATASET_ALIASES = {
    "reuters": "reut",
    "reut": "reut",
    "uat": "uat",
    "amap": "amap",
    "usps": "usps",
    "cora": "cora",
    "cite": "cite",
    "citeseer": "cite",
}
DATASET_LABELS = {
    "reut": "Reuters",
    "uat": "UAT",
    "amap": "AMAP",
    "usps": "USPS",
    "cora": "Cora",
    "cite": "Citeseer",
}
DATASET_ORDER = ("reut", "uat", "amap", "usps", "cora", "cite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed raw/AE fusion-weight curves on selected datasets, "
            "with dynamic attention as the adaptive reference."
        )
    )
    parser.add_argument("--datasets", default="reut,uat,amap,usps,cora,cite")
    parser.add_argument("--weights", default="0,0.1,0.25,0.5,0.75,0.9,1")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--timeout", type=int, default=0, help="Per train.py timeout in seconds; 0 disables timeout.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dcgl-mode",
        choices=("no_dcgl", "dcgl", "both"),
        default="no_dcgl",
        help="no_dcgl isolates fusion; dcgl uses the final DSAFC negative term; both runs both sets.",
    )
    parser.add_argument("--no-dynamic", action="store_true", help="Only run fixed weights, without attention baselines.")
    parser.add_argument("--update-raw-tables", action="store_true", help="Update docs/DSAFC_raw_tables.md Table 4-5a from this run.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_experiment_config() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("dsafc_experiment", ROOT / "experiment.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def parse_datasets(raw: str) -> tuple[str, ...]:
    datasets: list[str] = []
    for token in str(raw).replace(";", ",").split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in DATASET_ALIASES:
            raise ValueError(f"Unsupported dataset for this fixed-weight experiment: {token}")
        datasets.append(DATASET_ALIASES[key])
    return tuple(dict.fromkeys(datasets))


def parse_weights(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in str(raw).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = min(1.0, max(0.0, float(token)))
        values.append(value)
    if not values:
        raise ValueError("No fixed weights were provided.")
    return tuple(dict.fromkeys(values))


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


def dcgl_enabled_values(mode: str) -> tuple[bool, ...]:
    if mode == "no_dcgl":
        return (False,)
    if mode == "dcgl":
        return (True,)
    return (False, True)


def resolve_ae_graph(dataset: str) -> Path:
    path = ROOT / "data" / "ae_graph" / f"{dataset}_ae_graph.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing current AE graph asset: {path}")
    return path


def build_module_args(config: dict[str, Any], profile: dict[str, Any], *, enable_dcgl: bool) -> dict[str, Any]:
    if not enable_dcgl:
        return {}
    args = {"enable_dcgl_negative_loss": True}
    args.update(config.get("dcgl_negative_args", {}))
    args.update(profile.get("dcgl_negative_args", {}))
    return args


def build_train_command(
    config: dict[str, Any],
    dataset: str,
    args: argparse.Namespace,
    *,
    fusion_mode: str,
    fixed_raw_weight: float | None,
    enable_dcgl: bool,
    fusion_npz: Path,
) -> list[str]:
    profile = config["dataset_profiles"][dataset]
    train_args = merge_args(
        config.get("baseline_args", {}),
        config.get("train_common_args", {}),
        profile.get("train_args", {}),
    )
    cluster_num = int(profile["cluster_num"])
    cmd = [
        str(args.python),
        "train.py",
        "--dataset",
        dataset,
        "--cluster_num",
        str(cluster_num),
        "--graph_mode",
        "dual",
        "--ae_graph_path",
        str(resolve_ae_graph(dataset)),
        "--fusion_mode",
        fusion_mode,
        "--runs",
        str(args.runs),
        "--seed_start",
        str(args.seed_start),
        "--device",
        args.device,
        "--save_fusion_weights_path",
        str(fusion_npz),
    ]
    cmd.extend(dict_to_cli(train_args))
    cmd.extend(dict_to_cli(config.get("dual_args", {})))
    if fusion_mode == "fixed":
        cmd.extend(["--fixed_raw_weight", f"{float(fixed_raw_weight):.6f}"])
        cmd.extend(dict_to_cli(config.get("dual_mean_args", {})))
    elif fusion_mode == "attn":
        cmd.extend(dict_to_cli(merge_args(config.get("dual_attn_args", {}), profile.get("dual_attn_args", {}))))
    else:
        raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
    cmd.extend(dict_to_cli(build_module_args(config, profile, enable_dcgl=enable_dcgl)))
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


def parse_final_metrics(stdout: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    pattern = re.compile(r"^(ACC|NMI|ARI|F1)\s+\|\s+([+-]?\d+(?:\.\d+)?)\s+.+?\s+([+-]?\d+(?:\.\d+)?)$", re.MULTILINE)
    for match in pattern.finditer(stdout):
        metrics[match.group(1)] = {"mean": float(match.group(2)), "std": float(match.group(3))}
    return metrics


def score_metrics(metrics: dict[str, dict[str, float]]) -> float:
    if not all(metric in metrics for metric in METRICS):
        return float("-inf")
    return (
        metrics["ACC"]["mean"]
        + 0.4 * metrics["F1"]["mean"]
        + 0.2 * metrics["NMI"]["mean"]
        + 0.2 * metrics["ARI"]["mean"]
    )


def fusion_diag(path: Path, *, fallback_raw_weight: float | None = None) -> dict[str, Any]:
    if path.exists():
        data = np.load(path, allow_pickle=True)
        weights = np.asarray(data["fusion_weights"], dtype=np.float32)
        fusion_mean = np.asarray(data["fusion_mean"], dtype=np.float32)
        entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0)), axis=1)
        mean_diff = float(fusion_mean[0]) - float(fusion_mean[1])
        if abs(mean_diff) < FUSION_DOMINANCE_TOL:
            dominant_view = "balanced"
        else:
            dominant_view = "raw" if mean_diff > 0 else "ae"
        return {
            "mean_alpha_raw": float(np.mean(weights[:, 0])),
            "mean_alpha_ae": float(np.mean(weights[:, 1])),
            "mean_entropy": float(np.mean(entropy)),
            "dominant_view": dominant_view,
        }
    if fallback_raw_weight is None:
        return {}
    raw = float(fallback_raw_weight)
    ae = 1.0 - raw
    entropy = 0.0
    if raw > 0:
        entropy -= raw * math.log(raw)
    if ae > 0:
        entropy -= ae * math.log(ae)
    return {
        "mean_alpha_raw": raw,
        "mean_alpha_ae": ae,
        "mean_entropy": entropy,
        "dominant_view": "balanced" if abs(raw - ae) < FUSION_DOMINANCE_TOL else ("raw" if raw > ae else "ae"),
    }


def load_resume(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(row["dataset"], row["key"])] = row
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(run_dir: Path, rows: list[dict[str, Any]], *, datasets: tuple[str, ...], weights: tuple[float, ...], args: argparse.Namespace) -> None:
    lines = [
        "# Fixed Fusion Weight Ablation",
        "",
        f"- Generated at: `{datetime.now().isoformat()}`",
        f"- Datasets: `{', '.join(datasets)}`",
        f"- Raw weights: `{', '.join(f'{w:g}' for w in weights)}`",
        f"- Runs: `{args.runs}`",
        f"- Seeds: `{args.seed_start}..{args.seed_start + args.runs - 1}`",
        f"- DCGL mode: `{args.dcgl_mode}`",
        "- Current AE graph assets are reused from `data/ae_graph`.",
        "",
    ]
    for dataset in datasets:
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Setting | Raw weight | DCGL-negative | ACC | NMI | ARI | F1 | Score | Learned/used raw weight |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        dataset_rows = [row for row in rows if row["dataset"] == dataset and row["status"] == "ok"]
        def sort_key(row: dict[str, Any]) -> tuple[bool, str, float]:
            fixed_weight = row.get("fixed_raw_weight")
            fixed_value = -1.0 if fixed_weight is None else float(fixed_weight)
            return bool(row["enable_dcgl"]), str(row["setting"]), fixed_value

        dataset_rows.sort(key=sort_key)
        for row in dataset_rows:
            metrics = row["metrics"]
            diag = row.get("fusion_diag", {})
            fixed_weight = row.get("fixed_raw_weight")
            fixed_text = "--" if fixed_weight is None else f"{float(fixed_weight):.2f}"
            used_raw = diag.get("mean_alpha_raw")
            used_raw_text = "--" if used_raw is None else f"{float(used_raw):.4f}"
            lines.append(
                f"| {row['label']} | {fixed_text} | {row['enable_dcgl']} | "
                f"{metrics['ACC']['mean']:.2f}+-{metrics['ACC']['std']:.2f} | "
                f"{metrics['NMI']['mean']:.2f}+-{metrics['NMI']['std']:.2f} | "
                f"{metrics['ARI']['mean']:.2f}+-{metrics['ARI']['std']:.2f} | "
                f"{metrics['F1']['mean']:.2f}+-{metrics['F1']['std']:.2f} | "
                f"{row['score']:.2f} | {used_raw_text} |"
            )
        lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def maybe_plot(run_dir: Path, rows: list[dict[str, Any]], datasets: tuple[str, ...]) -> None:
    if not any(row.get("status") == "ok" for row in rows):
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    ok_rows = [row for row in rows if row["status"] == "ok"]
    for metric in ("ACC", "score"):
        all_values: list[float] = []
        for row in ok_rows:
            metrics = row.get("metrics") or {}
            if metric == "score":
                all_values.append(float(row["score"]))
            elif metric in metrics:
                all_values.append(float(metrics[metric]["mean"]))
        y_limits: tuple[float, float] | None = None
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            pad = max(2.0, (y_max - y_min) * 0.08)
            y_limits = (y_min - pad, y_max + pad)
        n_cols = 3 if len(datasets) > 1 else 1
        n_rows = int(math.ceil(len(datasets) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.9 * n_cols, 4.3 * n_rows), sharey=False, squeeze=False)
        axes_arr = axes.ravel()
        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        for ax, dataset in zip(axes_arr, datasets):
            for enable_dcgl, color in ((False, "#4c78a8"), (True, "#f58518")):
                fixed = [
                    row for row in ok_rows
                    if row["dataset"] == dataset and row["setting"] == "fixed" and row["enable_dcgl"] == enable_dcgl
                ]
                if fixed:
                    fixed.sort(key=lambda row: float(row["fixed_raw_weight"]))
                    x = [float(row["fixed_raw_weight"]) for row in fixed]
                    y = [float(row["score"] if metric == "score" else row["metrics"][metric]["mean"]) for row in fixed]
                    label = "fixed + DCGL" if enable_dcgl else "fixed"
                    line, = ax.plot(x, y, marker="o", color=color, label=label)
                    if label not in legend_labels:
                        legend_handles.append(line)
                        legend_labels.append(label)
                dynamic = [
                    row for row in ok_rows
                    if row["dataset"] == dataset and row["setting"] == "dynamic" and row["enable_dcgl"] == enable_dcgl
                ]
                for row in dynamic:
                    diag = row.get("fusion_diag", {})
                    x = float(diag.get("mean_alpha_raw", 0.5))
                    y = float(row["score"] if metric == "score" else row["metrics"][metric]["mean"])
                    label = "dynamic + DCGL" if enable_dcgl else "dynamic"
                    scatter = ax.scatter(
                        [x],
                        [y],
                        marker="o",
                        s=150,
                        facecolor="white",
                        edgecolor=color,
                        linewidth=2.2,
                        label=label,
                        zorder=4,
                    )
                    ax.scatter([x], [y], marker="o", s=42, facecolor="black", edgecolor="black", linewidth=0.0, zorder=5)
                    if label not in legend_labels:
                        legend_handles.append(scatter)
                        legend_labels.append(label)
            ax.set_title(DATASET_LABELS.get(dataset, dataset))
            ax.set_xlabel("Raw-graph weight")
            ax.set_ylabel("Score" if metric == "score" else metric)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(-0.03, 1.03)
            if y_limits is not None:
                ax.set_ylim(*y_limits)
        for ax in axes_arr[len(datasets):]:
            ax.axis("off")
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=min(4, len(legend_handles)),
                frameon=False,
                bbox_to_anchor=(0.5, 1.02),
            )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        fig.savefig(run_dir / f"{metric.lower()}_fixed_weight_curve.png", dpi=220)
        plt.close(fig)


def split_md_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def find_section(lines: list[str], title: str) -> tuple[int, int]:
    start = None
    for idx, line in enumerate(lines):
        if line.strip() in {f"## {title}", f"### {title}"}:
            start = idx
            break
    if start is None:
        raise ValueError(f"Section not found: {title}")
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            end = idx
            break
    return start, end


def find_table(lines: list[str], section_title: str) -> tuple[int, int]:
    start, end = find_section(lines, section_title)
    table_start = None
    for idx in range(start + 1, end):
        if lines[idx].strip().startswith("|"):
            table_start = idx
            break
    if table_start is None:
        raise ValueError(f"Table not found in section: {section_title}")
    table_end = table_start
    while table_end < end and lines[table_end].strip().startswith("|"):
        table_end += 1
    return table_start, table_end


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sort_row(row: dict[str, Any]) -> tuple[int, int, float]:
    dataset_order = {dataset: idx for idx, dataset in enumerate(DATASET_ORDER)}
    fixed_weight = row.get("fixed_raw_weight")
    fixed_value = -1.0 if fixed_weight is None else float(fixed_weight)
    return dataset_order.get(str(row.get("dataset")), 999), int(bool(row.get("enable_dcgl"))), fixed_value


def build_raw_table(rows: list[dict[str, Any]], run_dir: Path) -> list[str]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ok_rows.sort(key=sort_row)
    lines = [
        "| Dataset | Variant | $w_A$ | DCGL-negative | ACC | NMI | ARI | F1 | Score | Learned/used $w_A$ |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ok_rows:
        metrics = row["metrics"]
        diag = row.get("fusion_diag", {})
        fixed_weight = row.get("fixed_raw_weight")
        fixed_text = "--" if fixed_weight is None else f"{float(fixed_weight):.2f}"
        used_raw = diag.get("mean_alpha_raw")
        used_raw_text = "--" if used_raw is None else f"{float(used_raw):.4f}"
        variant = "Fixed" if row.get("setting") == "fixed" else str(row.get("label", "Dynamic"))
        dataset = DATASET_LABELS.get(str(row["dataset"]), str(row["dataset"]))
        lines.append(
            f"| {dataset} | {variant} | {fixed_text} | {row['enable_dcgl']} | "
            f"{metrics['ACC']['mean']:.2f}+-{metrics['ACC']['std']:.2f} | "
            f"{metrics['NMI']['mean']:.2f}+-{metrics['NMI']['std']:.2f} | "
            f"{metrics['ARI']['mean']:.2f}+-{metrics['ARI']['std']:.2f} | "
            f"{metrics['F1']['mean']:.2f}+-{metrics['F1']['std']:.2f} | "
            f"{row['score']:.2f} | {used_raw_text} |"
        )
    return lines


def update_raw_tables(raw_path: Path, run_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    section_title = "Table 4-5a Fixed Fusion Weight Ablation"
    start, _ = find_section(lines, section_title)
    table_start, table_end = find_table(lines, section_title)
    intro = [
        f"## {section_title}",
        "",
        "This table records the fixed raw-graph fusion weight sweep on the six final datasets.",
        "It is used in the KBS-version Section 4.4 to support the claim that a single manually fixed fusion ratio is not portable across datasets.",
        f"The source log is `{rel(run_dir / 'summary.md')}`.",
        "`w_A` denotes the raw-graph weight; the refined-graph weight is `1-w_A`.",
        "",
    ]
    new_lines = lines[:start] + intro + build_raw_table(rows, run_dir) + lines[table_end:]
    raw_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
    print(f"[UPDATED] {raw_path}")


def main() -> int:
    args = parse_args()
    config = load_experiment_config()
    datasets = parse_datasets(args.datasets)
    weights = parse_weights(args.weights)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (OUTPUT_ROOT / f"{stamp}_{'_'.join(datasets)}_fixed_weight")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = run_dir / "results.jsonl"
    resume_rows = load_resume(results_jsonl) if args.resume else {}

    rows: list[dict[str, Any]] = list(resume_rows.values())
    for dataset in datasets:
        for enable_dcgl in dcgl_enabled_values(args.dcgl_mode):
            dcgl_tag = "dcgl" if enable_dcgl else "no_dcgl"
            for raw_weight in weights:
                key = f"fixed_w{raw_weight:.3f}_{dcgl_tag}"
                cached = resume_rows.get((dataset, key))
                if cached is not None:
                    print(f"[SKIP] {dataset}/{key}", flush=True)
                    continue
                out_dir = run_dir / dataset / key
                out_dir.mkdir(parents=True, exist_ok=True)
                fusion_npz = out_dir / "fusion_best.npz"
                cmd = build_train_command(
                    config,
                    dataset,
                    args,
                    fusion_mode="fixed",
                    fixed_raw_weight=raw_weight,
                    enable_dcgl=enable_dcgl,
                    fusion_npz=fusion_npz,
                )
                if args.dry_run:
                    print(" ".join(cmd))
                    continue
                print(f"[RUN] {dataset}/{key}", flush=True)
                rc, stdout, stderr = run_command(cmd, cwd=ROOT, timeout=args.timeout)
                (out_dir / "train.log").write_text(
                    "\n".join(["[COMMAND]", " ".join(cmd), "", "[STDOUT]", stdout or "", "", "[STDERR]", stderr or ""]),
                    encoding="utf-8",
                )
                metrics = parse_final_metrics(stdout)
                row = {
                    "dataset": dataset,
                    "key": key,
                    "setting": "fixed",
                    "label": f"Fixed wA={raw_weight:.2f}",
                    "fixed_raw_weight": float(raw_weight),
                    "enable_dcgl": bool(enable_dcgl),
                    "metrics": metrics,
                    "score": score_metrics(metrics),
                    "fusion_diag": fusion_diag(fusion_npz, fallback_raw_weight=raw_weight),
                    "returncode": rc,
                    "status": "ok" if rc == 0 and metrics else "failed",
                    "log_path": str(out_dir / "train.log"),
                }
                append_jsonl(results_jsonl, row)
                rows.append(row)
                print(f"[{row['status'].upper()}] {dataset}/{key} score={row['score']:.2f}", flush=True)

            if args.no_dynamic:
                continue
            key = f"dynamic_{dcgl_tag}"
            cached = resume_rows.get((dataset, key))
            if cached is not None:
                print(f"[SKIP] {dataset}/{key}", flush=True)
                continue
            out_dir = run_dir / dataset / key
            out_dir.mkdir(parents=True, exist_ok=True)
            fusion_npz = out_dir / "fusion_best.npz"
            cmd = build_train_command(
                config,
                dataset,
                args,
                fusion_mode="attn",
                fixed_raw_weight=None,
                enable_dcgl=enable_dcgl,
                fusion_npz=fusion_npz,
            )
            if args.dry_run:
                print(" ".join(cmd))
                continue
            print(f"[RUN] {dataset}/{key}", flush=True)
            rc, stdout, stderr = run_command(cmd, cwd=ROOT, timeout=args.timeout)
            (out_dir / "train.log").write_text(
                "\n".join(["[COMMAND]", " ".join(cmd), "", "[STDOUT]", stdout or "", "", "[STDERR]", stderr or ""]),
                encoding="utf-8",
            )
            metrics = parse_final_metrics(stdout)
            label = "DSAFC dynamic" if enable_dcgl else "A-DSF dynamic"
            row = {
                "dataset": dataset,
                "key": key,
                "setting": "dynamic",
                "label": label,
                "fixed_raw_weight": None,
                "enable_dcgl": bool(enable_dcgl),
                "metrics": metrics,
                "score": score_metrics(metrics),
                "fusion_diag": fusion_diag(fusion_npz),
                "returncode": rc,
                "status": "ok" if rc == 0 and metrics else "failed",
                "log_path": str(out_dir / "train.log"),
            }
            append_jsonl(results_jsonl, row)
            rows.append(row)
            print(f"[{row['status'].upper()}] {dataset}/{key} score={row['score']:.2f}", flush=True)

    write_summary(run_dir, rows, datasets=datasets, weights=weights, args=args)
    if not args.dry_run:
        maybe_plot(run_dir, rows, datasets)
        if args.update_raw_tables:
            update_raw_tables(RAW_TABLES, run_dir, rows)
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "datasets": list(datasets),
        "weights": list(weights),
        "runs": args.runs,
        "seed_start": args.seed_start,
        "dcgl_mode": args.dcgl_mode,
        "results_jsonl": str(results_jsonl),
        "summary": str(run_dir / "summary.md"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] summary={run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
