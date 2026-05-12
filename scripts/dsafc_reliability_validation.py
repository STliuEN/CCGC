from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "reliability_validation"
METRICS = ("ACC", "NMI", "ARI", "F1")
DATASET_LABELS = {
    "reut": "Reuters",
    "uat": "UAT",
    "amap": "AMAP",
    "usps": "USPS",
    "eat": "EAT",
    "cora": "Cora",
    "cite": "Citeseer",
}
ACTIVE_DATASETS = ("reut", "uat", "amap", "usps", "cora", "cite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build reliability-definition validation tables from a unified "
            "DSAFC ablation run with exported fusion artifacts."
        )
    )
    parser.add_argument(
        "--ablation-run-dir",
        type=Path,
        default=ROOT / "experiment_output" / "dsafc_dual_structure_ablation" / "20260512_145444_all_all_unified",
        help="Unified ablation directory containing manifest.json and fusion artifacts.",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def metric_mean(row: dict[str, Any], metric: str) -> float:
    return float(((row.get("metrics") or {}).get(metric) or {}).get("mean", float("nan")))


def score(row: dict[str, Any]) -> float:
    metrics = row.get("metrics") or {}
    if not all(metric in metrics for metric in METRICS):
        return float("nan")
    return (
        float(metrics["ACC"]["mean"])
        + 0.4 * float(metrics["F1"]["mean"])
        + 0.2 * float(metrics["NMI"]["mean"])
        + 0.2 * float(metrics["ARI"]["mean"])
    )


def safe_corr(x_values: list[float], y_values: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(x_values, y_values) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_fusion_trace(fusion_npz: Path) -> dict[str, Any]:
    if not fusion_npz.exists():
        return {}
    data = np.load(fusion_npz, allow_pickle=True)
    out: dict[str, Any] = {}
    if "fusion_weights" in data:
        weights = np.asarray(data["fusion_weights"], dtype=np.float64)
        out["mean_alpha_raw"] = float(np.mean(weights[:, 0]))
        out["mean_alpha_ae"] = float(np.mean(weights[:, 1]))
        entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0)), axis=1)
        out["mean_entropy"] = float(np.mean(entropy))
    if "fusion_trace" in data:
        trace = np.asarray(data["fusion_trace"], dtype=np.float64)
        if trace.ndim == 2 and trace.shape[1] >= 3:
            out["start_alpha_raw"] = float(trace[0, 1])
            out["start_alpha_ae"] = float(trace[0, 2])
            out["end_alpha_raw"] = float(trace[-1, 1])
            out["end_alpha_ae"] = float(trace[-1, 2])
            out["max_alpha_raw"] = float(np.max(trace[:, 1]))
            out["max_alpha_ae"] = float(np.max(trace[:, 2]))
            out["trace_points"] = int(trace.shape[0])
    return out


def dataset_rows(manifest: dict[str, Any], ablation_run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest.get("rows", []):
        dataset = str(item["dataset"])
        if dataset not in ACTIVE_DATASETS:
            continue
        variants = item.get("variants") or {}
        structure = item.get("structure_diag") or {}
        dsafc = variants.get("dsafc") or {}
        a_dsf = variants.get("a_dsf") or {}
        osl = variants.get("osl") or {}
        rsl = variants.get("rsl") or {}
        fusion_diag = dsafc.get("fusion_diag") or {}
        fusion_npz = Path(dsafc.get("fusion_npz", ""))
        if not fusion_npz.is_absolute():
            fusion_npz = ROOT / fusion_npz
        trace_diag = load_fusion_trace(fusion_npz)
        alpha_raw = float(fusion_diag.get("mean_alpha_raw", trace_diag.get("mean_alpha_raw", float("nan"))))
        alpha_ae = float(fusion_diag.get("mean_alpha_ae", trace_diag.get("mean_alpha_ae", float("nan"))))
        hom_raw = float(structure.get("homophily_raw", float("nan")))
        hom_ae = float(structure.get("homophily_ae", float("nan")))
        score_osl = score(osl)
        score_rsl = score(rsl)
        score_adsf = score(a_dsf)
        score_dsafc = score(dsafc)
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "homophily_raw": hom_raw,
                "homophily_ae": hom_ae,
                "delta_homophily_ae_minus_raw": hom_ae - hom_raw,
                "alpha_raw": alpha_raw,
                "alpha_ae": alpha_ae,
                "alpha_gap_raw_minus_ae": alpha_raw - alpha_ae,
                "weight_entropy": float(fusion_diag.get("mean_entropy", trace_diag.get("mean_entropy", float("nan")))),
                "score_osl": score_osl,
                "score_rsl": score_rsl,
                "score_adsf": score_adsf,
                "score_dsafc": score_dsafc,
                "delta_score_rsl_minus_osl": score_rsl - score_osl,
                "delta_score_dsafc_minus_adsf": score_dsafc - score_adsf,
                "start_alpha_raw": trace_diag.get("start_alpha_raw", ""),
                "start_alpha_ae": trace_diag.get("start_alpha_ae", ""),
                "end_alpha_raw": trace_diag.get("end_alpha_raw", ""),
                "end_alpha_ae": trace_diag.get("end_alpha_ae", ""),
                "fusion_npz": rel(fusion_npz) if fusion_npz.exists() else "",
                "source": rel(ablation_run_dir / "summary.md"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fmt(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return "--"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "--"
    return f"{val:.{digits}f}"


def write_summary(run_dir: Path, rows: list[dict[str, Any]], ablation_run_dir: Path) -> None:
    alpha_gap = [float(row["alpha_gap_raw_minus_ae"]) for row in rows]
    delta_hom = [float(row["delta_homophily_ae_minus_raw"]) for row in rows]
    delta_perf = [float(row["delta_score_rsl_minus_osl"]) for row in rows]
    dsafc_gain = [float(row["delta_score_dsafc_minus_adsf"]) for row in rows]
    correlations = {
        "corr_alpha_gap_vs_delta_homophily": safe_corr(alpha_gap, delta_hom),
        "corr_alpha_gap_vs_delta_rsl_osl_score": safe_corr(alpha_gap, delta_perf),
        "corr_alpha_ae_vs_rsl_advantage": safe_corr([float(row["alpha_ae"]) for row in rows], delta_perf),
        "corr_entropy_vs_dsafc_gain": safe_corr([float(row["weight_entropy"]) for row in rows], dsafc_gain),
    }
    write_csv(
        run_dir / "reliability_validation.csv",
        rows,
        [
            "dataset",
            "dataset_label",
            "homophily_raw",
            "homophily_ae",
            "delta_homophily_ae_minus_raw",
            "alpha_raw",
            "alpha_ae",
            "alpha_gap_raw_minus_ae",
            "weight_entropy",
            "score_osl",
            "score_rsl",
            "score_adsf",
            "score_dsafc",
            "delta_score_rsl_minus_osl",
            "delta_score_dsafc_minus_adsf",
            "start_alpha_raw",
            "start_alpha_ae",
            "end_alpha_raw",
            "end_alpha_ae",
            "fusion_npz",
            "source",
        ],
    )
    write_csv(
        run_dir / "reliability_correlations.csv",
        [{"name": key, "pearson_r": value} for key, value in correlations.items()],
        ["name", "pearson_r"],
    )

    lines = [
        "# DSAFC Reliability Definition and Validation",
        "",
        f"- Source ablation run: `{rel(ablation_run_dir / 'summary.md')}`",
        "- Operational definition: a structure view is treated as more reliable when its post-hoc structural purity or single-view contribution is higher; DSAFC should assign larger average fusion weight to the more reliable view.",
        "- The following statistics are diagnostic only and do not participate in training.",
        "",
        "## Reliability Correlations",
        "",
        "| Diagnostic relation | Pearson r |",
        "| --- | ---: |",
        f"| $\\alpha^{{(A)}}-\\alpha^{{(A_E)}}$ vs. homophily$(A_E)-$homophily$(A)$ | {fmt(correlations['corr_alpha_gap_vs_delta_homophily'])} |",
        f"| $\\alpha^{{(A)}}-\\alpha^{{(A_E)}}$ vs. RSL-OSL score gap | {fmt(correlations['corr_alpha_gap_vs_delta_rsl_osl_score'])} |",
        f"| $\\alpha^{{(A_E)}}$ vs. RSL-OSL score gap | {fmt(correlations['corr_alpha_ae_vs_rsl_advantage'])} |",
        f"| fusion entropy vs. DSAFC-A-DSF score gain | {fmt(correlations['corr_entropy_vs_dsafc_gain'])} |",
        "",
        "## Per-Dataset Reliability Table",
        "",
        "| Dataset | $\\Delta$ homophily | RSL-OSL score | $\\alpha^{(A)}$ | $\\alpha^{(A_E)}$ | Entropy | DSAFC-A-DSF score | Source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset_label']} | {fmt(row['delta_homophily_ae_minus_raw'])} | "
            f"{fmt(row['delta_score_rsl_minus_osl'], 2)} | {fmt(row['alpha_raw'])} | "
            f"{fmt(row['alpha_ae'])} | {fmt(row['weight_entropy'])} | "
            f"{fmt(row['delta_score_dsafc_minus_adsf'], 2)} | `{row['source']}` |"
        )
    lines.extend(
        [
            "",
            "## Fusion Trajectory Summary",
            "",
            "| Dataset | Start $\\alpha^{(A)}$ | Start $\\alpha^{(A_E)}$ | End $\\alpha^{(A)}$ | End $\\alpha^{(A_E)}$ | Artifact |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset_label']} | {fmt(row['start_alpha_raw'])} | {fmt(row['start_alpha_ae'])} | "
            f"{fmt(row['end_alpha_raw'])} | {fmt(row['end_alpha_ae'])} | `{row['fusion_npz']}` |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Per-dataset CSV: `{rel(run_dir / 'reliability_validation.csv')}`",
            f"- Correlation CSV: `{rel(run_dir / 'reliability_correlations.csv')}`",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    ablation_run_dir = args.ablation_run_dir if args.ablation_run_dir.is_absolute() else ROOT / args.ablation_run_dir
    if args.run_dir is None:
        run_dir = OUTPUT_ROOT / ablation_run_dir.name
    else:
        run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(ablation_run_dir)
    rows = dataset_rows(manifest, ablation_run_dir)
    write_summary(run_dir, rows, ablation_run_dir)
    print(f"[DONE] summary={run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
