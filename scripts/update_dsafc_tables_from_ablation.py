from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_TABLES = ROOT / "docs" / "DSAFC_raw_tables.md"

METRICS = ("ACC", "NMI", "ARI", "F1")
VARIANTS = ("osl", "rsl", "f_dsf", "a_dsf", "dsafc")
VARIANT_LABELS = {
    "osl": "OSL",
    "rsl": "RSL",
    "f_dsf": "F-DSF",
    "a_dsf": "A-DSF",
    "dsafc": "DSAFC",
}
DATASET_LABELS = {
    "reut": "Reuters",
    "reuters": "Reuters",
    "uat": "UAT",
    "amap": "AMAP",
    "usps": "USPS",
    "eat": "EAT",
    "cora": "Cora",
    "cite": "Citeseer",
    "citeseer": "Citeseer",
}
DATASET_ORDER = ("Reuters", "UAT", "AMAP", "USPS", "EAT", "Cora", "Citeseer")
FUSION_DOMINANCE_TOL = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update DSAFC raw tables from one unified "
            "scripts/dsafc_dual_structure_ablation.py run."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw-tables", type=Path, default=RAW_TABLES)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def split_md_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def format_md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


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


def replace_table(lines: list[str], section_title: str, table_lines: list[str]) -> list[str]:
    table_start, table_end = find_table(lines, section_title)
    return lines[:table_start] + table_lines + lines[table_end:]


def normalize_dataset(raw: str) -> str:
    key = raw.strip().lower()
    if key not in DATASET_LABELS:
        raise ValueError(f"Unsupported dataset in ablation manifest: {raw}")
    return DATASET_LABELS[key]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def source_path(run_dir: Path) -> str:
    return f"`{rel(run_dir / 'summary.md')}`"


def metric_cell(value: dict[str, Any]) -> str:
    return f"{float(value['mean']):.2f}+-{float(value['std']):.2f}"


def mean_cell(value: dict[str, Any]) -> str:
    return f"{float(value['mean']):.2f}"


def dominance_label(raw_weight: float, ae_weight: float, *, tol: float = FUSION_DOMINANCE_TOL) -> str:
    diff = float(raw_weight) - float(ae_weight)
    if abs(diff) < tol:
        return "Balanced"
    return "Raw" if diff > 0 else "AE"


def load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_results(manifest: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_row in manifest.get("rows", []):
        dataset = normalize_dataset(str(dataset_row["dataset"]))
        out[dataset] = {}
        for variant, row in (dataset_row.get("variants") or {}).items():
            if variant in VARIANTS:
                out[dataset][variant] = row
    if not out:
        raise ValueError("Manifest has no dataset rows.")
    return out


def load_structure(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for dataset_row in manifest.get("rows", []):
        dataset = normalize_dataset(str(dataset_row["dataset"]))
        diag = dataset_row.get("structure_diag")
        if diag:
            out[dataset] = diag
    return out


def dataset_order_from_results(results: dict[str, dict[str, dict[str, Any]]]) -> tuple[str, ...]:
    ordered = tuple(dataset for dataset in DATASET_ORDER if dataset in results)
    extras = tuple(dataset for dataset in results if dataset not in DATASET_ORDER)
    return ordered + extras


def validate_complete(results: dict[str, dict[str, dict[str, Any]]], dataset_order: tuple[str, ...]) -> None:
    missing: list[str] = []
    for dataset in dataset_order:
        if dataset not in results:
            missing.append(f"{dataset}: all variants")
            continue
        for variant in VARIANTS:
            row = results[dataset].get(variant)
            if not row:
                missing.append(f"{dataset}: {VARIANT_LABELS[variant]}")
                continue
            if row.get("status") != "ok":
                missing.append(f"{dataset}: {VARIANT_LABELS[variant]} status={row.get('status')}")
                continue
            metrics = row.get("metrics") or {}
            for metric in METRICS:
                value = metrics.get(metric)
                if not value or "mean" not in value or "std" not in value:
                    missing.append(f"{dataset}: {VARIANT_LABELS[variant]} {metric}")
    if missing:
        preview = "\n".join(f"- {item}" for item in missing[:40])
        if len(missing) > 40:
            preview += f"\n- ... {len(missing) - 40} more"
        raise ValueError(f"Unified ablation run is incomplete:\n{preview}")


def update_main_table(lines: list[str], results: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    table_start, table_end = find_table(lines, "Table 4-2 Main Clustering Results")
    header = split_md_row(lines[table_start])
    dataset_col = header.index("Dataset")
    metric_col = header.index("Metric")
    ours_col = header.index("Ours")

    updated = lines[:]
    current_dataset = ""
    for idx in range(table_start + 2, table_end):
        row = split_md_row(lines[idx])
        if row[dataset_col]:
            current_dataset = row[dataset_col]
        metric = row[metric_col]
        if current_dataset in results and metric in METRICS:
            row[ours_col] = mean_cell(results[current_dataset]["dsafc"]["metrics"][metric])
            updated[idx] = format_md_row(row)
    return updated


def build_ours_source_table(
    results: dict[str, dict[str, dict[str, Any]]],
    run_dir: Path,
    dataset_order: tuple[str, ...],
) -> list[str]:
    lines = ["| Dataset | Ours | Source |", "| --- | --- | --- |"]
    src = source_path(run_dir)
    for dataset in dataset_order:
        metrics = results[dataset]["dsafc"]["metrics"]
        values = " / ".join(mean_cell(metrics[metric]) for metric in METRICS)
        lines.append(f"| {dataset} | {values} | {src} |")
    return lines


def build_ours_std_table(
    results: dict[str, dict[str, dict[str, Any]]],
    run_dir: Path,
    dataset_order: tuple[str, ...],
) -> list[str]:
    lines = [
        "| Dataset | ACC std | NMI std | ARI std | F1 std | Std source |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    src = source_path(run_dir)
    for dataset in dataset_order:
        metrics = results[dataset]["dsafc"]["metrics"]
        stds = " | ".join(f"{float(metrics[metric]['std']):.2f}" for metric in METRICS)
        lines.append(f"| {dataset} | {stds} | {src} |")
    return lines


def build_ablation_table(
    results: dict[str, dict[str, dict[str, Any]]],
    dataset_order: tuple[str, ...],
) -> list[str]:
    lines = [
        "| Dataset | Metric | OSL | RSL | F-DSF | A-DSF | DSAFC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in dataset_order:
        for metric_idx, metric in enumerate(METRICS):
            cells = [dataset if metric_idx == 0 else "", metric]
            for variant in VARIANTS:
                cells.append(metric_cell(results[dataset][variant]["metrics"][metric]))
            lines.append(format_md_row(cells))
    return lines


def build_dsafc_source_table(
    results: dict[str, dict[str, dict[str, Any]]],
    run_dir: Path,
    dataset_order: tuple[str, ...],
) -> list[str]:
    lines = ["| Dataset | DSAFC | Source |", "| --- | --- | --- |"]
    src = source_path(run_dir)
    for dataset in dataset_order:
        metrics = results[dataset]["dsafc"]["metrics"]
        values = " / ".join(mean_cell(metrics[metric]) for metric in METRICS)
        lines.append(f"| {dataset} | {values} | {src} |")
    return lines


def build_acc_plot_table(
    results: dict[str, dict[str, dict[str, Any]]],
    dataset_order: tuple[str, ...],
) -> list[str]:
    lines = [
        "| Dataset | OSL | RSL | F-DSF | A-DSF | DSAFC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in dataset_order:
        cells = [dataset]
        for variant in VARIANTS:
            cells.append(metric_cell(results[dataset][variant]["metrics"]["ACC"]))
        lines.append(format_md_row(cells))
    return lines


def build_structure_table(
    structure: dict[str, dict[str, Any]],
    run_dir: Path,
    dataset_order: tuple[str, ...],
) -> list[str]:
    lines = [
        "| Dataset | $\\lvert E_A\\rvert$ | $\\lvert E_E\\rvert$ | Edge overlap | New-edge ratio | Homophily$(A)$ | Homophily$(A_E)$ | Source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    src = source_path(run_dir)
    for dataset in dataset_order:
        diag = structure[dataset]
        lines.append(
            f"| {dataset} | {int(diag['raw_edges'])} | {int(diag['ae_edges'])} | "
            f"{float(diag['edge_overlap_ratio']):.4f} | {float(diag['new_edge_ratio']):.4f} | "
            f"{float(diag['homophily_raw']):.4f} | {float(diag['homophily_ae']):.4f} | {src} |"
        )
    return lines


def build_fusion_table(
    results: dict[str, dict[str, dict[str, Any]]],
    run_dir: Path,
    dataset_order: tuple[str, ...],
) -> list[str]:
    lines = [
        "| Dataset | Variant | Mean $\\alpha^{(A)}$ | Mean $\\alpha^{(A_E)}$ | Weight entropy | Dominant view | Source |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    src = source_path(run_dir)
    for dataset in dataset_order:
        diag = results[dataset]["dsafc"].get("fusion_diag")
        if not diag:
            raise ValueError(f"Missing DSAFC fusion diagnostics for {dataset}.")
        dominant = dominance_label(float(diag["mean_alpha_raw"]), float(diag["mean_alpha_ae"]))
        lines.append(
            f"| {dataset} | DSAFC | {float(diag['mean_alpha_raw']):.4f} | "
            f"{float(diag['mean_alpha_ae']):.4f} | {float(diag['mean_entropy']):.4f} | "
            f"{dominant} | {src} |"
        )
    return lines


def build_structure_fusion_table(
    structure: dict[str, dict[str, Any]],
    results: dict[str, dict[str, dict[str, Any]]],
    run_dir: Path,
    dataset_order: tuple[str, ...],
) -> list[str]:
    lines = [
        "| Dataset | Edge overlap | New-edge ratio | Homophily$(A)$ | Homophily$(A_E)$ | $\\Delta$ homophily | Mean $\\alpha^{(A)}$ | Mean $\\alpha^{(A_E)}$ | Weight entropy | View tendency | Source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    src = source_path(run_dir)
    for dataset in dataset_order:
        diag = structure[dataset]
        fusion = results[dataset]["dsafc"].get("fusion_diag")
        if not fusion:
            raise ValueError(f"Missing DSAFC fusion diagnostics for {dataset}.")
        hom_raw = float(diag["homophily_raw"])
        hom_ae = float(diag["homophily_ae"])
        alpha_raw = float(fusion["mean_alpha_raw"])
        alpha_ae = float(fusion["mean_alpha_ae"])
        lines.append(
            f"| {dataset} | {float(diag['edge_overlap_ratio']):.4f} | "
            f"{float(diag['new_edge_ratio']):.4f} | {hom_raw:.4f} | {hom_ae:.4f} | "
            f"{(hom_ae - hom_raw):+.4f} | {alpha_raw:.4f} | {alpha_ae:.4f} | "
            f"{float(fusion['mean_entropy']):.4f} | {dominance_label(alpha_raw, alpha_ae)} | {src} |"
        )
    return lines


def replace_diagnosis_tables(lines: list[str], table_lines: list[str]) -> list[str]:
    titles = (
        "Table 4-4 Structure Diagnosis",
        "Table 4-5 Fusion Weight Diagnosis",
        "Table 4-4 Structure-Fusion Reliability Diagnosis",
    )
    spans: list[tuple[int, int]] = []
    for title in titles:
        try:
            spans.append(find_section(lines, title))
        except ValueError:
            continue

    section = [
        "## Table 4-4 Structure-Fusion Reliability Diagnosis",
        "",
        "This merged table combines post-hoc graph-structure diagnostics and exported DSAFC fusion weights.",
        "The view tendency uses a 0.05 tolerance: nearly equal mean weights are reported as `Balanced`.",
        "",
        *table_lines,
        "",
    ]
    if spans:
        start = min(span[0] for span in spans)
        end = max(span[1] for span in spans)
        return lines[:start] + section + lines[end:]

    _, insert_at = find_section(lines, "Figure 4-1 ACC Plot Data")
    return lines[:insert_at] + ["", *section] + lines[insert_at:]


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    results = load_results(manifest)
    structure = load_structure(manifest)
    dataset_order = dataset_order_from_results(results)
    validate_complete(results, dataset_order)
    missing_structure = [dataset for dataset in dataset_order if dataset not in structure]
    if missing_structure:
        raise ValueError(f"Missing structure diagnostics: {', '.join(missing_structure)}")

    raw_path = args.raw_tables
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    lines = update_main_table(lines, results)
    lines = replace_table(lines, "Table 4-2 Ours Source Index", build_ours_source_table(results, run_dir, dataset_order))
    lines = replace_table(lines, "Table 4-2 Ours Std Availability", build_ours_std_table(results, run_dir, dataset_order))
    lines = replace_table(lines, "Table 4-3 Ablation Results", build_ablation_table(results, dataset_order))
    lines = replace_table(lines, "Table 4-3 DSAFC Source Index", build_dsafc_source_table(results, run_dir, dataset_order))
    lines = replace_table(lines, "Figure 4-1 ACC Plot Data", build_acc_plot_table(results, dataset_order))
    lines = replace_diagnosis_tables(lines, build_structure_fusion_table(structure, results, run_dir, dataset_order))

    output = "\n".join(lines).rstrip() + "\n"
    if args.dry_run:
        print(output)
    else:
        raw_path.write_text(output, encoding="utf-8")
        print(f"[UPDATED] {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
