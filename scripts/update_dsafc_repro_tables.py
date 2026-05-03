from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_TABLES = ROOT / "docs" / "DSAFC_raw_tables.md"
METRICS = ("ACC", "NMI", "ARI", "F1")
TARGET_DATASETS = ("Reuters", "USPS", "UAT", "AMAP", "EAT", "Cora", "Citeseer")
LITERATURE_STD_RECORDS = [
    {
        "dataset": dataset,
        "method": "DFCN",
        "status": "ok",
        "metrics": {
            "acc": {"std": acc},
            "nmi": {"std": nmi},
            "ari": {"std": ari},
            "f1": {"std": f1},
        },
        "source": "papers/Tu 等 - 2021 - Deep Fusion Clustering Network.pdf, Table 3",
        "config": {"source_type": "literature"},
    }
    for dataset, (acc, nmi, ari, f1) in {
        "Reuters": (0.20, 0.40, 0.40, 0.10),
        "USPS": (0.20, 0.30, 0.20, 0.20),
        "Citeseer": (0.20, 0.20, 0.30, 0.20),
    }.items()
]
REPRO_SOURCES = [
    ROOT / "other_projects" / "CCGC" / "results" / "ccgc_reut_usps_std_summary_latest.json",
    ROOT / "other_projects" / "SSGC" / "NodeClustering" / "results" / "ssgc_reut_usps_std_summary_latest.json",
    ROOT / "other_projects" / "SCGC-S" / "results" / "scgcs_reut_usps_std_summary_latest.json",
    ROOT / "other_projects" / "AGE" / "results" / "age_extended_std_summary_latest.json",
    ROOT / "other_projects" / "MVGRL" / "results" / "mvgrl_node_extended_std_summary_latest.json",
    ROOT / "other_projects" / "SCGC-N" / "results" / "scgcn_uat_eat_amap_cora_std_summary_latest.json",
    ROOT / "other_projects" / "GLAC-GCN" / "results" / "glacgcn_uat_eat_amap_std_summary_latest.json",
    ROOT / "other_projects" / "GLAC-GCN" / "results" / "glacgcn_paper_std_summary_latest.json",
]


def split_md_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def format_md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def find_section(lines: list[str], title: str) -> tuple[int, int] | None:
    start = None
    for idx, line in enumerate(lines):
        if line.strip() in {f"## {title}", f"### {title}"}:
            start = idx
            break
    if start is None:
        return None
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            end = idx
            break
    return start, end


def find_table(lines: list[str], section_title: str) -> tuple[int, int]:
    section = find_section(lines, section_title)
    if section is None:
        raise ValueError(f"Section not found: {section_title}")
    start, end = section
    table_start = None
    table_end = None
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


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = list(LITERATURE_STD_RECORDS)
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            record = dict(record)
            record["_summary_path"] = str(path)
            records.append(record)
    return records


def is_production_record(record: dict[str, Any]) -> bool:
    method = str(record.get("method", ""))
    config = record.get("config", {}) or {}
    if method == "CCGC":
        return int(config.get("epochs", 0)) >= 400 and int(config.get("runs", 0)) >= 10
    if method == "SCGC-S":
        return int(config.get("epochs", 0)) >= 400 and int(config.get("seeds", 0)) >= 10
    if method == "SSGC":
        return int(config.get("rep", 0)) >= 10
    if method == "AGE":
        return int(config.get("epochs", 0)) >= 400 and int(config.get("runs", 0)) >= 10
    if method == "MVGRL":
        return int(config.get("runs", 0)) >= 10
    if method in {"SCGC-N", "SCGC-N*"}:
        return int(config.get("epochs", 0)) >= 200 and int(config.get("iterations", 0)) >= 10
    if method == "GLAC-GCN":
        return int(config.get("epochs", 0)) >= 200 and int(config.get("runs", 0)) >= 10
    return True


def normalize_dataset(dataset: str) -> str:
    mapping = {
        "REUT": "Reuters",
        "REUTERS": "Reuters",
        "USPS": "USPS",
        "UAT": "UAT",
        "AMAP": "AMAP",
        "EAT": "EAT",
        "CORA": "Cora",
        "CITE": "Citeseer",
        "CITESEER": "Citeseer",
    }
    return mapping.get(dataset.upper(), dataset)


def metric_value(record: dict[str, Any], metric: str, key: str) -> float | None:
    row = record.get("metrics", {}).get(metric.lower())
    if not row:
        return None
    value = row.get(key)
    return None if value is None else float(value)


def successful_index(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        dataset = normalize_dataset(str(record.get("dataset", "")))
        method = str(record.get("method", ""))
        if dataset not in TARGET_DATASETS:
            continue
        for metric in METRICS:
            if metric_value(record, metric, "mean") is not None:
                index[(dataset, metric, method)] = record
    return index


def update_main_table(lines: list[str], records: list[dict[str, Any]]) -> list[str]:
    table_start, table_end = find_table(lines, "Table 4-2 Main Clustering Results")
    header = split_md_row(lines[table_start])
    dataset_col = header.index("Dataset")
    metric_col = header.index("Metric")
    col_by_method = {method: idx for idx, method in enumerate(header)}
    record_index = successful_index(records)

    current_dataset = ""
    updated = lines[:]
    for line_idx in range(table_start + 2, table_end):
        row = split_md_row(lines[line_idx])
        if row[dataset_col]:
            current_dataset = row[dataset_col]
        dataset = current_dataset
        metric = row[metric_col]
        for method, col_idx in col_by_method.items():
            if method in {"Dataset", "Metric"}:
                continue
            record = record_index.get((dataset, metric, method))
            if not record:
                continue
            value = metric_value(record, metric, "mean")
            if value is not None:
                row[col_idx] = f"{value:.2f}"
        updated[line_idx] = format_md_row(row)
    return updated


def build_std_section(records: list[dict[str, Any]]) -> list[str]:
    rows = [
        "### Table 4-2 Main Std Availability",
        "",
        "This auxiliary table is generated from manual reproduction logs under `other_projects`. Missing standard deviations are left out here, so the plotting script renders numeric cells as `value±--`.",
        "",
        "| Dataset | Metric | Method | Std | Source |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for record in records:
        dataset = normalize_dataset(str(record.get("dataset", "")))
        method = str(record.get("method", ""))
        if dataset not in TARGET_DATASETS:
            continue
        source = record.get("source") or record.get("_summary_path") or ""
        if record.get("status") != "ok":
            continue
        for metric in METRICS:
            std = metric_value(record, metric, "std")
            if std is None:
                continue
            if source.startswith("`") and source.endswith("`"):
                source_cell = source
            else:
                source_cell = f"`{source}`"
            rows.append(f"| {dataset} | {metric} | {method} | {std:.2f} | {source_cell} |")
    return rows


def replace_or_insert_std_section(lines: list[str], section_lines: list[str]) -> list[str]:
    section = find_section(lines, "Table 4-2 Main Std Availability")
    if section is not None:
        start, end = section
        return lines[:start] + section_lines + [""] + lines[end:]

    ours = find_section(lines, "Table 4-2 Ours Std Availability")
    if ours is not None:
        _, end = ours
        return lines[:end] + [""] + section_lines + lines[end:]

    table_start, table_end = find_table(lines, "Table 4-2 Main Clustering Results")
    return lines[:table_end] + [""] + section_lines + lines[table_end:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Update DSAFC main/std tables from unified reproduction JSON logs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_records(REPRO_SOURCES)
    records = [record for record in records if is_production_record(record)]
    if not records:
        raise RuntimeError("No unified reproduction JSON logs found. Run the other_projects one-click scripts first.")

    lines = RAW_TABLES.read_text(encoding="utf-8").splitlines()
    lines = update_main_table(lines, records)
    lines = replace_or_insert_std_section(lines, build_std_section(records))
    output = "\n".join(lines).rstrip() + "\n"
    if args.dry_run:
        print(output)
    else:
        RAW_TABLES.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
