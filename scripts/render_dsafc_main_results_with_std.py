from __future__ import annotations

import math
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW_TABLES = ROOT / "docs" / "DSAFC_raw_tables.md"
OUTPUT = ROOT / "assets" / "DSAFC_main_results_with_std.png"
METRICS = ("ACC", "NMI", "ARI", "F1")
STD_PLACEHOLDER = "--"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.render_dsafc_paper_figures import (
    FONT_SCALE,
    GROUP_GAP_RATIO,
    GROUP_LABEL_Y_OFFSET,
    OURS_COLUMN_FILL,
    TABLE_ROW_SCALE,
    _font,
    _parse_float,
    _wrap,
    collect_rank_styles,
    load_markdown_table,
    load_local_reproduction_pairs,
    ours_highlight_columns,
    reorder_main_table_columns,
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    }
)


def read_aux_markdown_table(section_title: str) -> tuple[list[str], list[list[str]]]:
    lines = RAW_TABLES.read_text(encoding="utf-8").splitlines()
    start = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == f"## {section_title}" or stripped == f"### {section_title}":
            start = idx + 1
            break
    if start is None:
        raise ValueError(f"Section not found: {section_title}")

    table_lines: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            if table_lines:
                break
            continue
        if stripped.startswith("## ") or stripped.startswith("### "):
            break
        if stripped.startswith("|"):
            table_lines.append(stripped)
        elif table_lines:
            break

    if len(table_lines) < 2:
        raise ValueError(f"No markdown table found under section: {section_title}")

    def split_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    return split_row(table_lines[0]), [split_row(line) for line in table_lines[2:]]


def _normalize_std_value(value: str) -> str | None:
    value = value.strip()
    if not value or value == "--":
        return None
    pm_match = re.search(r"(?:\u00b1|\u5364|\+/-)\s*([0-9]+(?:\.[0-9]+)?)", value)
    if pm_match:
        return f"{float(pm_match.group(1)):.2f}"
    std_match = re.search(r"\bstd\s*=\s*([0-9]+(?:\.[0-9]+)?)", value, flags=re.IGNORECASE)
    if std_match:
        return f"{float(std_match.group(1)):.2f}"
    numeric = _parse_float(value)
    return f"{numeric:.2f}" if numeric is not None else None


def _add_std_rows(
    index: dict[tuple[str, str, str], str],
    methods: list[str],
    rows_by_dataset: dict[str, dict[str, list[str]]],
) -> None:
    for dataset, metric_rows in rows_by_dataset.items():
        for metric in METRICS:
            stds = metric_rows[metric]
            for method, std in zip(methods, stds):
                normalized = _normalize_std_value(std)
                if normalized is not None:
                    index[(dataset, metric, method)] = normalized


def load_known_std_index() -> dict[tuple[str, str, str], str]:
    index: dict[tuple[str, str, str], str] = {}

    # DFCN Table 3 reports mean+std for its own covered datasets. These means
    # match the DFCN values retained from the aligned source tables where used.
    _add_std_rows(
        index,
        ["DFCN"],
        {
            "USPS": {
                "ACC": ["0.20"],
                "NMI": ["0.30"],
                "ARI": ["0.20"],
                "F1": ["0.20"],
            },
            "Reuters": {
                "ACC": ["0.20"],
                "NMI": ["0.40"],
                "ARI": ["0.40"],
                "F1": ["0.10"],
            },
            "Citeseer": {
                "ACC": ["0.20"],
                "NMI": ["0.20"],
                "ARI": ["0.30"],
                "F1": ["0.20"],
            },
            "ACM": {
                "ACC": ["0.20"],
                "NMI": ["0.40"],
                "ARI": ["0.40"],
                "F1": ["0.20"],
            },
            "DBLP": {
                "ACC": ["0.80"],
                "NMI": ["1.00"],
                "ARI": ["1.50"],
                "F1": ["0.80"],
            },
            "HHAR": {
                "ACC": ["0.10"],
                "NMI": ["0.10"],
                "ARI": ["0.10"],
                "F1": ["0.10"],
            },
        },
    )

    # SDCN Table 2 reports mean+std for USPS, Reuters, and Citeseer baselines.
    _add_std_rows(
        index,
        ["K-means", "AE", "DEC", "GAE", "DAEGC", "SDCN"],
        {
            "USPS": {
                "ACC": ["0.04", "0.03", "0.17", "0.33", "0.40", "0.19"],
                "NMI": ["0.05", "0.03", "0.25", "0.58", "0.24", "0.27"],
                "ARI": ["0.06", "0.05", "0.27", "0.55", "0.34", "0.24"],
                "F1": ["0.03", "0.03", "0.21", "0.43", "0.49", "0.18"],
            },
            "Reuters": {
                "ACC": ["0.01", "0.21", "0.13", "0.27", "0.13", "0.21"],
                "NMI": ["0.51", "0.29", "0.34", "0.41", "0.29", "0.21"],
                "ARI": ["0.38", "0.37", "0.14", "0.22", "0.18", "0.37"],
                "F1": ["2.43", "0.22", "0.22", "0.42", "0.13", "0.08"],
            },
            "Citeseer": {
                "ACC": ["3.17", "0.13", "0.20", "0.80", "1.39", "0.31"],
                "NMI": ["3.22", "0.08", "0.30", "0.65", "0.86", "0.32"],
                "ARI": ["3.02", "0.14", "0.36", "1.18", "1.24", "0.43"],
                "F1": ["3.53", "0.11", "0.17", "0.82", "1.32", "0.24"],
            },
        },
    )

    # Simple Contrastive Graph Clustering Table III gives mean+std for
    # Cora/Citeseer/AMAP/EAT/UAT and the SCGC-S column used in the table.
    _add_std_rows(
        index,
        ["K-means", "AE", "DEC", "SSGC", "GAE", "DAEGC", "SDCN", "DFCN", "AGE", "MVGRL", "SCGC-S"],
        {
            "Cora": {
                "ACC": ["2.71", "0.91", "0.26", "3.70", "2.11", "0.36", "2.83", "0.49", "1.83", "3.70", "0.88"],
                "NMI": ["3.43", "0.65", "0.34", "1.92", "2.97", "0.69", "1.91", "0.87", "1.42", "1.54", "0.72"],
                "ARI": ["1.95", "0.58", "0.42", "4.01", "1.65", "0.43", "3.24", "2.10", "2.14", "3.94", "1.59"],
                "F1": ["4.46", "1.05", "0.17", "5.53", "3.05", "0.57", "1.04", "0.50", "1.59", "1.86", "1.96"],
            },
            "Citeseer": {
                "ACC": ["3.17", "0.13", "0.20", "0.34", "0.80", "1.39", "0.31", "0.20", "0.24", "1.59", "0.77"],
                "NMI": ["3.22", "0.08", "0.30", "0.20", "0.65", "0.86", "0.32", "0.20", "0.53", "0.93", "0.45"],
                "ARI": ["3.02", "0.14", "0.36", "0.32", "1.18", "1.24", "0.43", "0.30", "0.41", "1.73", "1.13"],
                "F1": ["3.53", "0.11", "0.17", "0.27", "0.82", "1.32", "0.24", "0.20", "0.27", "2.17", "1.01"],
            },
            "AMAP": {
                "ACC": ["0.76", "0.08", "0.08", "0.19", "2.48", "0.23", "0.81", "0.23", "0.68", "3.12", "0.37"],
                "NMI": ["1.33", "0.30", "0.05", "0.15", "2.79", "0.45", "0.83", "1.21", "0.61", "3.94", "0.88"],
                "ARI": ["0.44", "0.47", "0.04", "0.47", "4.57", "0.24", "1.23", "0.74", "1.34", "2.34", "0.72"],
                "F1": ["0.51", "0.20", "0.12", "0.01", "1.76", "0.54", "1.49", "0.31", "0.93", "5.50", "0.97"],
            },
            "EAT": {
                "ACC": ["0.56", "2.32", "1.60", "0.45", "2.10", "0.15", "1.51", "0.19", "0.32", "0.71", "0.42"],
                "NMI": ["1.21", "2.80", "1.74", "0.21", "2.30", "0.06", "2.54", "0.41", "0.90", "1.08", "0.49"],
                "ARI": ["0.40", "2.65", "1.87", "0.04", "1.26", "0.08", "1.95", "0.18", "0.46", "1.30", "0.59"],
                "F1": ["0.92", "2.25", "1.28", "0.66", "3.26", "0.16", "3.10", "0.04", "0.40", "0.75", "0.46"],
            },
            "UAT": {
                "ACC": ["0.15", "1.14", "1.84", "0.81", "1.52", "0.49", "1.91", "0.09", "0.42", "1.38", "1.62"],
                "NMI": ["0.69", "1.60", "2.39", "0.18", "0.98", "0.44", "1.26", "0.41", "0.66", "0.94", "0.71"],
                "ARI": ["0.76", "2.02", "1.97", "0.27", "1.79", "0.51", "1.49", "0.23", "0.70", "1.46", "1.85"],
                "F1": ["0.22", "1.49", "1.51", "1.57", "1.52", "0.64", "3.54", "0.29", "0.73", "2.19", "0.87"],
            },
        },
    )

    # CCGC Table 1 reports the headline CCGC column with std.
    _add_std_rows(
        index,
        ["CCGC"],
        {
            "Cora": {"ACC": ["1.20"], "NMI": ["1.04"], "ARI": ["1.89"], "F1": ["2.79"]},
            "Citeseer": {"ACC": ["0.94"], "NMI": ["0.79"], "ARI": ["1.80"], "F1": ["2.06"]},
            "AMAP": {"ACC": ["0.41"], "NMI": ["0.48"], "ARI": ["0.66"], "F1": ["0.57"]},
            "EAT": {"ACC": ["0.66"], "NMI": ["0.87"], "ARI": ["0.41"], "F1": ["0.94"]},
            "UAT": {"ACC": ["1.11"], "NMI": ["1.92"], "ARI": ["2.09"], "F1": ["1.69"]},
        },
    )

    # SCGC Table 2 reports SCGC and SCGC*; the paper table names them
    # SCGC-N and SCGC-N* to disambiguate from SCGC-S.
    _add_std_rows(
        index,
        ["SCGC-N", "SCGC-N*"],
        {
            "USPS": {
                "ACC": ["0.08", "0.06"],
                "NMI": ["0.07", "0.10"],
                "ARI": ["0.11", "0.06"],
                "F1": ["0.05", "0.06"],
            },
            "Reuters": {
                "ACC": ["0.04", "0.00"],
                "NMI": ["0.05", "0.01"],
                "ARI": ["0.11", "0.01"],
                "F1": ["0.03", "0.01"],
            },
            "Citeseer": {
                "ACC": ["0.06", "0.01"],
                "NMI": ["0.10", "0.02"],
                "ARI": ["0.12", "0.02"],
                "F1": ["0.04", "0.01"],
            },
        },
    )

    return index


def _load_long_std_table(section_title: str) -> dict[tuple[str, str, str], str]:
    columns, rows = read_aux_markdown_table(section_title)
    column_index = {col.strip().lower(): idx for idx, col in enumerate(columns)}
    required = {"dataset", "metric", "method", "std"}
    if not required.issubset(column_index):
        return {}

    index: dict[tuple[str, str, str], str] = {}
    for row in rows:
        dataset = row[column_index["dataset"]]
        metric = row[column_index["metric"]]
        method = row[column_index["method"]]
        std = _normalize_std_value(row[column_index["std"]])
        if std is not None:
            index[(dataset, metric, method)] = std
    return index


def _load_wide_std_table(section_title: str) -> dict[tuple[str, str, str], str]:
    columns, rows = read_aux_markdown_table(section_title)
    lower_columns = {col.strip().lower() for col in columns}
    if {"dataset", "metric", "method", "std"}.issubset(lower_columns):
        return {}
    if "Dataset" not in columns or "Metric" not in columns:
        return {}

    dataset_col = columns.index("Dataset")
    metric_col = columns.index("Metric")
    index: dict[tuple[str, str, str], str] = {}
    current_dataset = ""
    for row in rows:
        if row[dataset_col]:
            current_dataset = row[dataset_col]
        dataset = current_dataset
        metric = row[metric_col]
        for col_idx, method in enumerate(columns):
            if col_idx in (dataset_col, metric_col):
                continue
            if method.lower() in {"source", "std source"}:
                continue
            std = _normalize_std_value(row[col_idx])
            if std is not None:
                index[(dataset, metric, method)] = std
    return index


def _load_ours_std_table() -> dict[tuple[str, str, str], str]:
    _, rows = read_aux_markdown_table("Table 4-2 Ours Std Availability")
    index: dict[tuple[str, str, str], str] = {}
    for row in rows:
        dataset = row[0]
        for metric, std in zip(METRICS, row[1:5]):
            normalized = _normalize_std_value(std)
            if normalized is not None:
                index[(dataset, metric, "Ours")] = normalized
    return index


def load_std_index() -> dict[tuple[str, str, str], str]:
    index = load_known_std_index()
    try:
        index.update(_load_long_std_table("Table 4-2 Main Std Availability"))
        index.update(_load_wide_std_table("Table 4-2 Main Std Availability"))
    except ValueError:
        pass
    try:
        index.update(_load_ours_std_table())
    except ValueError:
        pass
    return index


def build_display_rows(columns: list[str], base_rows: list[list[str]]) -> list[list[str]]:
    std_index = load_std_index()
    display_rows: list[list[str]] = []
    current_dataset = ""
    for row in base_rows:
        display_row = row.copy()
        if row[0]:
            current_dataset = row[0]
        dataset = current_dataset
        metric = row[1]
        for col_idx in range(2, len(columns)):
            value = row[col_idx]
            if _parse_float(value) is None:
                continue
            method = columns[col_idx]
            std = std_index.get((dataset, metric, method), STD_PLACEHOLDER)
            display_row[col_idx] = f"{value}±{std}"
        display_rows.append(display_row)
    return display_rows


def build_display_rows_with_local_marks(columns: list[str], base_rows: list[list[str]]) -> list[list[str]]:
    local_repro_pairs = load_local_reproduction_pairs()
    display_rows = build_display_rows(columns, base_rows)
    current_dataset = ""
    for row in display_rows:
        if row[0]:
            current_dataset = row[0]
        dataset = current_dataset
        for col_idx in range(2, len(columns)):
            method = columns[col_idx]
            if (dataset, method) not in local_repro_pairs:
                continue
            cell = row[col_idx]
            if _parse_float(cell) is None or cell.endswith("*"):
                continue
            if "±" in cell:
                value, std = cell.split("±", 1)
                row[col_idx] = f"{value}*±{std}"
            elif "¡À" in cell:
                value, std = cell.split("¡À", 1)
                row[col_idx] = f"{value}*¡À{std}"
            else:
                row[col_idx] = f"{cell}*"
    return display_rows


def render_table_with_display_rows(
    path: Path,
    columns: list[str],
    base_rows: list[list[str]],
    display_rows: list[list[str]],
    *,
    col_widths: list[float] | None = None,
    wrap_widths: list[int] | None = None,
    fig_width: float = 12.0,
    font_size: float = 13.0,
    header_size: float | None = None,
    row_unit: float = 0.44,
    header_unit: float = 0.62,
    vertical_after: tuple[int, ...] = (),
    group_col: int | None = None,
    bold_last_col: bool = False,
    bold_best: bool = False,
    underline_second: bool = False,
    numeric_start_col: int = 2,
    col_aligns: list[str] | None = None,
    highlight_col_fills: dict[int, str] | None = None,
) -> None:
    header_size = _font(header_size or font_size + 1.0)
    font_size = _font(font_size)
    row_unit *= TABLE_ROW_SCALE
    n_cols = len(columns)
    col_widths = col_widths or [1.0] * n_cols
    total_width = float(sum(col_widths))
    col_widths = [w / total_width for w in col_widths]
    wrap_widths = wrap_widths or [18] * n_cols
    col_aligns = col_aligns or ["center"] * n_cols
    highlight_col_fills = highlight_col_fills or {}

    wrapped_rows = [[_wrap(cell, wrap_widths[i]) for i, cell in enumerate(row)] for row in display_rows]
    row_heights = []
    for row in wrapped_rows:
        line_count = max((str(cell).count("\n") + 1 for cell in row), default=1)
        row_heights.append(row_unit * line_count)

    row_gaps_after = [0.0] * len(base_rows)
    if group_col is not None:
        for r_idx in range(len(base_rows) - 1):
            if base_rows[r_idx + 1][group_col]:
                row_gaps_after[r_idx] = row_unit * GROUP_GAP_RATIO

    top = 0.45
    bottom = 0.38
    body_height = sum(row_heights) + sum(row_gaps_after)
    total_height = top + header_unit + body_height + bottom
    fig_height = max(2.2, total_height * 0.54)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    x_edges = [0.0]
    for width in col_widths:
        x_edges.append(x_edges[-1] + width)
    x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(n_cols)]

    y_top = total_height - top
    y_header_bottom = y_top - header_unit
    y_bottom = y_header_bottom - body_height
    row_tops = []
    row_bottoms = []
    group_line_positions = []
    y_cursor = y_header_bottom
    for r_idx, row_h in enumerate(row_heights):
        row_tops.append(y_cursor)
        y_row_bottom = y_cursor - row_h
        row_bottoms.append(y_row_bottom)
        gap = row_gaps_after[r_idx]
        if gap:
            group_line_positions.append(y_row_bottom - gap / 2)
        y_cursor = y_row_bottom - gap

    for c_idx, fill in highlight_col_fills.items():
        if 0 <= c_idx < n_cols:
            ax.add_patch(
                plt.Rectangle(
                    (x_edges[c_idx], y_bottom),
                    x_edges[c_idx + 1] - x_edges[c_idx],
                    y_top - y_bottom,
                    facecolor=fill,
                    edgecolor="none",
                    zorder=0,
                )
            )

    ax.plot([0, 1], [y_top, y_top], color="black", lw=1.45, solid_capstyle="butt")
    ax.plot([0, 1], [y_header_bottom, y_header_bottom], color="black", lw=1.05, solid_capstyle="butt")
    ax.plot([0, 1], [y_bottom, y_bottom], color="black", lw=1.45, solid_capstyle="butt")
    for y_line in group_line_positions:
        ax.plot([0, 1], [y_line, y_line], color="black", lw=0.62, alpha=0.75, solid_capstyle="butt")

    for idx in vertical_after:
        x = x_edges[idx + 1]
        ax.plot([x, x], [y_top, y_bottom], color="black", lw=0.75, alpha=0.85)

    y = y_top - header_unit / 2
    for i, col in enumerate(columns):
        ax.text(
            x_centers[i],
            y,
            col,
            ha="center",
            va="center",
            fontsize=header_size,
            fontweight="bold",
        )

    bold_cells, underline_cells = collect_rank_styles(
        base_rows,
        numeric_start_col=numeric_start_col,
        bold_best=bold_best,
        underline_second=underline_second,
    )

    group_spans: list[tuple[str, int, int]] = []
    if group_col is not None:
        group_label = None
        group_start = 0
        for r_idx, row in enumerate(base_rows):
            if row[group_col]:
                if group_label is not None:
                    group_spans.append((group_label, group_start, r_idx))
                group_label = row[group_col]
                group_start = r_idx
        if group_label is not None:
            group_spans.append((group_label, group_start, len(base_rows)))

    for r_idx, row in enumerate(wrapped_rows):
        raw_row = base_rows[r_idx]
        y_mid = (row_tops[r_idx] + row_bottoms[r_idx]) / 2
        for c_idx, cell in enumerate(row):
            if group_col is not None and c_idx == group_col:
                continue
            weight = "normal"
            if c_idx == 0 and raw_row[c_idx]:
                weight = "bold"
            if bold_last_col and c_idx == n_cols - 1:
                weight = "bold"
            if (r_idx, c_idx) in bold_cells:
                weight = "bold"
            ha = col_aligns[c_idx]
            if ha == "left":
                x = x_edges[c_idx] + 0.012
            elif ha == "right":
                x = x_edges[c_idx + 1] - 0.012
            else:
                x = x_centers[c_idx]
            ax.text(
                x,
                y_mid,
                cell,
                ha=ha,
                va="center",
                fontsize=font_size,
                fontweight=weight,
                linespacing=1.18,
            )
            if (r_idx, c_idx) in underline_cells:
                cell_width = x_edges[c_idx + 1] - x_edges[c_idx]
                if ha == "left":
                    x0 = x_edges[c_idx] + 0.012
                    x1 = min(x_edges[c_idx + 1] - 0.012, x0 + cell_width * 0.56)
                elif ha == "right":
                    x1 = x_edges[c_idx + 1] - 0.012
                    x0 = max(x_edges[c_idx] + 0.012, x1 - cell_width * 0.56)
                else:
                    x0 = x_centers[c_idx] - cell_width * 0.23
                    x1 = x_centers[c_idx] + cell_width * 0.23
                y_line = y_mid - row_heights[r_idx] * 0.18
                ax.plot([x0, x1], [y_line, y_line], color="black", lw=1.05, solid_capstyle="butt")

    for label, start, end in group_spans:
        c_idx = group_col
        assert c_idx is not None
        y_mid = (row_tops[start] + row_bottoms[end - 1]) / 2 + GROUP_LABEL_Y_OFFSET
        cell = _wrap(label, wrap_widths[c_idx])
        ha = col_aligns[c_idx]
        if ha == "left":
            x = x_edges[c_idx] + 0.012
        elif ha == "right":
            x = x_edges[c_idx + 1] - 0.012
        else:
            x = x_centers[c_idx]
        ax.text(
            x,
            y_mid,
            cell,
            ha=ha,
            va="center",
            fontsize=font_size,
            fontweight="bold",
            linespacing=1.18,
        )

    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    columns, base_rows = load_markdown_table("Table 4-2 Main Clustering Results", group_col=0)
    columns, base_rows = reorder_main_table_columns(columns, base_rows)
    display_rows = build_display_rows_with_local_marks(columns, base_rows)
    render_table_with_display_rows(
        OUTPUT,
        columns,
        base_rows,
        display_rows,
        highlight_col_fills=ours_highlight_columns(columns),
        col_widths=[1.25, 0.85] + [1.0] * (len(columns) - 2),
        wrap_widths=[14, 8] + [12] * (len(columns) - 2),
        fig_width=30,
        font_size=11.7,
        header_size=12.8,
        row_unit=0.36,
        header_unit=0.54,
        vertical_after=(0, 1),
        group_col=0,
        bold_best=True,
        underline_second=True,
        numeric_start_col=2,
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
