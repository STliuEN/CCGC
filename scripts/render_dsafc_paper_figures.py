from __future__ import annotations

import csv
import math
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FONT_SCALE = 0.70
TABLE_ROW_SCALE = 1.12
GROUP_GAP_RATIO = 0.42
GROUP_LABEL_Y_OFFSET = -0.025
COMPETITIVE_RIGHT_BLOCK = ("DFCN", "CCGC", "SCGC-S", "GLAC-GCN", "SCGC-N", "SCGC-N*", "Ours")
MAIN_RED = "#d7191c"
RAW_BLUE = "#3f6fbd"
AE_ORANGE = "#df7d42"
FUSION_GREEN = "#73a96d"
CLUSTER_YELLOW = "#e6b94b"
SHARED_RED = "#d88f89"
NEUTRAL_LINE = "#8a8a8a"
SOFT_BLUE = "#e7eff9"
SOFT_ORANGE = "#fbefe7"
SOFT_GREEN = "#e6f2e8"
SOFT_YELLOW = "#fff5d8"
SOFT_GREY = "#f3f5f7"
OURS_COLUMN_FILL = SOFT_GREEN
LOCAL_REPRO_SOURCE_SECTION = "Table 4-2 Main Std Availability"
LOCAL_REPRO_SOURCE_TOKEN = "other_projects"
LOCAL_REPRO_EXCLUDE_METHODS = {"Ours"}
HYPERPARAM_SENSITIVITY_RUN = (
    ROOT
    / "experiment_output"
    / "hyperparameter_sensitivity_ofat"
    / "final_four_current_contract"
)
REQUIRED_HYPERPARAM_SENSITIVITY_PARAMS = {"t", "k_E", "dcgl_neg_weight", "fusion_temp"}
STRUCTURAL_UNCERTAINTY_ROOT = ROOT / "experiment_output" / "structural_uncertainty_robustness"
FINAL_DATASETS = {"reut", "uat", "amap", "usps", "cora", "cite"}
ACTIVE_DATASET_LABELS = ("Reuters", "UAT", "AMAP", "USPS", "Cora", "Citeseer")
ARCHIVED_DATASET_LABELS = {"EAT"}
ARCHIVED_METHODS = {"SCGC-N", "SCGC-N*"}
PRIMARY_BAR = "#505050"
SECONDARY_BAR = "#8d8d8d"
TERTIARY_BAR = "#d8d8d8"
FRAME_COLOR = "#000000"
GRID_COLOR = "#c7d5e9"
AXIS_COLOR = "#000000"
TEXT_COLOR = "#111111"
PLOT_COLORS = (
    RAW_BLUE,
    AE_ORANGE,
    FUSION_GREEN,
    CLUSTER_YELLOW,
    MAIN_RED,
    "#6ea8c7",
    "#7f8790",
    "#8a78a8",
)
ABLATION_COLORS = (
    RAW_BLUE,
    AE_ORANGE,
    "#95a9b8",
    CLUSTER_YELLOW,
    FUSION_GREEN,
)
CLUSTER_COLORS = (
    RAW_BLUE,
    AE_ORANGE,
    FUSION_GREEN,
    CLUSTER_YELLOW,
    MAIN_RED,
    "#66a9c9",
    "#7f8790",
    "#8a78a8",
    "#b98b5f",
    "#6f9c8b",
)
RANK_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "dsafc_rank",
    [SOFT_GREEN, SOFT_BLUE, SOFT_YELLOW, SOFT_ORANGE],
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": AXIS_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "axes.unicode_minus": False,
        "legend.frameon": False,
    }
)


def style_framed_axes(ax: plt.Axes, *, grid_axis: str | None = "y") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID_COLOR, linestyle="--", linewidth=0.55, alpha=0.72)
        ax.set_axisbelow(True)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(AXIS_COLOR)
        ax.spines[spine].set_linewidth(0.62)
    ax.tick_params(axis="both", colors=TEXT_COLOR, width=0.62)


def style_legend(legend, *, linewidth: float = 0.75) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#b5b5b5")
    frame.set_linewidth(linewidth)
    frame.set_alpha(1.0)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)


def _parse_float(value: str) -> float | None:
    match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", value or "")
    return float(match.group(1)) if match else None


def _parse_mean_std(value: str) -> tuple[float | None, float | None]:
    value = value or ""
    mean = _parse_float(value)
    std_match = re.search(r"(?:±|\+-)\s*([0-9]+(?:\.[0-9]+)?)", value)
    std = float(std_match.group(1)) if std_match else None
    return mean, std


def _wrap(text: str, width: int) -> str:
    text = str(text)
    if "\n" in text or len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def _font(size: float) -> float:
    return size * FONT_SCALE


def load_local_reproduction_pairs() -> set[tuple[str, str]]:
    raw_path = ROOT / "docs" / "DSAFC_raw_tables.md"
    lines = raw_path.read_text(encoding="utf-8").splitlines()

    start = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == f"## {LOCAL_REPRO_SOURCE_SECTION}" or stripped == f"### {LOCAL_REPRO_SOURCE_SECTION}":
            start = idx + 1
            break
    if start is None:
        return set()

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
        return set()

    def split_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    columns = split_row(table_lines[0])
    rows = [split_row(line) for line in table_lines[2:]]
    column_index = {col.strip().lower(): idx for idx, col in enumerate(columns)}
    required = {"dataset", "method", "source"}
    if not required.issubset(column_index):
        return set()

    marked_pairs: set[tuple[str, str]] = set()
    for row in rows:
        dataset = row[column_index["dataset"]].strip()
        method = row[column_index["method"]].strip()
        source = row[column_index["source"]].replace("\\", "/").lower()
        if not dataset or not method or method in LOCAL_REPRO_EXCLUDE_METHODS:
            continue
        if LOCAL_REPRO_SOURCE_TOKEN in source:
            marked_pairs.add((dataset, method))
    return marked_pairs


def mark_local_reproduction_cells(columns: list[str], rows: list[list[str]]) -> list[list[str]]:
    marked_pairs = load_local_reproduction_pairs()
    if not marked_pairs:
        return [row.copy() for row in rows]

    display_rows: list[list[str]] = []
    current_dataset = ""
    for row in rows:
        display_row = row.copy()
        if row[0]:
            current_dataset = row[0]
        dataset = current_dataset
        for c_idx in range(2, len(columns)):
            method = columns[c_idx]
            cell = display_row[c_idx]
            if (dataset, method) not in marked_pairs:
                continue
            if _parse_float(cell) is None or cell.endswith("*"):
                continue
            display_row[c_idx] = f"{cell}*"
        display_rows.append(display_row)
    return display_rows


def render_table(
    path: Path,
    columns: list[str],
    rows: list[list[str]],
    *,
    display_rows: list[list[str]] | None = None,
    rank_rows: list[list[str]] | None = None,
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
    display_rows = display_rows or rows
    rank_rows = rank_rows or rows
    highlight_col_fills = highlight_col_fills or {}

    wrapped_rows = [[_wrap(cell, wrap_widths[i]) for i, cell in enumerate(row)] for row in display_rows]
    row_heights = []
    for row in wrapped_rows:
        line_count = max((str(cell).count("\n") + 1 for cell in row), default=1)
        row_heights.append(row_unit * line_count)
    row_gaps_after = [0.0] * len(rows)
    if group_col is not None:
        for r_idx in range(len(rows) - 1):
            if rows[r_idx + 1][group_col]:
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
        rank_rows,
        numeric_start_col=numeric_start_col,
        bold_best=bold_best,
        underline_second=underline_second,
    )

    group_spans: list[tuple[str, int, int]] = []
    if group_col is not None:
        group_label = None
        group_start = 0
        for r_idx, row in enumerate(rows):
            if row[group_col]:
                if group_label is not None:
                    group_spans.append((group_label, group_start, r_idx))
                group_label = row[group_col]
                group_start = r_idx
        if group_label is not None:
            group_spans.append((group_label, group_start, len(rows)))

    for r_idx, row in enumerate(wrapped_rows):
        raw_row = rows[r_idx]
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
                weight = "heavy"
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
                y_line = y_mid - row_heights[r_idx] * 0.28
                ax.plot([x0, x1], [y_line, y_line], color="black", lw=1.35, solid_capstyle="butt")

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


def collect_rank_styles(
    rows: list[list[str]],
    *,
    numeric_start_col: int = 2,
    bold_best: bool = False,
    underline_second: bool = False,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    bold_cells: set[tuple[int, int]] = set()
    underline_cells: set[tuple[int, int]] = set()
    for r_idx, row in enumerate(rows):
        scored: list[tuple[int, float]] = []
        for c_idx in range(numeric_start_col, len(row)):
            val = _parse_float(row[c_idx])
            if val is not None:
                scored.append((c_idx, val))
        if not scored:
            continue
        unique_values = sorted({val for _, val in scored}, reverse=True)
        best = unique_values[0]
        second = unique_values[1] if len(unique_values) > 1 else None
        for c_idx, val in scored:
            if bold_best and math.isclose(val, best, rel_tol=0.0, abs_tol=1e-9):
                bold_cells.add((r_idx, c_idx))
            if underline_second and second is not None and math.isclose(val, second, rel_tol=0.0, abs_tol=1e-9):
                underline_cells.add((r_idx, c_idx))
    return bold_cells, underline_cells


def reorder_main_table_columns(columns: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    if len(columns) <= 2:
        return columns, rows
    methods = columns[2:]
    trailing = [method for method in COMPETITIVE_RIGHT_BLOCK if method in methods]
    leading = [method for method in methods if method not in trailing]
    reordered_methods = leading + trailing
    method_to_idx = {method: idx for idx, method in enumerate(methods, start=2)}
    new_indices = [0, 1] + [method_to_idx[method] for method in reordered_methods]
    new_columns = [columns[idx] for idx in new_indices]
    new_rows = [[row[idx] for idx in new_indices] for row in rows]
    return new_columns, new_rows


def filter_active_table(
    columns: list[str],
    rows: list[list[str]],
    *,
    drop_methods: set[str] | None = None,
    group_col: int | None = 0,
) -> tuple[list[str], list[list[str]]]:
    drop_methods = drop_methods or set()
    keep_indices = [idx for idx, col in enumerate(columns) if idx < 2 or col not in drop_methods]
    filtered_columns = [columns[idx] for idx in keep_indices]
    filtered_rows: list[list[str]] = []
    current_dataset = ""
    for row in rows:
        dataset_cell = row[group_col] if group_col is not None else row[0]
        if dataset_cell:
            current_dataset = dataset_cell
        if current_dataset in ARCHIVED_DATASET_LABELS:
            continue
        filtered_rows.append([row[idx] for idx in keep_indices])
    return filtered_columns, filtered_rows


def ours_highlight_columns(columns: list[str]) -> dict[int, str]:
    try:
        ours_idx = columns.index("Ours")
    except ValueError:
        return {}
    return {ours_idx: OURS_COLUMN_FILL}


def compute_method_average_ranks(columns: list[str], rows: list[list[str]]) -> dict[str, float]:
    methods = columns[2:]
    rank_history: dict[str, list[int]] = {method: [] for method in methods}
    for row in rows:
        values = {method: _parse_float(value) for method, value in zip(methods, row[2:])}
        for method, value in values.items():
            if value is None:
                continue
            rank = 1 + sum(
                1
                for other_method, other_value in values.items()
                if other_method != method and other_value is not None and other_value > value
            )
            rank_history[method].append(rank)
    return {
        method: float(sum(ranks)) / float(len(ranks))
        for method, ranks in rank_history.items()
        if ranks
    }


def compute_dataset_average_ranks(columns: list[str], rows: list[list[str]]) -> dict[str, dict[str, float]]:
    methods = columns[2:]
    dataset_history: dict[str, dict[str, list[int]]] = {}
    current_dataset = ""
    for row in rows:
        if row[0]:
            current_dataset = row[0]
        dataset_history.setdefault(current_dataset, {method: [] for method in methods})
        values = {method: _parse_float(value) for method, value in zip(methods, row[2:])}
        for method, value in values.items():
            if value is None:
                continue
            rank = 1 + sum(
                1
                for other_method, other_value in values.items()
                if other_method != method and other_value is not None and other_value > value
            )
            dataset_history[current_dataset][method].append(rank)
    dataset_avg: dict[str, dict[str, float]] = {}
    for dataset, history in dataset_history.items():
        dataset_avg[dataset] = {
            method: float(sum(ranks)) / float(len(ranks))
            for method, ranks in history.items()
            if ranks
        }
    return dataset_avg


def compute_win_tie_loss_against_ours(
    columns: list[str],
    rows: list[list[str]],
    *,
    tie_tol: float = 0.05,
) -> dict[str, tuple[int, int, int]]:
    if "Ours" not in columns:
        return {}
    methods = [method for method in columns[2:] if method != "Ours"]
    ours_idx = columns.index("Ours")
    summary: dict[str, list[int]] = {method: [0, 0, 0] for method in methods}
    for row in rows:
        ours_value = _parse_float(row[ours_idx])
        if ours_value is None:
            continue
        for method in methods:
            method_value = _parse_float(row[columns.index(method)])
            if method_value is None:
                continue
            if ours_value > method_value + tie_tol:
                summary[method][0] += 1
            elif method_value > ours_value + tie_tol:
                summary[method][2] += 1
            else:
                summary[method][1] += 1
    return {method: tuple(values) for method, values in summary.items()}


def build_rank_summary_rows(columns: list[str], rows: list[list[str]]) -> list[list[str]]:
    avg_ranks = compute_method_average_ranks(columns, rows)
    wtl = compute_win_tie_loss_against_ours(columns, rows)
    ordered = sorted(columns[2:], key=lambda method: (avg_ranks.get(method, float("inf")), method))
    summary_rows: list[list[str]] = []
    for method in ordered:
        wins, ties, losses = wtl.get(method, (0, 0, 0)) if method != "Ours" else (0, 0, 0)
        summary_rows.append(
            [
                method,
                f"{avg_ranks.get(method, float('nan')):.2f}",
                "-" if method == "Ours" else f"{wins}/{ties}/{losses}",
            ]
        )
    return summary_rows


def render_average_rank_chart(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    avg_ranks = compute_method_average_ranks(columns, rows)
    ordered = sorted(avg_ranks.items(), key=lambda item: item[1])
    methods = [item[0] for item in ordered]
    scores = [item[1] for item in ordered]
    colors = []
    for method in methods:
        if method == "Ours":
            colors.append(PLOT_COLORS[0])
        elif method in {"CCGC", "SCGC-S", "GLAC-GCN", "SCGC-N"}:
            colors.append(PLOT_COLORS[3])
        else:
            colors.append("#b9c1c8")

    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    y = np.arange(len(methods))
    bars = ax.barh(y, scores, color=colors, edgecolor=AXIS_COLOR, linewidth=0.35, height=0.72)
    ax.invert_yaxis()
    ax.set_yticks(y, labels=methods)
    ax.set_xlabel("Average rank (lower is better)", fontsize=_font(15.0), fontweight="bold")
    ax.set_title("Overall Average Rank Across 28 Tasks", fontsize=_font(17.0), fontweight="bold", pad=10)
    style_framed_axes(ax, grid_axis="x")
    for tick, method in zip(ax.get_yticklabels(), methods):
        if method == "Ours":
            tick.set_fontweight("bold")
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center",
            ha="left",
            fontsize=_font(12.2),
            color="#333333",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_dataset_rank_heatmap(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    dataset_avg = compute_dataset_average_ranks(columns, rows)
    datasets = list(dataset_avg.keys())
    methods = [method for method in ("CCGC", "SCGC-S", "GLAC-GCN", "SCGC-N", "Ours") if method in columns]
    matrix = np.array([[dataset_avg[dataset][method] for method in methods] for dataset in datasets], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(matrix, cmap=RANK_HEATMAP_CMAP, aspect="auto", vmin=np.min(matrix), vmax=np.max(matrix))
    ax.set_xticks(np.arange(len(methods)), labels=methods)
    ax.set_yticks(np.arange(len(datasets)), labels=datasets)
    ax.set_title("Dataset-wise Average Rank Heatmap", fontsize=_font(16.5), fontweight="bold", pad=10)
    style_framed_axes(ax, grid_axis=None)
    for tick, method in zip(ax.get_xticklabels(), methods):
        if method == "Ours":
            tick.set_fontweight("bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value <= (float(np.min(matrix)) + float(np.max(matrix))) / 2.0 else "#222222"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=_font(11.5), color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Average rank (lower is better)", fontsize=_font(12.0))
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def select_columns(
    columns: list[str],
    rows: list[list[str]],
    selected: list[str],
    *,
    display_columns: list[str] | None = None,
) -> tuple[list[str], list[list[str]]]:
    column_to_idx = {column: idx for idx, column in enumerate(columns)}
    indices = [column_to_idx[column] for column in selected]
    return display_columns or selected, [[row[idx] for idx in indices] for row in rows]


def render_ablation_acc_chart(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    datasets = [row[0] for row in rows]
    variants = columns[1:]
    parsed = [[_parse_mean_std(value) for value in row[1:]] for row in rows]
    matrix = np.array([[mean or 0.0 for mean, _ in row] for row in parsed], dtype=float)

    n_cols = 3
    n_rows = int(math.ceil(len(datasets) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0, 4.35), sharey=False, squeeze=False)
    axes_arr = axes.ravel()
    x = np.arange(len(variants))
    y_max = max(90.0, float(np.max(matrix)) + 6.0)
    legend_handles = []
    legend_labels = []

    for idx_ax, (ax, dataset, values) in enumerate(zip(axes_arr, datasets, matrix)):
        bars = ax.bar(
            x,
            values,
            width=0.72,
            color=[ABLATION_COLORS[idx % len(ABLATION_COLORS)] for idx in range(len(variants))],
            edgecolor="none",
            linewidth=0.0,
        )
        if not legend_handles:
            legend_handles = list(bars)
            legend_labels = list(variants)
        ax.set_title(dataset, fontsize=_font(11.2), fontweight="bold", pad=3)
        ax.set_xticks(x, labels=variants)
        ax.tick_params(axis="x", labelrotation=24, labelsize=_font(7.5), pad=1.0)
        ax.tick_params(axis="y", labelsize=_font(7.8), pad=1.0)
        ax.set_ylabel("ACC (%)", fontsize=_font(8.6), fontweight="bold", labelpad=1.0)
        ax.set_ylim(0, y_max)
        ax.set_box_aspect(1.0)
        style_framed_axes(ax, grid_axis="y")

    for ax in axes_arr[len(datasets):]:
        ax.axis("off")

    legend = fig.legend(
        legend_handles,
        legend_labels,
        ncol=len(variants),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        frameon=True,
        fontsize=_font(7.8),
        handlelength=0.95,
        handletextpad=0.25,
        columnspacing=0.48,
        borderpad=0.2,
        labelspacing=0.15,
    )
    style_legend(legend)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.08, top=0.90, wspace=0.02, hspace=0.30)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def render_method_positioning_table(path: Path) -> None:
    rows = [
        ["Deep graph clustering", "DAEGC, SDCN, DFCN", "No / partial", "No", "No / partial", "No"],
        ["Contrastive graph clustering", "MVGRL, CCGC, SCGC", "No / partial", "No", "No", "Partial"],
        ["Structure-refined graph clustering", "GLAC-GCN", "Yes", "No", "Partial", "Partial"],
        ["Multi-view graph clustering", "One2Multi, MAGCN, DMVCJ", "Partial", "No", "Mostly global", "No / partial"],
        ["Reliable pseudo-label / quality-aware MVC", "UPS, SPICE, HCPG-MVC, AWDC-MVC", "No / partial", "No", "Partial", "Sample / graph level"],
        ["DSAFC", "This paper", "Yes", "Yes", "Yes", "Yes"],
    ]
    render_table(
        path,
        [
            "Category",
            "Representative methods",
            "Structure\nrefinement",
            "Homologous\ndual structure",
            "Node-wise\nreliability fusion",
            "Pseudo-cluster\nreliability",
        ],
        rows,
        col_widths=[2.4, 2.6, 1.25, 1.45, 1.55, 1.45],
        wrap_widths=[22, 24, 14, 14, 15, 16],
        fig_width=14.6,
        font_size=11.8,
        header_size=12.8,
        row_unit=0.48,
        vertical_after=(0, 1),
        bold_last_col=False,
        col_aligns=["left", "left", "center", "center", "center", "center"],
    )


def render_dataset_statistics_table(path: Path, *, drop_datasets: set[str] | None = None) -> None:
    columns, rows = load_markdown_table("Table 4-1 Dataset Statistics")
    drop_datasets = drop_datasets or set()
    rows = [row for row in rows if row and row[0] not in drop_datasets]
    render_table(
        path,
        columns,
        rows,
        col_widths=[1.7, 1.25, 1.55, 2.1, 1.65, 1.0],
        fig_width=11.0,
        font_size=15.5,
        header_size=18,
        row_unit=0.48,
    )


def load_hyperparameter_sensitivity_rows() -> tuple[list[str], list[list[str]], list[dict[str, float | str]]]:
    aggregate_path = HYPERPARAM_SENSITIVITY_RUN / "aggregate.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing hyperparameter sensitivity aggregate: {aggregate_path}")

    label_map = {
        "t": r"$t$",
        "k_E": r"$k_E$",
        "dcgl_neg_weight": r"$\lambda_{\mathrm{sep}}$",
        "fusion_temp": r"$\tau_f$",
    }
    order = {"t": 0, "k_E": 1, "dcgl_neg_weight": 2, "fusion_temp": 3}

    records: list[dict[str, float | str]] = []
    with aggregate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            param = row["param"]
            if param not in order:
                continue
            records.append(
                {
                    "param": param,
                    "label": label_map[param],
                    "value": float(row["value"]),
                    "avg_acc": float(row["avg_acc"]),
                    "avg_acc_std": float(row.get("avg_acc_std") or 0.0),
                    "avg_nmi": float(row["avg_nmi"]),
                    "avg_ari": float(row["avg_ari"]),
                    "avg_f1": float(row["avg_f1"]),
                    "avg_score": float(row.get("avg_score") or 0.0),
                }
            )
    present_params = {str(item["param"]) for item in records}
    missing_params = REQUIRED_HYPERPARAM_SENSITIVITY_PARAMS - present_params
    if missing_params:
        missing = ", ".join(sorted(missing_params))
        raise ValueError(
            "Hyperparameter sensitivity run is incomplete. "
            f"Missing parameter(s): {missing}. Expected t, k_E, "
            f"dcgl_neg_weight(lambda_sep), and fusion_temp in {aggregate_path}."
        )

    per_dataset_path = HYPERPARAM_SENSITIVITY_RUN / "per_dataset.csv"
    if not per_dataset_path.exists():
        raise FileNotFoundError(f"Missing hyperparameter sensitivity per-dataset CSV: {per_dataset_path}")
    datasets: set[str] = set()
    with per_dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("returncode") == "0" and row.get("param") in REQUIRED_HYPERPARAM_SENSITIVITY_PARAMS:
                datasets.add(str(row.get("dataset", "")))
    missing_datasets = FINAL_DATASETS - datasets
    if missing_datasets:
        missing = ", ".join(sorted(missing_datasets))
        raise ValueError(
            "Hyperparameter sensitivity run is not a final six-dataset run. "
            f"Missing dataset(s): {missing}. Source: {per_dataset_path}."
        )

    records.sort(key=lambda item: (order[str(item["param"])], float(item["value"])))

    columns = ["Parameter", "Value", "Avg ACC", "Avg NMI", "Avg ARI", "Avg F1"]
    rows: list[list[str]] = []
    last_param = None
    for item in records:
        param = str(item["param"])
        value = int(item["value"]) if param in {"t", "k_E"} else float(item["value"])
        value_text = f"{value:d}" if isinstance(value, int) else f"{value:.2f}".rstrip("0").rstrip(".")
        rows.append(
            [
                str(item["label"]) if param != last_param else "",
                value_text,
                f"{float(item['avg_acc']):.2f}",
                f"{float(item['avg_nmi']):.2f}",
                f"{float(item['avg_ari']):.2f}",
                f"{float(item['avg_f1']):.2f}",
            ]
        )
        last_param = param
    return columns, rows, records


def load_hyperparameter_per_dataset_records() -> list[dict[str, float | str]]:
    per_dataset_path = HYPERPARAM_SENSITIVITY_RUN / "per_dataset.csv"
    if not per_dataset_path.exists():
        raise FileNotFoundError(f"Missing hyperparameter sensitivity per-dataset CSV: {per_dataset_path}")

    order = {"t": 0, "k_E": 1, "dcgl_neg_weight": 2, "fusion_temp": 3}
    records: list[dict[str, float | str]] = []
    with per_dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            param = row["param"]
            if param not in order or row.get("returncode") != "0":
                continue
            records.append(
                {
                    "dataset": row["dataset"],
                    "param": param,
                    "value": float(row["value"]),
                    "acc_mean": float(row["acc_mean"]),
                    "acc_std": float(row.get("acc_std") or 0.0),
                }
            )
    records.sort(key=lambda item: (order[str(item["param"])], str(item["dataset"]), float(item["value"])))
    return records


def render_hyperparameter_sensitivity_chart(path: Path, records: list[dict[str, float | str]]) -> None:
    grouped: dict[str, dict[str, list[dict[str, float | str]]]] = {}
    for item in records:
        grouped.setdefault(str(item["param"]), {}).setdefault(str(item["dataset"]), []).append(item)

    panels = [
        ("t", r"Propagation depth $t$"),
        ("k_E", r"Refined graph $k_E$"),
        ("dcgl_neg_weight", r"Separation strength $\lambda_{\mathrm{sep}}$"),
        ("fusion_temp", r"Fusion temperature $\tau_f$"),
    ]
    dataset_labels = {
        "reut": "Reuters",
        "uat": "UAT",
        "amap": "AMAP",
        "usps": "USPS",
        "cora": "Cora",
        "cite": "Citeseer",
    }
    dataset_order = ["reut", "uat", "amap", "usps", "cora", "cite"]
    dataset_colors = {
        "reut": RAW_BLUE,
        "uat": AE_ORANGE,
        "amap": FUSION_GREEN,
        "usps": CLUSTER_YELLOW,
        "cora": MAIN_RED,
        "cite": "#7f8790",
    }
    dataset_markers = {
        "reut": "o",
        "uat": "s",
        "amap": "D",
        "usps": "^",
        "cora": "v",
        "cite": "P",
    }

    fig, axes = plt.subplots(2, 2, figsize=(5.8, 4.35), sharey=False, squeeze=False)
    axes_arr = axes.ravel()
    for ax, (param, title) in zip(axes_arr, panels):
        dataset_items = grouped.get(param, {})
        if not dataset_items:
            ax.axis("off")
            continue

        all_y: list[float] = []
        all_x: list[float] = []
        for dataset in dataset_order:
            items = dataset_items.get(dataset, [])
            if not items:
                continue
            x = np.array([float(item["value"]) for item in items], dtype=float)
            y = np.array([float(item["acc_mean"]) for item in items], dtype=float)
            all_x.extend(x.tolist())
            all_y.extend(y.tolist())
            ax.plot(
                x,
                y,
                color=dataset_colors[dataset],
                linewidth=1.08,
                marker=dataset_markers[dataset],
                markersize=3.0,
                markerfacecolor=dataset_colors[dataset],
                markeredgecolor=dataset_colors[dataset],
                markeredgewidth=0.45,
                alpha=0.93,
                zorder=3,
            )

        if param == "k_E":
            ax.axvline(15, color="#777777", linestyle="--", linewidth=0.7, alpha=0.62, zorder=1)
        ax.set_title(title, fontsize=_font(11.6), fontweight="bold", pad=4)
        ax.set_xlabel("Value", fontsize=_font(9.2), fontweight="bold", labelpad=1.5)
        ax.set_ylabel("ACC (%)", fontsize=_font(9.2), fontweight="bold", labelpad=1.8)
        ax.tick_params(axis="both", labelsize=_font(8.0), pad=1.0)
        x_values = np.array(sorted(set(all_x)), dtype=float)
        ax.set_xticks(x_values)
        if param in {"fusion_temp", "dcgl_neg_weight"}:
            ax.set_xticklabels([f"{val:.2f}".rstrip("0").rstrip(".") for val in x_values])
        else:
            ax.set_xticklabels([str(int(val)) for val in x_values])
        y_min = float(np.min(all_y))
        y_max = float(np.max(all_y))
        pad = max(0.7, (y_max - y_min) * 0.10)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_box_aspect(0.82)
        style_framed_axes(ax, grid_axis="y")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=dataset_colors[dataset],
            marker=dataset_markers[dataset],
            linewidth=1.08,
            markersize=3.2,
            markerfacecolor=dataset_colors[dataset],
            markeredgecolor=dataset_colors[dataset],
            label=dataset_labels[dataset],
        )
        for dataset in dataset_order
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.992),
        frameon=True,
        fontsize=_font(7.0),
        handlelength=1.0,
        handletextpad=0.24,
        columnspacing=0.46,
        borderpad=0.22,
    )
    style_legend(legend, linewidth=0.55)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.07, top=0.88, wspace=0.15, hspace=0.35)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def resolve_structural_uncertainty_run() -> Path | None:
    if not STRUCTURAL_UNCERTAINTY_ROOT.exists():
        return None
    candidates = sorted(
        (path for path in STRUCTURAL_UNCERTAINTY_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        aggregate_path = candidate / "aggregate.csv"
        if not aggregate_path.exists() or aggregate_path.stat().st_size <= 0:
            continue
        with aggregate_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        if any(row.get("variant") == "dsafc" and row.get("avg_acc") for row in rows):
            return candidate
    return None


def load_structural_uncertainty_rows() -> tuple[list[str], list[list[str]]] | None:
    run_dir = resolve_structural_uncertainty_run()
    if run_dir is None:
        return None
    aggregate_path = run_dir / "aggregate.csv"
    records: list[dict[str, str]] = []
    with aggregate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(row)
    if not records:
        return None

    condition_order = {"edge_delete": 0, "edge_add": 1, "knn_k": 2}
    records.sort(key=lambda row: (condition_order.get(row.get("condition", ""), 99), float(row.get("value", 0.0))))
    label_map = {
        "edge_delete": "Edge deletion",
        "edge_add": "Edge addition",
        "knn_k": "KNN perturbation",
    }
    columns = ["Condition", "Value", "Variant", "Datasets", "Avg ACC", "Avg NMI", "Avg ARI", "Avg F1"]
    rows: list[list[str]] = []
    last_key: tuple[str, str] | None = None
    for row in records:
        condition = row.get("condition", "")
        value = float(row.get("value", 0.0))
        if condition in {"edge_delete", "edge_add"}:
            value_text = f"{int(round(value * 100))}%"
        else:
            value_text = str(int(value))
        key = (condition, value_text)
        rows.append(
            [
                label_map.get(condition, condition) if key != last_key else "",
                value_text if key != last_key else "",
                row.get("variant", ""),
                str(int(float(row.get("datasets") or 0))),
                f"{float(row.get('avg_acc') or 0.0):.2f}",
                f"{float(row.get('avg_nmi') or 0.0):.2f}",
                f"{float(row.get('avg_ari') or 0.0):.2f}",
                f"{float(row.get('avg_f1') or 0.0):.2f}",
            ]
        )
        last_key = key
    return columns, rows


def load_markdown_table(section_title: str, *, group_col: int | None = None) -> tuple[list[str], list[list[str]]]:
    raw_path = ROOT / "docs" / "DSAFC_raw_tables.md"
    lines = raw_path.read_text(encoding="utf-8").splitlines()

    start = None
    for idx, line in enumerate(lines):
        if line.strip() == f"## {section_title}":
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
        if stripped.startswith("## "):
            break
        if stripped.startswith("|"):
            table_lines.append(stripped)
        elif table_lines:
            break

    if len(table_lines) < 2:
        raise ValueError(f"No markdown table found under section: {section_title}")

    def split_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    columns = split_row(table_lines[0])
    rows = [split_row(line) for line in table_lines[2:]]

    if group_col is not None:
        last_group = None
        for row in rows:
            if row[group_col] == last_group:
                row[group_col] = ""
            else:
                last_group = row[group_col]

    return columns, rows


def render_algorithm(path: Path) -> None:
    lines = [
        (0, "Construct the refined graph $A_E$ from $X$ and $A$."),
        (0, "Generate graph-smoothed features $\\tilde{X}^{(A)}$ and $\\tilde{X}^{(A_E)}$."),
        (0, "Initialize the shared two-head encoder $f_\\theta$ and fusion network $g_\\phi$."),
        (0, "Initialize pseudo labels $\\hat{Y}$ and center distances by K-means on initial smoothed features."),
        (0, "for $e=1$ to $T$ do"),
        (1, "Encode two structure views to obtain $z_1^{(A)},z_2^{(A)},z_1^{(A_E)},z_2^{(A_E)}$."),
        (1, "Compute $H^{(A)}$, $H^{(A_E)}$, relation descriptors $R$, fusion weights $\\alpha$, and $H$."),
        (1, "if $e \\leq E_w$ then"),
        (2, "Minimize the dual-view warm-up objective."),
        (1, "else"),
        (2, "Update $\\hat{Y}$, center distances $d$, and high-confidence set $\\mathcal{H}$ on $H$."),
        (2, "Compute $\\mathcal{L}_{\\mathrm{cg}}^{(A)}$ and $\\mathcal{L}_{\\mathrm{cg}}^{(A_E)}$."),
        (2, "Compute $\\mathcal{L}_{\\mathrm{ins}}$, $\\mathcal{L}_{\\mathrm{clu}}$, and $\\mathcal{L}_{\\mathrm{bal}}$."),
        (2, "Form the total objective $\\mathcal{L}$."),
        (1, "end if"),
        (1, "Update all trainable parameters by Adam."),
        (0, "end for"),
        (0, "Apply K-means to the final fused representation $H$."),
        (0, "return $\\hat{Y}$"),
    ]

    fig_width = 11.5
    fig_height = 8.9
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([0.02, 0.98], [0.965, 0.965], color="black", lw=1.6)
    ax.text(0.025, 0.922, r"Algorithm 1: $\mathbf{DSAFC}$", fontsize=_font(23), va="center")
    ax.plot([0.02, 0.98], [0.885, 0.885], color="black", lw=1.1)

    ax.text(
        0.02,
        0.835,
        r"$\mathbf{Input:}$ graph data $\mathcal{G}=\{X,A\}$; cluster number $K$; epochs $T$; warm-up epochs $E_w$;",
        fontsize=_font(16),
        va="top",
    )
    ax.text(0.12, 0.792, r"hyperparameters for refined graph construction, fusion, and cluster guidance.", fontsize=_font(16), va="top")
    ax.text(
        0.02,
        0.740,
        r"$\mathbf{Output:}$ clustering assignment $\hat{Y}$.",
        fontsize=_font(16),
        va="top",
    )

    y = 0.675
    step = 0.028
    for idx, (indent, text) in enumerate(lines, start=1):
        weight = "bold" if text.startswith(("for ", "if ", "else", "end", "return")) else "normal"
        ax.text(0.035, y, f"{idx}:", fontsize=_font(16), va="top", ha="right")
        ax.text(0.06 + indent * 0.04, y, text, fontsize=_font(16), va="top", ha="left", fontweight=weight)
        y -= step

    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    render_method_positioning_table(ASSETS / "DSAFC_method_positioning.png")

    notation_rows = [
        ["Problem", r"$X\in\mathbb{R}^{N\times d}$", "Node attribute matrix."],
        ["", r"$A,\ A_E$", "Original graph and offline refined graph."],
        ["", r"$K,\ \hat{Y}$", "Cluster number and final clustering assignment."],
        ["Refined graph", r"$Z,\ \hat{X}$", "AE latent representation and reconstruction for building $A_E$."],
        ["", r"$A^k,\ A^c,\ k_E$", "Top-$k_E$ candidate graph, consensus cluster graph, and refined-graph neighbor number."],
        ["Dual-view encoding", r"$v\in\{A,A_E\},\ \hat{A}^{(v)}$", "Structure view index and normalized adjacency matrix."],
        ["", r"$\tilde{X}^{(v)}$", "Graph-smoothed features under view $v$."],
        ["", r"$f_\theta(\cdot),\ z_1^{(v)},z_2^{(v)}$", "Shared two-head encoder and projected embeddings."],
        ["", r"$H^{(v)},\ H$", "View-specific representation and adaptively fused representation."],
        ["Adaptive fusion", r"$R^i,\ g_\phi(\cdot)$", "Cross-view relation descriptor and lightweight fusion network."],
        ["", r"$\tau_f,\ \eta_f$", "Fusion temperature and minimum branch weight."],
        ["", r"$\alpha_i^{(A)},\alpha_i^{(A_E)}$", "Node-wise weights assigned to the two structure views."],
        ["Cluster guidance", r"$d_i,\ r_t,\ \tau_t,\ \mathcal{H},\mathcal{H}_c$", "Nearest-center distance, high-confidence ratio and threshold, high-confidence set, and its subset in cluster $c$."],
        ["", r"$u_{c,1}^{(v)},u_{c,2}^{(v)}$", "Cluster centers estimated from the two heads under view $v$."],
        ["", r"$s_c^{(v)},\omega_c^{(v)},\mathcal{B}_c^{(v)}$", "Reliability score, conservative gate, and negative center set for cluster $c$."],
        ["Optimization", r"$\mathcal{L}_{\mathrm{warm}}^{(v)},\mathcal{L}_{\mathrm{cg}}^{(v)}$", "Warm-up and cluster-guided discriminative objectives in view $v$."],
        ["", r"$\alpha_{\mathrm{cg}},\lambda_{\mathrm{sep}},\tau_{\mathrm{sep}}$", "Base cluster-guidance coefficient, cluster-separation strength, and center-contrast temperature."],
        ["", r"$\mathcal{L}_{\mathrm{ins}},\mathcal{L}_{\mathrm{clu}},\mathcal{L}_{\mathrm{bal}}$", "Instance consistency, cluster-distribution consistency, and fusion-balance regularizers."],
        ["", r"$T,\ E_w,\rho_e,\tau_d$", "Training epochs, warm-up epochs, ramp-up consistency coefficient, and cluster-distribution temperature."],
        ["", r"$\mathcal{L}$", "Final training objective."],
    ]
    render_table(
        ASSETS / "DSAFC_notation.png",
        ["Component", "Notation", "Definition"],
        notation_rows,
        col_widths=[1.3, 2.3, 5.5],
        wrap_widths=[14, 999, 44],
        fig_width=12.4,
        font_size=10.8,
        header_size=14,
        row_unit=0.46,
        vertical_after=(0, 1),
        col_aligns=["center", "center", "left"],
    )

    render_algorithm(ASSETS / "DSAFC_algorithm.png")
    render_dataset_statistics_table(ASSETS / "DSAFC_dataset_statistics.png", drop_datasets={"EAT"})
    render_dataset_statistics_table(ASSETS / "DSAFC_kbs_dataset_statistics.png", drop_datasets={"EAT"})

    main_columns, main_rows = load_markdown_table("Table 4-2 Main Clustering Results", group_col=0)
    main_columns, main_rows = filter_active_table(main_columns, main_rows, drop_methods=ARCHIVED_METHODS, group_col=0)
    main_columns, main_rows = reorder_main_table_columns(main_columns, main_rows)
    main_display_rows = mark_local_reproduction_cells(main_columns, main_rows)
    render_table(
        ASSETS / "DSAFC_main_results.png",
        main_columns,
        main_rows,
        display_rows=main_display_rows,
        rank_rows=main_rows,
        highlight_col_fills=ours_highlight_columns(main_columns),
        col_widths=[1.25, 0.85] + [1.0] * (len(main_columns) - 2),
        wrap_widths=[14, 8] + [9] * (len(main_columns) - 2),
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
    render_average_rank_chart(ASSETS / "DSAFC_average_rank.png", main_columns, main_rows)
    render_dataset_rank_heatmap(ASSETS / "DSAFC_rank_heatmap.png", main_columns, main_rows)

    ablation_columns, ablation_rows = load_markdown_table("Table 4-3 Ablation Results", group_col=0)
    ablation_columns, ablation_rows = filter_active_table(ablation_columns, ablation_rows, group_col=0)
    render_table(
        ASSETS / "DSAFC_ablation_results.png",
        ablation_columns,
        ablation_rows,
        col_widths=[1.45, 0.9, 1.28, 1.28, 1.28, 1.28, 1.32],
        fig_width=15.4,
        font_size=12.8,
        header_size=15.2,
        row_unit=0.42,
        vertical_after=(0, 1),
        group_col=0,
        bold_best=False,
        numeric_start_col=2,
    )
    render_table(
        ASSETS / "DSAFC_kbs_ablation_results.png",
        ablation_columns,
        ablation_rows,
        col_widths=[1.45, 0.9, 1.28, 1.28, 1.28, 1.28, 1.32],
        fig_width=15.4,
        font_size=12.8,
        header_size=15.2,
        row_unit=0.42,
        vertical_after=(0, 1),
        group_col=0,
        bold_best=False,
        numeric_start_col=2,
    )

    ablation_acc_columns, ablation_acc_rows = load_markdown_table("Figure 4-1 ACC Plot Data")
    ablation_acc_columns, ablation_acc_rows = filter_active_table(ablation_acc_columns, ablation_acc_rows, group_col=0)
    render_ablation_acc_chart(
        ASSETS / "DSAFC_ablation_acc_comparison.png",
        ablation_acc_columns,
        ablation_acc_rows,
    )

    structure_columns, structure_rows = load_markdown_table("Table 4-4 Structure-Fusion Reliability Diagnosis")
    structure_columns, structure_rows = filter_active_table(structure_columns, structure_rows, group_col=0)
    structure_columns, structure_rows = select_columns(
        structure_columns,
        structure_rows,
        [
            "Dataset",
            "Edge overlap",
            "New-edge ratio",
            "Homophily$(A)$",
            "Homophily$(A_E)$",
            r"$\Delta$ homophily",
            r"Mean $\alpha^{(A)}$",
            r"Mean $\alpha^{(A_E)}$",
            "Weight entropy",
            "View tendency",
        ],
        display_columns=[
            "Dataset",
            "Edge\noverlap",
            "New-edge\nratio",
            "Homophily\n$(A)$",
            "Homophily\n$(A_E)$",
            r"$\Delta$" + "\nhomophily",
            r"Mean $\alpha^{(A)}$",
            r"Mean $\alpha^{(A_E)}$",
            "Weight\nentropy",
            "View\ntendency",
        ],
    )
    render_table(
        ASSETS / "DSAFC_structure_fusion_reliability_diagnosis.png",
        structure_columns,
        structure_rows,
        col_widths=[1.45, 1.25, 1.3, 1.38, 1.44, 1.34, 1.5, 1.58, 1.25, 1.35],
        wrap_widths=[14, 12, 12, 12, 12, 12, 15, 15, 12, 12],
        fig_width=22.4,
        font_size=12.5,
        header_size=13.9,
        row_unit=0.48,
        header_unit=0.82,
        vertical_after=(0, 5),
        numeric_start_col=1,
    )
    render_table(
        ASSETS / "DSAFC_kbs_structure_fusion_reliability_diagnosis.png",
        structure_columns,
        structure_rows,
        col_widths=[1.45, 1.25, 1.3, 1.38, 1.44, 1.34, 1.5, 1.58, 1.25, 1.35],
        wrap_widths=[14, 12, 12, 12, 12, 12, 15, 15, 12, 12],
        fig_width=22.4,
        font_size=12.5,
        header_size=13.9,
        row_unit=0.48,
        header_unit=0.82,
        vertical_after=(0, 5),
        numeric_start_col=1,
    )

    reliability_columns, reliability_rows = load_markdown_table("Table 4-5 Reliability Validation")
    reliability_columns, reliability_rows = filter_active_table(reliability_columns, reliability_rows, group_col=0)
    render_table(
        ASSETS / "DSAFC_reliability_validation.png",
        reliability_columns,
        reliability_rows,
        col_widths=[1.25, 1.25, 1.25, 1.38, 1.42, 1.18, 1.35, 2.7],
        wrap_widths=[13, 12, 12, 14, 14, 12, 13, 24],
        fig_width=16.8,
        font_size=12.0,
        header_size=13.2,
        row_unit=0.42,
        vertical_after=(0, 5),
        numeric_start_col=1,
    )

    robustness_table = load_structural_uncertainty_rows()
    if robustness_table is not None:
        robustness_columns, robustness_rows = robustness_table
        render_table(
            ASSETS / "DSAFC_structural_uncertainty_robustness.png",
            robustness_columns,
            robustness_rows,
            col_widths=[1.7, 0.78, 1.0, 0.82, 1.02, 1.02, 1.02, 1.02],
            wrap_widths=[17, 8, 10, 8, 10, 10, 10, 10],
            fig_width=13.4,
            font_size=11.5,
            header_size=12.7,
            row_unit=0.34,
            vertical_after=(0, 1),
            group_col=0,
            numeric_start_col=4,
        )

    hyper_columns, hyper_rows, hyper_records = load_hyperparameter_sensitivity_rows()
    render_table(
        ASSETS / "DSAFC_hyperparameter_sensitivity_table.png",
        hyper_columns,
        hyper_rows,
        col_widths=[1.55, 0.9, 1.15, 1.15, 1.15, 1.15],
        fig_width=10.8,
        font_size=13.0,
        header_size=14.4,
        row_unit=0.36,
        vertical_after=(0, 1),
        group_col=0,
        bold_best=True,
        numeric_start_col=2,
    )
    render_hyperparameter_sensitivity_chart(
        ASSETS / "DSAFC_hyperparameter_sensitivity_acc.png",
        load_hyperparameter_per_dataset_records(),
    )


if __name__ == "__main__":
    main()
