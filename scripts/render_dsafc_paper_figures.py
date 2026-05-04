from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FONT_SCALE = 0.70
TABLE_ROW_SCALE = 1.12
GROUP_GAP_RATIO = 0.42
GROUP_LABEL_Y_OFFSET = -0.025

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    }
)


def _parse_float(value: str) -> float | None:
    match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", value or "")
    return float(match.group(1)) if match else None


def _wrap(text: str, width: int) -> str:
    text = str(text)
    if "\n" in text or len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def _font(size: float) -> float:
    return size * FONT_SCALE


def render_table(
    path: Path,
    columns: list[str],
    rows: list[list[str]],
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
    numeric_start_col: int = 2,
    col_aligns: list[str] | None = None,
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

    wrapped_rows = [[_wrap(cell, wrap_widths[i]) for i, cell in enumerate(row)] for row in rows]
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

    bold_cells: set[tuple[int, int]] = set()
    if bold_best:
        for r_idx, row in enumerate(rows):
            values = [_parse_float(v) for v in row[numeric_start_col:]]
            values = [v for v in values if v is not None]
            if values:
                best = max(values)
                for c_idx in range(numeric_start_col, n_cols):
                    val = _parse_float(row[c_idx])
                    if val is not None and math.isclose(val, best, rel_tol=0.0, abs_tol=1e-9):
                        bold_cells.add((r_idx, c_idx))

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
        ["", r"$\alpha_{\mathrm{cg}},\lambda_{\mathrm{neg}},\tau_{\mathrm{neg}}$", "Base cluster-guidance negative coefficient, negative weight, and center-contrast temperature."],
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

    dataset_rows = [
        ["Reuters", "Text", "10000", "2000", "46135", "4"],
        ["UAT", "Graph", "1190", "239", "13599", "4"],
        ["AMAP", "Graph", "7650", "745", "119081", "8"],
        ["USPS", "Image", "9298", "256", "34996", "10"],
        ["EAT", "Graph", "399", "203", "5993", "4"],
        ["Cora", "Graph", "2708", "1433", "5278", "7"],
        ["Citeseer", "Graph", "3327", "3703", "4552", "6"],
    ]
    render_table(
        ASSETS / "DSAFC_dataset_statistics.png",
        ["Dataset", "Type", "Sample", "Dimension", "Edge", "Class"],
        dataset_rows,
        col_widths=[1.7, 1.25, 1.55, 2.1, 1.65, 1.0],
        fig_width=11.0,
        font_size=15.5,
        header_size=18,
        row_unit=0.48,
    )

    main_columns, main_rows = load_markdown_table("Table 4-2 Main Clustering Results", group_col=0)
    render_table(
        ASSETS / "DSAFC_main_results.png",
        main_columns,
        main_rows,
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
        numeric_start_col=2,
    )

    ablation_rows = [
        ["Reuters", "ACC", "70.78", "71.07", "74.45", "76.12", "83.85"],
        ["", "NMI", "45.26", "46.97", "54.04", "52.65", "60.26"],
        ["", "ARI", "42.41", "39.28", "48.06", "49.97", "66.17"],
        ["", "F1", "59.53", "58.04", "62.59", "63.07", "72.27"],
        ["UAT", "ACC", "47.61", "54.10", "55.84", "54.76", "56.26"],
        ["", "NMI", "19.25", "27.21", "28.84", "26.96", "27.35"],
        ["", "ARI", "11.37", "24.51", "27.14", "23.08", "22.04"],
        ["", "F1", "44.20", "49.21", "50.23", "52.82", "56.43"],
        ["AMAP", "ACC", "77.09", "36.96", "55.17", "77.01", "77.47"],
        ["", "NMI", "66.86", "21.67", "48.49", "67.04", "67.61"],
        ["", "ARI", "57.66", "13.52", "33.41", "57.57", "58.39"],
        ["", "F1", "71.81", "30.70", "52.79", "71.62", "72.15"],
        ["USPS", "ACC", "80.92", "78.58", "79.55", "79.01", "82.40"],
        ["", "NMI", "73.12", "69.16", "71.00", "70.20", "73.29"],
        ["", "ARI", "66.67", "63.70", "65.24", "64.43", "68.39"],
        ["", "F1", "80.57", "77.28", "78.34", "77.83", "82.16"],
        ["EAT", "ACC", "55.16", "45.64", "50.65", "52.53", "54.76"],
        ["", "NMI", "32.81", "23.32", "26.87", "28.65", "31.97"],
        ["", "ARI", "26.50", "18.30", "19.79", "20.41", "24.32"],
        ["", "F1", "52.83", "36.67", "46.99", "50.26", "52.98"],
        ["Cora", "ACC", "72.56", "49.79", "62.98", "72.87", "73.51"],
        ["", "NMI", "55.21", "26.52", "42.15", "54.81", "55.54"],
        ["", "ARI", "49.64", "22.48", "35.59", "49.70", "50.37"],
        ["", "F1", "69.15", "43.72", "60.41", "70.02", "71.14"],
        ["Citeseer", "ACC", "70.31", "54.44", "67.05", "69.99", "71.18"],
        ["", "NMI", "44.44", "27.71", "40.78", "44.50", "45.14"],
        ["", "ARI", "44.60", "25.86", "40.59", "44.34", "46.09"],
        ["", "F1", "61.68", "51.54", "59.43", "61.68", "61.80"],
    ]
    render_table(
        ASSETS / "DSAFC_ablation_results.png",
        ["Dataset", "Metric", "OSL", "RSL", "F-DSF", "A-DSF", "DSAFC"],
        ablation_rows,
        col_widths=[1.45, 0.9, 1.1, 1.1, 1.1, 1.1, 1.2],
        fig_width=13.2,
        font_size=14.6,
        header_size=16.2,
        row_unit=0.42,
        vertical_after=(0, 1),
        group_col=0,
        bold_best=True,
        numeric_start_col=2,
    )

    hyper_rows = []
    hyper_data = {
        "fusion temp": [
            ("1.0", "68.88", "49.45", "44.22", "63.40"),
            ("1.3", "68.76", "49.08", "43.90", "63.47"),
            ("1.6", "68.97", "49.31", "44.14", "63.84"),
            ("1.9", "68.62", "49.16", "43.85", "63.43"),
            ("2.2", "68.99", "49.49", "44.43", "63.76"),
        ],
        "min view weight": [
            ("0.00", "68.55", "49.08", "44.29", "63.38"),
            ("0.05", "68.69", "49.15", "44.12", "63.55"),
            ("0.10", "68.90", "49.28", "44.26", "63.56"),
            ("0.15", "68.78", "49.21", "44.03", "63.36"),
            ("0.20", "68.28", "48.38", "43.24", "63.25"),
        ],
        "refined k": [
            ("5", "68.93", "49.24", "44.23", "63.59"),
            ("10", "68.90", "49.28", "44.26", "63.56"),
            ("15", "68.88", "49.24", "44.23", "63.54"),
            ("20", "68.92", "49.24", "44.22", "63.59"),
            ("25", "68.90", "49.28", "44.26", "63.57"),
        ],
        "neg weight": [
            ("0.0", "68.71", "49.20", "43.89", "63.45"),
            ("0.2", "68.64", "49.06", "43.90", "63.40"),
            ("0.4", "68.83", "49.27", "44.21", "63.97"),
            ("0.6", "68.90", "49.27", "44.26", "63.56"),
            ("0.8", "68.72", "49.19", "43.92", "63.78"),
            ("1.0", "69.09", "49.46", "44.28", "64.09"),
        ],
        "fusion balance": [
            ("0.00", "68.89", "49.38", "44.16", "63.46"),
            ("0.05", "68.87", "49.27", "44.11", "63.52"),
            ("0.10", "68.74", "49.20", "43.99", "63.49"),
            ("0.20", "68.90", "49.19", "43.93", "63.42"),
            ("0.35", "68.91", "49.28", "44.10", "63.54"),
        ],
        "lambda inst": [
            ("0.00", "68.92", "49.24", "44.20", "63.45"),
            ("0.03", "68.92", "49.23", "44.23", "63.65"),
            ("0.06", "68.84", "49.22", "44.15", "63.45"),
            ("0.08", "68.85", "49.28", "44.17", "63.54"),
            ("0.10", "68.83", "49.23", "44.08", "63.46"),
        ],
        "lambda clu": [
            ("0.00", "68.95", "49.28", "44.21", "63.63"),
            ("0.01", "68.93", "49.24", "44.23", "63.59"),
            ("0.03", "68.91", "49.30", "44.36", "63.49"),
            ("0.05", "68.90", "49.22", "44.25", "63.53"),
            ("0.07", "68.93", "49.22", "44.23", "63.65"),
        ],
        "neg tau": [
            ("0.20", "68.67", "49.18", "44.18", "63.54"),
            ("0.35", "68.81", "49.31", "44.06", "63.81"),
            ("0.50", "68.90", "49.28", "44.26", "63.56"),
            ("0.75", "68.72", "49.21", "44.20", "63.86"),
            ("1.00", "68.75", "49.13", "44.10", "63.56"),
        ],
    }
    for param, records in hyper_data.items():
        for idx, record in enumerate(records):
            hyper_rows.append([param if idx == 0 else "", *record])

    render_table(
        ASSETS / "DSAFC_hyperparameter_sensitivity_table.png",
        ["Parameter", "Value", "Avg ACC", "Avg NMI", "Avg ARI", "Avg F1"],
        hyper_rows,
        col_widths=[1.9, 1.0, 1.15, 1.15, 1.15, 1.15],
        fig_width=11.6,
        font_size=13.6,
        header_size=15.2,
        row_unit=0.35,
        vertical_after=(0, 1),
        group_col=0,
        bold_best=True,
        numeric_start_col=2,
    )


if __name__ == "__main__":
    main()
