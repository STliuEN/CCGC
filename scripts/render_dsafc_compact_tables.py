from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_dsafc_main_results_with_std import build_display_rows_with_local_marks
from scripts.render_dsafc_paper_figures import (
    ASSETS,
    load_markdown_table,
    mark_local_reproduction_cells,
    ours_highlight_columns,
    render_table,
    reorder_main_table_columns,
)


RAW_TABLES = ROOT / "docs" / "DSAFC_raw_tables.md"
SOURCE_SECTION = "Table 4-2 Main Clustering Results"
COMPACT_MAIN_SECTION = "Table 4-2 Compact Main Clustering Results"
COMPACT_STD_SECTION = "Table 4-2 Compact Main Results With Std"
MAIN_OUTPUT = ASSETS / "DSAFC_main_results_compact.png"
STD_OUTPUT = ASSETS / "DSAFC_main_results_with_std_compact.png"

DROP_DATASETS = {"EAT"}
DROP_METHODS = {"SCGC-N", "SCGC-N*"}

COMPACT_NOTE = (
    "This compact display-only variant removes dataset `EAT` and drops "
    "the `SCGC-N` / `SCGC-N*` columns. The original full tables above remain "
    "the canonical raw record."
)


def _filter_table(columns: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    keep_indices = [
        idx
        for idx, col in enumerate(columns)
        if idx < 2 or col not in DROP_METHODS
    ]
    filtered_columns = [columns[idx] for idx in keep_indices]
    filtered_rows = [row for row in rows if row[0] not in DROP_DATASETS]
    filtered_rows = [[row[idx] for idx in keep_indices] for row in filtered_rows]
    return filtered_columns, filtered_rows


def _sparsify_group_col(rows: list[list[str]], group_col: int = 0) -> list[list[str]]:
    sparse_rows: list[list[str]] = []
    last_group = None
    for row in rows:
        sparse_row = row.copy()
        if sparse_row[group_col] == last_group:
            sparse_row[group_col] = ""
        else:
            last_group = sparse_row[group_col]
        sparse_rows.append(sparse_row)
    return sparse_rows


def _markdown_table(columns: list[str], rows: list[list[str]]) -> str:
    align_row = []
    for idx, _ in enumerate(columns):
        align_row.append("---" if idx < 2 else "---:")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(align_row) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _section_block(title: str, table_md: str) -> str:
    return f"## {title}\n\n{COMPACT_NOTE}\n\n{table_md}\n"


def _upsert_section(path: Path, title: str, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    heading = f"## {title}"
    section_text = _section_block(title, body)
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == heading:
            start = idx
            break
    if start is None:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + section_text
        path.write_text(text, encoding="utf-8")
        return

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("## "):
            end = idx
            break

    new_lines = lines[:start] + section_text.strip().splitlines() + [""] + lines[end:]
    path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    columns_full, rows_full = load_markdown_table(SOURCE_SECTION)
    columns_full, rows_full = _filter_table(columns_full, rows_full)
    columns_full, rows_full = reorder_main_table_columns(columns_full, rows_full)

    main_display_full = mark_local_reproduction_cells(columns_full, rows_full)
    std_display_full = build_display_rows_with_local_marks(columns_full, rows_full)

    _upsert_section(RAW_TABLES, COMPACT_MAIN_SECTION, _markdown_table(columns_full, main_display_full))
    _upsert_section(RAW_TABLES, COMPACT_STD_SECTION, _markdown_table(columns_full, std_display_full))

    base_rows_grouped = _sparsify_group_col(rows_full)
    main_display_grouped = _sparsify_group_col(main_display_full)
    std_display_grouped = _sparsify_group_col(std_display_full)

    render_table(
        MAIN_OUTPUT,
        columns_full,
        base_rows_grouped,
        display_rows=main_display_grouped,
        rank_rows=base_rows_grouped,
        highlight_col_fills=ours_highlight_columns(columns_full),
        col_widths=[1.25, 0.85] + [1.0] * (len(columns_full) - 2),
        wrap_widths=[14, 8] + [9] * (len(columns_full) - 2),
        fig_width=27,
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

    render_table(
        STD_OUTPUT,
        columns_full,
        base_rows_grouped,
        display_rows=std_display_grouped,
        rank_rows=base_rows_grouped,
        highlight_col_fills=ours_highlight_columns(columns_full),
        col_widths=[1.25, 0.85] + [1.0] * (len(columns_full) - 2),
        wrap_widths=[14, 8] + [12] * (len(columns_full) - 2),
        fig_width=27,
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

    print(MAIN_OUTPUT)
    print(STD_OUTPUT)


if __name__ == "__main__":
    main()
