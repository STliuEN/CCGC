from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCGC_ENV_PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
DEFAULT_PYTHON = SCGC_ENV_PYTHON if SCGC_ENV_PYTHON.exists() else Path(sys.executable)
OUTPUT_ROOT = ROOT / "experiment_output" / "dsafc_dual_structure_ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one complete DSAFC reproduction and update main/ablation "
            "raw tables from that same run."
        )
    )
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--update-tables",
        action="store_true",
        help="Update docs/DSAFC_raw_tables.md after training. Defaults to true only for all datasets and all variants.",
    )
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def run_step(name: str, cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    print(f"[STEP] {name}", flush=True)
    print(" ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[STEP] {name}\n")
        handle.write("[COMMAND]\n")
        handle.write(" ".join(cmd) + "\n\n")
        handle.write("[OUTPUT]\n")
        handle.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        returncode = proc.wait()
        handle.write(f"\n[RETURNCODE] {returncode}\n")
        handle.flush()

    if returncode != 0:
        raise RuntimeError(f"{name} failed with return code {returncode}; see {log_path}")


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = str(args.dataset).replace(",", "_").replace(";", "_")
    variant_tag = str(args.variants).replace(",", "_").replace(";", "_")
    run_dir = args.run_dir or (OUTPUT_ROOT / f"{stamp}_{dataset_tag}_{variant_tag}_unified")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = run_dir / "_pipeline_logs"
    ablation_cmd = [
        str(args.python),
        "scripts/dsafc_dual_structure_ablation.py",
        "--dataset",
        args.dataset,
        "--variants",
        args.variants,
        "--runs",
        str(args.runs),
        "--seed-start",
        str(args.seed_start),
        "--device",
        args.device,
        "--run-dir",
        str(run_dir),
        "--export-fusion-artifacts",
        "--reuse-current-ae-assets",
    ]
    if args.timeout:
        ablation_cmd.extend(["--timeout", str(args.timeout)])
    if args.resume:
        ablation_cmd.extend(["--resume-jsonl", str(run_dir / "results.jsonl")])

    run_step("train_ablation_chain", ablation_cmd, cwd=ROOT, log_path=logs_dir / "01_train_ablation_chain.log")

    should_update = args.update_tables or (
        str(args.dataset).strip().lower() == "all" and str(args.variants).strip().lower() == "all"
    )
    if should_update:
        update_cmd = [
            str(args.python),
            "scripts/update_dsafc_tables_from_ablation.py",
            "--run-dir",
            str(run_dir),
        ]
        run_step("update_raw_tables", update_cmd, cwd=ROOT, log_path=logs_dir / "02_update_raw_tables.log")

    if should_update and not args.skip_render:
        render_cmds = [
            ("render_paper_figures", [str(args.python), "scripts/render_dsafc_paper_figures.py"]),
            ("render_main_std", [str(args.python), "scripts/render_dsafc_main_results_with_std.py"]),
            ("render_compact_tables", [str(args.python), "scripts/render_dsafc_compact_tables.py"]),
        ]
        for idx, (name, cmd) in enumerate(render_cmds, start=3):
            run_step(name, cmd, cwd=ROOT, log_path=logs_dir / f"{idx:02d}_{name}.log")

    report = run_dir / "unified_reproduction_report.md"
    report.write_text(
        "\n".join(
            [
                "# DSAFC Unified Reproduction Report",
                "",
                f"- Run dir: `{run_dir}`",
                f"- Summary: `{run_dir / 'summary.md'}`",
                f"- Runs: `{args.runs}`",
                f"- Seeds: `{args.seed_start}..{args.seed_start + args.runs - 1}`",
                f"- Dataset argument: `{args.dataset}`",
                f"- Variant argument: `{args.variants}`",
                f"- Raw tables updated: `{should_update}`",
                f"- Rendered assets refreshed: `{should_update and not args.skip_render}`",
                "",
                "For full all-dataset/all-variant runs, the main-table `Ours` column and the ablation-table `DSAFC` column are from the same DSAFC rows in this run.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] report={report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
