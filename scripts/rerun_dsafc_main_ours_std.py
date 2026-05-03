from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "dsafc_best_10run_attn_dcgl_only" / "rerun_current_specs"
SCGC_ENV_PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
DEFAULT_PYTHON = SCGC_ENV_PYTHON if SCGC_ENV_PYTHON.exists() else Path(sys.executable)
METRICS = ("ACC", "NMI", "ARI", "F1")


@dataclass(frozen=True)
class RerunSpec:
    dataset_key: str
    dataset_label: str
    selected_point: str
    source_log: str
    note: str
    command_args: tuple[str, ...]


SPECS: tuple[RerunSpec, ...] = (
    RerunSpec(
        dataset_key="reut",
        dataset_label="Reuters",
        selected_point="83.20 / 59.82 / 66.01 / 70.57",
        source_log="experiment_output/reut/reut_train_with_dual_attn_20260423_110149.txt",
        note="Selected best complete 10-run attention/DCGL-negative-only row from all experiment_output logs.",
        command_args=(
            "--dataset", "reut",
            "--cluster_num", "4",
            "--graph_mode", "dual",
            "--fusion_mode", "attn",
            "--t", "4",
            "--linlayers", "1",
            "--epochs", "400",
            "--dims", "500",
            "--lr", "0.0001",
            "--threshold", "0.4",
            "--alpha", "0.5",
            "--knn_k", "5",
            "--enable_dcgl_negative_loss",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
            "--warmup_epochs", "35",
            "--fusion_hidden", "64",
            "--fusion_temp", "1.6",
            "--fusion_balance", "0.25",
            "--fusion_min_weight", "0.15",
            "--lambda_inst", "0.08",
            "--lambda_clu", "0.06",
        ),
    ),
    RerunSpec(
        dataset_key="uat",
        dataset_label="UAT",
        selected_point="56.24 / 27.31 / 21.98 / 56.44",
        source_log="experiment_output/uat/uat_train_with_dual_attn_20260423_115353.txt",
        note="Selected best complete 10-run attention/DCGL-negative-only row from all experiment_output logs.",
        command_args=(
            "--dataset", "uat",
            "--cluster_num", "4",
            "--graph_mode", "dual",
            "--fusion_mode", "attn",
            "--t", "4",
            "--linlayers", "1",
            "--epochs", "400",
            "--dims", "500",
            "--lr", "0.0001",
            "--threshold", "0.4",
            "--alpha", "0.5",
            "--knn_k", "5",
            "--enable_dcgl_negative_loss",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
            "--warmup_epochs", "35",
            "--fusion_hidden", "64",
            "--fusion_temp", "1.9",
            "--fusion_balance", "0.35",
            "--fusion_min_weight", "0.2",
            "--lambda_inst", "0.08",
            "--lambda_clu", "0.07",
        ),
    ),
    RerunSpec(
        dataset_key="amap",
        dataset_label="AMAP",
        selected_point="77.39 / 67.22 / 58.25 / 71.69",
        source_log=(
            "experiment_output/paper_param_sensitivity_all_datasets/fusion_min_weight_0p05/"
            "fusion_min_weight_0p05_run1/amap/amap_train_with_dual_attn_20260430_031034.txt"
        ),
        note="Selected best complete 10-run attention/DCGL-negative-only row from all experiment_output logs.",
        command_args=(
            "--dataset", "amap",
            "--cluster_num", "8",
            "--graph_mode", "dual",
            "--fusion_mode", "attn",
            "--t", "4",
            "--linlayers", "1",
            "--epochs", "400",
            "--dims", "500",
            "--lr", "0.0001",
            "--threshold", "0.4",
            "--alpha", "0.5",
            "--knn_k", "5",
            "--enable_dcgl_negative_loss",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
            "--fusion_hidden", "64",
            "--fusion_temp", "1.25",
            "--fusion_balance", "0.08",
            "--lambda_inst", "0.07",
            "--lambda_clu", "0.035",
            "--warmup_epochs", "35",
            "--fusion_min_weight", "0.05",
        ),
    ),
    RerunSpec(
        dataset_key="usps",
        dataset_label="USPS",
        selected_point="82.40 / 73.29 / 68.39 / 82.16",
        source_log="experiment_output/usps/usps_train_with_dual_attn_20260420_092257.txt",
        note="Selected best complete 10-run attention/DCGL-negative-only row from all experiment_output logs.",
        command_args=(
            "--dataset", "usps",
            "--cluster_num", "10",
            "--graph_mode", "dual",
            "--fusion_mode", "attn",
            "--t", "4",
            "--linlayers", "1",
            "--epochs", "400",
            "--dims", "500",
            "--lr", "0.0001",
            "--threshold", "0.4",
            "--alpha", "0.5",
            "--knn_k", "5",
            "--enable_dcgl_negative_loss",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
            "--fusion_hidden", "64",
            "--fusion_temp", "1.8",
            "--fusion_balance", "0.35",
            "--lambda_inst", "0.09",
            "--lambda_clu", "0.09",
            "--warmup_epochs", "35",
            "--fusion_min_weight", "0.2",
        ),
    ),
    RerunSpec(
        dataset_key="eat",
        dataset_label="EAT",
        selected_point="54.76 / 31.97 / 24.32 / 52.98",
        source_log="experiment_output/eat/eat_train_with_dual_attn_20260418_135705.txt",
        note="Selected best complete 10-run attention/DCGL-negative-only row from all experiment_output logs.",
        command_args=(
            "--dataset", "eat",
            "--cluster_num", "4",
            "--graph_mode", "dual",
            "--fusion_mode", "attn",
            "--t", "4",
            "--linlayers", "1",
            "--epochs", "400",
            "--dims", "500",
            "--lr", "0.0001",
            "--threshold", "0.4",
            "--alpha", "0.5",
            "--knn_k", "5",
            "--enable_dcgl_negative_loss",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
            "--fusion_hidden", "64",
            "--fusion_temp", "2",
            "--fusion_balance", "0.35",
            "--lambda_inst", "0.08",
            "--lambda_clu", "0.08",
            "--warmup_epochs", "35",
            "--fusion_min_weight", "0.2",
        ),
    ),
    RerunSpec(
        dataset_key="cora",
        dataset_label="Cora",
        selected_point="73.49 / 55.55 / 50.14 / 71.17",
        source_log="experiment_output/cora/cora_train_with_dual_attn_20260425_105003.txt",
        note="Selected best complete 10-run attention/DCGL-negative-only row from all experiment_output logs.",
        command_args=(
            "--dataset", "cora",
            "--cluster_num", "7",
            "--graph_mode", "dual",
            "--fusion_mode", "attn",
            "--t", "4",
            "--linlayers", "1",
            "--epochs", "400",
            "--dims", "500",
            "--lr", "0.0001",
            "--threshold", "0.4",
            "--alpha", "0.5",
            "--knn_k", "5",
            "--enable_dcgl_negative_loss",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
            "--fusion_hidden", "64",
            "--fusion_temp", "1.3",
            "--fusion_balance", "0.0",
            "--lambda_inst", "0.03",
            "--lambda_clu", "0.01",
            "--warmup_epochs", "70",
            "--fusion_min_weight", "0.0",
            "--enable_branch_bias_fusion",
            "--branch_bias_target", "raw",
            "--branch_bias_cap", "0.1",
        ),
    ),
    RerunSpec(
        dataset_key="cite",
        dataset_label="Citeseer",
        selected_point="70.74 / 45.28 / 45.18 / 61.56",
        source_log="experiment_output/cite/cite_train_with_dual_attn_20260425_004609.txt",
        note="Selected best complete 10-run attention/DCGL-negative-only row from all experiment_output logs.",
        command_args=(
            "--dataset", "cite",
            "--cluster_num", "6",
            "--graph_mode", "dual",
            "--fusion_mode", "attn",
            "--t", "4",
            "--linlayers", "1",
            "--epochs", "400",
            "--dims", "500",
            "--lr", "0.0001",
            "--threshold", "0.4",
            "--alpha", "0.5",
            "--knn_k", "5",
            "--enable_dcgl_negative_loss",
            "--dcgl_neg_tau", "0.5",
            "--dcgl_neg_weight", "0.6",
            "--fusion_hidden", "64",
            "--fusion_temp", "1.8",
            "--fusion_balance", "0.15",
            "--lambda_inst", "0.045",
            "--lambda_clu", "0.02",
            "--warmup_epochs", "55",
            "--fusion_min_weight", "0.1",
            "--enable_branch_bias_fusion",
            "--branch_bias_target", "raw",
            "--branch_bias_cap", "0.15",
        ),
    ),
)


def parse_final_metrics(text: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        match = re.match(
            r"^\s*(ACC|NMI|ARI|F1)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*(?:±|卤|\+/-)\s*([0-9]+(?:\.[0-9]+)?)",
            line,
        )
        if match:
            metrics[match.group(1)] = {
                "mean": float(match.group(2)),
                "std": float(match.group(3)),
            }
            continue
        if re.match(r"^\s*(ACC|NMI|ARI|F1)\s*\|", line):
            metric_name = line.split("|", 1)[0].strip()
            nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", line.split("|", 1)[1])
            if metric_name in METRICS and len(nums) >= 2:
                metrics[metric_name] = {"mean": float(nums[0]), "std": float(nums[1])}
    return metrics


def parse_per_seed(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"Run\s+(\d+)\s+Done\s+\|\s+Seed:\s+(-?\d+)\s+\|\s+"
        r"ACC:\s+([0-9]+(?:\.[0-9]+)?)\s+\|\s+"
        r"NMI:\s+([0-9]+(?:\.[0-9]+)?)\s+\|\s+"
        r"ARI:\s+([0-9]+(?:\.[0-9]+)?)\s+\|\s+"
        r"F1:\s+([0-9]+(?:\.[0-9]+)?)"
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "run": int(match.group(1)),
                "seed": int(match.group(2)),
                "metrics": {
                    "ACC": float(match.group(3)),
                    "NMI": float(match.group(4)),
                    "ARI": float(match.group(5)),
                    "F1": float(match.group(6)),
                },
            }
        )
    return rows


def metrics_from_per_seed(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    metrics: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = [float(row["metrics"][metric]) for row in rows if metric in row.get("metrics", {})]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        metrics[metric] = {"mean": mean, "std": variance ** 0.5}
    return metrics


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    record = dict(record)
    metrics = record.get("metrics") or {}
    per_seed = record.get("per_seed") or []
    if per_seed:
        metrics = metrics_from_per_seed(per_seed)
        record["metrics"] = metrics
    if int(record.get("returncode", 0) or 0) == 0 and len(record.get("metrics") or {}) == len(METRICS):
        record["status"] = "OK"
    return record


def command_for_spec(spec: RerunSpec, python: Path, runs: int, seed_start: int, device: str) -> list[str]:
    cmd = [str(python), "train.py"]
    args = list(spec.command_args)
    args.extend(["--device", device, "--runs", str(runs), "--seed_start", str(seed_start)])
    return cmd + args


def write_summary(out_dir: Path, records: list[dict[str, Any]], runs: int, seed_start: int) -> Path:
    summary_path = out_dir / "summary.md"
    lines = [
        "# DSAFC Main Ours 10-Run Recheck",
        "",
        f"- Started/updated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Runs per dataset: {runs}",
        f"- Seed window: {seed_start}..{seed_start + runs - 1}",
        "- Policy: rerun the selected best complete 10-run attention/DCGL-negative-only parameters with `train.py --runs 10`.",
        "- Scope: Reuters, UAT, AMAP, USPS, EAT, Cora, Citeseer.",
        "",
        "## Main Summary",
        "",
        "| Dataset | Status | Selected complete 10-run row | Rerun ACC | Rerun NMI | Rerun ARI | Rerun F1 | Seeds | Log |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for record in records:
        metrics = record.get("metrics", {})

        def cell(metric: str) -> str:
            row = metrics.get(metric)
            if not row:
                return "--"
            return f"{row['mean']:.2f} +/- {row['std']:.2f}"

        lines.append(
            f"| {record['dataset_label']} | {record['status']} | {record['selected_point']} | "
            f"{cell('ACC')} | {cell('NMI')} | {cell('ARI')} | {cell('F1')} | "
            f"{record['seed_start']}..{record['seed_end']} | `{record['log_path']}` |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Results",
            "",
        ]
    )
    for record in records:
        lines.extend(
            [
                f"### {record['dataset_label']}",
                "",
                f"- Source best log: `{record['source_log']}`",
                f"- Note: {record['note']}",
                "",
                "| Run | Seed | ACC | NMI | ARI | F1 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in record.get("per_seed", []):
            metric = row["metrics"]
            lines.append(
                f"| {row['run']} | {row['seed']} | {metric['ACC']:.2f} | {metric['NMI']:.2f} | "
                f"{metric['ARI']:.2f} | {metric['F1']:.2f} |"
            )
        if not record.get("per_seed"):
            lines.append("| -- | -- | -- | -- | -- | -- |")
        lines.append("")

    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return summary_path


def load_existing_records(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "results.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    records = payload.get("records", [])
    return records if isinstance(records, list) else []


def merge_records(existing: list[dict[str, Any]], new_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in existing + new_records:
        record = normalize_record(record)
        dataset_key = str(record.get("dataset_key", ""))
        if not dataset_key:
            continue
        if dataset_key not in by_dataset:
            order.append(dataset_key)
        by_dataset[dataset_key] = record
    spec_order = [spec.dataset_key for spec in SPECS]
    ordered_keys = sorted(order, key=lambda key: spec_order.index(key) if key in spec_order else len(spec_order))
    return [by_dataset[key] for key in ordered_keys]


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun selected DSAFC/Ours rows with train.py 10-run mean/std.")
    parser.add_argument("--datasets", default="all", help="Comma list of dataset keys, or all.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    requested = {item.strip().lower() for item in str(args.datasets).split(",") if item.strip()}
    selected = list(SPECS) if requested == {"all"} else [spec for spec in SPECS if spec.dataset_key in requested]
    if not selected:
        raise ValueError(f"No matching datasets for: {args.datasets}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (OUTPUT_ROOT / timestamp)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_records = load_existing_records(out_dir)
    records: list[dict[str, Any]] = []
    for spec in selected:
        cmd = command_for_spec(spec, args.python, args.runs, args.seed_start, args.device)
        log_path = out_dir / f"{spec.dataset_key}_runs{args.runs}_seed{args.seed_start}.txt"
        rel_log = str(log_path.relative_to(ROOT))
        print(f"[RUN] {spec.dataset_label}: {rel_log}", flush=True)
        if args.dry_run:
            log_path.write_text("COMMAND: " + " ".join(cmd) + "\n", encoding="utf-8")
            records.append(
                {
                    "dataset_key": spec.dataset_key,
                    "dataset_label": spec.dataset_label,
                    "selected_point": spec.selected_point,
                    "source_log": spec.source_log,
                    "note": spec.note,
                    "status": "DRY-RUN",
                    "seed_start": args.seed_start,
                    "seed_end": args.seed_start + args.runs - 1,
                    "cmd": cmd,
                    "log_path": rel_log,
                    "metrics": {},
                    "per_seed": [],
                }
            )
            continue

        start = datetime.now()
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout if args.timeout > 0 else None,
            )
            output = proc.stdout or ""
            returncode = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            output += f"\n[TIMEOUT] Dataset exceeded {args.timeout} seconds.\n"
            returncode = 124
            timed_out = True
        elapsed = (datetime.now() - start).total_seconds()
        log_path.write_text(
            "\n".join(
                [
                    "COMMAND: " + " ".join(cmd),
                    f"RETURN_CODE: {returncode}",
                    f"ELAPSED_SEC: {elapsed:.2f}",
                    f"TIMED_OUT: {'YES' if timed_out else 'NO'}",
                    "=" * 80,
                    output,
                ]
            ),
            encoding="utf-8",
            errors="replace",
        )
        metrics = parse_final_metrics(output)
        status = "OK" if returncode == 0 and len(metrics) == len(METRICS) else f"FAIL({returncode})"
        records.append(
            {
                "dataset_key": spec.dataset_key,
                "dataset_label": spec.dataset_label,
                "selected_point": spec.selected_point,
                "source_log": spec.source_log,
                "note": spec.note,
                "status": status,
                "returncode": returncode,
                "elapsed_sec": elapsed,
                "timed_out": timed_out,
                "seed_start": args.seed_start,
                "seed_end": args.seed_start + args.runs - 1,
                "cmd": cmd,
                "log_path": rel_log,
                "metrics": metrics,
                "per_seed": parse_per_seed(output),
            }
        )
        merged_records = merge_records(existing_records, records)
        (out_dir / "results.jsonl").write_text(
            "\n".join(json.dumps(record, ensure_ascii=True) for record in merged_records) + "\n",
            encoding="utf-8",
        )
        write_summary(out_dir, merged_records, args.runs, args.seed_start)

    merged_records = merge_records(existing_records, records)
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "started_or_updated": datetime.now().isoformat(timespec="seconds"),
                "runs": args.runs,
                "seed_start": args.seed_start,
                "seed_end": args.seed_start + args.runs - 1,
                "records": merged_records,
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = write_summary(out_dir, merged_records, args.runs, args.seed_start)
    print(f"[SUMMARY] {summary_path}", flush=True)
    return 0 if all(record["status"] in {"OK", "DRY-RUN"} for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
