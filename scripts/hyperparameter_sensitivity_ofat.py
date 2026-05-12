from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "hyperparameter_sensitivity_ofat"
SCGC_ENV_PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
DEFAULT_PYTHON = SCGC_ENV_PYTHON if SCGC_ENV_PYTHON.exists() else Path(sys.executable)

METRICS = ("ACC", "NMI", "ARI", "F1")
DEFAULT_DATASETS = ("reut", "uat", "amap", "usps", "cora", "cite")
DATASET_ALIASES = {
    "reuters": "reut",
    "reut": "reut",
    "uat": "uat",
    "amap": "amap",
    "camap": "amap",
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
PARAM_LABELS = {
    "t": "propagation depth t",
    "k_E": "refined graph k_E",
    "fusion_temp": "fusion temperature tau_f",
    "dcgl_neg_weight": "cluster-separation strength lambda_sep",
    "lambda_inst": "lambda_inst",
    "lambda_clu": "lambda_clu",
    "warmup_epochs": "warmup epochs",
}
DEFAULT_VALUES = {
    "t": (1, 2, 3, 4, 5, 6),
    "k_E": (5, 10, 15, 20, 25, 30),
    "fusion_temp": (1.0, 1.3, 1.6, 1.9, 2.2),
    "dcgl_neg_weight": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "lambda_inst": (0.0, 0.03, 0.06, 0.09, 0.12),
    "lambda_clu": (0.0, 0.02, 0.04, 0.06, 0.08),
    "warmup_epochs": (25, 35, 45, 55, 70, 85),
}
RESOURCE_LABEL_TO_KEY = {
    "Wall time (sec)": "wall_time_sec",
    "CPU util (%)": "cpu_percent",
    "Process CPU (%)": "process_cpu_percent",
    "RAM used (GB)": "ram_used_gb",
    "RAM total (GB)": "ram_total_gb",
    "RAM util (%)": "ram_percent",
    "Process RSS (GB)": "process_rss_gb",
    "GPU util (%)": "gpu_util_percent",
    "GPU memory used (GB)": "gpu_memory_used_gb",
    "GPU memory total (GB)": "gpu_memory_total_gb",
    "Torch max allocated (GB)": "torch_gpu_allocated_gb",
    "Torch max reserved (GB)": "torch_gpu_reserved_gb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-factor-at-a-time hyperparameter sensitivity for the final "
            "DSAFC dual-attention setting. By default this reuses current AE "
            "graph assets and varies t/k_E/fusion_temp/lambda_sep."
        )
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Dataset key, comma-separated keys, or 'all'.",
    )
    parser.add_argument(
        "--params",
        default="t,k_E,dcgl_neg_weight,fusion_temp",
        help=(
            "Comma-separated parameters to sweep. Supported: t, k_E, fusion_temp, "
            "dcgl_neg_weight, lambda_inst, lambda_clu, warmup_epochs. "
            "The code name dcgl_neg_weight is reported as lambda_sep in tables."
        ),
    )
    parser.add_argument(
        "--include-ke",
        action="store_true",
        help="Also run k_E sensitivity with isolated AE graph generation in the run directory.",
    )
    parser.add_argument(
        "--values",
        action="append",
        default=[],
        metavar="PARAM=v1,v2,...",
        help="Override sweep values for one parameter, e.g. --values fusion_temp=1.0,1.6,2.2.",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--timeout", type=int, default=0, help="Per command timeout in seconds; 0 disables timeout.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_experiment_config() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("dsafc_experiment", ROOT / "experiment.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def parse_dataset_list(raw: str, config: dict[str, Any]) -> tuple[str, ...]:
    profiles = config.get("dataset_profiles", {})
    if str(raw).strip().lower() == "all":
        return tuple(dataset for dataset in DEFAULT_DATASETS if dataset in profiles)

    datasets: list[str] = []
    for token in str(raw).replace(";", ",").split(","):
        key = token.strip().lower().replace("-", "_")
        if not key:
            continue
        key = DATASET_ALIASES.get(key, key)
        if key not in profiles:
            raise ValueError(f"Unsupported dataset '{token}'.")
        datasets.append(key)
    if not datasets:
        raise ValueError("No valid datasets were provided.")
    return tuple(dict.fromkeys(datasets))


def parse_param_list(raw: str, include_ke: bool) -> tuple[str, ...]:
    aliases = {
        "ke": "k_E",
        "k_e": "k_E",
        "k": "k_E",
        "refined_k": "k_E",
        "ae_k": "k_E",
        "propagation_depth": "t",
        "smooth_t": "t",
        "t": "t",
        "fusion_temp": "fusion_temp",
        "tf": "fusion_temp",
        "tau_f": "fusion_temp",
        "dcgl_neg_weight": "dcgl_neg_weight",
        "neg_weight": "dcgl_neg_weight",
        "negative_weight": "dcgl_neg_weight",
        "lambda_sep": "dcgl_neg_weight",
        "sep": "dcgl_neg_weight",
        "separation_weight": "dcgl_neg_weight",
        "cluster_separation": "dcgl_neg_weight",
        "lambda_inst": "lambda_inst",
        "lambda_clu": "lambda_clu",
        "warmup": "warmup_epochs",
        "warmup_epoch": "warmup_epochs",
        "warmup_epochs": "warmup_epochs",
    }
    params: list[str] = []
    for token in str(raw).replace(";", ",").split(","):
        key = token.strip()
        if not key:
            continue
        canonical = aliases.get(key.lower(), key)
        if canonical not in DEFAULT_VALUES:
            raise ValueError(f"Unsupported sensitivity parameter '{token}'.")
        params.append(canonical)
    if include_ke and "k_E" not in params:
        params.append("k_E")
    if not params:
        raise ValueError("No valid sensitivity parameters were provided.")
    return tuple(dict.fromkeys(params))


def parse_value_overrides(raw_values: list[str]) -> dict[str, tuple[float | int, ...]]:
    overrides: dict[str, tuple[float | int, ...]] = {}
    aliases = {
        "ke": "k_E",
        "k_e": "k_E",
        "k": "k_E",
        "refined_k": "k_E",
        "ae_k": "k_E",
        "propagation_depth": "t",
        "smooth_t": "t",
        "tf": "fusion_temp",
        "tau_f": "fusion_temp",
        "dcgl_neg_weight": "dcgl_neg_weight",
        "neg_weight": "dcgl_neg_weight",
        "negative_weight": "dcgl_neg_weight",
        "lambda_sep": "dcgl_neg_weight",
        "sep": "dcgl_neg_weight",
        "separation_weight": "dcgl_neg_weight",
        "cluster_separation": "dcgl_neg_weight",
    }
    for item in raw_values:
        if "=" not in item:
            raise ValueError(f"Invalid --values entry '{item}', expected PARAM=v1,v2,...")
        raw_key, raw_vals = item.split("=", 1)
        key = aliases.get(raw_key.strip().lower(), raw_key.strip())
        if key not in DEFAULT_VALUES:
            raise ValueError(f"Unsupported --values parameter '{raw_key}'.")
        parsed: list[float | int] = []
        for token in raw_vals.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            if key in {"k_E", "t", "warmup_epochs"}:
                parsed.append(int(float(token)))
            else:
                parsed.append(float(token))
        if not parsed:
            raise ValueError(f"No values were provided for {key}.")
        overrides[key] = tuple(dict.fromkeys(parsed))
    return overrides


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


def has_npy_triplet(dataset: str) -> bool:
    dataset_dir = ROOT / "data" / "full_dataset" / dataset
    if not dataset_dir.exists():
        return False
    for feat_path in dataset_dir.rglob("*_feat.npy"):
        prefix = feat_path.name[: -len("_feat.npy")]
        if (feat_path.parent / f"{prefix}_label.npy").exists() and (feat_path.parent / f"{prefix}_adj.npy").exists():
            return True
    return False


def resolve_current_ae_graph(dataset: str) -> Path:
    path = ROOT / "data" / "ae_graph" / f"{dataset}_ae_graph.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing current AE graph asset: {path}")
    return path


def build_module_args(config: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}

    legacy_enabled = bool(config.get("enable_improved_module", False))
    legacy_args = config.get("improved_module_args", {})

    if bool(config.get("enable_dynamic_threshold_module", legacy_enabled)):
        args["enable_dynamic_threshold"] = True
        args.update(config.get("dynamic_threshold_args", {}))
        for key in ("dynamic_threshold_start", "dynamic_threshold_end"):
            if key in legacy_args and key not in args:
                args[key] = legacy_args[key]

    if bool(config.get("enable_ema_prototypes_module", legacy_enabled)):
        args["enable_ema_prototypes"] = True
        args.update(config.get("ema_prototypes_args", {}))
        if "ema_proto_momentum" in legacy_args and "ema_proto_momentum" not in args:
            args["ema_proto_momentum"] = legacy_args["ema_proto_momentum"]

    if bool(config.get("enable_dcgl_negative_module", False)):
        args["enable_dcgl_negative_loss"] = True
        args.update(config.get("dcgl_negative_args", {}))
        args.update(profile.get("dcgl_negative_args", {}))
        if args.pop("disable_dcgl_neg_reliability_gate", False):
            args["disable_dcgl_neg_reliability_gate"] = True

    if bool(config.get("enable_dcgl_cluster_module", False)):
        args["enable_dcgl_cluster_level"] = True
        args.update(config.get("dcgl_cluster_args", {}))
        args.update(profile.get("dcgl_cluster_args", {}))

    if bool(config.get("enable_gcn_backbone_module", False)):
        args["enable_gcn_backbone"] = True
        args.update(config.get("gcn_backbone_args", {}))
        args.update(profile.get("gcn_backbone_args", {}))

    return args


def build_train_command(
    config: dict[str, Any],
    dataset: str,
    ae_graph_path: Path,
    param: str,
    value: float | int,
    args: argparse.Namespace,
    fusion_weight_path: Path,
) -> list[str]:
    profile = config["dataset_profiles"][dataset]
    cluster_num = int(profile["cluster_num"])

    train_args = merge_args(
        config.get("baseline_args", {}),
        config.get("train_common_args", {}),
        profile.get("train_args", {}),
    )
    train_args["device"] = args.device
    train_args["runs"] = int(args.runs)
    train_args["seed_start"] = int(args.seed_start)
    if param == "t":
        train_args["t"] = int(value)

    dual_attn_args = merge_args(config.get("dual_attn_args", {}), profile.get("dual_attn_args", {}))
    if param in dual_attn_args or param in {"fusion_temp", "lambda_inst", "lambda_clu"}:
        dual_attn_args[param] = value
    elif param not in {"k_E", "t", "dcgl_neg_weight"}:
        raise ValueError(f"Cannot apply parameter '{param}' to train.py command.")

    module_args = build_module_args(config, profile)
    if param == "dcgl_neg_weight":
        module_args["dcgl_neg_weight"] = float(value)

    train_args = merge_args(train_args, config.get("dual_args", {}), dual_attn_args)

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
        str(ae_graph_path),
        "--fusion_mode",
        "attn",
        "--save_fusion_weights_path",
        str(fusion_weight_path),
    ]
    cmd.extend(dict_to_cli(train_args))
    cmd.extend(dict_to_cli(module_args))
    return cmd


def build_pretrain_command(
    config: dict[str, Any],
    dataset: str,
    k_e: int,
    args: argparse.Namespace,
    graph_path: Path,
    model_path: Path,
) -> list[str]:
    profile = config["dataset_profiles"][dataset]
    cluster_num = int(profile["cluster_num"])
    ae_args = merge_args(config.get("ae_args", {}), profile.get("ae_args", {}))
    if has_npy_triplet(dataset):
        ae_args.pop("base_graph_path", None)
    ae_args["ae_k"] = int(k_e)
    ae_args["out_graph_path"] = graph_path
    ae_args["model_save_path"] = model_path
    ae_args.setdefault("n_z", cluster_num)
    ae_args["device"] = args.device

    cmd = [
        str(args.python),
        "data/pretrain_optimize_A_graph.py",
        "--dataset",
        dataset,
        "--cluster_num",
        str(cluster_num),
    ]
    cmd.extend(dict_to_cli(ae_args))
    return cmd


def run_command(cmd: list[str], *, timeout: int) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=None if timeout <= 0 else timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_final_metrics(stdout: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    pattern = re.compile(r"^(ACC|NMI|ARI|F1)\s+\|\s+([+-]?\d+(?:\.\d+)?)\s+.+?\s+([+-]?\d+(?:\.\d+)?)$", re.MULTILINE)
    for match in pattern.finditer(stdout or ""):
        metrics[match.group(1)] = {"mean": float(match.group(2)), "std": float(match.group(3))}
    return metrics


def parse_resource_summary(stdout: str) -> dict[str, float]:
    resource: dict[str, float] = {}
    pattern = re.compile(r"^RESOURCE\s+\|\s+(.+?)\s+\|\s+([+-]?\d+(?:\.\d+)?)\s*$", re.MULTILINE)
    for match in pattern.finditer(stdout or ""):
        key = RESOURCE_LABEL_TO_KEY.get(match.group(1).strip())
        if key:
            resource[key] = float(match.group(2))
    return resource


def metric_text(metrics: dict[str, dict[str, float]], metric: str) -> str:
    item = metrics.get(metric)
    if not item:
        return ""
    return f"{item['mean']:.2f}+-{item['std']:.2f}"


def score_metrics(metrics: dict[str, dict[str, float]]) -> float:
    if not all(metric in metrics for metric in METRICS):
        return float("-inf")
    return (
        metrics["ACC"]["mean"]
        + 0.4 * metrics["F1"]["mean"]
        + 0.2 * metrics["NMI"]["mean"]
        + 0.2 * metrics["ARI"]["mean"]
    )


def safe_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): safe_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(item) for item in obj]
    return obj


def job_key(dataset: str, param: str, value: float | int) -> str:
    if param in {"k_E", "t"}:
        value_text = str(int(value))
    else:
        value_text = f"{float(value):.8g}"
    return f"{dataset}|{param}|{value_text}"


def load_finished(results_path: Path) -> dict[str, dict[str, Any]]:
    if not results_path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("returncode") == 0 and row.get("metrics"):
            rows[row["job_key"]] = row
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_json(row), ensure_ascii=False) + "\n")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float | int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("returncode") != 0 or not row.get("metrics"):
            continue
        grouped.setdefault((row["param"], row["value"]), []).append(row)

    out: list[dict[str, Any]] = []
    for (param, value), items in sorted(grouped.items(), key=lambda x: (x[0][0], float(x[0][1]))):
        record: dict[str, Any] = {"param": param, "value": value, "datasets": len(items)}
        for metric in METRICS:
            means = [item["metrics"][metric]["mean"] for item in items if metric in item.get("metrics", {})]
            stds = [item["metrics"][metric]["std"] for item in items if metric in item.get("metrics", {})]
            if means:
                record[f"avg_{metric.lower()}"] = sum(means) / len(means)
            if stds:
                record[f"avg_{metric.lower()}_std"] = sum(stds) / len(stds)
        score_values = [score_metrics(item["metrics"]) for item in items]
        score_values = [score for score in score_values if score != float("-inf")]
        if score_values:
            record["avg_score"] = sum(score_values) / len(score_values)
        out.append(record)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(run_dir: Path, rows: list[dict[str, Any]], aggregate: list[dict[str, Any]], args: argparse.Namespace) -> None:
    datasets = tuple(dict.fromkeys(row["dataset"] for row in rows))
    params = tuple(dict.fromkeys(row["param"] for row in rows))
    lines: list[str] = []
    lines.append("# DSAFC Hyperparameter Sensitivity OFAT")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Datasets: {', '.join(DATASET_LABELS.get(dataset, dataset) for dataset in datasets)}")
    lines.append(f"- Runs per setting: {args.runs}")
    lines.append(f"- Train seeds: {args.seed_start}..{args.seed_start + args.runs - 1}")
    lines.append(f"- Device: {args.device}")
    lines.append(f"- Parameters: {', '.join(PARAM_LABELS.get(param, param) for param in params)}")
    lines.append("- Protocol: one-factor-at-a-time around current experiment.py final DSAFC dual-attention settings.")
    lines.append("- AE assets: current project AE graph is reused except k_E, whose AE graph is generated inside this run directory.")
    lines.append("- Naming note: code parameter `dcgl_neg_weight` is reported as the paper-facing cluster-separation strength `lambda_sep`.")
    lines.append("")

    lines.append("## Aggregate By Parameter Value")
    lines.append("")
    lines.append("| Parameter | Value | Datasets | Avg ACC | Avg NMI | Avg ARI | Avg F1 | Avg Score |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in aggregate:
        lines.append(
            "| {param} | {value} | {datasets} | {acc:.2f} | {nmi:.2f} | {ari:.2f} | {f1:.2f} | {score:.2f} |".format(
                param=PARAM_LABELS.get(str(item["param"]), str(item["param"])),
                value=item["value"],
                datasets=int(item.get("datasets", 0)),
                acc=float(item.get("avg_acc", 0.0)),
                nmi=float(item.get("avg_nmi", 0.0)),
                ari=float(item.get("avg_ari", 0.0)),
                f1=float(item.get("avg_f1", 0.0)),
                score=float(item.get("avg_score", 0.0)),
            )
        )
    lines.append("")

    lines.append("## Best Aggregate Value")
    lines.append("")
    lines.append("| Parameter | Best by Avg ACC | Avg ACC | Best by Avg Score | Avg Score |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for param in params:
        items = [item for item in aggregate if item["param"] == param]
        if not items:
            continue
        best_acc = max(items, key=lambda item: float(item.get("avg_acc", float("-inf"))))
        best_score = max(items, key=lambda item: float(item.get("avg_score", float("-inf"))))
        lines.append(
            "| {param} | {acc_value} | {acc:.2f} | {score_value} | {score:.2f} |".format(
                param=PARAM_LABELS.get(param, param),
                acc_value=best_acc["value"],
                acc=float(best_acc.get("avg_acc", 0.0)),
                score_value=best_score["value"],
                score=float(best_score.get("avg_score", 0.0)),
            )
        )
    lines.append("")

    lines.append("## Per-Dataset Results")
    lines.append("")
    lines.append("| Dataset | Parameter | Value | ACC | NMI | ARI | F1 | Score | AE graph | Log |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in sorted(rows, key=lambda item: (item["dataset"], item["param"], float(item["value"]))):
        metrics = row.get("metrics", {})
        lines.append(
            "| {dataset} | {param} | {value} | {acc} | {nmi} | {ari} | {f1} | {score:.2f} | `{ae}` | `{log}` |".format(
                dataset=DATASET_LABELS.get(row["dataset"], row["dataset"]),
                param=PARAM_LABELS.get(row["param"], row["param"]),
                value=row["value"],
                acc=metric_text(metrics, "ACC"),
                nmi=metric_text(metrics, "NMI"),
                ari=metric_text(metrics, "ARI"),
                f1=metric_text(metrics, "F1"),
                score=score_metrics(metrics),
                ae=rel(Path(row.get("ae_graph_path", ""))),
                log=rel(Path(row.get("stdout_log", ""))),
            )
        )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Raw JSONL: `{rel(run_dir / 'results.jsonl')}`")
    lines.append(f"- Aggregate CSV: `{rel(run_dir / 'aggregate.csv')}`")
    lines.append(f"- Per-setting CSV: `{rel(run_dir / 'per_dataset.csv')}`")
    (run_dir / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def persist_outputs(run_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    aggregate = aggregate_rows(rows)
    per_dataset_rows: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics", {})
        record = {
            "dataset": row.get("dataset"),
            "param": row.get("param"),
            "value": row.get("value"),
            "returncode": row.get("returncode"),
            "score": score_metrics(metrics),
            "ae_graph_path": row.get("ae_graph_path"),
            "stdout_log": row.get("stdout_log"),
        }
        for metric in METRICS:
            record[f"{metric.lower()}_mean"] = metrics.get(metric, {}).get("mean", "")
            record[f"{metric.lower()}_std"] = metrics.get(metric, {}).get("std", "")
        per_dataset_rows.append(record)

    write_csv(
        run_dir / "per_dataset.csv",
        per_dataset_rows,
        [
            "dataset",
            "param",
            "value",
            "returncode",
            "acc_mean",
            "acc_std",
            "nmi_mean",
            "nmi_std",
            "ari_mean",
            "ari_std",
            "f1_mean",
            "f1_std",
            "score",
            "ae_graph_path",
            "stdout_log",
        ],
    )
    write_csv(
        run_dir / "aggregate.csv",
        aggregate,
        [
            "param",
            "value",
            "datasets",
            "avg_acc",
            "avg_acc_std",
            "avg_nmi",
            "avg_nmi_std",
            "avg_ari",
            "avg_ari_std",
            "avg_f1",
            "avg_f1_std",
            "avg_score",
        ],
    )
    write_summary(run_dir, rows, aggregate, args)


def materialize_ae_for_ke(
    config: dict[str, Any],
    dataset: str,
    k_e: int,
    args: argparse.Namespace,
    run_dir: Path,
) -> Path:
    graph_path = run_dir / "ae_assets" / dataset / f"k_E_{int(k_e)}" / f"{dataset}_ae_k{int(k_e)}_graph.txt"
    model_path = run_dir / "ae_assets" / dataset / f"k_E_{int(k_e)}" / f"{dataset}_ae_k{int(k_e)}_pretrain.pkl"
    if graph_path.exists():
        return graph_path

    cmd = build_pretrain_command(config, dataset, int(k_e), args, graph_path, model_path)
    log_dir = run_dir / "logs" / dataset / "ae_pretrain"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"k_E_{int(k_e)}.stdout.log"
    stderr_log = log_dir / f"k_E_{int(k_e)}.stderr.log"
    command_log = log_dir / f"k_E_{int(k_e)}.cmd.txt"
    command_log.write_text(" ".join(cmd) + "\n", encoding="utf-8")

    print(f"[AE] {dataset} k_E={int(k_e)} -> {rel(graph_path)}", flush=True)
    if args.dry_run:
        return graph_path

    rc, stdout, stderr = run_command(cmd, timeout=args.timeout)
    stdout_log.write_text(stdout or "", encoding="utf-8")
    stderr_log.write_text(stderr or "", encoding="utf-8")
    if rc != 0:
        raise RuntimeError(f"AE pretrain failed for {dataset} k_E={k_e}; see {stderr_log}")
    if not graph_path.exists():
        raise FileNotFoundError(f"AE pretrain completed but graph was not created: {graph_path}")
    return graph_path


def main() -> None:
    args = parse_args()
    config = load_experiment_config()
    datasets = parse_dataset_list(args.datasets, config)
    params = parse_param_list(args.params, args.include_ke)
    value_overrides = parse_value_overrides(args.values)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_dir is not None:
        run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    else:
        suffix = "_".join(params).replace("k_E", "kE")
        run_dir = OUTPUT_ROOT / f"{stamp}_{'_'.join(datasets)}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    finished = load_finished(results_path) if args.resume else {}
    rows = list(finished.values())

    print(f"[RUN] output={run_dir}", flush=True)
    print(f"[RUN] datasets={','.join(datasets)} params={','.join(params)} runs={args.runs}", flush=True)

    for dataset in datasets:
        profile = config["dataset_profiles"][dataset]
        for param in params:
            values = value_overrides.get(param, DEFAULT_VALUES[param])
            for value in values:
                key = job_key(dataset, param, value)
                if key in finished:
                    print(f"[SKIP] {key}", flush=True)
                    continue

                if param == "k_E":
                    ae_graph_path = materialize_ae_for_ke(config, dataset, int(value), args, run_dir)
                else:
                    ae_graph_path = resolve_current_ae_graph(dataset)

                log_dir = run_dir / "logs" / dataset / param
                log_dir.mkdir(parents=True, exist_ok=True)
                value_text = str(int(value)) if param == "k_E" else f"{float(value):.8g}".replace(".", "p")
                stdout_log = log_dir / f"{value_text}.stdout.log"
                stderr_log = log_dir / f"{value_text}.stderr.log"
                command_log = log_dir / f"{value_text}.cmd.txt"
                fusion_path = run_dir / "fusion_weights" / dataset / param / f"{value_text}.npz"

                cmd = build_train_command(config, dataset, ae_graph_path, param, value, args, fusion_path)
                command_log.write_text(" ".join(cmd) + "\n", encoding="utf-8")
                print(f"[TRAIN] {dataset} {param}={value} runs={args.runs}", flush=True)

                if args.dry_run:
                    row = {
                        "job_key": key,
                        "dataset": dataset,
                        "param": param,
                        "value": value,
                        "returncode": 0,
                        "metrics": {},
                        "resource": {},
                        "ae_graph_path": ae_graph_path,
                        "stdout_log": stdout_log,
                        "stderr_log": stderr_log,
                        "command_log": command_log,
                        "note": "dry-run",
                    }
                    append_jsonl(results_path, row)
                    rows.append(row)
                    continue

                rc, stdout, stderr = run_command(cmd, timeout=args.timeout)
                stdout_log.write_text(stdout or "", encoding="utf-8")
                stderr_log.write_text(stderr or "", encoding="utf-8")
                metrics = parse_final_metrics(stdout)
                resource = parse_resource_summary(stdout)
                row = {
                    "job_key": key,
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(dataset, dataset),
                    "param": param,
                    "param_label": PARAM_LABELS.get(param, param),
                    "value": value,
                    "returncode": rc,
                    "metrics": metrics,
                    "score": score_metrics(metrics),
                    "resource": resource,
                    "ae_graph_path": ae_graph_path,
                    "stdout_log": stdout_log,
                    "stderr_log": stderr_log,
                    "command_log": command_log,
                    "train_profile_args": {
                        "train_args": profile.get("train_args", {}),
                        "dual_attn_args": profile.get("dual_attn_args", {}),
                        "dcgl_negative_args": profile.get("dcgl_negative_args", {}),
                    },
                }
                append_jsonl(results_path, row)
                rows.append(row)
                persist_outputs(run_dir, rows, args)

                if rc != 0:
                    print(f"[WARN] failed {dataset} {param}={value}; stderr={rel(stderr_log)}", flush=True)
                else:
                    acc = metrics.get("ACC", {}).get("mean")
                    acc_text = "NA" if acc is None else f"{acc:.2f}"
                    print(f"[OK] {dataset} {param}={value} ACC={acc_text}", flush=True)

    persist_outputs(run_dir, rows, args)
    print(f"[DONE] summary={run_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
