from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "ae_k_seed_grid_validation"
SCGC_ENV_PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
DEFAULT_PYTHON = SCGC_ENV_PYTHON if SCGC_ENV_PYTHON.exists() else Path(sys.executable)
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
DEFAULT_DATASETS = ("reut", "uat", "amap", "usps", "cora", "cite")
DEFAULT_K_VALUES = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
METRICS = ("ACC", "NMI", "ARI", "F1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate AE graph k_E with multiple random AE seeds. Every generated "
            "AE graph/checkpoint is saved in the output directory, then evaluated "
            "with dual-attn + DCGL-negative only."
        )
    )
    parser.add_argument("--dataset", "--datasets", dest="datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--seeds-per-dataset", type=int, default=5)
    parser.add_argument(
        "--ae-seeds",
        default="",
        help="Optional explicit AE seeds shared by all datasets. If empty, sample random seeds per dataset.",
    )
    parser.add_argument("--random-seed-base", type=int, default=0)
    parser.add_argument("--random-seed-max", type=int, default=2_147_483_647)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--timeout", type=int, default=0, help="Per subprocess timeout in seconds; 0 disables timeout.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-ae", action="store_true")
    parser.add_argument("--train-only-existing-ae", action="store_true")
    parser.add_argument("--update-summary-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_experiment_config() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("dsafc_experiment", ROOT / "experiment.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def parse_datasets(raw: str, config: dict[str, Any]) -> tuple[str, ...]:
    profiles = config.get("dataset_profiles", {})
    out: list[str] = []
    for token in str(raw).replace(";", ",").split(","):
        key = token.strip().lower().replace("-", "_")
        if not key:
            continue
        key = DATASET_ALIASES.get(key, key)
        if key not in profiles:
            raise ValueError(f"Unsupported dataset '{token}'.")
        out.append(key)
    if not out:
        raise ValueError("No datasets were provided.")
    return tuple(dict.fromkeys(out))


def parse_int_list(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in str(raw).replace(";", ",").split(","):
        token = token.strip()
        if token:
            values.append(int(float(token)))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return tuple(dict.fromkeys(values))


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


def merge_args(*arg_dicts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for arg_dict in arg_dicts:
        if arg_dict:
            merged.update(arg_dict)
    return merged


def has_npy_triplet(dataset: str) -> bool:
    dataset_dir = ROOT / "data" / "full_dataset" / dataset
    if not dataset_dir.exists():
        return False
    buckets: dict[str, set[str]] = {}
    for npy_path in dataset_dir.rglob("*.npy"):
        name = npy_path.name
        for key in ("feat", "label", "adj"):
            suffix = f"_{key}.npy"
            if name.endswith(suffix):
                buckets.setdefault(name[: -len(suffix)], set()).add(key)
                break
    preferred = [dataset]
    if dataset == "cite":
        preferred.append("citeseer")
    preferred.extend(sorted(buckets))
    return any(buckets.get(key) == {"feat", "label", "adj"} for key in preferred)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): safe_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(item) for item in obj]
    return obj


def load_or_create_dataset_seeds(
    run_dir: Path,
    datasets: tuple[str, ...],
    args: argparse.Namespace,
) -> dict[str, tuple[int, ...]]:
    seed_manifest = run_dir / "ae_seed_manifest.json"
    if args.resume and seed_manifest.exists():
        payload = json.loads(seed_manifest.read_text(encoding="utf-8"))
        return {dataset: tuple(int(seed) for seed in payload["dataset_ae_seeds"][dataset]) for dataset in datasets}

    explicit = parse_int_list(args.ae_seeds) if str(args.ae_seeds).strip() else None
    if explicit is not None:
        dataset_seeds = {dataset: explicit for dataset in datasets}
    else:
        rng = random.SystemRandom()
        lower = int(args.random_seed_base)
        upper = int(args.random_seed_max)
        if upper < lower:
            raise ValueError("--random-seed-max must be >= --random-seed-base")
        population = upper - lower + 1
        if int(args.seeds_per_dataset) > population:
            raise ValueError("Requested more seeds than the sampling range contains.")
        dataset_seeds = {
            dataset: tuple(rng.sample(range(lower, upper + 1), int(args.seeds_per_dataset)))
            for dataset in datasets
        }

    seed_manifest.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "seed_mode": "explicit_shared" if explicit is not None else "random_per_dataset",
                "dataset_ae_seeds": {dataset: list(seeds) for dataset, seeds in dataset_seeds.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_seeds


def asset_paths(run_dir: Path, dataset: str, k_value: int, ae_seed: int) -> tuple[Path, Path]:
    asset_dir = run_dir / "ae_assets" / dataset / f"k_{int(k_value)}" / f"seed_{int(ae_seed)}"
    return (
        asset_dir / f"{dataset}_ae_k{int(k_value)}_seed{int(ae_seed)}_graph.txt",
        asset_dir / f"{dataset}_ae_k{int(k_value)}_seed{int(ae_seed)}_pretrain.pkl",
    )


def build_ae_command(
    config: dict[str, Any],
    dataset: str,
    k_value: int,
    ae_seed: int,
    graph_path: Path,
    model_path: Path,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    profile = config["dataset_profiles"][dataset]
    ae_args = merge_args(config.get("ae_args", {}), profile.get("ae_args", {}))
    if has_npy_triplet(dataset):
        ae_args.pop("base_graph_path", None)
    ae_args.update(
        {
            "dataset": dataset,
            "cluster_num": int(profile["cluster_num"]),
            "ae_k": int(k_value),
            "pretrain_seed": int(ae_seed),
            "graph_seed": int(ae_seed),
            "out_graph_path": graph_path,
            "model_save_path": model_path,
            "device": args.device,
        }
    )
    return [str(args.python), "data/pretrain_optimize_A_graph.py"] + dict_to_cli(ae_args), ae_args


def build_dcgl_negative_args(config: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    dcgl_args: dict[str, Any] = {"enable_dcgl_negative_loss": True}
    dcgl_args.update(config.get("dcgl_negative_args", {}))
    dcgl_args.update(profile.get("dcgl_negative_args", {}))
    if dcgl_args.pop("disable_dcgl_neg_reliability_gate", False):
        dcgl_args["disable_dcgl_neg_reliability_gate"] = True
    return dcgl_args


def build_train_command(
    config: dict[str, Any],
    dataset: str,
    graph_path: Path,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    profile = config["dataset_profiles"][dataset]
    train_args = merge_args(
        config.get("baseline_args", {}),
        config.get("train_common_args", {}),
        profile.get("train_args", {}),
        config.get("dual_args", {}),
        config.get("dual_attn_args", {}),
        profile.get("dual_attn_args", {}),
        build_dcgl_negative_args(config, profile),
    )
    train_args.update({"device": args.device, "runs": int(args.runs), "seed_start": int(args.seed_start)})
    cmd = [
        str(args.python),
        "train.py",
        "--dataset",
        dataset,
        "--cluster_num",
        str(int(profile["cluster_num"])),
        "--graph_mode",
        "dual",
        "--ae_graph_path",
        str(graph_path),
        "--fusion_mode",
        "attn",
    ] + dict_to_cli(train_args)
    return cmd, train_args


def run_subprocess(cmd: list[str], log_path: Path, timeout: int) -> tuple[int, float, bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
            timeout=timeout if timeout > 0 else None,
        )
        output = proc.stdout or ""
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        output += f"\n[TIMEOUT] exceeded {timeout} seconds.\n"
        returncode = 124
        timed_out = True
    elapsed = (datetime.now() - start).total_seconds()
    log_path.write_text(
        "COMMAND: "
        + " ".join(map(str, cmd))
        + f"\nRETURN_CODE: {returncode}\nELAPSED_SEC: {elapsed:.2f}\nTIMED_OUT: {'YES' if timed_out else 'NO'}\n"
        + "=" * 80
        + "\n"
        + output,
        encoding="utf-8",
        errors="replace",
    )
    return returncode, elapsed, timed_out, output


def parse_metrics(text: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    pattern = re.compile(r"^(ACC|NMI|ARI|F1)\s+\|\s+([+-]?\d+(?:\.\d+)?)\s+.+?\s+([+-]?\d+(?:\.\d+)?)$", re.MULTILINE)
    for match in pattern.finditer(text or ""):
        metrics[match.group(1)] = {"mean": float(match.group(2)), "std": float(match.group(3))}
    return metrics


def score_metrics(metrics: dict[str, dict[str, float]]) -> float:
    if not all(metric in metrics for metric in METRICS):
        return float("-inf")
    return (
        metrics["ACC"]["mean"]
        + 0.4 * metrics["F1"]["mean"]
        + 0.2 * metrics["NMI"]["mean"]
        + 0.2 * metrics["ARI"]["mean"]
        - 0.25 * (metrics["ACC"]["std"] + metrics["F1"]["std"])
    )


def result_key(dataset: str, k_value: int, ae_seed: int, args: argparse.Namespace) -> str:
    return json.dumps(
        {
            "dataset": dataset,
            "k": int(k_value),
            "ae_seed": int(ae_seed),
            "runs": int(args.runs),
            "seed_start": int(args.seed_start),
            "seed_end": int(args.seed_start) + int(args.runs) - 1,
        },
        sort_keys=True,
    )


def load_completed(path: Path, args: argparse.Namespace) -> tuple[set[str], list[dict[str, Any]]]:
    if not path.exists() or not args.resume:
        return set(), []
    completed: set[str] = set()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(item)
        if item.get("status") == "done":
            completed.add(result_key(str(item["dataset"]), int(item["k_value"]), int(item["ae_seed"]), args))
    return completed, rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_json(row), ensure_ascii=False) + "\n")


def fmt_metric(metrics: dict[str, dict[str, float]], metric: str) -> str:
    if metric not in metrics:
        return "-"
    value = metrics[metric]
    return f"{value['mean']:.2f}+/-{value['std']:.2f}"


def write_csvs(run_dir: Path, results: list[dict[str, Any]]) -> None:
    metric_rows = [row for row in results if row.get("metrics")]
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "k_value",
            "ae_seed",
            "status",
            "score",
            "acc_mean",
            "acc_std",
            "nmi_mean",
            "nmi_std",
            "ari_mean",
            "ari_std",
            "f1_mean",
            "f1_std",
            "ae_graph_path",
            "ae_model_path",
            "train_log_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            metrics = row.get("metrics", {})
            writer.writerow(
                {
                    "dataset": row.get("dataset"),
                    "k_value": row.get("k_value"),
                    "ae_seed": row.get("ae_seed"),
                    "status": row.get("status"),
                    "score": row.get("score"),
                    "acc_mean": metrics.get("ACC", {}).get("mean", ""),
                    "acc_std": metrics.get("ACC", {}).get("std", ""),
                    "nmi_mean": metrics.get("NMI", {}).get("mean", ""),
                    "nmi_std": metrics.get("NMI", {}).get("std", ""),
                    "ari_mean": metrics.get("ARI", {}).get("mean", ""),
                    "ari_std": metrics.get("ARI", {}).get("std", ""),
                    "f1_mean": metrics.get("F1", {}).get("mean", ""),
                    "f1_std": metrics.get("F1", {}).get("std", ""),
                    "ae_graph_path": row.get("ae_graph_path"),
                    "ae_model_path": row.get("ae_model_path"),
                    "train_log_path": row.get("train_log_path"),
                }
            )


def write_summary(run_dir: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    write_csvs(run_dir, results)
    summary_path = run_dir / "summary.md"
    rows = [row for row in results if row.get("metrics")]
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    by_dataset_k: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        dataset = str(row["dataset"])
        k_value = int(row["k_value"])
        by_dataset.setdefault(dataset, []).append(row)
        by_dataset_k.setdefault((dataset, k_value), []).append(row)

    lines = [
        "# AE k x AE Seed Grid Validation",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Runs per AE asset: {args.runs}",
        f"- Main train seeds: {args.seed_start}..{args.seed_start + args.runs - 1}",
        "- Train mode: dual-attn + DCGL-negative only.",
        f"- Results JSONL: `{rel(run_dir / 'results.jsonl')}`",
        f"- Results CSV: `{rel(run_dir / 'results.csv')}`",
        f"- AE seed manifest: `{rel(run_dir / 'ae_seed_manifest.json')}`",
        "",
    ]
    for dataset in sorted(by_dataset):
        dataset_rows = sorted(by_dataset[dataset], key=lambda item: float(item.get("score", float("-inf"))), reverse=True)
        lines.extend([f"## {dataset}", ""])
        lines.append(f"- Completed metric rows: {len(dataset_rows)}")
        if dataset_rows:
            best = dataset_rows[0]
            lines.append(
                "- Best row: "
                f"k={best['k_value']}, ae_seed={best['ae_seed']}, "
                f"ACC={fmt_metric(best['metrics'], 'ACC')}, "
                f"NMI={fmt_metric(best['metrics'], 'NMI')}, "
                f"ARI={fmt_metric(best['metrics'], 'ARI')}, "
                f"F1={fmt_metric(best['metrics'], 'F1')}, "
                f"score={float(best.get('score', float('-inf'))):.3f}"
            )
            lines.append(f"- Best AE graph: `{best.get('ae_graph_path')}`")
        lines.append("")
        lines.append("| k | Seeds done | Best seed | Best ACC | Best NMI | Best ARI | Best F1 | Best graph |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for (ds, k_value), k_rows in sorted(by_dataset_k.items()):
            if ds != dataset:
                continue
            best_k = max(k_rows, key=lambda item: float(item.get("score", float("-inf"))))
            lines.append(
                f"| {k_value} | {len(k_rows)} | {best_k['ae_seed']} | "
                f"{fmt_metric(best_k['metrics'], 'ACC')} | {fmt_metric(best_k['metrics'], 'NMI')} | "
                f"{fmt_metric(best_k['metrics'], 'ARI')} | {fmt_metric(best_k['metrics'], 'F1')} | "
                f"`{best_k.get('ae_graph_path')}` |"
            )
        lines.append("")
        lines.append("### Top 10 Rows")
        lines.append("")
        lines.append("| Rank | k | AE seed | Score | ACC | NMI | ARI | F1 | AE graph |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for rank, item in enumerate(dataset_rows[:10], start=1):
            lines.append(
                f"| {rank} | {item['k_value']} | {item['ae_seed']} | {float(item.get('score', float('-inf'))):.3f} | "
                f"{fmt_metric(item['metrics'], 'ACC')} | {fmt_metric(item['metrics'], 'NMI')} | "
                f"{fmt_metric(item['metrics'], 'ARI')} | {fmt_metric(item['metrics'], 'F1')} | "
                f"`{item.get('ae_graph_path')}` |"
            )
        lines.append("")
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return summary_path


def run_job(
    config: dict[str, Any],
    dataset: str,
    k_value: int,
    ae_seed: int,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    graph_path, model_path = asset_paths(run_dir, dataset, k_value, ae_seed)
    ae_cmd, ae_args = build_ae_command(config, dataset, k_value, ae_seed, graph_path, model_path, args)
    train_cmd, train_args = build_train_command(config, dataset, graph_path, args)
    ae_log = run_dir / dataset / "ae_pretrain" / f"{dataset}_k{k_value}_seed{ae_seed}.txt"
    train_log = run_dir / dataset / "train_eval" / f"{dataset}_k{k_value}_seed{ae_seed}_train_{args.seed_start}_{args.seed_start + args.runs - 1}.txt"

    result: dict[str, Any] = {
        "dataset": dataset,
        "k_value": int(k_value),
        "ae_seed": int(ae_seed),
        "pretrain_seed": int(ae_seed),
        "graph_seed": int(ae_seed),
        "runs": int(args.runs),
        "seed_start": int(args.seed_start),
        "seed_end": int(args.seed_start) + int(args.runs) - 1,
        "ae_graph_path": rel(graph_path),
        "ae_model_path": rel(model_path),
        "ae_log_path": rel(ae_log),
        "train_log_path": rel(train_log),
        "ae_args": ae_args,
        "train_args": train_args,
        "ae_cmd": [str(part) for part in ae_cmd],
        "train_cmd": [str(part) for part in train_cmd],
    }

    graph_ready = graph_path.exists() and model_path.exists()
    if args.train_only_existing_ae and not graph_ready:
        result.update({"status": "missing_existing_ae_asset", "metrics": {}, "score": float("-inf")})
        return result

    if args.force_ae or not graph_ready:
        print(f"[AE] {dataset} k={k_value} ae_seed={ae_seed} -> {rel(graph_path)}", flush=True)
        if args.dry_run:
            result.update({"status": "dry_run", "metrics": {}, "score": float("-inf")})
            return result
        rc, elapsed, timed_out, _ = run_subprocess(ae_cmd, ae_log, int(args.timeout))
        result.update({"ae_returncode": rc, "ae_elapsed": elapsed, "ae_timed_out": timed_out})
        if rc != 0 or not graph_path.exists():
            result.update({"status": "ae_failed", "metrics": {}, "score": float("-inf")})
            return result
    else:
        result.update({"ae_returncode": 0, "ae_elapsed": 0.0, "ae_timed_out": False, "ae_reused": True})

    print(
        f"[TRAIN] {dataset} k={k_value} ae_seed={ae_seed} train_seeds={args.seed_start}..{args.seed_start + args.runs - 1}",
        flush=True,
    )
    if args.dry_run:
        result.update({"status": "dry_run", "metrics": {}, "score": float("-inf")})
        return result
    rc, elapsed, timed_out, output = run_subprocess(train_cmd, train_log, int(args.timeout))
    metrics = parse_metrics(output)
    result.update(
        {
            "train_returncode": rc,
            "train_elapsed": elapsed,
            "train_timed_out": timed_out,
            "metrics": metrics,
            "score": score_metrics(metrics),
            "status": "done" if rc == 0 and metrics else "train_failed",
        }
    )
    return result


def main() -> None:
    args = parse_args()
    config = load_experiment_config()
    datasets = parse_datasets(args.datasets, config)
    k_values = parse_int_list(args.k_values)
    if args.run_dir is not None:
        run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_ROOT / f"{stamp}_{'_'.join(datasets)}_k_grid"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_seeds = load_or_create_dataset_seeds(run_dir, datasets, args)
    results_path = run_dir / "results.jsonl"
    completed, results = load_completed(results_path, args)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "k_values": k_values,
        "dataset_ae_seeds": dataset_seeds,
        "runs": int(args.runs),
        "seed_start": int(args.seed_start),
        "train_mode": "dual-attn + DCGL-negative only",
        "python": str(args.python),
        "device": args.device,
    }
    (run_dir / "manifest.json").write_text(json.dumps(safe_json(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[START] output={rel(run_dir)}", flush=True)
    print(f"[START] datasets={datasets}", flush=True)
    print(f"[START] k_values={k_values}", flush=True)
    print(f"[START] dataset_ae_seeds={dataset_seeds}", flush=True)

    done_count = 0
    for dataset in datasets:
        for k_value in k_values:
            for ae_seed in dataset_seeds[dataset]:
                key = result_key(dataset, k_value, ae_seed, args)
                if key in completed:
                    print(f"[SKIP] {dataset} k={k_value} ae_seed={ae_seed}", flush=True)
                    continue
                try:
                    result = run_job(config, dataset, k_value, ae_seed, run_dir, args)
                except Exception as exc:
                    result = {
                        "dataset": dataset,
                        "k_value": int(k_value),
                        "ae_seed": int(ae_seed),
                        "status": "exception",
                        "error": repr(exc),
                        "metrics": {},
                        "score": float("-inf"),
                    }
                    print(f"[ERROR] {dataset} k={k_value} ae_seed={ae_seed}: {exc!r}", flush=True)
                append_jsonl(results_path, result)
                results.append(result)
                done_count += 1
                if int(args.update_summary_every) > 0 and done_count % int(args.update_summary_every) == 0:
                    write_summary(run_dir, results, args)
                metrics = result.get("metrics", {})
                acc = metrics.get("ACC", {}).get("mean") if metrics else None
                acc_text = "NA" if acc is None else f"{float(acc):.2f}"
                print(f"[ROW] {dataset} k={k_value} ae_seed={ae_seed} status={result.get('status')} ACC={acc_text}", flush=True)

    summary = write_summary(run_dir, results, args)
    print(f"[DONE] summary={summary}", flush=True)


if __name__ == "__main__":
    main()
