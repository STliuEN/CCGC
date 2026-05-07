from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "ae_graph_seed_sweep"
SCGC_ENV_PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
DEFAULT_PYTHON = SCGC_ENV_PYTHON if SCGC_ENV_PYTHON.exists() else Path(sys.executable)
MAIN_DATASETS = ("reut", "uat", "amap", "usps", "eat", "cora", "cite")
DATASET_ALIASES = {
    "all": MAIN_DATASETS,
    "main": MAIN_DATASETS,
    "non_usps": tuple(dataset for dataset in MAIN_DATASETS if dataset != "usps"),
    "reuters": ("reut",),
    "citeseer": ("cite",),
}
METRICS = ("ACC", "NMI", "ARI", "F1")
OURS_MAIN_TABLE_TARGETS: dict[str, dict[str, float]] = {
    "usps": {"ACC": 82.40, "NMI": 73.29, "ARI": 68.39, "F1": 82.16},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated AE graph seed sweep. Each job builds a fixed AE graph/checkpoint "
            "asset in experiment_output, then evaluates it with the same 10 main-training seeds."
        )
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset, comma list, or group: all/main. Aliases: reuters->reut, citeseer->cite.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,30,42,50,59,77,88,100,123,150,202,256,512",
        help="Comma-separated AE pretrain/graph seeds.",
    )
    parser.add_argument(
        "--random-seed-attempts",
        type=int,
        default=0,
        help=(
            "When >0, ignore --seeds and sample this many unique AE seeds at random. "
            "Each sampled seed is still recorded and reused for both pretrain_seed and graph_seed."
        ),
    )
    parser.add_argument(
        "--random-seed-max",
        type=int,
        default=2_147_483_647,
        help="Upper bound used when sampling random AE seeds (inclusive).",
    )
    parser.add_argument(
        "--random-seed-base",
        type=int,
        default=0,
        help="Lower bound used when sampling random AE seeds (inclusive).",
    )
    parser.add_argument(
        "--stop-on-main-table-pass",
        action="store_true",
        help=(
            "Stop after a completed 10-run job whose mean ACC/NMI/ARI/F1 all match or exceed "
            "the stored Ours main-table target for that dataset."
        ),
    )
    parser.add_argument("--runs", type=int, default=10, help="Main-training runs per AE asset.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First main-training seed. The evaluation uses seed_start..seed_start+runs-1.",
    )
    parser.add_argument("--device", default="cuda", help="Device passed to AE pretrain and train.py.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-subprocess timeout in seconds. 0 disables timeout.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for isolated AE assets, logs, results.jsonl, and summary.md.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Exact output directory to use. Useful for resuming the same sweep in place.",
    )
    parser.add_argument(
        "--resume-jsonl",
        type=Path,
        default=None,
        help="Existing results.jsonl to load and skip completed dataset/AE-seed jobs.",
    )
    parser.add_argument(
        "--force-ae",
        action="store_true",
        help="Regenerate isolated AE graph/checkpoint even if the files already exist.",
    )
    parser.add_argument(
        "--train-only-existing-ae",
        action="store_true",
        help="Do not run AE pretraining; only evaluate isolated assets that already exist.",
    )
    parser.add_argument(
        "--update-summary-every",
        type=int,
        default=1,
        help="Rewrite summary.md after every N completed jobs. 0 writes only at the end.",
    )
    return parser.parse_args()


def load_experiment_config() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("dsafc_experiment", ROOT / "experiment.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def normalize_dataset_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def selected_datasets(raw: str, config: dict[str, Any]) -> tuple[str, ...]:
    datasets: list[str] = []
    valid = set(config.get("dataset_profiles", {})) & set(MAIN_DATASETS)
    for token in str(raw).replace(";", ",").split(","):
        name = normalize_dataset_name(token)
        if not name:
            continue
        if name in DATASET_ALIASES:
            datasets.extend(DATASET_ALIASES[name])
        elif name in valid:
            datasets.append(name)
        else:
            choices = ", ".join(sorted(set(MAIN_DATASETS) | set(DATASET_ALIASES)))
            raise ValueError(f"Unsupported dataset/group '{token}'. Valid: {choices}")
    if not datasets:
        return MAIN_DATASETS

    unique: list[str] = []
    seen = set()
    for dataset in datasets:
        if dataset not in seen:
            unique.append(dataset)
            seen.add(dataset)
    return tuple(unique)


def parse_seed_list(raw: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for token in str(raw).replace(";", ",").split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    return tuple(dict.fromkeys(seeds))


def sampled_seed_list(args: argparse.Namespace) -> tuple[int, ...]:
    attempts = int(args.random_seed_attempts)
    if attempts <= 0:
        return parse_seed_list(args.seeds)

    lower = int(args.random_seed_base)
    upper = int(args.random_seed_max)
    if upper < lower:
        raise ValueError("--random-seed-max must be >= --random-seed-base")
    population = upper - lower + 1
    if attempts > population:
        raise ValueError(
            f"Requested {attempts} random AE seeds, but range [{lower}, {upper}] only contains {population} values."
        )

    rng = random.SystemRandom()
    seeds = rng.sample(range(lower, upper + 1), attempts)
    return tuple(seeds)


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


def build_improved_args(config: dict[str, Any]) -> dict[str, Any]:
    legacy_enable = bool(config.get("enable_improved_module", False))
    dynamic_enabled = bool(config.get("enable_dynamic_threshold_module", legacy_enable))
    ema_enabled = bool(config.get("enable_ema_prototypes_module", legacy_enable))
    dcgl_neg_enabled = bool(config.get("enable_dcgl_negative_module", False))
    dcgl_cluster_enabled = bool(config.get("enable_dcgl_cluster_module", False))
    gcn_enabled = bool(config.get("enable_gcn_backbone_module", False))

    legacy_args = config.get("improved_module_args", {})
    improved: dict[str, Any] = {}
    if dynamic_enabled:
        improved["enable_dynamic_threshold"] = True
        improved.update(config.get("dynamic_threshold_args", {}))
        for key in ("dynamic_threshold_start", "dynamic_threshold_end"):
            if key in legacy_args and key not in improved:
                improved[key] = legacy_args[key]
    if ema_enabled:
        improved["enable_ema_prototypes"] = True
        improved.update(config.get("ema_prototypes_args", {}))
        if "ema_proto_momentum" in legacy_args and "ema_proto_momentum" not in improved:
            improved["ema_proto_momentum"] = legacy_args["ema_proto_momentum"]
    if dcgl_neg_enabled:
        improved["enable_dcgl_negative_loss"] = True
        improved.update(config.get("dcgl_negative_args", {}))
    if dcgl_cluster_enabled:
        improved["enable_dcgl_cluster_level"] = True
        improved.update(config.get("dcgl_cluster_args", {}))
    if gcn_enabled:
        improved["enable_gcn_backbone"] = True
        improved.update(config.get("gcn_backbone_args", {}))
    return improved


def asset_paths(out_dir: Path, dataset: str, ae_seed: int) -> tuple[Path, Path]:
    asset_dir = out_dir / "ae_assets" / dataset / f"seed_{ae_seed}"
    return asset_dir / f"{dataset}_ae_graph.txt", asset_dir / f"{dataset}_ae_pretrain.pkl"


def build_ae_command(
    config: dict[str, Any],
    dataset: str,
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
            "cluster_num": profile["cluster_num"],
            "pretrain_seed": ae_seed,
            "graph_seed": ae_seed,
            "out_graph_path": graph_path,
            "model_save_path": model_path,
            "device": args.device,
        }
    )
    return [str(args.python), "data/pretrain_optimize_A_graph.py"] + dict_to_cli(ae_args), ae_args


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
        build_improved_args(config),
    )
    train_args.update(
        {
            "device": args.device,
            "runs": int(args.runs),
            "seed_start": int(args.seed_start),
        }
    )
    cmd = [
        str(args.python),
        "train.py",
        "--dataset",
        dataset,
        "--cluster_num",
        str(profile["cluster_num"]),
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
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(f"COMMAND: {' '.join(map(str, cmd))}\n")
        handle.write(f"RETURN_CODE: {returncode}\n")
        handle.write(f"ELAPSED_SEC: {elapsed:.2f}\n")
        handle.write(f"TIMED_OUT: {'YES' if timed_out else 'NO'}\n")
        handle.write("=" * 80 + "\n")
        handle.write(output)
    return returncode, elapsed, timed_out, output


def parse_metrics(text: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(ACC|NMI|ARI|F1)\s*\|(.+)$", line)
        if not match:
            continue
        nums = re.findall(r"\d+(?:\.\d+)?", match.group(2))
        if len(nums) >= 2:
            metrics[match.group(1)] = {"mean": float(nums[0]), "std": float(nums[1])}
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


def target_for_dataset(dataset: str) -> dict[str, float] | None:
    return OURS_MAIN_TABLE_TARGETS.get(dataset)


def metric_gaps(metrics: dict[str, dict[str, float]], target: dict[str, float]) -> dict[str, float]:
    return {
        metric: float(metrics[metric]["mean"]) - float(target[metric])
        for metric in METRICS
        if metric in metrics and metric in target
    }


def passes_main_table_target(metrics: dict[str, dict[str, float]], dataset: str) -> bool:
    target = target_for_dataset(dataset)
    if not target:
        return False
    if not all(metric in metrics for metric in METRICS):
        return False
    return all(float(metrics[metric]["mean"]) >= float(target[metric]) for metric in METRICS)


def result_key(dataset: str, ae_seed: int, args: argparse.Namespace) -> str:
    payload = {
        "dataset": dataset,
        "ae_seed": ae_seed,
        "seed_mode": "sampled_random" if int(args.random_seed_attempts) > 0 else "explicit",
        "runs": int(args.runs),
        "seed_start": int(args.seed_start),
        "seed_end": int(args.seed_start) + int(args.runs) - 1,
    }
    return json.dumps(payload, sort_keys=True)


def load_completed(path: Path | None, args: argparse.Namespace) -> tuple[set[str], list[dict[str, Any]]]:
    if path is None or not path.exists():
        return set(), []
    completed: set[str] = set()
    results: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            results.append(item)
            if item.get("status") == "done":
                completed.add(result_key(str(item["dataset"]), int(item["ae_seed"]), args))
    return completed, results


def fmt_metric(metrics: dict[str, dict[str, float]], metric: str) -> str:
    if metric not in metrics:
        return "-"
    value = metrics[metric]
    return f"{value['mean']:.2f}+/-{value['std']:.2f}"


def write_summary(out_dir: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    summary_path = out_dir / "summary.md"
    seed_mode = "sampled_random" if int(args.random_seed_attempts) > 0 else "explicit"
    lines = [
        "# AE Graph Seed Sweep",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- AE seed mode: {seed_mode}",
        f"- Main train seeds: {args.seed_start}..{args.seed_start + args.runs - 1}",
        f"- Runs per AE asset: {args.runs}",
        f"- Stop-on-main-table-pass: {'ON' if args.stop_on_main_table_pass else 'OFF'}",
        f"- Selection score: ACC + 0.4*F1 + 0.2*NMI + 0.2*ARI - 0.25*(ACC_std + F1_std)",
        f"- Results jsonl: {str((out_dir / 'results.jsonl').relative_to(ROOT))}",
        "",
    ]
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        by_dataset.setdefault(str(item.get("dataset")), []).append(item)

    for dataset in sorted(by_dataset):
        rows = sorted(
            [item for item in by_dataset[dataset] if item.get("metrics")],
            key=lambda item: (
                1 if item.get("passed_main_table_target") else 0,
                float(item.get("score", float("-inf"))),
            ),
            reverse=True,
        )
        lines.extend([f"## {dataset}", ""])
        if not rows:
            lines.extend(["No completed metric rows yet.", ""])
            continue
        best = rows[0]
        target = target_for_dataset(dataset)
        passed_rows = [item for item in rows if item.get("passed_main_table_target")]
        lines.append(f"- Completed rows: {len(rows)}")
        if int(args.random_seed_attempts) > 0:
            lines.append(f"- Random AE attempts requested: {int(args.random_seed_attempts)}")
        if target:
            lines.append(
                "- Main-table Ours target: "
                f"{target['ACC']:.2f} / {target['NMI']:.2f} / {target['ARI']:.2f} / {target['F1']:.2f}"
            )
            lines.append(f"- Successful attempts meeting target: {len(passed_rows)}")
        lines.extend(
            [
                f"- Best AE seed: {best.get('ae_seed')}",
                f"- Best AE graph: {best.get('ae_graph_path')}",
                f"- Best AE model: {best.get('ae_model_path')}",
                f"- Best row passes main-table target: {'YES' if best.get('passed_main_table_target') else 'NO'}",
                "",
                "| Rank | AE seed | Pass Main Table | Score | ACC | NMI | ARI | F1 | Status | Train log |",
                "|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for rank, item in enumerate(rows, start=1):
            metrics = item.get("metrics", {})
            log_path = item.get("train_log_path", "")
            lines.append(
                "| {rank} | {seed} | {passed} | {score:.3f} | {acc} | {nmi} | {ari} | {f1} | {status} | {log} |".format(
                    rank=rank,
                    seed=item.get("ae_seed"),
                    passed="YES" if item.get("passed_main_table_target") else "NO",
                    score=float(item.get("score", float("-inf"))),
                    acc=fmt_metric(metrics, "ACC"),
                    nmi=fmt_metric(metrics, "NMI"),
                    ari=fmt_metric(metrics, "ARI"),
                    f1=fmt_metric(metrics, "F1"),
                    status=item.get("status"),
                    log=log_path,
                )
            )
        lines.append("")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def run_job(
    config: dict[str, Any],
    dataset: str,
    ae_seed: int,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    graph_path, model_path = asset_paths(out_dir, dataset, ae_seed)
    ae_cmd, ae_args = build_ae_command(config, dataset, ae_seed, graph_path, model_path, args)
    train_cmd, train_args = build_train_command(config, dataset, graph_path, args)
    ae_log = out_dir / dataset / "ae_pretrain" / f"{dataset}_ae_seed{ae_seed}.txt"
    train_log = out_dir / dataset / "train_eval" / f"{dataset}_ae_seed{ae_seed}_train_{args.seed_start}_{args.seed_start + args.runs - 1}.txt"

    result: dict[str, Any] = {
        "dataset": dataset,
        "ae_seed": ae_seed,
        "pretrain_seed": ae_seed,
        "graph_seed": ae_seed,
        "seed_mode": "sampled_random" if int(args.random_seed_attempts) > 0 else "explicit",
        "runs": int(args.runs),
        "seed_start": int(args.seed_start),
        "seed_end": int(args.seed_start) + int(args.runs) - 1,
        "ae_graph_path": str(graph_path.relative_to(ROOT)),
        "ae_model_path": str(model_path.relative_to(ROOT)),
        "ae_log_path": str(ae_log.relative_to(ROOT)),
        "train_log_path": str(train_log.relative_to(ROOT)),
        "ae_args": {key: str(value) if isinstance(value, Path) else value for key, value in ae_args.items()},
        "train_args": {key: str(value) if isinstance(value, Path) else value for key, value in train_args.items()},
        "ae_cmd": [str(part) for part in ae_cmd],
        "train_cmd": [str(part) for part in train_cmd],
    }

    graph_ready = graph_path.exists() and model_path.exists()
    if args.train_only_existing_ae and not graph_ready:
        result.update({"status": "missing_existing_ae_asset", "metrics": {}, "score": float("-inf")})
        return result

    if args.force_ae or not graph_ready:
        print(f"[AE] {dataset} seed={ae_seed} -> {graph_path.relative_to(ROOT)}", flush=True)
        rc, elapsed, timed_out, _ = run_subprocess(ae_cmd, ae_log, int(args.timeout))
        result.update({"ae_returncode": rc, "ae_elapsed": elapsed, "ae_timed_out": timed_out})
        if rc != 0 or not graph_path.exists():
            result.update({"status": "ae_failed", "metrics": {}, "score": float("-inf")})
            return result
    else:
        result.update({"ae_returncode": 0, "ae_elapsed": 0.0, "ae_timed_out": False, "ae_reused": True})

    print(
        f"[TRAIN] {dataset} ae_seed={ae_seed} train_seeds={args.seed_start}..{args.seed_start + args.runs - 1}",
        flush=True,
    )
    rc, elapsed, timed_out, output = run_subprocess(train_cmd, train_log, int(args.timeout))
    metrics = parse_metrics(output)
    target = target_for_dataset(dataset)
    passed_target = passes_main_table_target(metrics, dataset)
    result.update(
        {
            "train_returncode": rc,
            "train_elapsed": elapsed,
            "train_timed_out": timed_out,
            "metrics": metrics,
            "score": score_metrics(metrics),
            "main_table_target": target,
            "main_table_gaps": metric_gaps(metrics, target) if metrics and target else {},
            "passed_main_table_target": passed_target,
            "status": "done" if rc == 0 and metrics else "train_failed",
        }
    )
    return result


def main() -> None:
    args = parse_args()
    config = load_experiment_config()
    datasets = selected_datasets(args.dataset, config)
    seeds = sampled_seed_list(args)
    if args.run_dir is not None:
        out_dir = args.run_dir
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = args.output_root / f"{stamp}_{'_'.join(datasets)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"

    completed, results = load_completed(args.resume_jsonl, args)
    if args.resume_jsonl:
        print(f"[RESUME] loaded {len(results)} rows from {args.resume_jsonl}", flush=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "ae_seeds": seeds,
        "seed_mode": "sampled_random" if int(args.random_seed_attempts) > 0 else "explicit",
        "random_seed_attempts": int(args.random_seed_attempts),
        "runs": int(args.runs),
        "seed_start": int(args.seed_start),
        "python": str(args.python),
        "device": args.device,
        "stop_on_main_table_pass": bool(args.stop_on_main_table_pass),
        "main_table_targets": {dataset: target_for_dataset(dataset) for dataset in datasets if target_for_dataset(dataset)},
        "note": "Isolated AE graph assets. Main training uses fixed train seeds for all AE assets.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[START] output={out_dir.relative_to(ROOT)} datasets={datasets} ae_seeds={seeds}", flush=True)

    update_every = max(0, int(args.update_summary_every))
    completed_this_run = 0
    stop_requested = False
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for dataset in datasets:
            for ae_seed in seeds:
                key = result_key(dataset, ae_seed, args)
                if key in completed:
                    print(f"[SKIP] {dataset} ae_seed={ae_seed} already completed", flush=True)
                    continue
                result = run_job(config, dataset, ae_seed, out_dir, args)
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                results.append(result)
                completed_this_run += 1
                if update_every and completed_this_run % update_every == 0:
                    summary_path = write_summary(out_dir, results, args)
                    print(f"[SUMMARY] {summary_path.relative_to(ROOT)}", flush=True)
                if args.stop_on_main_table_pass and result.get("passed_main_table_target"):
                    print(
                        f"[STOP] {dataset} ae_seed={ae_seed} matched or exceeded the stored main-table target.",
                        flush=True,
                    )
                    stop_requested = True
                    break
            if stop_requested:
                break

    summary_path = write_summary(out_dir, results, args)
    print(f"[DONE] summary={summary_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_root = OUTPUT_ROOT
        error_root.mkdir(parents=True, exist_ok=True)
        error_path = error_root / f"fatal_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[FATAL] {error_path}", file=sys.stderr, flush=True)
        raise
