from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "base_knn_k_ablation_usps_reut"
SCGC_ENV_PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
DEFAULT_PYTHON = SCGC_ENV_PYTHON if SCGC_ENV_PYTHON.exists() else Path(sys.executable)
DEFAULT_DATASETS = ("reut", "usps")
METRICS = ("ACC", "NMI", "ARI", "F1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated base-KNN-k ablation for Reuters/USPS. The k value controls the "
            "raw/base graph A and train --knn_k. By default the AE branch reuses the "
            "fixed main-flow asset under data/ae_graph, so only the base graph k changes."
        )
    )
    parser.add_argument("--datasets", default="reut,usps", help="Comma-separated datasets. Default: reut,usps.")
    parser.add_argument("--ks", default="1,3,5,10,15,20", help="Comma-separated base KNN k values.")
    parser.add_argument("--ae-k", type=int, default=15, help="AE optimized graph top-k metadata when regenerating AE assets. Default: 15.")
    parser.add_argument(
        "--ae-asset-mode",
        choices=("fixed", "regenerate"),
        default="fixed",
        help="fixed reuses data/ae_graph/<dataset>_ae_graph.txt; regenerate rebuilds isolated AE assets per base k.",
    )
    parser.add_argument("--runs", type=int, default=10, help="Main-training runs per k value.")
    parser.add_argument("--seed-start", type=int, default=0, help="First main-training seed.")
    parser.add_argument("--pretrain-seed", type=int, default=None, help="Override AE pretrain seed.")
    parser.add_argument("--graph-seed", type=int, default=None, help="Override AE graph seed.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--timeout", type=int, default=0, help="Per-subprocess timeout seconds. 0 disables timeout.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None, help="Exact output directory.")
    parser.add_argument("--resume-jsonl", type=Path, default=None, help="Existing results.jsonl for skipping done jobs.")
    parser.add_argument("--force-ae", action="store_true", help="Regenerate isolated AE graph/checkpoint.")
    parser.add_argument("--train-only-existing-ae", action="store_true", help="Only train if isolated AE assets exist.")
    parser.add_argument("--update-summary-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them.")
    return parser.parse_args()


def load_experiment_config() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("dsafc_experiment", ROOT / "experiment.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def parse_csv_ints(raw: str, *, name: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in str(raw).replace(";", ",").split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return tuple(dict.fromkeys(values))


def parse_datasets(raw: str, config: dict[str, Any]) -> tuple[str, ...]:
    valid = set(config.get("dataset_profiles", {}))
    aliases = {"reuters": "reut", "reut": "reut", "usps": "usps"}
    datasets: list[str] = []
    for token in str(raw).replace(";", ",").split(","):
        name = token.strip().lower().replace("-", "_")
        if not name:
            continue
        name = aliases.get(name, name)
        if name not in valid:
            raise ValueError(f"Unsupported dataset '{token}'.")
        datasets.append(name)
    return tuple(dict.fromkeys(datasets or DEFAULT_DATASETS))


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


def build_module_args(config: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
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
        improved.update(profile.get("dcgl_negative_args", {}))
    if dcgl_cluster_enabled:
        improved["enable_dcgl_cluster_level"] = True
        improved.update(config.get("dcgl_cluster_args", {}))
        improved.update(profile.get("dcgl_cluster_args", {}))
    if gcn_enabled:
        improved["enable_gcn_backbone"] = True
        improved.update(config.get("gcn_backbone_args", {}))
        improved.update(profile.get("gcn_backbone_args", {}))
    return improved


def existing_base_graph(dataset: str, k: int) -> Path | None:
    path = ROOT / "data" / "graph" / f"{dataset}{k}_graph.txt"
    return path if path.exists() else None


def feature_path(dataset: str) -> Path:
    path = ROOT / "data" / "data" / f"{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    return path


def generate_base_graph(dataset: str, k: int, out_dir: Path) -> Path:
    out_path = out_dir / "base_graphs" / dataset / f"{dataset}{k}_graph.txt"
    if out_path.exists():
        return out_path

    print(f"[BASE] generate {dataset} k={k} -> {out_path.relative_to(ROOT)}", flush=True)
    features = np.loadtxt(feature_path(dataset), dtype=np.float32)
    n_neighbors = min(int(k) + 1, int(features.shape[0]))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(features)
    graph = nbrs.kneighbors_graph(features)
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph = graph + graph.T
    graph.data[:] = 1
    graph = graph.tocoo()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row, col in zip(graph.row.tolist(), graph.col.tolist()):
            if row != col:
                handle.write(f"{int(row)} {int(col)}\n")
    return out_path


def resolve_base_graph(dataset: str, k: int, out_dir: Path, *, generate_missing: bool = True) -> tuple[Path, str]:
    path = existing_base_graph(dataset, k)
    if path is not None:
        return path, "prebuilt"
    if generate_missing:
        return generate_base_graph(dataset, k, out_dir), "generated_main_knn"
    return ROOT / "data" / "graph" / f"{dataset}{k}_graph.txt", "online_main_knn"


def fixed_ae_graph_path(dataset: str) -> Path:
    path = ROOT / "data" / "ae_graph" / f"{dataset}_ae_graph.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed AE graph asset: {path}")
    return path


def asset_paths(out_dir: Path, dataset: str, base_k: int, ae_k: int) -> tuple[Path, Path]:
    asset_dir = out_dir / "ae_assets" / dataset / f"base_k_{base_k}_ae_k_{ae_k}"
    return (
        asset_dir / f"{dataset}_base_k{base_k}_ae_k{ae_k}_ae_graph.txt",
        asset_dir / f"{dataset}_base_k{base_k}_ae_k{ae_k}_ae_pretrain.pkl",
    )


def build_ae_command(
    config: dict[str, Any],
    dataset: str,
    k: int,
    base_graph_path: Path,
    graph_path: Path,
    model_path: Path,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    profile = config["dataset_profiles"][dataset]
    ae_args = merge_args(config.get("ae_args", {}), profile.get("ae_args", {}))
    ae_args.update(
        {
            "dataset": dataset,
            "cluster_num": profile["cluster_num"],
            "base_graph_path": base_graph_path,
            "ae_k": int(args.ae_k),
            "out_graph_path": graph_path,
            "model_save_path": model_path,
            "device": args.device,
        }
    )
    if args.pretrain_seed is not None:
        ae_args["pretrain_seed"] = int(args.pretrain_seed)
    if args.graph_seed is not None:
        ae_args["graph_seed"] = int(args.graph_seed)
    return [str(args.python), "data/pretrain_optimize_A_graph.py"] + dict_to_cli(ae_args), ae_args


def build_train_command(
    config: dict[str, Any],
    dataset: str,
    k: int,
    ae_graph_path: Path,
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
        build_module_args(config, profile),
    )
    train_args.update(
        {
            "knn_k": int(k),
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
        str(ae_graph_path),
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


def result_key(dataset: str, k: int, args: argparse.Namespace, ae_args: dict[str, Any]) -> str:
    payload = {
        "dataset": dataset,
        "base_knn_k": int(k),
        "ae_k": int(args.ae_k),
        "ae_asset_mode": args.ae_asset_mode,
        "ae_graph_path": str(ae_args.get("out_graph_path", "")),
        "runs": int(args.runs),
        "seed_start": int(args.seed_start),
        "seed_end": int(args.seed_start) + int(args.runs) - 1,
        "pretrain_seed": ae_args.get("pretrain_seed"),
        "graph_seed": ae_args.get("graph_seed"),
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
                completed.add(
                    json.dumps(
                        {
                            "dataset": item.get("dataset"),
                            "base_knn_k": item.get("base_knn_k"),
                            "ae_k": item.get("ae_k"),
                            "ae_asset_mode": item.get("ae_asset_mode"),
                            "ae_graph_path": item.get("ae_graph_path"),
                            "runs": int(args.runs),
                            "seed_start": int(args.seed_start),
                            "seed_end": int(args.seed_start) + int(args.runs) - 1,
                            "pretrain_seed": item.get("pretrain_seed"),
                            "graph_seed": item.get("graph_seed"),
                        },
                        sort_keys=True,
                    )
                )
    return completed, results


def fmt_metric(metrics: dict[str, dict[str, float]], metric: str) -> str:
    if metric not in metrics:
        return "-"
    value = metrics[metric]
    return f"{value['mean']:.2f}+/-{value['std']:.2f}"


def write_summary(out_dir: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    summary_path = out_dir / "summary.md"
    lines = [
        "# Base KNN k Ablation: USPS and Reuters",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Main train seeds: {args.seed_start}..{args.seed_start + args.runs - 1}",
        f"- Runs per k: {args.runs}",
        "- k policy: base graph KNN k is ablated.",
        f"- AE asset policy: {args.ae_asset_mode}.",
        f"- AE k metadata: {args.ae_k}.",
        "- Base graph policy: use data/graph/<dataset><k>_graph.txt if present; otherwise use train.py online KNN fallback unless AE regeneration needs an isolated base graph file.",
        f"- Results jsonl: {str((out_dir / 'results.jsonl').relative_to(ROOT))}",
        "",
    ]
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        by_dataset.setdefault(str(item.get("dataset")), []).append(item)

    for dataset in sorted(by_dataset):
        rows = sorted(
            [item for item in by_dataset[dataset] if item.get("metrics")],
            key=lambda item: float(item.get("score", float("-inf"))),
            reverse=True,
        )
        lines.extend([f"## {dataset}", ""])
        if not rows:
            lines.extend(["No completed metric rows yet.", ""])
            continue
        best = rows[0]
        lines.extend(
            [
                f"- Best k: {best.get('base_knn_k')}",
                f"- Best AE graph: {best.get('ae_graph_path')}",
                "",
                "| Rank | k | Base graph | Score | ACC | NMI | ARI | F1 | Status | Train log |",
                "|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for rank, item in enumerate(rows, start=1):
            metrics = item.get("metrics", {})
            lines.append(
                "| {rank} | {k} | {base_source} | {score:.3f} | {acc} | {nmi} | {ari} | {f1} | {status} | {log} |".format(
                    rank=rank,
                    k=item.get("base_knn_k"),
                    base_source=item.get("base_graph_source", ""),
                    score=float(item.get("score", float("-inf"))),
                    acc=fmt_metric(metrics, "ACC"),
                    nmi=fmt_metric(metrics, "NMI"),
                    ari=fmt_metric(metrics, "ARI"),
                    f1=fmt_metric(metrics, "F1"),
                    status=item.get("status"),
                    log=item.get("train_log_path", ""),
                )
            )
        lines.append("")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def run_job(config: dict[str, Any], dataset: str, k: int, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    regenerate_ae = args.ae_asset_mode == "regenerate"
    base_graph_path, base_graph_source = resolve_base_graph(dataset, k, out_dir, generate_missing=regenerate_ae)
    if regenerate_ae:
        ae_graph_path, ae_model_path = asset_paths(out_dir, dataset, k, int(args.ae_k))
        ae_cmd, ae_args = build_ae_command(config, dataset, k, base_graph_path, ae_graph_path, ae_model_path, args)
    else:
        ae_graph_path = fixed_ae_graph_path(dataset)
        ae_model_path = ROOT / "data" / "ae_graph" / f"{dataset}_fixed_asset_no_model.pkl"
        ae_cmd = []
        ae_args = {
            "dataset": dataset,
            "cluster_num": config["dataset_profiles"][dataset]["cluster_num"],
            "base_graph_path": base_graph_path,
            "ae_k": int(args.ae_k),
            "out_graph_path": ae_graph_path,
            "model_save_path": "",
            "asset_mode": "fixed",
        }
    train_cmd, train_args = build_train_command(config, dataset, k, ae_graph_path, args)
    ae_log = out_dir / dataset / "ae_pretrain" / f"{dataset}_base_k{k}_ae_k{args.ae_k}.txt"
    train_log = out_dir / dataset / "train_eval" / f"{dataset}_base_k{k}_train_{args.seed_start}_{args.seed_start + args.runs - 1}.txt"

    result: dict[str, Any] = {
        "dataset": dataset,
        "base_knn_k": int(k),
        "ae_k": int(args.ae_k),
        "ae_asset_mode": args.ae_asset_mode,
        "base_graph_path": str(base_graph_path.relative_to(ROOT)) if base_graph_path.is_relative_to(ROOT) else str(base_graph_path),
        "base_graph_source": base_graph_source,
        "pretrain_seed": ae_args.get("pretrain_seed"),
        "graph_seed": ae_args.get("graph_seed"),
        "runs": int(args.runs),
        "seed_start": int(args.seed_start),
        "seed_end": int(args.seed_start) + int(args.runs) - 1,
        "ae_graph_path": str(ae_graph_path.relative_to(ROOT)) if ae_graph_path.is_relative_to(ROOT) else str(ae_graph_path),
        "ae_model_path": str(ae_model_path.relative_to(ROOT)) if ae_model_path.is_relative_to(ROOT) else str(ae_model_path),
        "ae_log_path": str(ae_log.relative_to(ROOT)) if regenerate_ae else "",
        "train_log_path": str(train_log.relative_to(ROOT)),
        "ae_args": {key: str(value) if isinstance(value, Path) else value for key, value in ae_args.items()},
        "train_args": {key: str(value) if isinstance(value, Path) else value for key, value in train_args.items()},
        "ae_cmd": [str(part) for part in ae_cmd],
        "train_cmd": [str(part) for part in train_cmd],
    }

    graph_ready = ae_graph_path.exists() and (ae_model_path.exists() if regenerate_ae else True)
    if args.train_only_existing_ae and not graph_ready:
        result.update({"status": "missing_existing_ae_asset", "metrics": {}, "score": float("-inf")})
        return result

    if not regenerate_ae:
        result.update({"ae_returncode": 0, "ae_elapsed": 0.0, "ae_timed_out": False, "ae_reused": True})
        print(
            f"[AE] {dataset} base_k={k} reuse fixed {ae_graph_path.relative_to(ROOT) if ae_graph_path.is_relative_to(ROOT) else ae_graph_path}",
            flush=True,
        )
    elif args.force_ae or not graph_ready:
        print(
            f"[AE] {dataset} base_k={k} ae_k={args.ae_k} base={base_graph_path.relative_to(ROOT) if base_graph_path.is_relative_to(ROOT) else base_graph_path}",
            flush=True,
        )
        rc, elapsed, timed_out, _ = run_subprocess(ae_cmd, ae_log, int(args.timeout))
        result.update({"ae_returncode": rc, "ae_elapsed": elapsed, "ae_timed_out": timed_out})
        if rc != 0 or not ae_graph_path.exists():
            result.update({"status": "ae_failed", "metrics": {}, "score": float("-inf")})
            return result
    else:
        result.update({"ae_returncode": 0, "ae_elapsed": 0.0, "ae_timed_out": False, "ae_reused": True})

    print(f"[TRAIN] {dataset} base_k={k} ae_k={args.ae_k} train_seeds={args.seed_start}..{args.seed_start + args.runs - 1}", flush=True)
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


def main() -> int:
    args = parse_args()
    config = load_experiment_config()
    datasets = parse_datasets(args.datasets, config)
    ks = parse_csv_ints(args.ks, name="--ks")

    if args.run_dir is not None:
        out_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = args.output_root / f"{stamp}_{'_'.join(datasets)}_k_{'_'.join(map(str, ks))}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "k_values": ks,
        "fixed_ae_k": int(args.ae_k),
        "ae_asset_mode": args.ae_asset_mode,
        "runs": int(args.runs),
        "seed_start": int(args.seed_start),
        "python": str(args.python),
        "device": args.device,
        "note": "Base graph KNN k is ablated while the AE branch reuses fixed main-flow assets by default.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[START] output={out_dir.relative_to(ROOT)} datasets={datasets} k={ks}", flush=True)

    completed, results = load_completed(args.resume_jsonl, args)
    if args.resume_jsonl:
        print(f"[RESUME] loaded {len(results)} rows from {args.resume_jsonl}", flush=True)

    if args.dry_run:
        for dataset in datasets:
            for k in ks:
                regenerate_ae = args.ae_asset_mode == "regenerate"
                base_graph_path, base_graph_source = resolve_base_graph(dataset, k, out_dir, generate_missing=regenerate_ae)
                if regenerate_ae:
                    ae_graph_path, ae_model_path = asset_paths(out_dir, dataset, k, int(args.ae_k))
                    ae_cmd, ae_args = build_ae_command(config, dataset, k, base_graph_path, ae_graph_path, ae_model_path, args)
                else:
                    ae_graph_path = fixed_ae_graph_path(dataset)
                    ae_cmd = []
                    ae_args = {"ae_k": int(args.ae_k), "out_graph_path": ae_graph_path, "asset_mode": "fixed"}
                train_cmd, train_args = build_train_command(config, dataset, k, ae_graph_path, args)
                print(json.dumps({"dataset": dataset, "k": k, "base_graph": str(base_graph_path), "base_source": base_graph_source, "ae_args": ae_args, "train_args": train_args}, default=str))
                print("AE_CMD:", " ".join(map(str, ae_cmd)) if ae_cmd else f"REUSE {ae_graph_path}")
                print("TRAIN_CMD:", " ".join(map(str, train_cmd)))
        return 0

    update_every = max(0, int(args.update_summary_every))
    completed_this_run = 0
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for dataset in datasets:
            for k in ks:
                regenerate_ae = args.ae_asset_mode == "regenerate"
                base_graph_path, _ = resolve_base_graph(dataset, k, out_dir, generate_missing=regenerate_ae)
                if regenerate_ae:
                    ae_graph_path, ae_model_path = asset_paths(out_dir, dataset, k, int(args.ae_k))
                    _, ae_args_for_key = build_ae_command(config, dataset, k, base_graph_path, ae_graph_path, ae_model_path, args)
                else:
                    ae_graph_path = fixed_ae_graph_path(dataset)
                    ae_args_for_key = {"ae_k": int(args.ae_k), "out_graph_path": ae_graph_path, "asset_mode": "fixed"}
                key = result_key(dataset, k, args, ae_args_for_key)
                if key in completed:
                    print(f"[SKIP] {dataset} k={k} already completed", flush=True)
                    continue
                result = run_job(config, dataset, k, out_dir, args)
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                results.append(result)
                completed_this_run += 1
                if update_every and completed_this_run % update_every == 0:
                    summary_path = write_summary(out_dir, results, args)
                    print(f"[SUMMARY] {summary_path.relative_to(ROOT)}", flush=True)

    summary_path = write_summary(out_dir, results, args)
    print(f"[DONE] summary={summary_path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        error_path = OUTPUT_ROOT / f"fatal_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[FATAL] {error_path}", file=sys.stderr, flush=True)
        raise
