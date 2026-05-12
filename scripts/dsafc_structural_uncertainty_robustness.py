from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "structural_uncertainty_robustness"
SCGC_ENV_PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
DEFAULT_PYTHON = SCGC_ENV_PYTHON if SCGC_ENV_PYTHON.exists() else Path(sys.executable)
METRICS = ("ACC", "NMI", "ARI", "F1")
DATASET_ORDER = ("reut", "uat", "amap", "usps", "cora", "cite")
DATASET_ALIASES = {
    "reuters": "reut",
    "reut": "reut",
    "uat": "uat",
    "amap": "amap",
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


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    graph_mode: str
    fusion_mode: str | None
    enable_dcgl_negative: bool


VARIANTS = (
    Variant("osl", "OSL", "raw", None, False),
    Variant("rsl", "RSL", "ae", None, False),
    Variant("f_dsf", "F-DSF", "dual", "mean", False),
    Variant("a_dsf", "A-DSF", "dual", "attn", False),
    Variant("dsafc", "DSAFC", "dual", "attn", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled structural-uncertainty robustness experiment. "
            "Perturb the raw graph A while reusing the current AE graph A_E."
        )
    )
    parser.add_argument("--dataset", "--datasets", dest="datasets", default="all")
    parser.add_argument("--variants", default="osl,rsl,f_dsf,a_dsf,dsafc")
    parser.add_argument("--edge-rates", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--knn-values", default="3,5,10,15,20")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--perturb-seed", type=int, default=20260512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
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
    if str(raw).strip().lower() == "all":
        return tuple(dataset for dataset in DATASET_ORDER if dataset in profiles)
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
        raise ValueError("No datasets were provided.")
    return tuple(dict.fromkeys(datasets))


def parse_variants(raw: str) -> tuple[Variant, ...]:
    by_key = {variant.key: variant for variant in VARIANTS}
    if str(raw).strip().lower() == "all":
        return VARIANTS
    variants: list[Variant] = []
    for token in str(raw).replace(";", ",").split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in by_key:
            raise ValueError(f"Unsupported variant '{token}'.")
        variants.append(by_key[key])
    if not variants:
        raise ValueError("No variants were provided.")
    return tuple(dict.fromkeys(variants))


def parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(token.strip()) for token in str(raw).replace(";", ",").split(",") if token.strip())


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(float(token.strip())) for token in str(raw).replace(";", ",").split(",") if token.strip())


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


def discover_npy_triplet(dataset: str) -> tuple[Path, Path, Path]:
    dataset_dir = ROOT / "data" / "full_dataset" / dataset
    for feat_path in dataset_dir.rglob("*_feat.npy"):
        prefix = feat_path.name[: -len("_feat.npy")]
        label_path = feat_path.parent / f"{prefix}_label.npy"
        adj_path = feat_path.parent / f"{prefix}_adj.npy"
        if label_path.exists() and adj_path.exists():
            return feat_path, label_path, adj_path
    raise FileNotFoundError(f"Missing npy triplet for {dataset}")


def resolve_existing_base_graph(dataset: str, knn_k: int) -> Path | None:
    default_path = ROOT / "data" / "graph" / f"{dataset}_graph.txt"
    if default_path.exists():
        return default_path
    k_path = ROOT / "data" / "graph" / f"{dataset}{int(knn_k)}_graph.txt"
    if k_path.exists():
        return k_path
    return None


def postprocess_adj(adj: sp.spmatrix) -> sp.csr_matrix:
    adj = adj.tocsr().astype(np.float32)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj = adj + adj.T
    adj = adj.sign().tocsr()
    adj.eliminate_zeros()
    return adj


def read_edge_list_to_adj(path: Path, num_nodes: int) -> sp.csr_matrix:
    try:
        edges = np.loadtxt(path, dtype=int)
    except ValueError:
        edges = np.empty((0, 2), dtype=int)
    if edges.size == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    if edges.ndim == 1:
        edges = edges.reshape(1, 2)
    values = np.ones(edges.shape[0], dtype=np.float32)
    return postprocess_adj(sp.csr_matrix((values, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes)))


def load_dataset_arrays(dataset: str, knn_k: int) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    if has_npy_triplet(dataset):
        feat_path, label_path, adj_path = discover_npy_triplet(dataset)
        features = np.asarray(np.load(feat_path, allow_pickle=True), dtype=np.float32)
        labels = np.asarray(np.load(label_path, allow_pickle=True))
        if labels.ndim == 2:
            labels = labels[:, 0] if labels.shape[1] == 1 else np.argmax(labels, axis=1)
        adj_obj = np.load(adj_path, allow_pickle=True)
        if isinstance(adj_obj, np.ndarray) and adj_obj.dtype == object and adj_obj.shape == ():
            adj_obj = adj_obj.item()
        adj = adj_obj.tocsr().astype(np.float32) if sp.issparse(adj_obj) else sp.csr_matrix(np.asarray(adj_obj, dtype=np.float32))
        return features, labels.reshape(-1).astype(np.int64), postprocess_adj(adj)

    feat_path = ROOT / "data" / "data" / f"{dataset}.txt"
    label_path = ROOT / "data" / "data" / f"{dataset}_label.txt"
    features = np.loadtxt(feat_path, dtype=float).astype(np.float32)
    labels = np.loadtxt(label_path, dtype=int).reshape(-1).astype(np.int64)
    base_graph = resolve_existing_base_graph(dataset, knn_k)
    if base_graph is None:
        raw_adj = build_knn_adj(features, knn_k)
    else:
        raw_adj = read_edge_list_to_adj(base_graph, labels.shape[0])
    return features, labels, raw_adj


def build_knn_adj(features: np.ndarray, knn_k: int) -> sp.csr_matrix:
    nbrs = NearestNeighbors(n_neighbors=int(knn_k) + 1, algorithm="ball_tree").fit(features)
    return postprocess_adj(nbrs.kneighbors_graph(features))


def edge_pairs(adj: sp.csr_matrix) -> np.ndarray:
    coo = sp.triu(adj, k=1).tocoo()
    return np.column_stack([coo.row, coo.col]).astype(np.int64)


def adj_from_pairs(pairs: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    if pairs.size == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    values = np.ones(pairs.shape[0], dtype=np.float32)
    return postprocess_adj(sp.csr_matrix((values, (pairs[:, 0], pairs[:, 1])), shape=(num_nodes, num_nodes)))


def write_edge_list(adj: sp.csr_matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = edge_pairs(adj)
    if pairs.size == 0:
        path.write_text("", encoding="utf-8")
    else:
        np.savetxt(path, pairs, fmt="%d")


def perturb_delete(adj: sp.csr_matrix, rate: float, rng: np.random.Generator) -> sp.csr_matrix:
    pairs = edge_pairs(adj)
    if pairs.size == 0:
        return adj.copy()
    keep_count = max(1, int(round(pairs.shape[0] * (1.0 - float(rate)))))
    keep_idx = rng.choice(pairs.shape[0], size=keep_count, replace=False)
    return adj_from_pairs(pairs[np.sort(keep_idx)], adj.shape[0])


def perturb_add(adj: sp.csr_matrix, rate: float, rng: np.random.Generator) -> sp.csr_matrix:
    pairs = edge_pairs(adj)
    add_count = int(round(pairs.shape[0] * float(rate)))
    if add_count <= 0:
        return adj.copy()
    n = adj.shape[0]
    existing = {tuple(pair) for pair in pairs.tolist()}
    added: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(1000, add_count * 50)
    while len(added) < add_count and attempts < max_attempts:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        attempts += 1
        if u == v:
            continue
        if u > v:
            u, v = v, u
        pair = (u, v)
        if pair in existing or pair in added:
            continue
        added.add(pair)
    if not added:
        return adj.copy()
    add_pairs = np.asarray(sorted(added), dtype=np.int64)
    all_pairs = np.vstack([pairs, add_pairs]) if pairs.size else add_pairs
    return adj_from_pairs(all_pairs, n)


def edge_homophily(adj: sp.csr_matrix, labels: np.ndarray) -> float:
    pairs = edge_pairs(adj)
    if pairs.size == 0:
        return 0.0
    return float(np.mean(labels[pairs[:, 0]] == labels[pairs[:, 1]]))


def edge_overlap_ratio(adj_a: sp.csr_matrix, adj_b: sp.csr_matrix) -> float:
    denom = max(1, edge_pairs(adj_a).shape[0])
    return float(edge_pairs(adj_a.multiply(adj_b)).shape[0] / denom)


def resolve_ae_graph(dataset: str) -> Path:
    path = ROOT / "data" / "ae_graph" / f"{dataset}_ae_graph.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing current AE graph asset: {path}")
    return path


def build_module_args(config: dict[str, Any], profile: dict[str, Any], *, enable_dcgl_negative: bool) -> dict[str, Any]:
    if not enable_dcgl_negative:
        return {}
    args = {"enable_dcgl_negative_loss": True}
    args.update(config.get("dcgl_negative_args", {}))
    args.update(profile.get("dcgl_negative_args", {}))
    if args.pop("disable_dcgl_neg_reliability_gate", False):
        args["disable_dcgl_neg_reliability_gate"] = True
    return args


def build_train_command(
    config: dict[str, Any],
    dataset: str,
    variant: Variant,
    args: argparse.Namespace,
    raw_graph_path: Path | None,
    fusion_path: Path | None,
) -> list[str]:
    profile = config["dataset_profiles"][dataset]
    cluster_num = int(profile["cluster_num"])
    train_args = merge_args(config.get("baseline_args", {}), config.get("train_common_args", {}), profile.get("train_args", {}))
    train_args["device"] = args.device
    train_args["runs"] = int(args.runs)
    train_args["seed_start"] = int(args.seed_start)

    cmd = [
        str(args.python),
        "train.py",
        "--dataset",
        dataset,
        "--cluster_num",
        str(cluster_num),
        "--graph_mode",
        variant.graph_mode,
    ]
    if raw_graph_path is not None and variant.graph_mode in {"raw", "dual"}:
        cmd.extend(["--raw_graph_path", str(raw_graph_path)])

    if variant.graph_mode == "ae":
        cmd.extend(dict_to_cli(train_args))
        cmd.extend(["--ae_graph_path", str(resolve_ae_graph(dataset))])
    elif variant.graph_mode == "dual":
        merged_dual_args = merge_args(train_args, config.get("dual_args", {}))
        cmd.extend(["--ae_graph_path", str(resolve_ae_graph(dataset))])
        if variant.fusion_mode is None:
            raise ValueError("Dual variant requires fusion mode.")
        cmd.extend(["--fusion_mode", variant.fusion_mode])
        if variant.fusion_mode == "attn":
            merged_dual_args = merge_args(
                merged_dual_args,
                config.get("dual_attn_args", {}),
                profile.get("dual_attn_args", {}),
            )
            if fusion_path is not None:
                cmd.extend(["--save_fusion_weights_path", str(fusion_path)])
        elif variant.fusion_mode == "mean":
            merged_dual_args = merge_args(merged_dual_args, config.get("dual_mean_args", {}))
        cmd.extend(dict_to_cli(merged_dual_args))
    else:
        cmd.extend(dict_to_cli(train_args))
    cmd.extend(dict_to_cli(build_module_args(config, profile, enable_dcgl_negative=variant.enable_dcgl_negative)))
    return cmd


def run_command(cmd: list[str], timeout: int) -> tuple[int, str, str]:
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


def score_metrics(metrics: dict[str, dict[str, float]]) -> float:
    if not all(metric in metrics for metric in METRICS):
        return float("-inf")
    return metrics["ACC"]["mean"] + 0.4 * metrics["F1"]["mean"] + 0.2 * metrics["NMI"]["mean"] + 0.2 * metrics["ARI"]["mean"]


def safe_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(key): safe_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(item) for item in obj]
    return obj


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + int(base_seed)) % (2**32)


def job_key(dataset: str, condition: str, value: float | int, variant: str) -> str:
    value_text = f"{float(value):.8g}" if isinstance(value, float) else str(value)
    return f"{dataset}|{condition}|{value_text}|{variant}"


def load_finished(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("returncode") == 0 and row.get("metrics"):
            rows[str(row["job_key"])] = row
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_json(row), ensure_ascii=False) + "\n")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("returncode") != 0 or not row.get("metrics"):
            continue
        grouped.setdefault((row["condition"], str(row["value"]), row["variant"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (condition, value, variant), items in sorted(grouped.items()):
        record: dict[str, Any] = {
            "condition": condition,
            "value": value,
            "variant": variant,
            "datasets": len(items),
        }
        for metric in METRICS:
            vals = [item["metrics"][metric]["mean"] for item in items if metric in item.get("metrics", {})]
            if vals:
                record[f"avg_{metric.lower()}"] = sum(vals) / len(vals)
        scores = [score_metrics(item["metrics"]) for item in items]
        scores = [score for score in scores if math.isfinite(score)]
        if scores:
            record["avg_score"] = sum(scores) / len(scores)
        out.append(record)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def metric_text(metrics: dict[str, dict[str, float]], metric: str) -> str:
    item = metrics.get(metric)
    if not item:
        return ""
    return f"{item['mean']:.2f}+-{item['std']:.2f}"


def persist_outputs(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    per_rows: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics", {})
        record = {
            "dataset": row.get("dataset"),
            "condition": row.get("condition"),
            "value": row.get("value"),
            "variant": row.get("variant"),
            "returncode": row.get("returncode"),
            "score": score_metrics(metrics),
            "raw_graph_path": row.get("raw_graph_path"),
            "stdout_log": row.get("stdout_log"),
        }
        for metric in METRICS:
            record[f"{metric.lower()}_mean"] = metrics.get(metric, {}).get("mean", "")
            record[f"{metric.lower()}_std"] = metrics.get(metric, {}).get("std", "")
        per_rows.append(record)

    write_csv(
        run_dir / "per_dataset.csv",
        per_rows,
        [
            "dataset",
            "condition",
            "value",
            "variant",
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
            "raw_graph_path",
            "stdout_log",
        ],
    )
    aggregate = aggregate_rows(rows)
    write_csv(
        run_dir / "aggregate.csv",
        aggregate,
        ["condition", "value", "variant", "datasets", "avg_acc", "avg_nmi", "avg_ari", "avg_f1", "avg_score"],
    )

    lines = [
        "# Structural Uncertainty Robustness",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- Protocol: perturb only the original graph A; reuse the current project A_E asset.",
        "- Conditions: edge deletion, edge addition, and KNN perturbation.",
        "",
        "## Aggregate",
        "",
        "| Condition | Value | Variant | Datasets | Avg ACC | Avg NMI | Avg ARI | Avg F1 | Avg Score |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in aggregate:
        lines.append(
            "| {condition} | {value} | {variant} | {datasets} | {acc:.2f} | {nmi:.2f} | {ari:.2f} | {f1:.2f} | {score:.2f} |".format(
                condition=item["condition"],
                value=item["value"],
                variant=item["variant"],
                datasets=int(item.get("datasets", 0)),
                acc=float(item.get("avg_acc", 0.0)),
                nmi=float(item.get("avg_nmi", 0.0)),
                ari=float(item.get("avg_ari", 0.0)),
                f1=float(item.get("avg_f1", 0.0)),
                score=float(item.get("avg_score", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Dataset",
            "",
            "| Dataset | Condition | Value | Variant | ACC | NMI | ARI | F1 | Score | Raw graph | Log |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["dataset"], item["condition"], str(item["value"]), item["variant"])):
        metrics = row.get("metrics", {})
        lines.append(
            "| {dataset} | {condition} | {value} | {variant} | {acc} | {nmi} | {ari} | {f1} | {score:.2f} | `{raw}` | `{log}` |".format(
                dataset=DATASET_LABELS.get(row["dataset"], row["dataset"]),
                condition=row["condition"],
                value=row["value"],
                variant=row["variant_label"],
                acc=metric_text(metrics, "ACC"),
                nmi=metric_text(metrics, "NMI"),
                ari=metric_text(metrics, "ARI"),
                f1=metric_text(metrics, "F1"),
                score=score_metrics(metrics),
                raw=rel(Path(row.get("raw_graph_path", ""))) if row.get("raw_graph_path") else "",
                log=rel(Path(row.get("stdout_log", ""))),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Raw JSONL: `{rel(run_dir / 'results.jsonl')}`",
            f"- Per-dataset CSV: `{rel(run_dir / 'per_dataset.csv')}`",
            f"- Aggregate CSV: `{rel(run_dir / 'aggregate.csv')}`",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def materialize_condition_graphs(
    dataset: str,
    features: np.ndarray,
    labels: np.ndarray,
    raw_adj: sp.csr_matrix,
    edge_rates: tuple[float, ...],
    knn_values: tuple[int, ...],
    run_dir: Path,
    seed: int,
) -> dict[tuple[str, float | int], dict[str, Any]]:
    graphs: dict[tuple[str, float | int], dict[str, Any]] = {}
    for rate in edge_rates:
        for condition, builder in (("edge_delete", perturb_delete), ("edge_add", perturb_add)):
            rng = np.random.default_rng(stable_seed(seed, dataset, condition, f"{float(rate):.8g}"))
            adj = builder(raw_adj, float(rate), rng)
            graph_path = run_dir / "raw_graphs" / dataset / condition / f"{float(rate):.2f}".replace(".", "p") / f"{dataset}_{condition}_{float(rate):.2f}.txt"
            write_edge_list(adj, graph_path)
            graphs[(condition, float(rate))] = {
                "path": graph_path,
                "edges": int(edge_pairs(adj).shape[0]),
                "homophily": edge_homophily(adj, labels),
                "overlap_with_clean": edge_overlap_ratio(raw_adj, adj),
            }
    for k_value in knn_values:
        adj = build_knn_adj(features, int(k_value))
        graph_path = run_dir / "raw_graphs" / dataset / "knn_k" / str(int(k_value)) / f"{dataset}_knn_k{int(k_value)}.txt"
        write_edge_list(adj, graph_path)
        graphs[("knn_k", int(k_value))] = {
            "path": graph_path,
            "edges": int(edge_pairs(adj).shape[0]),
            "homophily": edge_homophily(adj, labels),
            "overlap_with_clean": edge_overlap_ratio(raw_adj, adj),
        }
    return graphs


def main() -> int:
    args = parse_args()
    config = load_experiment_config()
    datasets = parse_datasets(args.datasets, config)
    variants = parse_variants(args.variants)
    edge_rates = parse_float_list(args.edge_rates)
    knn_values = parse_int_list(args.knn_values)

    if args.run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_ROOT / f"{stamp}_{'_'.join(datasets)}_structural_uncertainty"
    else:
        run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    finished = load_finished(results_path) if args.resume else {}
    rows = list(finished.values())

    print(f"[RUN] output={run_dir}", flush=True)
    print(f"[RUN] datasets={','.join(datasets)} variants={','.join(v.key for v in variants)}", flush=True)

    for dataset in datasets:
        profile = config["dataset_profiles"][dataset]
        base_k = int(profile.get("train_args", {}).get("knn_k", config.get("train_common_args", {}).get("knn_k", 5)))
        features, labels, raw_adj = load_dataset_arrays(dataset, base_k)
        graph_meta = materialize_condition_graphs(dataset, features, labels, raw_adj, edge_rates, knn_values, run_dir, args.perturb_seed)
        rsl_reference_row: dict[str, Any] | None = None

        for (condition, value), meta in graph_meta.items():
            raw_graph_path = Path(meta["path"])
            for variant in variants:
                key = job_key(dataset, condition, value, variant.key)
                if key in finished:
                    print(f"[SKIP] {key}", flush=True)
                    continue
                if variant.key == "rsl" and rsl_reference_row is not None:
                    row = dict(rsl_reference_row)
                    row.update(
                        {
                            "job_key": key,
                            "condition": condition,
                            "value": value,
                            "raw_graph_path": raw_graph_path,
                            "graph_diag": meta,
                            "note": "RSL is independent of raw-graph perturbation; reused from the first RSL run.",
                        }
                    )
                    rows.append(row)
                    append_jsonl(results_path, row)
                    print(f"[REUSE] {dataset} {condition}={value} RSL", flush=True)
                    continue
                log_dir = run_dir / "logs" / dataset / condition / str(value).replace(".", "p") / variant.key
                log_dir.mkdir(parents=True, exist_ok=True)
                stdout_log = log_dir / "stdout.log"
                stderr_log = log_dir / "stderr.log"
                cmd_log = log_dir / "cmd.txt"
                fusion_path = log_dir / "fusion_weights.npz" if variant.graph_mode == "dual" and variant.fusion_mode == "attn" else None
                cmd = build_train_command(config, dataset, variant, args, raw_graph_path, fusion_path)
                cmd_log.write_text(" ".join(cmd) + "\n", encoding="utf-8")
                print(f"[TRAIN] {dataset} {condition}={value} {variant.label}", flush=True)
                if args.dry_run:
                    row = {
                        "job_key": key,
                        "dataset": dataset,
                        "condition": condition,
                        "value": value,
                        "variant": variant.key,
                        "variant_label": variant.label,
                        "returncode": 0,
                        "metrics": {},
                        "raw_graph_path": raw_graph_path,
                        "stdout_log": stdout_log,
                        "stderr_log": stderr_log,
                        "command_log": cmd_log,
                        "graph_diag": meta,
                    }
                    rows.append(row)
                    append_jsonl(results_path, row)
                    continue

                rc, stdout, stderr = run_command(cmd, args.timeout)
                stdout_log.write_text(stdout or "", encoding="utf-8")
                stderr_log.write_text(stderr or "", encoding="utf-8")
                metrics = parse_final_metrics(stdout)
                row = {
                    "job_key": key,
                    "dataset": dataset,
                    "condition": condition,
                    "value": value,
                    "variant": variant.key,
                    "variant_label": variant.label,
                    "returncode": rc,
                    "metrics": metrics,
                    "score": score_metrics(metrics),
                    "raw_graph_path": raw_graph_path,
                    "stdout_log": stdout_log,
                    "stderr_log": stderr_log,
                    "command_log": cmd_log,
                    "graph_diag": meta,
                }
                rows.append(row)
                append_jsonl(results_path, row)
                if variant.key == "rsl" and rc == 0 and metrics and rsl_reference_row is None:
                    rsl_reference_row = dict(row)
                if args.update_summary_every > 0 and len(rows) % args.update_summary_every == 0:
                    persist_outputs(run_dir, rows)
                if rc == 0:
                    print(f"[OK] {dataset} {condition}={value} {variant.label} ACC={metrics.get('ACC', {}).get('mean', 'NA')}", flush=True)
                else:
                    print(f"[WARN] {dataset} {condition}={value} {variant.label} failed; see {rel(stderr_log)}", flush=True)

    persist_outputs(run_dir, rows)
    print(f"[DONE] summary={run_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
