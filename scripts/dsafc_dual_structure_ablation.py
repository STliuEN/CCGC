from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "dsafc_dual_structure_ablation"
SCGC_ENV_PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
DEFAULT_PYTHON = SCGC_ENV_PYTHON if SCGC_ENV_PYTHON.exists() else Path(sys.executable)
METRICS = ("ACC", "NMI", "ARI", "F1")
ABLATION_ORDER = ("osl", "rsl", "f_dsf", "a_dsf", "dsafc")
ABLATION_LABELS = {
    "osl": "OSL",
    "rsl": "RSL",
    "f_dsf": "F-DSF",
    "a_dsf": "A-DSF",
    "dsafc": "DSAFC",
}
FUSION_DOMINANCE_TOL = 0.05


@dataclass(frozen=True)
class AblationVariant:
    key: str
    label: str
    graph_mode: str
    fusion_mode: str | None
    enable_dcgl_negative: bool
    description: str


VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant(
        key="osl",
        label="OSL",
        graph_mode="raw",
        fusion_mode=None,
        enable_dcgl_negative=False,
        description="Original-Structure Learning using only raw graph A.",
    ),
    AblationVariant(
        key="rsl",
        label="RSL",
        graph_mode="ae",
        fusion_mode=None,
        enable_dcgl_negative=False,
        description="Refined-Structure Learning using only refined graph A_E.",
    ),
    AblationVariant(
        key="f_dsf",
        label="F-DSF",
        graph_mode="dual",
        fusion_mode="mean",
        enable_dcgl_negative=False,
        description="Fixed Dual-Structure Fusion with 0.5/0.5 branch weights.",
    ),
    AblationVariant(
        key="a_dsf",
        label="A-DSF",
        graph_mode="dual",
        fusion_mode="attn",
        enable_dcgl_negative=False,
        description="Adaptive Dual-Structure Fusion without DCGL-negative term.",
    ),
    AblationVariant(
        key="dsafc",
        label="DSAFC",
        graph_mode="dual",
        fusion_mode="attn",
        enable_dcgl_negative=True,
        description="Full current DSAFC narrative: adaptive fusion + DCGL-negative only.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current DSAFC ablation chain (OSL/RSL/F-DSF/A-DSF/DSAFC) "
            "and export raw metrics, structure diagnostics, homophily diagnostics, "
            "and fusion-weight diagnostics for the current experiment.py settings."
        )
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset key, comma-separated dataset keys, or 'all'.",
    )
    parser.add_argument("--runs", type=int, default=10, help="Training runs per ablation variant.")
    parser.add_argument("--seed-start", type=int, default=0, help="First training seed.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--timeout", type=int, default=0, help="Per-training timeout in seconds. 0 disables timeout.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None, help="Exact output directory.")
    parser.add_argument("--resume-jsonl", type=Path, default=None, help="Optional existing results.jsonl to skip finished jobs.")
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated subset of osl,rsl,f_dsf,a_dsf,dsafc or 'all'.",
    )
    parser.add_argument(
        "--export-fusion-artifacts",
        action="store_true",
        help="Save per-node fusion weights and branch embeddings for dual variants.",
    )
    parser.add_argument(
        "--reuse-current-ae-assets",
        action="store_true",
        help="Use the current project AE graph/checkpoint directly instead of per-dataset overrides.",
    )
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
    aliases = {
        "reuters": "reut",
        "reut": "reut",
        "uat": "uat",
        "amap": "amap",
        "usps": "usps",
        "eat": "eat",
        "cora": "cora",
        "cite": "cite",
        "citeseer": "cite",
    }
    if str(raw).strip().lower() == "all":
        preferred = ["reut", "uat", "amap", "usps", "eat", "cora", "cite"]
        return tuple(name for name in preferred if name in profiles)
    out: list[str] = []
    for token in str(raw).replace(";", ",").split(","):
        name = token.strip().lower().replace("-", "_")
        if not name:
            continue
        name = aliases.get(name, name)
        if name not in profiles:
            raise ValueError(f"Unsupported dataset '{token}'.")
        out.append(name)
    if not out:
        raise ValueError("No valid datasets were provided.")
    return tuple(dict.fromkeys(out))


def parse_variant_list(raw: str) -> tuple[AblationVariant, ...]:
    by_key = {variant.key: variant for variant in VARIANTS}
    if str(raw).strip().lower() == "all":
        return VARIANTS
    picked: list[AblationVariant] = []
    for token in str(raw).replace(";", ",").split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in by_key:
            raise ValueError(f"Unsupported ablation variant '{token}'.")
        picked.append(by_key[key])
    if not picked:
        raise ValueError("No valid ablation variants were provided.")
    seen: dict[str, AblationVariant] = {}
    for item in picked:
        seen[item.key] = item
    return tuple(seen[key] for key in ABLATION_ORDER if key in seen)


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
        prefix = feat_path.name[:-len("_feat.npy")]
        if (feat_path.parent / f"{prefix}_label.npy").exists() and (feat_path.parent / f"{prefix}_adj.npy").exists():
            return True
    return False


def resolve_existing_base_graph(dataset: str, knn_k: int) -> Path | None:
    default_path = ROOT / "data" / "graph" / f"{dataset}_graph.txt"
    if default_path.exists():
        return default_path
    k_path = ROOT / "data" / "graph" / f"{dataset}{int(knn_k)}_graph.txt"
    if k_path.exists():
        return k_path
    return None


def read_edge_list_to_adj(path: Path, num_nodes: int) -> sp.csr_matrix:
    try:
        edges = np.loadtxt(path, dtype=int)
    except ValueError:
        edges = np.empty((0, 2), dtype=int)
    if edges.size == 0:
        adj = sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    else:
        if edges.ndim == 1:
            if edges.shape[0] != 2:
                raise ValueError(f"Invalid edge list format in {path}")
            edges = edges.reshape(1, 2)
        values = np.ones(edges.shape[0], dtype=np.float32)
        adj = sp.csr_matrix((values, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj = adj + adj.T
    adj = adj.sign().tocsr()
    return adj


def load_labels_and_raw_adj(dataset: str, knn_k: int) -> tuple[np.ndarray, sp.csr_matrix]:
    if has_npy_triplet(dataset):
        dataset_dir = ROOT / "data" / "full_dataset" / dataset
        feat_paths = list(dataset_dir.rglob("*_feat.npy"))
        if not feat_paths:
            raise FileNotFoundError(f"Missing npy features for {dataset}")
        feat_path = feat_paths[0]
        prefix = feat_path.name[:-len("_feat.npy")]
        label_path = feat_path.parent / f"{prefix}_label.npy"
        adj_path = feat_path.parent / f"{prefix}_adj.npy"
        labels = np.load(label_path, allow_pickle=True)
        labels = np.asarray(labels)
        if labels.ndim == 2:
            if labels.shape[1] == 1:
                labels = labels[:, 0]
            else:
                labels = np.argmax(labels, axis=1)
        labels = labels.reshape(-1).astype(np.int64)
        adj_obj = np.load(adj_path, allow_pickle=True)
        if isinstance(adj_obj, np.ndarray) and adj_obj.dtype == object and adj_obj.shape == ():
            adj_obj = adj_obj.item()
        raw_adj = sp.csr_matrix(np.asarray(adj_obj, dtype=np.float32)) if not sp.issparse(adj_obj) else adj_obj.tocsr().astype(np.float32)
        raw_adj.eliminate_zeros()
        raw_adj = raw_adj - sp.dia_matrix((raw_adj.diagonal()[np.newaxis, :], [0]), shape=raw_adj.shape)
        raw_adj.eliminate_zeros()
        raw_adj = raw_adj + raw_adj.T
        raw_adj = raw_adj.sign().tocsr()
        return labels, raw_adj

    label_path = ROOT / "data" / "data" / f"{dataset}_label.txt"
    labels = np.loadtxt(label_path, dtype=int).reshape(-1).astype(np.int64)
    base_graph = resolve_existing_base_graph(dataset, knn_k)
    if base_graph is None:
        raise FileNotFoundError(f"Missing base graph asset for {dataset} with k={knn_k}")
    return labels, read_edge_list_to_adj(base_graph, labels.shape[0])


def edge_count(adj: sp.csr_matrix) -> int:
    return int(adj.nnz // 2)


def edge_overlap_ratio(adj_a: sp.csr_matrix, adj_b: sp.csr_matrix) -> float:
    overlap = adj_a.multiply(adj_b)
    denom = max(1, edge_count(adj_a))
    return float(edge_count(overlap) / denom)


def new_edge_ratio(adj_a: sp.csr_matrix, adj_b: sp.csr_matrix) -> float:
    new_edges = adj_b - adj_b.multiply(adj_a)
    new_edges.data[:] = 1
    denom = max(1, edge_count(adj_b))
    return float(edge_count(new_edges) / denom)


def edge_homophily(adj: sp.csr_matrix, labels: np.ndarray) -> float:
    coo = sp.triu(adj, k=1).tocoo()
    if coo.nnz == 0:
        return 0.0
    same = np.sum(labels[coo.row] == labels[coo.col])
    return float(same / coo.nnz)


def parse_final_metrics(stdout: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    pattern = re.compile(r"^(ACC|NMI|ARI|F1)\s+\|\s+([+-]?\d+(?:\.\d+)?)\s+.+?\s+([+-]?\d+(?:\.\d+)?)$", re.MULTILINE)
    for match in pattern.finditer(stdout):
        metrics[match.group(1)] = {
            "mean": float(match.group(2)),
            "std": float(match.group(3)),
        }
    return metrics


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


def parse_resource_summary(stdout: str) -> dict[str, float]:
    resource: dict[str, float] = {}
    pattern = re.compile(r"^RESOURCE\s+\|\s+(.+?)\s+\|\s+([+-]?\d+(?:\.\d+)?)\s*$", re.MULTILINE)
    for match in pattern.finditer(stdout or ""):
        key = RESOURCE_LABEL_TO_KEY.get(match.group(1).strip())
        if key:
            resource[key] = float(match.group(2))
    return resource


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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(key): safe_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(item) for item in obj]
    return obj


def load_resume(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[(row["dataset"], row["variant"])] = row
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_json(row), ensure_ascii=False) + "\n")


def run_command(cmd: list[str], *, cwd: Path, timeout: int) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=None if timeout <= 0 else timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def build_module_args(config: dict[str, Any], profile: dict[str, Any], *, enable_dcgl_negative: bool) -> dict[str, Any]:
    dynamic_enabled = bool(config.get("enable_dynamic_threshold_module", False))
    ema_enabled = bool(config.get("enable_ema_prototypes_module", False))
    dcgl_cluster_enabled = False
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
    if enable_dcgl_negative:
        improved["enable_dcgl_negative_loss"] = True
        improved.update(config.get("dcgl_negative_args", {}))
        improved.update(profile.get("dcgl_negative_args", {}))
        if improved.pop("disable_dcgl_neg_reliability_gate", False):
            improved["disable_dcgl_neg_reliability_gate"] = True
    if dcgl_cluster_enabled:
        improved["enable_dcgl_cluster_level"] = True
        improved.update(config.get("dcgl_cluster_args", {}))
        improved.update(profile.get("dcgl_cluster_args", {}))
    if gcn_enabled:
        improved["enable_gcn_backbone"] = True
        improved.update(config.get("gcn_backbone_args", {}))
        improved.update(profile.get("gcn_backbone_args", {}))
    return improved


def resolve_ae_graph_and_model(dataset: str, config: dict[str, Any], profile: dict[str, Any], reuse_current: bool) -> tuple[Path, Path | None]:
    if reuse_current:
        graph_path = ROOT / "data" / "ae_graph" / f"{dataset}_ae_graph.txt"
        model_path = ROOT / "pretrain_graph" / f"{dataset}_ae_pretrain.pkl"
        return graph_path, model_path if model_path.exists() else None

    ae_args = merge_args(config.get("ae_args", {}), profile.get("ae_args", {}))
    out_graph = str(ae_args.get("out_graph_path", "")).strip()
    model_path = str(ae_args.get("model_save_path", "")).strip()
    if out_graph:
        graph_path = (ROOT / out_graph).resolve()
    else:
        graph_path = ROOT / "data" / "ae_graph" / f"{dataset}_ae_graph.txt"
    if model_path:
        resolved_model = (ROOT / model_path).resolve()
    else:
        default_model = ROOT / "pretrain_graph" / f"{dataset}_ae_pretrain.pkl"
        resolved_model = default_model if default_model.exists() else None
    return graph_path, resolved_model


def compute_structure_diag(dataset: str, knn_k: int, ae_graph_path: Path) -> dict[str, Any]:
    labels, raw_adj = load_labels_and_raw_adj(dataset, knn_k)
    ae_adj = read_edge_list_to_adj(ae_graph_path, labels.shape[0])
    avg_degree_raw = 0.0 if labels.shape[0] == 0 else float(2.0 * edge_count(raw_adj) / labels.shape[0])
    avg_degree_ae = 0.0 if labels.shape[0] == 0 else float(2.0 * edge_count(ae_adj) / labels.shape[0])
    return {
        "num_nodes": int(labels.shape[0]),
        "raw_edges": edge_count(raw_adj),
        "ae_edges": edge_count(ae_adj),
        "edge_overlap_ratio": edge_overlap_ratio(raw_adj, ae_adj),
        "new_edge_ratio": new_edge_ratio(raw_adj, ae_adj),
        "avg_degree_raw": avg_degree_raw,
        "avg_degree_ae": avg_degree_ae,
        "homophily_raw": edge_homophily(raw_adj, labels),
        "homophily_ae": edge_homophily(ae_adj, labels),
    }


def compute_fusion_diag(fusion_npz: Path) -> dict[str, Any]:
    data = np.load(fusion_npz, allow_pickle=True)
    weights = np.asarray(data["fusion_weights"], dtype=np.float32)
    fusion_mean = np.asarray(data["fusion_mean"], dtype=np.float32)
    entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0)), axis=1)
    diff = weights[:, 0] - weights[:, 1]
    dominant = np.where(np.abs(diff) < FUSION_DOMINANCE_TOL, "balanced", np.where(diff > 0, "raw", "ae"))
    mean_diff = float(fusion_mean[0]) - float(fusion_mean[1])
    if abs(mean_diff) < FUSION_DOMINANCE_TOL:
        dominant_view = "balanced"
    else:
        dominant_view = "raw" if mean_diff > 0 else "ae"
    return {
        "mean_alpha_raw": float(np.mean(weights[:, 0])),
        "mean_alpha_ae": float(np.mean(weights[:, 1])),
        "std_alpha_raw": float(np.std(weights[:, 0])),
        "std_alpha_ae": float(np.std(weights[:, 1])),
        "mean_entropy": float(np.mean(entropy)),
        "std_entropy": float(np.std(entropy)),
        "dominant_view": dominant_view,
        "dominant_raw_ratio": float(np.mean(dominant == "raw")),
        "dominant_ae_ratio": float(np.mean(dominant == "ae")),
        "dominant_balanced_ratio": float(np.mean(dominant == "balanced")),
    }


def fixed_mean_fusion_diag() -> dict[str, Any]:
    entropy = -2.0 * 0.5 * math.log(0.5)
    return {
        "mean_alpha_raw": 0.5,
        "mean_alpha_ae": 0.5,
        "std_alpha_raw": 0.0,
        "std_alpha_ae": 0.0,
        "mean_entropy": float(entropy),
        "std_entropy": 0.0,
        "dominant_view": "balanced",
        "dominant_raw_ratio": 0.0,
        "dominant_ae_ratio": 0.0,
        "dominant_balanced_ratio": 1.0,
    }


def build_train_command(
    config: dict[str, Any],
    dataset: str,
    variant: AblationVariant,
    args: argparse.Namespace,
    profile: dict[str, Any],
    ae_graph_path: Path,
    fusion_npz_path: Path | None,
) -> list[str]:
    cluster_num = int(profile["cluster_num"])
    train_args = merge_args(
        config.get("baseline_args", {}),
        config.get("train_common_args", {}),
        profile.get("train_args", {}),
    )
    module_args = build_module_args(config, profile, enable_dcgl_negative=variant.enable_dcgl_negative)

    cmd = [
        str(args.python),
        "train.py",
        "--dataset",
        dataset,
        "--cluster_num",
        str(cluster_num),
        "--graph_mode",
        variant.graph_mode,
        "--runs",
        str(args.runs),
        "--seed_start",
        str(args.seed_start),
        "--device",
        args.device,
    ]
    cmd.extend(dict_to_cli(train_args))

    if variant.graph_mode == "ae":
        cmd.extend(["--ae_graph_path", str(ae_graph_path)])
    if variant.graph_mode == "dual":
        cmd.extend(["--ae_graph_path", str(ae_graph_path)])
        cmd.extend(dict_to_cli(config.get("dual_args", {})))
        if variant.fusion_mode is None:
            raise ValueError("Dual variant requires fusion_mode.")
        cmd.extend(["--fusion_mode", variant.fusion_mode])
        if variant.fusion_mode == "attn":
            merged_dual_attn = merge_args(
                config.get("dual_attn_args", {}),
                profile.get("dual_attn_args", {}),
            )
            cmd.extend(dict_to_cli(merged_dual_attn))
            if fusion_npz_path is not None:
                cmd.extend(["--save_fusion_weights_path", str(fusion_npz_path)])
        elif variant.fusion_mode == "mean":
            cmd.extend(dict_to_cli(config.get("dual_mean_args", {})))
    cmd.extend(dict_to_cli(module_args))
    return cmd


def format_metric_cell(value: dict[str, float] | None) -> str:
    if not value:
        return "--"
    return f"{value['mean']:.2f}+-{value['std']:.2f}"


def format_resource_cell(resource: dict[str, Any], key: str, digits: int = 2) -> str:
    value = resource.get(key)
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def write_summary(
    out_dir: Path,
    dataset_rows: list[dict[str, Any]],
    *,
    datasets: tuple[str, ...],
    variants: tuple[AblationVariant, ...],
    args: argparse.Namespace,
) -> None:
    summary_path = out_dir / "summary.md"
    lines = [
        "# DSAFC Dual-Structure Ablation Summary",
        "",
        f"- Generated at: `{datetime.now().isoformat()}`",
        f"- Datasets: `{', '.join(datasets)}`",
        f"- Variants: `{', '.join(variant.label for variant in variants)}`",
        f"- Runs per variant: `{args.runs}`",
        f"- Seeds: `{args.seed_start}..{args.seed_start + args.runs - 1}`",
        "",
    ]

    for row in dataset_rows:
        lines.append(f"## {row['dataset']}")
        lines.append("")
        lines.append("| Variant | ACC | NMI | ARI | F1 | Score |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for variant in variants:
            result = row["variants"].get(variant.key)
            metrics = result.get("metrics", {}) if result else {}
            score_text = "--"
            if result and isinstance(result.get("score"), (int, float)) and math.isfinite(float(result["score"])):
                score_text = f"{float(result['score']):.2f}"
            lines.append(
                f"| {variant.label} | "
                f"{format_metric_cell(metrics.get('ACC'))} | "
                f"{format_metric_cell(metrics.get('NMI'))} | "
                f"{format_metric_cell(metrics.get('ARI'))} | "
                f"{format_metric_cell(metrics.get('F1'))} | "
                f"{score_text} |"
            )
        lines.append("")

        diag = row.get("structure_diag", {})
        if diag:
            lines.append("### Structure Diagnosis")
            lines.append("")
            lines.append("| |E_A| | |E_E| | Overlap | New-edge ratio | Homophily(A) | Homophily(A_E) |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            lines.append(
                f"| {row['dataset']} | {diag['raw_edges']} | {diag['ae_edges']} | "
                f"{diag['edge_overlap_ratio']:.4f} | {diag['new_edge_ratio']:.4f} | "
                f"{diag['homophily_raw']:.4f} | {diag['homophily_ae']:.4f} |"
            )
            lines.append("")

        fusion_rows = [
            (variant.label, row["variants"][variant.key].get("fusion_diag"))
            for variant in variants
            if variant.key in row["variants"] and row["variants"][variant.key].get("fusion_diag")
        ]
        if fusion_rows:
            lines.append("### Fusion Weight Diagnosis")
            lines.append("")
            lines.append("| Variant | mean alpha(A) | mean alpha(A_E) | mean entropy | dominant view | raw-dom ratio | balanced ratio | ae-dom ratio |")
            lines.append("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
            for label, diag_row in fusion_rows:
                lines.append(
                    f"| {label} | {diag_row['mean_alpha_raw']:.4f} | {diag_row['mean_alpha_ae']:.4f} | "
                    f"{diag_row['mean_entropy']:.4f} | {diag_row['dominant_view']} | "
                    f"{diag_row['dominant_raw_ratio']:.4f} | "
                    f"{diag_row.get('dominant_balanced_ratio', 0.0):.4f} | "
                    f"{diag_row['dominant_ae_ratio']:.4f} |"
                )
            lines.append("")

        resource_rows = [
            (variant.label, row["variants"][variant.key].get("resource", {}))
            for variant in variants
            if variant.key in row["variants"] and row["variants"][variant.key].get("resource")
        ]
        if resource_rows:
            lines.append("### Training Resource Monitor")
            lines.append("")
            lines.append("| Variant | Wall time (s) | CPU (%) | Proc CPU (%) | RAM used (GB) | Proc RSS (GB) | GPU (%) | GPU mem (GB) | Torch peak (GB) |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for label, resource in resource_rows:
                lines.append(
                    f"| {label} | "
                    f"{format_resource_cell(resource, 'wall_time_sec', 2)} | "
                    f"{format_resource_cell(resource, 'cpu_percent', 1)} | "
                    f"{format_resource_cell(resource, 'process_cpu_percent', 1)} | "
                    f"{format_resource_cell(resource, 'ram_used_gb', 3)} | "
                    f"{format_resource_cell(resource, 'process_rss_gb', 3)} | "
                    f"{format_resource_cell(resource, 'gpu_util_percent', 1)} | "
                    f"{format_resource_cell(resource, 'gpu_memory_used_gb', 3)} | "
                    f"{format_resource_cell(resource, 'torch_gpu_allocated_gb', 3)} |"
                )
            lines.append("")

    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_experiment_config()
    datasets = parse_dataset_list(args.dataset, config)
    variants = parse_variant_list(args.variants)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.run_dir or (args.output_root / f"{stamp}_{'_'.join(datasets)}_{'_'.join(variant.key for variant in variants)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl = out_dir / "results.jsonl"
    resume_rows = load_resume(args.resume_jsonl or results_jsonl)
    dataset_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        profile = config["dataset_profiles"][dataset]
        knn_k = int(profile.get("train_args", {}).get("knn_k", config.get("train_common_args", {}).get("knn_k", 5)))
        ae_graph_path, ae_model_path = resolve_ae_graph_and_model(dataset, config, profile, args.reuse_current_ae_assets)
        structure_diag = compute_structure_diag(dataset, knn_k, ae_graph_path)

        dataset_row = {
            "dataset": dataset,
            "dataset_label": dataset,
            "ae_graph_path": str(ae_graph_path),
            "ae_model_path": str(ae_model_path) if ae_model_path else None,
            "structure_diag": structure_diag,
            "variants": {},
        }

        for variant in variants:
            cached = resume_rows.get((dataset, variant.key))
            if cached is not None:
                print(f"[SKIP] {dataset}/{variant.key} cached", flush=True)
                dataset_row["variants"][variant.key] = cached
                continue

            variant_dir = out_dir / dataset / variant.key
            variant_dir.mkdir(parents=True, exist_ok=True)
            fusion_npz = variant_dir / "fusion_diag_best.npz" if (args.export_fusion_artifacts and variant.graph_mode == "dual" and variant.fusion_mode == "attn") else None
            cmd = build_train_command(
                config,
                dataset,
                variant,
                args,
                profile,
                ae_graph_path,
                fusion_npz,
            )
            log_path = variant_dir / "train.log"

            if args.dry_run:
                row = {
                    "dataset": dataset,
                    "variant": variant.key,
                    "label": variant.label,
                    "description": variant.description,
                    "cmd": cmd,
                    "status": "dry_run",
                }
                dataset_row["variants"][variant.key] = row
                continue

            print(f"[RUN] {dataset}/{variant.key} runs={args.runs} seed_start={args.seed_start}", flush=True)
            rc, stdout, stderr = run_command(cmd, cwd=ROOT, timeout=args.timeout)
            log_path.write_text(
                "\n".join(
                    [
                        "=" * 80,
                        "[COMMAND]",
                        " ".join(cmd),
                        "",
                        "=" * 80,
                        "[STDOUT]",
                        stdout or "",
                        "",
                        "=" * 80,
                        "[STDERR]",
                        stderr or "",
                    ]
                ),
                encoding="utf-8",
            )

            metrics = parse_final_metrics(stdout)
            resource = parse_resource_summary(stdout)
            row = {
                "dataset": dataset,
                "variant": variant.key,
                "label": variant.label,
                "description": variant.description,
                "cmd": cmd,
                "log_path": str(log_path),
                "returncode": rc,
                "metrics": metrics,
                "resource": resource,
                "score": score_metrics(metrics),
                "status": "ok" if rc == 0 and metrics else "failed",
            }
            if variant.key == "f_dsf" and rc == 0 and metrics:
                row["fusion_diag"] = fixed_mean_fusion_diag()
            if fusion_npz is not None and fusion_npz.exists():
                row["fusion_npz"] = str(fusion_npz)
                row["fusion_diag"] = compute_fusion_diag(fusion_npz)

            dataset_row["variants"][variant.key] = row
            append_jsonl(results_jsonl, row)
            print(f"[{row['status'].upper()}] {dataset}/{variant.key} score={row['score']:.2f}", flush=True)

        dataset_rows.append(dataset_row)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "datasets": list(datasets),
        "variants": [variant.key for variant in variants],
        "runs": args.runs,
        "seed_start": args.seed_start,
        "reuse_current_ae_assets": bool(args.reuse_current_ae_assets),
        "export_fusion_artifacts": bool(args.export_fusion_artifacts),
        "results_jsonl": str(results_jsonl),
        "rows": dataset_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(safe_json(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(out_dir, dataset_rows, datasets=datasets, variants=variants, args=args)
    print(f"[DONE] summary={out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
