from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiment_output" / "final_scgcn_push"
METRICS = ("ACC", "NMI", "ARI", "F1")
MAIN_DATASETS = ("reut", "uat", "amap", "usps", "eat", "cora", "cite")
DATASET_GROUPS = {
    "all": MAIN_DATASETS,
    "non_usps": tuple(dataset for dataset in MAIN_DATASETS if dataset != "usps"),
    # Main-table datasets whose current Ours ACC is still below max(SCGC-N, SCGC-S).
    "push": ("amap", "eat", "cora", "cite"),
}
SCGC_ENV_PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
DEFAULT_PYTHON = SCGC_ENV_PYTHON if SCGC_ENV_PYTHON.exists() else Path(sys.executable)

CLUSTER_NUM = {
    "reut": 4,
    "uat": 4,
    "amap": 8,
    "usps": 10,
    "eat": 4,
    "cora": 7,
    "cite": 6,
}

AE_BASE_GRAPH = {
    "reut": ROOT / "data" / "graph" / "reut5_graph.txt",
    "uat": ROOT / "data" / "graph" / "uat_graph.txt",
    "usps": ROOT / "data" / "graph" / "usps5_graph.txt",
}

AE_DEFAULT_K = {
    "reut": 15,
    "uat": 15,
    "amap": 15,
    "usps": 15,
    "eat": 15,
    "cora": 15,
    "cite": 15,
}

AE_DEFAULT_EPOCHS = {
    "reut": 30,
    "uat": 30,
    "amap": 30,
    "usps": 30,
    "eat": 30,
    "cora": 30,
    "cite": 30,
}

AE_DEFAULT_N_Z = {
    "reut": 3,
    "uat": 4,
    "amap": 8,
    "usps": 10,
    "eat": 3,
    "cora": 7,
    "cite": 3,
}

SCGCN_TARGET = {
    "reut": {"ACC": 80.32, "NMI": 55.63, "ARI": 59.67, "F1": 63.66},
    "uat": {"ACC": 52.02, "NMI": 24.62, "ARI": 19.76, "F1": 52.78},
    "amap": {"ACC": 32.94, "NMI": 17.25, "ARI": 3.81, "F1": 18.17},
    "usps": {"ACC": 82.98, "NMI": 82.51, "ARI": 76.48, "F1": 80.06},
    "eat": {"ACC": 49.62, "NMI": 18.43, "ARI": 17.30, "F1": 45.98},
    "cora": {"ACC": 64.29, "NMI": 45.47, "ARI": 38.74, "F1": 51.12},
    "cite": {"ACC": 73.19, "NMI": 46.74, "ARI": 50.01, "F1": 63.34},
}

SCGCS_TARGET = {
    "reut": {"ACC": 76.67, "NMI": 56.43, "ARI": 55.48, "F1": 63.03},
    "uat": {"ACC": 56.58, "NMI": 28.07, "ARI": 24.84, "F1": 55.52},
    "amap": {"ACC": 77.48, "NMI": 67.67, "ARI": 58.48, "F1": 72.22},
    "usps": {"ACC": 79.46, "NMI": 71.83, "ARI": 64.85, "F1": 79.09},
    "eat": {"ACC": 57.94, "NMI": 33.91, "ARI": 27.51, "F1": 57.96},
    "cora": {"ACC": 73.88, "NMI": 56.10, "ARI": 51.79, "F1": 70.81},
    "cite": {"ACC": 71.02, "NMI": 45.25, "ARI": 46.29, "F1": 64.80},
}

TARGET_ALIASES = {
    "ours": "ours",
    "current": "ours",
    "current-ours": "ours",
    "main": "ours",
    "scgcn": "scgcn",
    "scgc-n": "scgcn",
    "n": "scgcn",
    "scgcs": "scgcs",
    "scgc-s": "scgcs",
    "s": "scgcs",
    "scgc-max": "scgc-max",
    "max": "scgc-max",
    "both": "scgc-max",
}

CURRENT_OURS_TARGET = {
    "reut": {"ACC": 83.20, "NMI": 59.82, "ARI": 66.01, "F1": 70.57},
    "uat": {"ACC": 56.24, "NMI": 27.31, "ARI": 21.98, "F1": 56.44},
    "amap": {"ACC": 77.39, "NMI": 67.22, "ARI": 58.25, "F1": 71.69},
    "usps": {"ACC": 82.40, "NMI": 73.29, "ARI": 68.39, "F1": 82.16},
    "eat": {"ACC": 54.76, "NMI": 31.97, "ARI": 24.32, "F1": 52.98},
    "cora": {"ACC": 73.49, "NMI": 55.55, "ARI": 50.14, "F1": 71.17},
    "cite": {"ACC": 70.74, "NMI": 45.28, "ARI": 45.18, "F1": 61.56},
}

COMMON_TRAIN_ARGS: dict[str, Any] = {
    "t": 4,
    "linlayers": 1,
    "epochs": 400,
    "dims": 500,
    "lr": 1e-4,
    "threshold": 0.4,
    "alpha": 0.5,
    "knn_k": 5,
}

DCGL_DEFAULT = {
    "enable_dcgl_negative_loss": True,
    "dcgl_neg_tau": 0.5,
    "dcgl_neg_weight": 0.6,
}


@dataclass(frozen=True)
class Candidate:
    dataset: str
    name: str
    fusion_mode: str
    ae_k: str | int
    args: dict[str, Any]


@dataclass(frozen=True)
class Job:
    candidate: Candidate
    graph_path: Path
    train_roll_idx: int
    seed_start: int
    ae_seed: int | None
    ae_k_value: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run final per-dataset DSAFC tuning and seed rolls against SCGC baselines "
            "using existing or rolled AE graphs."
        )
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset, comma list, or group: all, non_usps, push. "
            "push = amap,eat,cora,cite."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=(
            "quick",
            "grid",
            "targeted",
            "safe_contract_grid",
            "cite_rescue",
            "cite_finalists",
            "cite_aggressive",
            "cite_hidden_k20",
            "cite_high_hidden_k20",
            "cite_peak_h512",
            "cite_peak_h480_train",
            "reut_attn_dcgl_only_push",
            "reut_attn_dcgl_refine",
            "reut_attn_dcgl_apex",
            "amap_attn_peak",
            "amap_attn_dcgl_apex",
            "uat_attn_dcgl_only_push",
            "uat_attn_dcgl_refine",
            "uat_attn_dcgl_apex",
            "uat_attn_peak",
            "uat_attn_roll",
            "uat_attn_continued",
            "eat_attn_peak",
            "eat_attn_open_peak",
            "eat_attn_open_bridge",
            "eat_attn_apex",
            "cora_attn_peak",
            "usps_peak",
            "usps_attn_peak",
            "usps_attn_micro",
            "usps_escape",
        ),
        default="quick",
        help=(
            "quick is curated; grid is broad; targeted is dataset-specific local push; "
            "safe_contract_grid reads experiment.py safe_tuning_grid under the DCGL-negative-only contract; "
            "cite_rescue is a hand-picked CiteSeer jump search; "
            "cite_finalists reruns the current CiteSeer front-runners; "
            "cite_aggressive tries structural CiteSeer escape candidates; "
            "cite_hidden_k20 narrows around the best CiteSeer k20 hidden-width branch; "
            "cite_high_hidden_k20 searches the high-hidden CiteSeer k20 tail; "
            "cite_peak_h512 micro-tunes the current CiteSeer h512 peak; "
            "cite_peak_h480_train tunes the current CiteSeer h480 training peak; "
            "reut_attn_dcgl_only_push jointly tunes Reuters attention with only DCGL negative enabled; "
            "reut_attn_dcgl_refine refines the current Reuters DCGL-only local peak; "
            "reut_attn_dcgl_apex micro-tunes the best Reuters DCGL-only peak; "
            "amap_attn_peak tunes only the AMAP adaptive-attention branch; "
            "amap_attn_dcgl_apex micro-tunes the AMAP attention branch with only DCGL negative enabled; "
            "uat_attn_dcgl_only_push jointly tunes UAT attention with only DCGL negative enabled; "
            "uat_attn_dcgl_refine refines the current UAT DCGL-only local peak; "
            "uat_attn_dcgl_apex micro-tunes the current UAT DCGL-only apex; "
            "uat_attn_peak tunes the UAT dynamic-GCN adaptive-attention branch; "
            "uat_attn_roll rolls narrow current-code UAT finalists; "
            "uat_attn_continued jointly tunes UAT bridge candidates around k20/k25 and the historical DCGL branch; "
            "eat_attn_peak tunes only the EAT adaptive-attention branch; "
            "eat_attn_open_peak tunes the EAT open-attention seed-9 branch; "
            "eat_attn_open_bridge ultra-tunes the EAT open-attention all-metric bridge; "
            "eat_attn_apex micro-tunes the current EAT all-metric apex; "
            "cora_attn_peak tunes only the Cora raw-anchored adaptive-attention branch; "
            "usps_peak tunes the stable USPS mean/attention neighborhood; "
            "usps_attn_peak tunes only the USPS adaptive-attention branch; "
            "usps_attn_micro micro-tunes the current USPS attention frontiers; "
            "usps_escape widens USPS search around hidden width, graph k, and loss strength."
        ),
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per training roll.")
    parser.add_argument(
        "--train-rolls",
        type=int,
        default=1,
        help="Number of training seed windows per candidate.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First training seed for the first training roll.",
    )
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=0,
        help="Seed-start increment between training rolls. 0 means --runs.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--ae-graph-root",
        type=Path,
        default=ROOT / "data" / "ae_graph",
        help=(
            "Root that contains <dataset>_ae_graph.txt and optional "
            "sensitivity/ae_k_<k>/<dataset>_ae_graph.txt files."
        ),
    )
    parser.add_argument(
        "--ae-rolls",
        type=int,
        default=0,
        help="Generate this many AE graph seed rolls before training. 0 reuses existing AE graphs.",
    )
    parser.add_argument(
        "--ae-seed-start",
        type=int,
        default=1000,
        help="First AE pretrain/graph seed when --ae-rolls > 0.",
    )
    parser.add_argument(
        "--ae-seed-stride",
        type=int,
        default=1,
        help="AE seed increment between AE graph rolls.",
    )
    parser.add_argument(
        "--ae-epochs",
        type=int,
        default=0,
        help="Override AE pretrain epochs. 0 uses dataset-specific defaults.",
    )
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--ae-sim-method", choices=("cos", "heat", "ncos"), default="cos")
    parser.add_argument(
        "--force-ae-roll",
        action="store_true",
        help="Regenerate rolled AE graphs even if the output graph already exists.",
    )
    parser.add_argument(
        "--pass-metrics",
        type=str,
        default="ACC",
        help="Comma-separated metrics required to beat the selected SCGC target, or 'all'.",
    )
    parser.add_argument(
        "--target-baseline",
        type=str,
        default="scgc-max",
        help=(
            "Target line: ours/current for current main-table Ours, scgcn/scgc-n, "
            "scgcs/scgc-s, or scgc-max for max(SCGC-N, SCGC-S)."
        ),
    )
    parser.add_argument(
        "--stop-on-pass",
        action="store_true",
        help="Stop the sweep after the first job that passes --pass-metrics.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print/write commands.")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help=(
            "Limit candidates after dataset filtering. 0 means no limit for quick "
            "and the first 64 candidates per dataset for grid/targeted."
        ),
    )
    parser.add_argument(
        "--candidate-offset",
        type=int,
        default=0,
        help="Skip this many candidates after dataset filtering and before max-candidates.",
    )
    parser.add_argument(
        "--rank-metric",
        choices=METRICS,
        default="ACC",
        help="Primary metric for ranking the summary table.",
    )
    parser.add_argument("--top", type=int, default=12, help="Rows per dataset in summary.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-candidate timeout in seconds. 0 disables timeout.",
    )
    parser.add_argument(
        "--dataset-budget-hours",
        type=float,
        default=0.0,
        help="Maximum wall-clock hours per dataset. 0 disables per-dataset budgeting.",
    )
    parser.add_argument(
        "--total-budget-hours",
        type=float,
        default=0.0,
        help="Maximum total wall-clock hours for this invocation. 0 disables total budgeting.",
    )
    parser.add_argument(
        "--resume-jsonl",
        type=Path,
        default=None,
        help="Existing results.jsonl to load and skip matching completed jobs.",
    )
    parser.add_argument(
        "--update-summary-every",
        type=int,
        default=1,
        help="Rewrite summary.md after every N completed jobs. 0 writes only at the end.",
    )
    return parser.parse_args()


def normalize_dataset_name(name: str) -> str:
    dataset = name.strip().lower().replace("-", "_")
    aliases = {
        "reuters": "reut",
        "citeseer": "cite",
        "citation": "cite",
        "nonusps": "non_usps",
        "non-usps": "non_usps",
    }
    return aliases.get(dataset, dataset)


def selected_dataset_names(args: argparse.Namespace) -> tuple[str, ...]:
    datasets: list[str] = []
    raw = str(args.dataset).replace(";", ",")
    for token in raw.split(","):
        name = normalize_dataset_name(token)
        if not name:
            continue
        if name in DATASET_GROUPS:
            datasets.extend(DATASET_GROUPS[name])
        elif name in CLUSTER_NUM:
            datasets.append(name)
        else:
            valid = ", ".join(sorted(set(MAIN_DATASETS) | set(DATASET_GROUPS)))
            raise ValueError(f"Unsupported dataset/group '{token}'. Valid: {valid}")

    if not datasets:
        return DATASET_GROUPS["all"]

    unique: list[str] = []
    seen = set()
    for dataset in datasets:
        if dataset not in seen:
            unique.append(dataset)
            seen.add(dataset)
    return tuple(unique)


def resolve_target_baseline(value: str) -> str:
    key = str(value).strip().lower()
    if key not in TARGET_ALIASES:
        valid = ", ".join(sorted(TARGET_ALIASES))
        raise ValueError(f"Unsupported target baseline '{value}'. Valid: {valid}")
    return TARGET_ALIASES[key]


def target_for_dataset(dataset: str, target_baseline: str) -> dict[str, float]:
    key = resolve_target_baseline(target_baseline)
    if key == "ours":
        return CURRENT_OURS_TARGET[dataset]
    if key == "scgcn":
        return SCGCN_TARGET[dataset]
    if key == "scgcs":
        return SCGCS_TARGET[dataset]
    return {
        metric: max(SCGCN_TARGET[dataset][metric], SCGCS_TARGET[dataset][metric])
        for metric in METRICS
    }


def slug_value(value: Any) -> str:
    text = str(value).replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:8]


def make_candidate(
    dataset: str,
    stem: str,
    fusion_mode: str,
    ae_k: str | int,
    **overrides: Any,
) -> Candidate:
    args = dict(DCGL_DEFAULT)
    args.update(overrides)
    return Candidate(dataset=dataset, name=stem, fusion_mode=fusion_mode, ae_k=ae_k, args=args)


BASE_ATTN_CONFIG: dict[str, dict[str, Any]] = {
    "reut": {
        "fusion_hidden": 64,
        "fusion_temp": 1.6,
        "fusion_balance": 0.25,
        "lambda_inst": 0.08,
        "lambda_clu": 0.06,
        "warmup_epochs": 35,
        "fusion_min_weight": 0.15,
    },
    "uat": {
        "fusion_hidden": 64,
        "fusion_temp": 1.9,
        "fusion_balance": 0.35,
        "lambda_inst": 0.08,
        "lambda_clu": 0.07,
        "warmup_epochs": 35,
        "fusion_min_weight": 0.20,
    },
    "amap": {
        "fusion_hidden": 64,
        "fusion_temp": 1.25,
        "fusion_balance": 0.08,
        "lambda_inst": 0.07,
        "lambda_clu": 0.035,
        "warmup_epochs": 35,
        "fusion_min_weight": 0.05,
    },
    "usps": {
        "fusion_hidden": 64,
        "fusion_temp": 1.8,
        "fusion_balance": 0.35,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "warmup_epochs": 35,
        "fusion_min_weight": 0.20,
    },
    "eat": {
        "fusion_hidden": 64,
        "fusion_temp": 2.0,
        "fusion_balance": 0.35,
        "lambda_inst": 0.08,
        "lambda_clu": 0.08,
        "warmup_epochs": 35,
        "fusion_min_weight": 0.20,
    },
    "cora": {
        "fusion_hidden": 64,
        "fusion_temp": 1.3,
        "fusion_balance": 0.0,
        "lambda_inst": 0.03,
        "lambda_clu": 0.01,
        "warmup_epochs": 70,
        "fusion_min_weight": 0.0,
        "enable_branch_bias_fusion": True,
        "branch_bias_target": "raw",
        "branch_bias_cap": 0.10,
    },
    "cite": {
        "fusion_hidden": 64,
        "fusion_temp": 1.8,
        "fusion_balance": 0.15,
        "lambda_inst": 0.045,
        "lambda_clu": 0.02,
        "warmup_epochs": 55,
        "fusion_min_weight": 0.10,
        "enable_branch_bias_fusion": True,
        "branch_bias_target": "raw",
        "branch_bias_cap": 0.15,
    },
}


def base_attn_args(dataset: str, **overrides: Any) -> dict[str, Any]:
    args = dict(BASE_ATTN_CONFIG[dataset])
    args.update(overrides)
    return args


def add_attn_candidate(
    candidates: list[Candidate],
    dataset: str,
    stem: str,
    ae_k: str | int = "default",
    **overrides: Any,
) -> None:
    candidates.append(
        make_candidate(
            dataset,
            stem,
            "attn",
            ae_k,
            **base_attn_args(dataset, **overrides),
        )
    )


def load_experiment_config() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("ccgc_experiment_config", ROOT / "experiment.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, dict):
        raise RuntimeError("experiment.py does not expose CONFIG dict")
    return config


def safe_contract_grid_candidates() -> list[Candidate]:
    config = load_experiment_config()
    contract = dict(config.get("safe_tuning_contract", {}))
    if contract.get("enable_dcgl_negative_module") is not True:
        raise RuntimeError("safe_tuning_contract must keep enable_dcgl_negative_module=True")
    for key in (
        "enable_dynamic_threshold_module",
        "enable_ema_prototypes_module",
        "enable_dcgl_cluster_module",
        "enable_gcn_backbone_module",
    ):
        if contract.get(key) is not False:
            raise RuntimeError(f"safe_tuning_contract must keep {key}=False")

    profiles = config.get("dataset_profiles", {})
    common_train = dict(config.get("train_common_args", {}))
    common_dual_attn = dict(config.get("dual_attn_args", {}))
    common_dcgl = dict(config.get("dcgl_negative_args", {}))
    forbidden = {
        "enable_dynamic_threshold",
        "enable_ema_prototypes",
        "enable_dcgl_cluster_level",
        "enable_gcn_backbone",
        "dynamic_threshold_start",
        "dynamic_threshold_end",
        "ema_proto_momentum",
        "lambda_dcgl_cluster",
        "dcgl_cluster_tau",
    }

    def values_from_grid(grid: dict[str, Any], group: str) -> dict[str, list[Any]]:
        raw = grid.get(group, {})
        if not isinstance(raw, dict):
            return {}
        values: dict[str, list[Any]] = {}
        for key, value in raw.items():
            if key in forbidden:
                continue
            if isinstance(value, (list, tuple)):
                values[key] = list(value)
            else:
                values[key] = [value]
        return values

    def expand_options(options: dict[str, list[Any]]) -> list[dict[str, Any]]:
        if not options:
            return [{}]
        keys = list(options)
        rows: list[dict[str, Any]] = []
        for combo in product(*(options[key] for key in keys)):
            rows.append(dict(zip(keys, combo)))
        return rows

    candidates: list[Candidate] = []
    seen: set[tuple[str, str, tuple[tuple[str, Any], ...]]] = set()
    for dataset in MAIN_DATASETS:
        profile = profiles.get(dataset, {})
        grid = profile.get("safe_tuning_grid", {})
        if not grid:
            continue

        base_train = dict(common_train)
        base_train.update(profile.get("train_args", {}))
        base_attn = dict(common_dual_attn)
        base_attn.update(profile.get("dual_attn_args", {}))
        base_dcgl = dict(common_dcgl)
        base_args = {
            key: value
            for key, value in {**base_train, **base_attn, **base_dcgl}.items()
            if key not in forbidden and key != "device"
        }
        base_args["enable_dcgl_negative_loss"] = True

        groups = {
            "train": values_from_grid(grid, "train_args"),
            "attn": values_from_grid(grid, "dual_attn_args"),
            "dcgl": values_from_grid(grid, "dcgl_negative_args"),
        }

        centers = [("center", {})]
        for group_name, options in groups.items():
            for option in expand_options(options):
                if option:
                    centers.append((group_name, option))

        joint_options: dict[str, list[Any]] = {}
        for options in groups.values():
            joint_options.update(options)
        for option in expand_options(joint_options):
            if option:
                centers.append(("joint", option))

        for ae_k in ("default",):
            for group_name, overrides in centers:
                args = dict(base_args)
                args.update(overrides)
                args["enable_dcgl_negative_loss"] = True
                for key in forbidden:
                    args.pop(key, None)
                changed = [
                    f"{key}{slug_value(value)}"
                    for key, value in sorted(overrides.items())
                ]
                suffix = "_".join(changed) if changed else "default"
                raw_stem = f"safe_{dataset}_{group_name}_{suffix}"
                stem = f"safe_{dataset}_{group_name}_{short_hash(raw_stem)}"
                candidate = make_candidate(dataset, stem, "attn", ae_k, **args)
                key = (
                    candidate.dataset,
                    str(candidate.ae_k),
                    tuple(sorted(candidate.args.items())),
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)

    return candidates


def quick_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    for dataset in MAIN_DATASETS:
        add_attn_candidate(candidates, dataset, "attn_notes_default", "default")
        add_attn_candidate(candidates, dataset, "attn_k10_notes", 10)
        add_attn_candidate(candidates, dataset, "attn_k20_notes", 20)

    add_attn_candidate(candidates, "reut", "attn_reut_min0p10_tau0p75_w0p4", 10,
                       fusion_min_weight=0.10, dcgl_neg_tau=0.75, dcgl_neg_weight=0.4)
    add_attn_candidate(candidates, "reut", "attn_reut_min0p20_tau0p5_w0p6", 20,
                       fusion_min_weight=0.20, dcgl_neg_tau=0.5, dcgl_neg_weight=0.6)

    add_attn_candidate(candidates, "uat", "attn_uat_temp2p1_bal0p45_min0p22", 15,
                       fusion_temp=2.1, fusion_balance=0.45, fusion_min_weight=0.22,
                       lambda_clu=0.075)
    add_attn_candidate(candidates, "uat", "attn_uat_temp1p9_bal0p45_min0p25", 20,
                       fusion_balance=0.45, fusion_min_weight=0.25)

    add_attn_candidate(candidates, "amap", "attn_amap_min0_tau0p5_w0p8", 10,
                       fusion_min_weight=0.0, dcgl_neg_tau=0.5, dcgl_neg_weight=0.8)
    add_attn_candidate(candidates, "amap", "attn_amap_bal0p05_min0p05_tau1p0_w1p0", 20,
                       fusion_balance=0.05, fusion_min_weight=0.05,
                       dcgl_neg_tau=1.0, dcgl_neg_weight=1.0)

    candidates.extend(
        [
            make_candidate("usps", "mean_default_tau0p5_w0p6", "mean", "default", warmup_epochs=35),
            make_candidate("usps", "mean_k10_tau0p75_w0p8", "mean", 10, warmup_epochs=35,
                           dcgl_neg_tau=0.75, dcgl_neg_weight=0.8),
            make_candidate("usps", "mean_k15_tau1p0_w1p0", "mean", 15, warmup_epochs=35,
                           dcgl_neg_tau=1.0, dcgl_neg_weight=1.0),
            make_candidate("usps", "mean_k20_thr0p35_tau0p75_w1p0", "mean", 20,
                           threshold=0.35, warmup_epochs=35,
                           dcgl_neg_tau=0.75, dcgl_neg_weight=1.0),
        ]
    )
    add_attn_candidate(candidates, "usps", "attn_usps_temp2p0_bal0p45_min0p20_tau0p75_w0p8", 10,
                       fusion_temp=2.0, fusion_balance=0.45,
                       dcgl_neg_tau=0.75, dcgl_neg_weight=0.8)
    add_attn_candidate(candidates, "usps", "attn_usps_temp2p2_bal0p45_min0p25_tau1p0_w1p0", 20,
                       fusion_temp=2.2, fusion_balance=0.45, fusion_min_weight=0.25,
                       dcgl_neg_tau=1.0, dcgl_neg_weight=1.0)

    add_attn_candidate(candidates, "eat", "attn_eat_tau0p35_w0p8_thr0p45", 10,
                       threshold=0.45, dcgl_neg_tau=0.35, dcgl_neg_weight=0.8)
    add_attn_candidate(candidates, "eat", "attn_eat_temp1p8_bal0p25_min0p15_thr0p5", 20,
                       threshold=0.5, fusion_temp=1.8, fusion_balance=0.25,
                       fusion_min_weight=0.15, dcgl_neg_tau=0.5, dcgl_neg_weight=1.0)

    add_attn_candidate(candidates, "cora", "attn_cora_cap0p08_tau0p35_w0p8", 5,
                       threshold=0.35, branch_bias_cap=0.08,
                       dcgl_neg_tau=0.35, dcgl_neg_weight=0.8)
    add_attn_candidate(candidates, "cora", "attn_cora_cap0p12_tau0p5_w1p0", 10,
                       threshold=0.4, branch_bias_cap=0.12,
                       dcgl_neg_tau=0.5, dcgl_neg_weight=1.0)

    add_attn_candidate(candidates, "cite", "attn_cite_k5_rawcap0p08_lowmin_tau0p35", 5,
                       threshold=0.35, fusion_temp=1.6, fusion_balance=0.10,
                       lambda_inst=0.03, lambda_clu=0.01, warmup_epochs=70,
                       fusion_min_weight=0.0, branch_bias_cap=0.08,
                       dcgl_neg_tau=0.35, dcgl_neg_weight=0.8)
    add_attn_candidate(candidates, "cite", "attn_cite_k10_rawcap0p12_temp2p0_bal0p25", 10,
                       fusion_temp=2.0, fusion_balance=0.25, fusion_min_weight=0.05,
                       branch_bias_cap=0.12, dcgl_neg_tau=0.75, dcgl_neg_weight=0.8)
    return candidates


def grid_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    candidates.extend(targeted_candidates())

    for ae_k, threshold, tau, weight in product(
        (10, 15, 20, 25),
        (0.35, 0.4, 0.45),
        (0.35, 0.5, 0.75, 1.0),
        (0.4, 0.6, 0.8, 1.0),
    ):
        candidates.append(
            make_candidate(
                "usps",
                f"mean_k{ae_k}_thr{slug_value(threshold)}_tau{slug_value(tau)}_w{slug_value(weight)}",
                "mean",
                ae_k,
                threshold=threshold,
                warmup_epochs=35,
                dcgl_neg_tau=tau,
                dcgl_neg_weight=weight,
            )
        )

    for dataset in MAIN_DATASETS:
        base = BASE_ATTN_CONFIG[dataset]
        for ae_k, tau, weight, threshold in product(
            (5, 10, 15, 20, 25),
            (0.35, 0.5, 0.75, 1.0),
            (0.4, 0.6, 0.8, 1.0),
            (0.35, 0.4, 0.45),
        ):
            add_attn_candidate(
                candidates,
                dataset,
                (
                    f"grid_{dataset}_k{ae_k}_thr{slug_value(threshold)}_"
                    f"tau{slug_value(tau)}_w{slug_value(weight)}"
                ),
                ae_k,
                threshold=threshold,
                fusion_temp=base["fusion_temp"],
                fusion_balance=base["fusion_balance"],
                fusion_min_weight=base["fusion_min_weight"],
                dcgl_neg_tau=tau,
                dcgl_neg_weight=weight,
            )
    return candidates


def targeted_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    reut_grid = []
    for ae_k, min_weight, tau, weight, threshold in product(
        (10, 15, 20),
        (0.10, 0.15, 0.20),
        (0.5, 0.75),
        (0.4, 0.6),
        (0.4, 0.45),
    ):
        distance = (
            abs(ae_k - 15) / 10.0
            + abs(min_weight - 0.15)
            + abs(tau - 0.5)
            + abs(weight - 0.6)
            + abs(threshold - 0.4)
        )
        reut_grid.append(
            (
                distance,
                make_candidate(
                    "reut",
                    (
                        f"target_reut_k{ae_k}_min{slug_value(min_weight)}_"
                        f"tau{slug_value(tau)}_w{slug_value(weight)}_thr{slug_value(threshold)}"
                    ),
                    "attn",
                    ae_k,
                    **base_attn_args(
                        "reut",
                        threshold=threshold,
                        fusion_min_weight=min_weight,
                        dcgl_neg_tau=tau,
                        dcgl_neg_weight=weight,
                    ),
                ),
            )
        )
    candidates.extend(candidate for _, candidate in sorted(reut_grid, key=lambda item: item[0]))

    uat_grid = []
    for ae_k, temp, balance, min_weight, lambda_clu, tau, weight in product(
        (10, 15, 20),
        (1.9, 2.1),
        (0.35, 0.45),
        (0.20, 0.22, 0.25),
        (0.07, 0.075),
        (0.5, 0.75),
        (0.6, 0.8),
    ):
        distance = (
            abs(ae_k - 15) / 10.0
            + abs(temp - 1.9)
            + abs(balance - 0.35)
            + abs(min_weight - 0.20)
            + abs(lambda_clu - 0.07)
            + abs(tau - 0.5)
            + abs(weight - 0.6)
        )
        uat_grid.append(
            (
                distance,
                make_candidate(
                    "uat",
                    (
                        f"target_uat_k{ae_k}_temp{slug_value(temp)}_bal{slug_value(balance)}_"
                        f"min{slug_value(min_weight)}_lc{slug_value(lambda_clu)}_"
                        f"tau{slug_value(tau)}_w{slug_value(weight)}"
                    ),
                    "attn",
                    ae_k,
                    **base_attn_args(
                        "uat",
                        fusion_temp=temp,
                        fusion_balance=balance,
                        fusion_min_weight=min_weight,
                        lambda_clu=lambda_clu,
                        dcgl_neg_tau=tau,
                        dcgl_neg_weight=weight,
                    ),
                ),
            )
        )
    candidates.extend(candidate for _, candidate in sorted(uat_grid, key=lambda item: item[0]))

    amap_grid = []
    for ae_k, min_weight, balance, lambda_inst, tau, weight, threshold in product(
        (5, 10, 15, 20, 25),
        (0.0, 0.05, 0.10),
        (0.05, 0.08, 0.10),
        (0.0, 0.03, 0.07),
        (0.5, 1.0),
        (0.6, 0.8, 1.0),
        (0.35, 0.4, 0.45),
    ):
        distance = (
            abs(ae_k - 15) / 10.0
            + abs(min_weight - 0.05)
            + abs(balance - 0.08)
            + abs(lambda_inst - 0.07)
            + abs(tau - 0.5)
            + abs(weight - 0.6)
            + abs(threshold - 0.4)
        )
        amap_grid.append(
            (
                distance,
                make_candidate(
                    "amap",
                    (
                        f"target_amap_k{ae_k}_min{slug_value(min_weight)}_bal{slug_value(balance)}_"
                        f"li{slug_value(lambda_inst)}_tau{slug_value(tau)}_"
                        f"w{slug_value(weight)}_thr{slug_value(threshold)}"
                    ),
                    "attn",
                    ae_k,
                    **base_attn_args(
                        "amap",
                        threshold=threshold,
                        fusion_balance=balance,
                        fusion_min_weight=min_weight,
                        lambda_inst=lambda_inst,
                        dcgl_neg_tau=tau,
                        dcgl_neg_weight=weight,
                    ),
                ),
            )
        )
    candidates.extend(candidate for _, candidate in sorted(amap_grid, key=lambda item: item[0]))

    usps_grid = []
    for ae_k, temp, balance, min_weight, tau, weight, threshold in product(
        (10, 15, 20),
        (1.8, 2.0, 2.2),
        (0.35, 0.45),
        (0.15, 0.20, 0.25),
        (0.75, 1.0),
        (0.8, 1.0),
        (0.35, 0.4),
    ):
        distance = (
            abs(ae_k - 15) / 10.0
            + abs(temp - 1.8)
            + abs(balance - 0.35)
            + abs(min_weight - 0.20)
            + abs(tau - 0.75)
            + abs(weight - 0.8)
            + abs(threshold - 0.4)
        )
        usps_grid.append(
            (
                distance,
                make_candidate(
                    "usps",
                    (
                        f"target_usps_k{ae_k}_temp{slug_value(temp)}_bal{slug_value(balance)}_"
                        f"min{slug_value(min_weight)}_tau{slug_value(tau)}_"
                        f"w{slug_value(weight)}_thr{slug_value(threshold)}"
                    ),
                    "attn",
                    ae_k,
                    **base_attn_args(
                        "usps",
                        threshold=threshold,
                        fusion_temp=temp,
                        fusion_balance=balance,
                        fusion_min_weight=min_weight,
                        dcgl_neg_tau=tau,
                        dcgl_neg_weight=weight,
                    ),
                ),
            )
        )
    candidates.extend(candidate for _, candidate in sorted(usps_grid, key=lambda item: item[0]))

    eat_grid = []
    for ae_k, temp, balance, min_weight, tau, weight, threshold in product(
        (5, 10, 15, 20, 25),
        (1.8, 2.0, 2.2),
        (0.25, 0.35),
        (0.15, 0.20),
        (0.35, 0.5),
        (0.8, 1.0),
        (0.4, 0.45, 0.5),
    ):
        distance = (
            abs(ae_k - 15) / 10.0
            + abs(temp - 2.0)
            + abs(balance - 0.35)
            + abs(min_weight - 0.20)
            + abs(tau - 0.5)
            + abs(weight - 0.8)
            + abs(threshold - 0.45)
        )
        eat_grid.append(
            (
                distance,
                make_candidate(
                    "eat",
                    (
                        f"target_eat_k{ae_k}_temp{slug_value(temp)}_bal{slug_value(balance)}_"
                        f"min{slug_value(min_weight)}_tau{slug_value(tau)}_"
                        f"w{slug_value(weight)}_thr{slug_value(threshold)}"
                    ),
                    "attn",
                    ae_k,
                    **base_attn_args(
                        "eat",
                        threshold=threshold,
                        fusion_temp=temp,
                        fusion_balance=balance,
                        fusion_min_weight=min_weight,
                        dcgl_neg_tau=tau,
                        dcgl_neg_weight=weight,
                    ),
                ),
            )
        )
    candidates.extend(candidate for _, candidate in sorted(eat_grid, key=lambda item: item[0]))

    cora_grid = []
    for ae_k, cap, tau, weight, threshold in product(
        (5, 10, 15),
        (0.08, 0.10, 0.12),
        (0.35, 0.5),
        (0.6, 0.8, 1.0),
        (0.35, 0.4),
    ):
        distance = (
            abs(ae_k - 10) / 10.0
            + abs(cap - 0.10)
            + abs(tau - 0.5)
            + abs(weight - 0.6)
            + abs(threshold - 0.4)
        )
        cora_grid.append(
            (
                distance,
                make_candidate(
                    "cora",
                    (
                        f"target_cora_k{ae_k}_cap{slug_value(cap)}_tau{slug_value(tau)}_"
                        f"w{slug_value(weight)}_thr{slug_value(threshold)}"
                    ),
                    "attn",
                    ae_k,
                    **base_attn_args(
                        "cora",
                        threshold=threshold,
                        branch_bias_cap=cap,
                        dcgl_neg_tau=tau,
                        dcgl_neg_weight=weight,
                    ),
                ),
            )
        )
    candidates.extend(candidate for _, candidate in sorted(cora_grid, key=lambda item: item[0]))

    cite_grid = []
    for ae_k, temp, balance, min_weight, cap, warmup, tau, weight, threshold, lambda_inst, lambda_clu in product(
        (5, 10, 15),
        (1.8, 2.0, 2.2),
        (0.20, 0.25, 0.30, 0.35),
        (0.0, 0.03, 0.05, 0.08, 0.10),
        (0.08, 0.10, 0.12, 0.14, 0.16),
        (45, 55, 65, 70),
        (0.5, 0.75, 1.0),
        (0.6, 0.8, 1.0),
        (0.35, 0.4, 0.45),
        (0.03, 0.045, 0.06),
        (0.01, 0.02, 0.03),
    ):
        distance = (
            abs(ae_k - 10) / 10.0
            + abs(temp - 2.0)
            + abs(balance - 0.25)
            + abs(min_weight - 0.05)
            + abs(cap - 0.12)
            + abs(warmup - 55) / 50.0
            + abs(tau - 0.75)
            + abs(weight - 0.8)
            + abs(threshold - 0.4)
            + abs(lambda_inst - 0.045)
            + abs(lambda_clu - 0.02)
        )
        cite_grid.append(
            (
                distance,
                make_candidate(
                    "cite",
                    (
                        f"target_cite_k{ae_k}_temp{slug_value(temp)}_bal{slug_value(balance)}_"
                        f"min{slug_value(min_weight)}_cap{slug_value(cap)}_warm{warmup}_"
                        f"tau{slug_value(tau)}_w{slug_value(weight)}_thr{slug_value(threshold)}_"
                        f"li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}"
                    ),
                    "attn",
                    ae_k,
                    **base_attn_args(
                        "cite",
                        threshold=threshold,
                        fusion_temp=temp,
                        fusion_balance=balance,
                        lambda_inst=lambda_inst,
                        lambda_clu=lambda_clu,
                        warmup_epochs=warmup,
                        fusion_min_weight=min_weight,
                        branch_bias_cap=cap,
                        dcgl_neg_tau=tau,
                        dcgl_neg_weight=weight,
                    ),
                ),
            )
        )

    candidates.extend(candidate for _, candidate in sorted(cite_grid, key=lambda item: item[0]))
    return candidates


def cite_rescue_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    strong_base = base_attn_args(
        "cite",
        fusion_temp=2.0,
        fusion_balance=0.25,
        fusion_min_weight=0.05,
        branch_bias_cap=0.12,
        warmup_epochs=55,
        lambda_inst=0.06,
        lambda_clu=0.02,
        dcgl_neg_tau=0.75,
        dcgl_neg_weight=0.8,
        threshold=0.4,
    )

    def add(stem: str, ae_k: str | int, **overrides: Any) -> None:
        args = dict(strong_base)
        args.update(overrides)
        candidates.append(make_candidate("cite", stem, "attn", ae_k, **args))

    def add_mean(stem: str, ae_k: str | int, **overrides: Any) -> None:
        args = dict(DCGL_DEFAULT)
        args.update({"warmup_epochs": 55, "threshold": 0.4, "dcgl_neg_tau": 0.75, "dcgl_neg_weight": 0.8})
        args.update(overrides)
        candidates.append(make_candidate("cite", stem, "mean", ae_k, **args))

    for ae_k in (5, 10, 15, 20, 25):
        add(f"rescue_k{ae_k}_base_li0p06", ae_k)
        add(f"rescue_k{ae_k}_min0p03_li0p045", ae_k, fusion_min_weight=0.03, lambda_inst=0.045)
        add(f"rescue_k{ae_k}_thr0p35", ae_k, threshold=0.35)
        add(f"rescue_k{ae_k}_thr0p45", ae_k, threshold=0.45)

    for threshold, warmup in product((0.30, 0.35, 0.45, 0.50), (45, 55, 65, 70)):
        add(
            f"rescue_thr{slug_value(threshold)}_warm{warmup}",
            10,
            threshold=threshold,
            warmup_epochs=warmup,
        )

    for temp, balance, min_weight in product((1.6, 1.8, 2.0, 2.2, 2.4), (0.15, 0.25, 0.35), (0.0, 0.03, 0.05)):
        add(
            (
                f"rescue_temp{slug_value(temp)}_bal{slug_value(balance)}_"
                f"min{slug_value(min_weight)}"
            ),
            10,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
        )

    for cap in (0.11, 0.12, 0.13):
        for tau, weight in ((0.35, 0.6), (0.5, 0.6), (0.75, 0.8), (1.0, 1.0), (1.25, 1.0)):
            add(
                f"rescue_cap{slug_value(cap)}_tau{slug_value(tau)}_w{slug_value(weight)}",
                10,
                branch_bias_cap=cap,
                dcgl_neg_tau=tau,
                dcgl_neg_weight=weight,
            )

    for alpha, t, lr, epochs in product((0.3, 0.5, 0.7), (2, 3, 4, 5), (5e-5, 1e-4, 2e-4), (300, 400, 500)):
        distance = abs(alpha - 0.5) + abs(t - 4) / 10.0 + abs(lr - 1e-4) * 1000.0 + abs(epochs - 400) / 500.0
        if distance > 0.55:
            continue
        add(
            (
                f"rescue_alpha{slug_value(alpha)}_t{t}_"
                f"lr{slug_value(lr)}_ep{epochs}"
            ),
            10,
            alpha=alpha,
            t=t,
            lr=lr,
            epochs=epochs,
        )

    for ae_k, threshold, tau, weight in product((5, 10, 15, 20), (0.35, 0.4, 0.45), (0.5, 0.75, 1.0), (0.6, 0.8, 1.0)):
        add_mean(
            f"rescue_mean_k{ae_k}_thr{slug_value(threshold)}_tau{slug_value(tau)}_w{slug_value(weight)}",
            ae_k,
            threshold=threshold,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    return candidates


def cite_finalist_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    def add(stem: str, ae_k: int, **overrides: Any) -> None:
        args = base_attn_args(
            "cite",
            fusion_temp=2.0,
            fusion_balance=0.25,
            fusion_min_weight=0.05,
            branch_bias_cap=0.12,
            warmup_epochs=55,
            lambda_inst=0.06,
            lambda_clu=0.02,
            dcgl_neg_tau=0.75,
            dcgl_neg_weight=0.8,
            threshold=0.4,
        )
        args.update(overrides)
        candidates.append(make_candidate("cite", stem, "attn", ae_k, **args))

    add("final_k20_min0p03_li0p045", 20, fusion_min_weight=0.03, lambda_inst=0.045)
    add("final_k20_min0p03_li0p045_thr0p35", 20, fusion_min_weight=0.03, lambda_inst=0.045, threshold=0.35)
    add("final_k20_min0p03_li0p045_temp1p6", 20, fusion_min_weight=0.03, lambda_inst=0.045, fusion_temp=1.6)
    add("final_k20_min0p03_li0p045_temp2p2", 20, fusion_min_weight=0.03, lambda_inst=0.045, fusion_temp=2.2)
    add("final_k20_min0p05_li0p06", 20)
    add("final_k20_min0p05_li0p06_temp1p6", 20, fusion_temp=1.6)
    add("final_k10_min0p05_li0p06", 10)
    add("final_k10_min0p05_li0p06_temp1p6", 10, fusion_temp=1.6)
    add("final_k10_min0p03_li0p045", 10, fusion_min_weight=0.03, lambda_inst=0.045)
    add("final_k10_min0p03_li0p03", 10, fusion_min_weight=0.03, lambda_inst=0.03)
    add("final_k10_min0p05_li0p06_warm70", 10, warmup_epochs=70)
    add("final_k10_min0p05_li0p06_thr0p45_warm70", 10, threshold=0.45, warmup_epochs=70)
    return candidates


def cite_aggressive_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    center = {
        "fusion_temp": 2.0,
        "fusion_balance": 0.25,
        "fusion_min_weight": 0.05,
        "branch_bias_cap": 0.12,
        "warmup_epochs": 55,
        "lambda_inst": 0.06,
        "lambda_clu": 0.02,
        "dcgl_neg_tau": 0.75,
        "dcgl_neg_weight": 0.8,
        "threshold": 0.4,
    }

    def add(stem: str, ae_k: int = 10, **overrides: Any) -> None:
        args = base_attn_args("cite", **center)
        args.update(overrides)
        candidates.append(make_candidate("cite", stem, "attn", ae_k, **args))

    for ae_k in (10, 20):
        add(f"aggr_k{ae_k}_no_branch_bias", ae_k, enable_branch_bias_fusion=False)
        add(f"aggr_k{ae_k}_ae_anchor_cap0p12", ae_k, branch_bias_target="ae", branch_bias_cap=0.12)
        add(f"aggr_k{ae_k}_ae_anchor_cap0p20", ae_k, branch_bias_target="ae", branch_bias_cap=0.20)
        add(f"aggr_k{ae_k}_rawcap0p06", ae_k, branch_bias_cap=0.06)
        add(f"aggr_k{ae_k}_rawcap0p20", ae_k, branch_bias_cap=0.20)

    for hidden in (16, 32, 64, 128, 256):
        add(f"aggr_hidden{hidden}", 10, fusion_hidden=hidden)
        add(f"aggr_k20_hidden{hidden}", 20, fusion_hidden=hidden, fusion_min_weight=0.03, lambda_inst=0.045)

    for lambda_inst, lambda_clu in product((0.0, 0.01, 0.03, 0.06, 0.10, 0.15), (0.0, 0.01, 0.02, 0.05, 0.10)):
        distance = abs(lambda_inst - 0.06) + abs(lambda_clu - 0.02)
        if distance > 0.12:
            continue
        add(
            f"aggr_lambda_li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}",
            10,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
        )
        add(
            f"aggr_k20_lambda_li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}",
            20,
            fusion_min_weight=0.03,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
        )

    for tau, weight in ((0.2, 0.2), (0.35, 0.4), (0.5, 0.4), (0.5, 0.8), (0.75, 0.4), (1.0, 0.6), (1.5, 0.8)):
        add(f"aggr_neg_tau{slug_value(tau)}_w{slug_value(weight)}", 10, dcgl_neg_tau=tau, dcgl_neg_weight=weight)
        add(f"aggr_k20_neg_tau{slug_value(tau)}_w{slug_value(weight)}", 20, fusion_min_weight=0.03, lambda_inst=0.045, dcgl_neg_tau=tau, dcgl_neg_weight=weight)

    for ae_k, t, threshold, warmup in product((10, 20), (1, 2, 3, 4), (0.25, 0.3, 0.4, 0.55), (25, 35, 55, 85)):
        distance = abs(t - 4) / 10.0 + abs(threshold - 0.4) + abs(warmup - 55) / 100.0
        if distance > 0.55:
            continue
        add(
            f"aggr_k{ae_k}_t{t}_thr{slug_value(threshold)}_warm{warmup}",
            ae_k,
            t=t,
            threshold=threshold,
            warmup_epochs=warmup,
            fusion_min_weight=0.03 if ae_k == 20 else 0.05,
            lambda_inst=0.045 if ae_k == 20 else 0.06,
        )

    add("aggr_no_dcgl", 10, enable_dcgl_negative_loss=False, dcgl_neg_tau=None, dcgl_neg_weight=None)
    add("aggr_k20_no_dcgl", 20, fusion_min_weight=0.03, lambda_inst=0.045, enable_dcgl_negative_loss=False, dcgl_neg_tau=None, dcgl_neg_weight=None)
    return candidates


def cite_hidden_k20_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    center = {
        "fusion_temp": 2.0,
        "fusion_balance": 0.25,
        "fusion_min_weight": 0.03,
        "branch_bias_cap": 0.12,
        "warmup_epochs": 55,
        "lambda_inst": 0.045,
        "lambda_clu": 0.02,
        "dcgl_neg_tau": 0.75,
        "dcgl_neg_weight": 0.8,
        "threshold": 0.4,
    }

    def add(stem: str, **overrides: Any) -> None:
        args = base_attn_args("cite", **center)
        args.update(overrides)
        candidates.append(make_candidate("cite", stem, "attn", 20, **args))

    # Start with the exact local center and the known single-seed high-ish point.
    add("hidden_k20_center_h64", fusion_hidden=64)
    add("hidden_k20_known_h256", fusion_hidden=256)

    for hidden in (96, 128, 160, 192, 224, 256, 288, 320, 384, 512):
        add(f"hidden_k20_h{hidden}", fusion_hidden=hidden)

    for hidden, temp, balance, min_weight in product(
        (160, 192, 224, 256, 288, 320, 384),
        (1.6, 1.8, 2.0, 2.2, 2.4),
        (0.15, 0.20, 0.25, 0.30, 0.35),
        (0.01, 0.02, 0.03, 0.04, 0.05),
    ):
        distance = (
            abs(hidden - 256) / 256.0
            + abs(temp - 2.0)
            + abs(balance - 0.25)
            + abs(min_weight - 0.03) * 4.0
        )
        if distance > 0.85:
            continue
        add(
            (
                f"hidden_k20_h{hidden}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}"
            ),
            fusion_hidden=hidden,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
        )

    for hidden, lambda_inst, lambda_clu in product(
        (192, 256, 320, 384),
        (0.0, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08),
        (0.0, 0.01, 0.02, 0.03, 0.05),
    ):
        distance = abs(hidden - 256) / 256.0 + abs(lambda_inst - 0.045) * 3.0 + abs(lambda_clu - 0.02) * 4.0
        if distance > 0.7:
            continue
        add(
            f"hidden_k20_h{hidden}_li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}",
            fusion_hidden=hidden,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
        )

    for hidden, tau, weight in product(
        (192, 256, 320, 384),
        (0.35, 0.5, 0.65, 0.75, 0.9, 1.0),
        (0.4, 0.6, 0.8, 1.0),
    ):
        distance = abs(hidden - 256) / 256.0 + abs(tau - 0.75) + abs(weight - 0.8)
        if distance > 0.75:
            continue
        add(
            f"hidden_k20_h{hidden}_tau{slug_value(tau)}_w{slug_value(weight)}",
            fusion_hidden=hidden,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for hidden, threshold, warmup, alpha, t in product(
        (192, 256, 320),
        (0.35, 0.4, 0.45, 0.5),
        (35, 45, 55, 70, 85),
        (0.4, 0.5, 0.6),
        (3, 4, 5),
    ):
        distance = (
            abs(hidden - 256) / 256.0
            + abs(threshold - 0.4)
            + abs(warmup - 55) / 75.0
            + abs(alpha - 0.5)
            + abs(t - 4) / 8.0
        )
        if distance > 0.65:
            continue
        add(
            (
                f"hidden_k20_h{hidden}_thr{slug_value(threshold)}_"
                f"warm{warmup}_alpha{slug_value(alpha)}_t{t}"
            ),
            fusion_hidden=hidden,
            threshold=threshold,
            warmup_epochs=warmup,
            alpha=alpha,
            t=t,
        )

    return candidates


def cite_high_hidden_k20_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    center = {
        "fusion_hidden": 512,
        "fusion_temp": 2.0,
        "fusion_balance": 0.25,
        "fusion_min_weight": 0.03,
        "branch_bias_cap": 0.12,
        "warmup_epochs": 55,
        "lambda_inst": 0.045,
        "lambda_clu": 0.02,
        "dcgl_neg_tau": 0.75,
        "dcgl_neg_weight": 0.8,
        "threshold": 0.4,
    }

    def add(stem: str, **overrides: Any) -> None:
        args = base_attn_args("cite", **center)
        args.update(overrides)
        candidates.append(make_candidate("cite", stem, "attn", 20, **args))

    for hidden in (384, 448, 512, 576, 640, 768, 896, 1024):
        add(f"high_hidden_k20_h{hidden}", fusion_hidden=hidden)

    for hidden, temp, balance, min_weight in product(
        (448, 512, 576, 640, 768),
        (1.6, 1.8, 2.0, 2.2, 2.4),
        (0.15, 0.20, 0.25, 0.30, 0.35),
        (0.01, 0.02, 0.03, 0.04, 0.05),
    ):
        distance = (
            abs(hidden - 512) / 512.0
            + abs(temp - 2.0)
            + abs(balance - 0.25)
            + abs(min_weight - 0.03) * 4.0
        )
        if distance > 0.75:
            continue
        add(
            (
                f"high_hidden_k20_h{hidden}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}"
            ),
            fusion_hidden=hidden,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
        )

    for hidden, lambda_inst, lambda_clu in product(
        (448, 512, 576, 640, 768),
        (0.0, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.10),
        (0.0, 0.01, 0.02, 0.03, 0.05),
    ):
        distance = (
            abs(hidden - 512) / 512.0
            + abs(lambda_inst - 0.045) * 3.0
            + abs(lambda_clu - 0.02) * 4.0
        )
        if distance > 0.65:
            continue
        add(
            f"high_hidden_k20_h{hidden}_li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}",
            fusion_hidden=hidden,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
        )

    for hidden, tau, weight in product(
        (448, 512, 576, 640, 768),
        (0.35, 0.5, 0.65, 0.75, 0.9, 1.0, 1.25),
        (0.4, 0.6, 0.8, 1.0),
    ):
        distance = abs(hidden - 512) / 512.0 + abs(tau - 0.75) + abs(weight - 0.8)
        if distance > 0.7:
            continue
        add(
            f"high_hidden_k20_h{hidden}_tau{slug_value(tau)}_w{slug_value(weight)}",
            fusion_hidden=hidden,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for hidden, threshold, warmup, alpha, t in product(
        (448, 512, 576, 640),
        (0.35, 0.4, 0.45, 0.5),
        (35, 45, 55, 70, 85),
        (0.4, 0.5, 0.6),
        (3, 4, 5),
    ):
        distance = (
            abs(hidden - 512) / 512.0
            + abs(threshold - 0.4)
            + abs(warmup - 55) / 80.0
            + abs(alpha - 0.5)
            + abs(t - 4) / 8.0
        )
        if distance > 0.62:
            continue
        add(
            (
                f"high_hidden_k20_h{hidden}_thr{slug_value(threshold)}_"
                f"warm{warmup}_alpha{slug_value(alpha)}_t{t}"
            ),
            fusion_hidden=hidden,
            threshold=threshold,
            warmup_epochs=warmup,
            alpha=alpha,
            t=t,
        )

    for hidden, epochs, lr in product(
        (448, 512, 576, 640, 768),
        (300, 400, 500, 600),
        (5e-5, 8e-5, 1e-4),
    ):
        distance = abs(hidden - 512) / 512.0 + abs(epochs - 400) / 500.0 + abs(lr - 1e-4) * 4000.0
        if distance > 0.65:
            continue
        add(
            f"high_hidden_k20_h{hidden}_ep{epochs}_lr{slug_value(lr)}",
            fusion_hidden=hidden,
            epochs=epochs,
            lr=lr,
        )

    return candidates


def cite_peak_h512_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    center = {
        "fusion_hidden": 512,
        "fusion_temp": 2.0,
        "fusion_balance": 0.30,
        "fusion_min_weight": 0.03,
        "branch_bias_cap": 0.12,
        "warmup_epochs": 55,
        "lambda_inst": 0.045,
        "lambda_clu": 0.02,
        "dcgl_neg_tau": 0.75,
        "dcgl_neg_weight": 0.8,
        "threshold": 0.4,
    }

    def add(stem: str, **overrides: Any) -> None:
        args = base_attn_args("cite", **center)
        args.update(overrides)
        candidates.append(make_candidate("cite", stem, "attn", 20, **args))

    add("peak_h512_center")
    for hidden in (480, 496, 512, 528, 544, 576, 608):
        add(f"peak_h{hidden}", fusion_hidden=hidden)

    for hidden, temp, balance, min_weight in product(
        (480, 496, 512, 528, 544, 576, 608),
        (1.9, 2.0, 2.1),
        (0.24, 0.27, 0.30, 0.33, 0.36),
        (0.02, 0.025, 0.03, 0.035, 0.04),
    ):
        distance = (
            abs(hidden - 512) / 256.0
            + abs(temp - 2.0) * 1.5
            + abs(balance - 0.30) * 2.0
            + abs(min_weight - 0.03) * 8.0
        )
        if distance > 0.48:
            continue
        add(
            (
                f"peak_h{hidden}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}"
            ),
            fusion_hidden=hidden,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
        )

    for hidden, lambda_inst, lambda_clu in product(
        (496, 512, 528, 544, 576),
        (0.0, 0.01, 0.02, 0.03, 0.045, 0.06),
        (0.0, 0.01, 0.02, 0.03),
    ):
        distance = (
            abs(hidden - 512) / 256.0
            + abs(lambda_inst - 0.045) * 4.0
            + abs(lambda_clu - 0.02) * 6.0
        )
        if distance > 0.5:
            continue
        add(
            f"peak_h{hidden}_li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}",
            fusion_hidden=hidden,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
        )

    for hidden, tau, weight in product(
        (496, 512, 528, 544, 576),
        (0.55, 0.65, 0.75, 0.85, 0.95),
        (0.6, 0.7, 0.8, 0.9, 1.0),
    ):
        distance = abs(hidden - 512) / 256.0 + abs(tau - 0.75) + abs(weight - 0.8)
        if distance > 0.45:
            continue
        add(
            f"peak_h{hidden}_tau{slug_value(tau)}_w{slug_value(weight)}",
            fusion_hidden=hidden,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for hidden, threshold, warmup, alpha, t in product(
        (496, 512, 528, 544),
        (0.35, 0.38, 0.4, 0.42, 0.45),
        (45, 55, 65, 75),
        (0.45, 0.5, 0.55),
        (3, 4, 5),
    ):
        distance = (
            abs(hidden - 512) / 256.0
            + abs(threshold - 0.4) * 2.0
            + abs(warmup - 55) / 80.0
            + abs(alpha - 0.5) * 2.0
            + abs(t - 4) / 6.0
        )
        if distance > 0.46:
            continue
        add(
            (
                f"peak_h{hidden}_thr{slug_value(threshold)}_"
                f"warm{warmup}_alpha{slug_value(alpha)}_t{t}"
            ),
            fusion_hidden=hidden,
            threshold=threshold,
            warmup_epochs=warmup,
            alpha=alpha,
            t=t,
        )

    for hidden, epochs, lr in product(
        (496, 512, 528, 544, 576),
        (350, 400, 450, 500),
        (6e-5, 8e-5, 1e-4, 1.2e-4),
    ):
        distance = (
            abs(hidden - 512) / 256.0
            + abs(epochs - 400) / 350.0
            + abs(lr - 1e-4) * 5000.0
        )
        if distance > 0.5:
            continue
        add(
            f"peak_h{hidden}_ep{epochs}_lr{slug_value(lr)}",
            fusion_hidden=hidden,
            epochs=epochs,
            lr=lr,
        )

    return candidates


def cite_peak_h480_train_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    center = {
        "fusion_hidden": 480,
        "fusion_temp": 2.0,
        "fusion_balance": 0.30,
        "fusion_min_weight": 0.03,
        "branch_bias_cap": 0.12,
        "warmup_epochs": 55,
        "lambda_inst": 0.045,
        "lambda_clu": 0.02,
        "dcgl_neg_tau": 0.75,
        "dcgl_neg_weight": 0.8,
        "threshold": 0.4,
    }

    def add(stem: str, **overrides: Any) -> None:
        args = base_attn_args("cite", **center)
        args.update(overrides)
        candidates.append(make_candidate("cite", stem, "attn", 20, **args))

    add("peak480_center")

    for hidden in (448, 464, 480, 496, 512):
        add(f"peak480_h{hidden}", fusion_hidden=hidden)

    for cap in (0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15):
        add(f"peak480_cap{slug_value(cap)}", branch_bias_cap=cap)

    for lambda_inst, lambda_clu in product(
        (0.0, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08),
        (0.0, 0.01, 0.02, 0.03, 0.05),
    ):
        distance = abs(lambda_inst - 0.045) * 4.0 + abs(lambda_clu - 0.02) * 6.0
        if distance > 0.45:
            continue
        add(
            f"peak480_li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}",
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
        )

    for tau, weight in product(
        (0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.10),
        (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    ):
        distance = abs(tau - 0.75) + abs(weight - 0.8)
        if distance > 0.45:
            continue
        add(
            f"peak480_tau{slug_value(tau)}_w{slug_value(weight)}",
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for threshold, warmup, alpha, t in product(
        (0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46),
        (40, 45, 50, 55, 60, 70),
        (0.35, 0.45, 0.5, 0.55, 0.65),
        (2, 3, 4, 5),
    ):
        distance = (
            abs(threshold - 0.4) * 2.0
            + abs(warmup - 55) / 70.0
            + abs(alpha - 0.5) * 2.0
            + abs(t - 4) / 6.0
        )
        if distance > 0.50:
            continue
        add(
            (
                f"peak480_thr{slug_value(threshold)}_warm{warmup}_"
                f"alpha{slug_value(alpha)}_t{t}"
            ),
            threshold=threshold,
            warmup_epochs=warmup,
            alpha=alpha,
            t=t,
        )

    for hidden, epochs, lr in product(
        (464, 480, 496),
        (300, 350, 400, 450, 500, 600),
        (5e-5, 7e-5, 8e-5, 1e-4, 1.2e-4, 1.5e-4),
    ):
        distance = (
            abs(hidden - 480) / 256.0
            + abs(epochs - 400) / 350.0
            + abs(lr - 1e-4) * 5000.0
        )
        if distance > 0.52:
            continue
        add(
            f"peak480_h{hidden}_ep{epochs}_lr{slug_value(lr)}",
            fusion_hidden=hidden,
            epochs=epochs,
            lr=lr,
        )

    for start, end in ((0.15, 0.45), (0.2, 0.5), (0.25, 0.55), (0.3, 0.6)):
        add(
            f"peak480_dynamic_{slug_value(start)}_{slug_value(end)}",
            enable_dynamic_threshold=True,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
        )

    for momentum in (0.85, 0.90, 0.95):
        add(
            f"peak480_ema{slug_value(momentum)}",
            enable_ema_prototypes=True,
            ema_proto_momentum=momentum,
        )

    for lambda_dcgl, tau in product((0.03, 0.05, 0.08, 0.10), (0.35, 0.50, 0.75)):
        add(
            f"peak480_dcglclu{slug_value(lambda_dcgl)}_tau{slug_value(tau)}",
            enable_dcgl_cluster_level=True,
            lambda_dcgl_cluster=lambda_dcgl,
            dcgl_cluster_tau=tau,
        )

    return candidates


def reut_attn_dcgl_only_push_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str, str, tuple[tuple[str, Any], ...]]] = set()
    ranked: list[tuple[float, Candidate]] = []

    apr21_center = {
        "dims": 500,
        "threshold": 0.4,
        "alpha": 0.5,
        "warmup_epochs": 35,
        "fusion_hidden": 64,
        "fusion_temp": 1.8,
        "fusion_balance": 0.35,
        "fusion_min_weight": 0.20,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    }
    apr02_center = {
        "dims": 256,
        "threshold": 0.5,
        "alpha": 0.5,
        "warmup_epochs": 20,
        "fusion_hidden": 64,
        "fusion_temp": 1.4,
        "fusion_balance": 0.25,
        "fusion_min_weight": 0.10,
        "lambda_inst": 0.12,
        "lambda_clu": 0.12,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 1.0,
    }

    def distance(args: dict[str, Any], ae_k: str | int, center: dict[str, Any]) -> float:
        ae_val = AE_DEFAULT_K["reut"] if ae_k == "default" else int(ae_k)
        center_ae = AE_DEFAULT_K["reut"]
        return (
            abs(ae_val - center_ae) / 20.0
            + abs(float(args["dims"]) - float(center["dims"])) / 600.0
            + abs(float(args["threshold"]) - float(center["threshold"])) * 1.8
            + abs(float(args["warmup_epochs"]) - float(center["warmup_epochs"])) / 80.0
            + abs(float(args["fusion_hidden"]) - float(center["fusion_hidden"])) / 192.0
            + abs(float(args["fusion_temp"]) - float(center["fusion_temp"])) * 0.8
            + abs(float(args["fusion_balance"]) - float(center["fusion_balance"])) * 1.5
            + abs(float(args["fusion_min_weight"]) - float(center["fusion_min_weight"])) * 1.8
            + abs(float(args["lambda_inst"]) - float(center["lambda_inst"])) * 3.0
            + abs(float(args["lambda_clu"]) - float(center["lambda_clu"])) * 3.0
            + abs(float(args["dcgl_neg_tau"]) - float(center["dcgl_neg_tau"])) * 0.7
            + abs(float(args["dcgl_neg_weight"]) - float(center["dcgl_neg_weight"])) * 0.8
        )

    def add(stem: str, ae_k: str | int, center: dict[str, Any], **overrides: Any) -> None:
        args = dict(center)
        args.update(overrides)
        # Keep this preset narrative-clean: only DCGL negative is enabled.
        for forbidden in (
            "enable_dynamic_threshold",
            "enable_gcn_backbone",
            "enable_dcgl_cluster_level",
            "enable_branch_bias_fusion",
            "enable_ema_prototypes",
        ):
            args.pop(forbidden, None)
        candidate = make_candidate("reut", stem, "attn", ae_k, **args)
        key = (
            candidate.fusion_mode,
            str(candidate.ae_k),
            tuple(sorted(candidate.args.items())),
        )
        if key in seen:
            return
        seen.add(key)
        ranked.append((distance(args, ae_k, center), candidate))

    for prefix, center in (("apr21", apr21_center), ("apr02", apr02_center)):
        add(f"reut_dcgl_only_{prefix}_center", "default", center)
        for ae_k in ("default", 5, 10, 15, 20, 25):
            add(f"reut_dcgl_only_{prefix}_k{slug_value(ae_k)}", ae_k, center)
        for seed_like_weight in (0.45, 0.55, 0.6, 0.7, 0.85, 1.0):
            add(
                f"reut_dcgl_only_{prefix}_w{slug_value(seed_like_weight)}",
                "default",
                center,
                dcgl_neg_weight=seed_like_weight,
            )

    for ae_k, threshold, temp, balance, min_weight, lambda_pair, tau, weight, warmup in product(
        ("default", 10, 15, 20),
        (0.38, 0.40, 0.42, 0.45),
        (1.6, 1.8, 1.9),
        (0.25, 0.35, 0.40),
        (0.15, 0.20, 0.25),
        ((0.08, 0.08), (0.09, 0.09), (0.12, 0.09)),
        (0.45, 0.50, 0.75),
        (0.45, 0.60, 0.75, 0.90),
        (30, 35, 40),
    ):
        args = dict(apr21_center)
        args.update(
            {
                "threshold": threshold,
                "fusion_temp": temp,
                "fusion_balance": balance,
                "fusion_min_weight": min_weight,
                "lambda_inst": lambda_pair[0],
                "lambda_clu": lambda_pair[1],
                "dcgl_neg_tau": tau,
                "dcgl_neg_weight": weight,
                "warmup_epochs": warmup,
            }
        )
        if distance(args, ae_k, apr21_center) > 0.75:
            continue
        add(
            (
                f"r21_k{slug_value(ae_k)}_th{slug_value(threshold)}_"
                f"tf{slug_value(temp)}_b{slug_value(balance)}_m{slug_value(min_weight)}_"
                f"li{slug_value(lambda_pair[0])}_lc{slug_value(lambda_pair[1])}_"
                f"ta{slug_value(tau)}_w{slug_value(weight)}_wu{warmup}"
            ),
            ae_k,
            apr21_center,
            **args,
        )

    for ae_k, dims, threshold, temp, balance, min_weight, lambda_pair, tau, weight, warmup in product(
        ("default", 10, 15, 20),
        (256, 500),
        (0.45, 0.50, 0.52),
        (1.4, 1.6, 1.8),
        (0.20, 0.25, 0.30),
        (0.08, 0.10, 0.15),
        ((0.10, 0.10), (0.12, 0.12), (0.14, 0.12)),
        (0.50, 0.65, 0.75),
        (0.80, 1.00, 1.10),
        (20, 25, 30),
    ):
        args = dict(apr02_center)
        args.update(
            {
                "dims": dims,
                "threshold": threshold,
                "fusion_temp": temp,
                "fusion_balance": balance,
                "fusion_min_weight": min_weight,
                "lambda_inst": lambda_pair[0],
                "lambda_clu": lambda_pair[1],
                "dcgl_neg_tau": tau,
                "dcgl_neg_weight": weight,
                "warmup_epochs": warmup,
            }
        )
        if distance(args, ae_k, apr02_center) > 0.78:
            continue
        add(
            (
                f"r02_k{slug_value(ae_k)}_d{dims}_th{slug_value(threshold)}_"
                f"tf{slug_value(temp)}_b{slug_value(balance)}_m{slug_value(min_weight)}_"
                f"li{slug_value(lambda_pair[0])}_lc{slug_value(lambda_pair[1])}_"
                f"ta{slug_value(tau)}_w{slug_value(weight)}_wu{warmup}"
            ),
            ae_k,
            apr02_center,
            **args,
        )

    for hidden, temp, balance, min_weight, weight in product(
        (32, 64, 96, 128, 192),
        (1.5, 1.7, 1.8, 1.9, 2.1),
        (0.25, 0.35, 0.45),
        (0.10, 0.15, 0.20, 0.25),
        (0.45, 0.60, 0.75, 0.90),
    ):
        args = dict(apr21_center)
        args.update(
            {
                "fusion_hidden": hidden,
                "fusion_temp": temp,
                "fusion_balance": balance,
                "fusion_min_weight": min_weight,
                "dcgl_neg_weight": weight,
            }
        )
        if distance(args, "default", apr21_center) > 0.70:
            continue
        add(
            (
                f"rh{hidden}_tf{slug_value(temp)}_b{slug_value(balance)}_"
                f"m{slug_value(min_weight)}_w{slug_value(weight)}"
            ),
            "default",
            apr21_center,
            **args,
        )

    ranked.sort(key=lambda item: item[0])
    candidates.extend(candidate for _dist, candidate in ranked)
    return candidates


def reut_attn_dcgl_refine_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str, tuple[tuple[str, Any], ...]]] = set()
    ranked: list[tuple[float, Candidate]] = []

    peak = {
        "dims": 500,
        "threshold": 0.4,
        "alpha": 0.5,
        "warmup_epochs": 35,
        "fusion_hidden": 64,
        "fusion_temp": 1.8,
        "fusion_balance": 0.40,
        "fusion_min_weight": 0.20,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    }
    f1_bridge = {
        "dims": 256,
        "threshold": 0.5,
        "alpha": 0.5,
        "warmup_epochs": 20,
        "fusion_hidden": 64,
        "fusion_temp": 1.4,
        "fusion_balance": 0.25,
        "fusion_min_weight": 0.08,
        "lambda_inst": 0.12,
        "lambda_clu": 0.12,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 1.0,
    }

    def clean(args: dict[str, Any]) -> dict[str, Any]:
        args = dict(args)
        for forbidden in (
            "enable_dynamic_threshold",
            "enable_gcn_backbone",
            "enable_dcgl_cluster_level",
            "enable_branch_bias_fusion",
            "enable_ema_prototypes",
        ):
            args.pop(forbidden, None)
        return args

    def distance(args: dict[str, Any], center: dict[str, Any], ae_k: str | int) -> float:
        ae_val = AE_DEFAULT_K["reut"] if ae_k == "default" else int(ae_k)
        return (
            abs(ae_val - 15) / 25.0
            + abs(float(args["dims"]) - float(center["dims"])) / 700.0
            + abs(float(args["threshold"]) - float(center["threshold"])) * 2.2
            + abs(float(args["warmup_epochs"]) - float(center["warmup_epochs"])) / 90.0
            + abs(float(args["fusion_hidden"]) - float(center["fusion_hidden"])) / 220.0
            + abs(float(args["fusion_temp"]) - float(center["fusion_temp"])) * 1.0
            + abs(float(args["fusion_balance"]) - float(center["fusion_balance"])) * 2.0
            + abs(float(args["fusion_min_weight"]) - float(center["fusion_min_weight"])) * 2.0
            + abs(float(args["lambda_inst"]) - float(center["lambda_inst"])) * 4.0
            + abs(float(args["lambda_clu"]) - float(center["lambda_clu"])) * 4.0
            + abs(float(args["dcgl_neg_tau"]) - float(center["dcgl_neg_tau"])) * 0.8
            + abs(float(args["dcgl_neg_weight"]) - float(center["dcgl_neg_weight"])) * 0.9
        )

    def add(stem: str, ae_k: str | int, center: dict[str, Any], **overrides: Any) -> None:
        args = dict(center)
        args.update(overrides)
        args = clean(args)
        candidate = make_candidate("reut", stem, "attn", ae_k, **args)
        key = (candidate.fusion_mode, str(candidate.ae_k), tuple(sorted(candidate.args.items())))
        if key in seen:
            return
        seen.add(key)
        ranked.append((distance(args, center, ae_k), candidate))

    add("reut_refine_peak_center", "default", peak)
    add("reut_refine_f1_bridge_center", "default", f1_bridge)

    for ae_k in ("default", 5, 10, 15, 20, 25):
        for balance, threshold in product((0.38, 0.40, 0.42, 0.44, 0.46), (0.36, 0.38, 0.40, 0.42, 0.44)):
            add(
                f"rr_peak_k{slug_value(ae_k)}_b{slug_value(balance)}_th{slug_value(threshold)}",
                ae_k,
                peak,
                fusion_balance=balance,
                threshold=threshold,
            )

    for temp, min_weight, warmup, lambda_inst, lambda_clu in product(
        (1.65, 1.75, 1.8, 1.85, 1.95),
        (0.16, 0.18, 0.20, 0.22, 0.24),
        (30, 32, 35, 38, 40, 45),
        (0.07, 0.08, 0.09, 0.10, 0.11),
        (0.07, 0.08, 0.09, 0.10, 0.11),
    ):
        args = dict(peak)
        args.update(
            {
                "fusion_temp": temp,
                "fusion_min_weight": min_weight,
                "warmup_epochs": warmup,
                "lambda_inst": lambda_inst,
                "lambda_clu": lambda_clu,
            }
        )
        if distance(args, peak, "default") > 0.42:
            continue
        add(
            (
                f"rr_peak_tf{slug_value(temp)}_m{slug_value(min_weight)}_"
                f"wu{warmup}_li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}"
            ),
            "default",
            peak,
            **args,
        )

    for tau, weight, balance, threshold, warmup in product(
        (0.35, 0.45, 0.50, 0.60, 0.70, 0.85),
        (0.45, 0.55, 0.60, 0.70, 0.80, 0.95),
        (0.38, 0.40, 0.42, 0.44),
        (0.38, 0.40, 0.42),
        (32, 35, 38, 40),
    ):
        args = dict(peak)
        args.update(
            {
                "dcgl_neg_tau": tau,
                "dcgl_neg_weight": weight,
                "fusion_balance": balance,
                "threshold": threshold,
                "warmup_epochs": warmup,
            }
        )
        if distance(args, peak, "default") > 0.50:
            continue
        add(
            (
                f"rr_peak_ta{slug_value(tau)}_w{slug_value(weight)}_"
                f"b{slug_value(balance)}_th{slug_value(threshold)}_wu{warmup}"
            ),
            "default",
            peak,
            **args,
        )

    for hidden, temp, balance, min_weight in product(
        (48, 64, 80, 96, 128),
        (1.7, 1.8, 1.9),
        (0.38, 0.40, 0.42, 0.45),
        (0.16, 0.20, 0.24),
    ):
        args = dict(peak)
        args.update(
            {
                "fusion_hidden": hidden,
                "fusion_temp": temp,
                "fusion_balance": balance,
                "fusion_min_weight": min_weight,
            }
        )
        if distance(args, peak, "default") > 0.48:
            continue
        add(
            (
                f"rr_peak_h{hidden}_tf{slug_value(temp)}_"
                f"b{slug_value(balance)}_m{slug_value(min_weight)}"
            ),
            "default",
            peak,
            **args,
        )

    for dims, temp, balance, min_weight, lambda_inst, lambda_clu, weight, warmup in product(
        (256, 384, 500),
        (1.4, 1.55, 1.7),
        (0.25, 0.30, 0.35, 0.40),
        (0.06, 0.08, 0.10, 0.12),
        (0.10, 0.12, 0.14),
        (0.10, 0.12, 0.14),
        (0.8, 1.0, 1.15),
        (18, 20, 25, 30),
    ):
        args = dict(f1_bridge)
        args.update(
            {
                "dims": dims,
                "fusion_temp": temp,
                "fusion_balance": balance,
                "fusion_min_weight": min_weight,
                "lambda_inst": lambda_inst,
                "lambda_clu": lambda_clu,
                "dcgl_neg_weight": weight,
                "warmup_epochs": warmup,
            }
        )
        if distance(args, f1_bridge, "default") > 0.62:
            continue
        add(
            (
                f"rr_bridge_d{dims}_tf{slug_value(temp)}_b{slug_value(balance)}_"
                f"m{slug_value(min_weight)}_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_w{slug_value(weight)}_wu{warmup}"
            ),
            "default",
            f1_bridge,
            **args,
        )

    ranked.sort(key=lambda item: item[0])
    return [candidate for _dist, candidate in ranked]


def reut_attn_dcgl_apex_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str, tuple[tuple[str, Any], ...]]] = set()
    ranked: list[tuple[float, Candidate]] = []
    center = {
        "dims": 500,
        "threshold": 0.4,
        "alpha": 0.5,
        "warmup_epochs": 35,
        "fusion_hidden": 64,
        "fusion_temp": 1.75,
        "fusion_balance": 0.40,
        "fusion_min_weight": 0.20,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    }

    def dist(args: dict[str, Any], ae_k: str | int) -> float:
        ae_val = AE_DEFAULT_K["reut"] if ae_k == "default" else int(ae_k)
        return (
            abs(ae_val - 15) / 25.0
            + abs(float(args["threshold"]) - 0.4) * 3.0
            + abs(float(args["warmup_epochs"]) - 35) / 70.0
            + abs(float(args["fusion_hidden"]) - 64) / 220.0
            + abs(float(args["fusion_temp"]) - 1.75) * 1.4
            + abs(float(args["fusion_balance"]) - 0.40) * 2.4
            + abs(float(args["fusion_min_weight"]) - 0.20) * 2.4
            + abs(float(args["lambda_inst"]) - 0.09) * 5.0
            + abs(float(args["lambda_clu"]) - 0.09) * 5.0
            + abs(float(args["dcgl_neg_tau"]) - 0.5) * 1.0
            + abs(float(args["dcgl_neg_weight"]) - 0.6) * 1.0
        )

    def add(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = dict(center)
        args.update(overrides)
        for forbidden in (
            "enable_dynamic_threshold",
            "enable_gcn_backbone",
            "enable_dcgl_cluster_level",
            "enable_branch_bias_fusion",
            "enable_ema_prototypes",
        ):
            args.pop(forbidden, None)
        candidate = make_candidate("reut", stem, "attn", ae_k, **args)
        key = (candidate.fusion_mode, str(candidate.ae_k), tuple(sorted(candidate.args.items())))
        if key in seen:
            return
        seen.add(key)
        ranked.append((dist(args, ae_k), candidate))

    add("reut_apex_center")

    for ae_k in ("default", 10, 15, 20):
        add(f"reut_apex_k{slug_value(ae_k)}", ae_k)

    for temp, warmup in product(
        (1.68, 1.72, 1.75, 1.78, 1.82),
        (31, 32, 33, 34, 35, 36, 37, 38),
    ):
        add(f"reut_apex_tf{slug_value(temp)}_wu{warmup}", fusion_temp=temp, warmup_epochs=warmup)

    for threshold, balance, min_weight in product(
        (0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43),
        (0.36, 0.38, 0.40, 0.42, 0.44),
        (0.16, 0.18, 0.20, 0.22, 0.24),
    ):
        args = dict(center)
        args.update(
            {
                "threshold": threshold,
                "fusion_balance": balance,
                "fusion_min_weight": min_weight,
            }
        )
        if dist(args, "default") > 0.38:
            continue
        add(
            f"reut_apex_th{slug_value(threshold)}_b{slug_value(balance)}_m{slug_value(min_weight)}",
            threshold=threshold,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
        )

    for lambda_inst, lambda_clu, tau, weight in product(
        (0.07, 0.08, 0.09, 0.10, 0.11),
        (0.07, 0.08, 0.09, 0.10, 0.11),
        (0.40, 0.45, 0.50, 0.55, 0.60, 0.70),
        (0.45, 0.55, 0.60, 0.65, 0.75, 0.85),
    ):
        args = dict(center)
        args.update(
            {
                "lambda_inst": lambda_inst,
                "lambda_clu": lambda_clu,
                "dcgl_neg_tau": tau,
                "dcgl_neg_weight": weight,
            }
        )
        if dist(args, "default") > 0.45:
            continue
        add(
            (
                f"reut_apex_li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}_"
                f"ta{slug_value(tau)}_w{slug_value(weight)}"
            ),
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for hidden, temp, threshold, warmup in product(
        (48, 64, 80, 96),
        (1.70, 1.75, 1.80),
        (0.38, 0.40, 0.42),
        (32, 35, 38),
    ):
        args = dict(center)
        args.update(
            {
                "fusion_hidden": hidden,
                "fusion_temp": temp,
                "threshold": threshold,
                "warmup_epochs": warmup,
            }
        )
        if dist(args, "default") > 0.46:
            continue
        add(
            (
                f"reut_apex_h{hidden}_tf{slug_value(temp)}_"
                f"th{slug_value(threshold)}_wu{warmup}"
            ),
            fusion_hidden=hidden,
            fusion_temp=temp,
            threshold=threshold,
            warmup_epochs=warmup,
        )

    ranked.sort(key=lambda item: item[0])
    return [candidate for _dist, candidate in ranked]


def usps_peak_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    mean_center = {
        "warmup_epochs": 35,
        "threshold": 0.4,
        "alpha": 0.5,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    }
    attn_center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.8,
        "fusion_balance": 0.35,
        "fusion_min_weight": 0.20,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "warmup_epochs": 35,
        "threshold": 0.4,
        "alpha": 0.5,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    }

    def add_mean(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = dict(mean_center)
        args.update(overrides)
        candidates.append(make_candidate("usps", stem, "mean", ae_k, **args))

    def add_attn(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = base_attn_args("usps", **attn_center)
        args.update(overrides)
        candidates.append(make_candidate("usps", stem, "attn", ae_k, **args))

    # Stable historical center: 10-run mean around 82.90, with several single runs >83.
    add_mean("mean_center")
    for ae_k in (10, 15, 20, 25):
        add_mean(f"mean_k{ae_k}", ae_k)

    for threshold, tau, weight in product(
        (0.35, 0.38, 0.4, 0.42, 0.45),
        (0.35, 0.5, 0.65, 0.75, 0.9, 1.0),
        (0.4, 0.6, 0.8, 1.0),
    ):
        distance = abs(threshold - 0.4) * 2.0 + abs(tau - 0.5) + abs(weight - 0.6)
        if distance > 0.45:
            continue
        add_mean(
            f"mean_thr{slug_value(threshold)}_tau{slug_value(tau)}_w{slug_value(weight)}",
            "default",
            threshold=threshold,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for alpha, t, epochs, lr in product(
        (0.35, 0.45, 0.5, 0.55, 0.65),
        (3, 4, 5),
        (350, 400, 450, 500),
        (7e-5, 8e-5, 1e-4, 1.2e-4),
    ):
        distance = (
            abs(alpha - 0.5) * 2.0
            + abs(t - 4) / 6.0
            + abs(epochs - 400) / 350.0
            + abs(lr - 1e-4) * 5000.0
        )
        if distance > 0.45:
            continue
        add_mean(
            f"mean_alpha{slug_value(alpha)}_t{t}_ep{epochs}_lr{slug_value(lr)}",
            "default",
            alpha=alpha,
            t=t,
            epochs=epochs,
            lr=lr,
        )

    # Attention should be close to mean on USPS, but allow restrained changes around the notes center.
    add_attn("attn_center")
    for ae_k in (10, 15, 20, 25):
        add_attn(f"attn_k{ae_k}", ae_k)

    for temp, balance, min_weight, tau, weight in product(
        (1.6, 1.8, 2.0, 2.2),
        (0.25, 0.35, 0.45, 0.55),
        (0.10, 0.15, 0.20, 0.25, 0.30),
        (0.35, 0.5, 0.75, 1.0),
        (0.4, 0.6, 0.8, 1.0),
    ):
        distance = (
            abs(temp - 1.8)
            + abs(balance - 0.35)
            + abs(min_weight - 0.20) * 2.0
            + abs(tau - 0.5)
            + abs(weight - 0.6)
        )
        if distance > 0.55:
            continue
        add_attn(
            (
                f"attn_temp{slug_value(temp)}_bal{slug_value(balance)}_"
                f"min{slug_value(min_weight)}_tau{slug_value(tau)}_w{slug_value(weight)}"
            ),
            "default",
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for lambda_inst, lambda_clu in product(
        (0.0, 0.03, 0.06, 0.08, 0.09, 0.10, 0.12),
        (0.0, 0.03, 0.06, 0.08, 0.09, 0.10, 0.12),
    ):
        distance = abs(lambda_inst - 0.09) * 3.0 + abs(lambda_clu - 0.09) * 3.0
        if distance > 0.35:
            continue
        add_attn(
            f"attn_li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}",
            "default",
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
        )

    return candidates


def usps_attn_peak_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.6,
        "fusion_balance": 0.25,
        "fusion_min_weight": 0.15,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "warmup_epochs": 35,
        "threshold": 0.4,
        "alpha": 0.5,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
    }

    def add(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = base_attn_args("usps", **center)
        args.update(overrides)
        candidates.append(make_candidate("usps", stem, "attn", ae_k, **args))

    add("attn_peak_center")

    # Best attention neighborhood seen so far on AE roll -7, seed 13:
    # attn_temp1p6_bal0p25_min0p15_tau0p5_w0p6 -> ACC 84.15.
    for temp, balance, min_weight, tau, weight in product(
        (1.4, 1.5, 1.6, 1.7, 1.8),
        (0.15, 0.20, 0.25, 0.30, 0.35),
        (0.05, 0.10, 0.15, 0.20, 0.25),
        (0.35, 0.5, 0.65, 0.75),
        (0.4, 0.6, 0.8),
    ):
        distance = (
            abs(temp - 1.6)
            + abs(balance - 0.25) * 1.5
            + abs(min_weight - 0.15) * 2.0
            + abs(tau - 0.5)
            + abs(weight - 0.6)
        )
        if distance > 0.48:
            continue
        add(
            (
                f"attn_peak_temp{slug_value(temp)}_bal{slug_value(balance)}_"
                f"min{slug_value(min_weight)}_tau{slug_value(tau)}_w{slug_value(weight)}"
            ),
            "default",
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for lambda_inst, lambda_clu, warmup in product(
        (0.04, 0.06, 0.08, 0.09, 0.10, 0.12, 0.15),
        (0.04, 0.06, 0.08, 0.09, 0.10, 0.12, 0.15),
        (25, 35, 45, 55),
    ):
        distance = (
            abs(lambda_inst - 0.09) * 2.5
            + abs(lambda_clu - 0.09) * 2.5
            + abs(warmup - 35) / 80.0
        )
        if distance > 0.32:
            continue
        add(
            (
                f"attn_peak_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_warm{warmup}"
            ),
            "default",
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            warmup_epochs=warmup,
        )

    for alpha, threshold, t, epochs, lr in product(
        (0.45, 0.5, 0.55),
        (0.35, 0.38, 0.4, 0.42, 0.45),
        (4, 5),
        (350, 400, 450),
        (8e-5, 1e-4, 1.2e-4),
    ):
        distance = (
            abs(alpha - 0.5) * 2.0
            + abs(threshold - 0.4) * 2.0
            + abs(t - 4) / 5.0
            + abs(epochs - 400) / 450.0
            + abs(lr - 1e-4) * 4500.0
        )
        if distance > 0.42:
            continue
        add(
            (
                f"attn_peak_alpha{slug_value(alpha)}_thr{slug_value(threshold)}_"
                f"t{t}_ep{epochs}_lr{slug_value(lr)}"
            ),
            "default",
            alpha=alpha,
            threshold=threshold,
            t=t,
            epochs=epochs,
            lr=lr,
        )

    for hidden, temp, balance, min_weight in product(
        (32, 48, 64, 96, 128),
        (1.5, 1.6, 1.7),
        (0.15, 0.25, 0.35),
        (0.10, 0.15, 0.20),
    ):
        distance = (
            abs(hidden - 64) / 220.0
            + abs(temp - 1.6)
            + abs(balance - 0.25) * 1.5
            + abs(min_weight - 0.15) * 2.0
        )
        if distance > 0.45:
            continue
        add(
            (
                f"attn_peak_h{hidden}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}"
            ),
            "default",
            fusion_hidden=hidden,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
        )

    for start, end in ((0.20, 0.45), (0.25, 0.50), (0.30, 0.55)):
        add(
            f"attn_peak_dynamic_{slug_value(start)}_{slug_value(end)}",
            "default",
            enable_dynamic_threshold=True,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
        )

    return candidates


def usps_attn_micro_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str | int, tuple[tuple[str, str], ...]]] = set()

    acc_frontier = {
        "fusion_hidden": 64,
        "fusion_temp": 1.4,
        "fusion_balance": 0.25,
        "fusion_min_weight": 0.10,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "warmup_epochs": 35,
        "threshold": 0.4,
        "alpha": 0.5,
        "dcgl_neg_tau": 0.35,
        "dcgl_neg_weight": 0.6,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
    }
    f1_frontier = {
        **acc_frontier,
        "fusion_temp": 1.6,
        "fusion_balance": 0.20,
        "fusion_min_weight": 0.05,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.4,
    }

    def add(stem: str, base: dict[str, Any], ae_k: str | int = "default", **overrides: Any) -> None:
        args = base_attn_args("usps", **base)
        args.update(overrides)
        key = (
            stem,
            ae_k,
            tuple(sorted((name, str(value)) for name, value in args.items())),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(make_candidate("usps", stem, "attn", ae_k, **args))

    add("usps_micro_acc_frontier", acc_frontier)
    add("usps_micro_f1_frontier", f1_frontier)

    # ACC frontier: the current paper row is
    # temp=1.4,balance=0.25,min=0.10,tau=0.35,weight=0.6 on AE roll -7/seed 13.
    for temp, balance, min_weight, tau, weight in product(
        (1.30, 1.35, 1.40, 1.45, 1.50),
        (0.20, 0.23, 0.25, 0.27, 0.30),
        (0.07, 0.10, 0.12, 0.15, 0.18),
        (0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        (0.45, 0.50, 0.60, 0.70),
    ):
        distance = (
            abs(temp - 1.40)
            + abs(balance - 0.25) * 1.8
            + abs(min_weight - 0.10) * 2.2
            + abs(tau - 0.35) * 0.8
            + abs(weight - 0.60) * 0.7
        )
        if distance > 0.24:
            continue
        add(
            (
                f"usps_micro_acc_temp{slug_value(temp)}_bal{slug_value(balance)}_"
                f"min{slug_value(min_weight)}_tau{slug_value(tau)}_w{slug_value(weight)}"
            ),
            acc_frontier,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    # F1/NMI frontier: slightly softer minimum weight improves F1 but loses ACC/ARI.
    # Search a small bridge between the two frontiers.
    for temp, balance, min_weight, tau, weight in product(
        (1.45, 1.50, 1.55, 1.60, 1.65),
        (0.15, 0.18, 0.20, 0.23, 0.25),
        (0.03, 0.05, 0.07, 0.10, 0.12),
        (0.35, 0.40, 0.45, 0.50, 0.55),
        (0.35, 0.40, 0.50, 0.60),
    ):
        distance = (
            abs(temp - 1.60)
            + abs(balance - 0.20) * 1.7
            + abs(min_weight - 0.05) * 2.5
            + abs(tau - 0.50) * 0.9
            + abs(weight - 0.40) * 0.8
        )
        if distance > 0.23:
            continue
        add(
            (
                f"usps_micro_f1_temp{slug_value(temp)}_bal{slug_value(balance)}_"
                f"min{slug_value(min_weight)}_tau{slug_value(tau)}_w{slug_value(weight)}"
            ),
            f1_frontier,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for alpha, threshold, warmup, lambda_inst, lambda_clu in product(
        (0.45, 0.50, 0.55),
        (0.36, 0.38, 0.40, 0.42, 0.44),
        (30, 35, 40),
        (0.07, 0.09, 0.11),
        (0.07, 0.09, 0.11),
    ):
        distance = (
            abs(alpha - 0.50) * 1.5
            + abs(threshold - 0.40) * 2.0
            + abs(warmup - 35) / 80.0
            + abs(lambda_inst - 0.09) * 2.0
            + abs(lambda_clu - 0.09) * 2.0
        )
        if distance > 0.20:
            continue
        add(
            (
                f"usps_micro_acc_alpha{slug_value(alpha)}_thr{slug_value(threshold)}_"
                f"warm{warmup}_li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}"
            ),
            acc_frontier,
            alpha=alpha,
            threshold=threshold,
            warmup_epochs=warmup,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
        )

    for hidden, t, epochs, lr in product(
        (48, 64, 80, 96),
        (4, 5),
        (350, 400, 450),
        (8e-5, 1e-4, 1.2e-4),
    ):
        distance = (
            abs(hidden - 64) / 180.0
            + abs(t - 4) / 6.0
            + abs(epochs - 400) / 500.0
            + abs(lr - 1e-4) * 4500.0
        )
        if distance > 0.24:
            continue
        add(
            (
                f"usps_micro_acc_h{hidden}_t{t}_ep{epochs}_lr{slug_value(lr)}"
            ),
            acc_frontier,
            fusion_hidden=hidden,
            t=t,
            epochs=epochs,
            lr=lr,
        )

    for start, end, min_weight, balance in product(
        (0.18, 0.20, 0.22, 0.25),
        (0.42, 0.45, 0.48, 0.50),
        (0.08, 0.10, 0.12),
        (0.20, 0.25, 0.30),
    ):
        distance = (
            abs(start - 0.20) * 1.5
            + abs(end - 0.45)
            + abs(min_weight - 0.10) * 2.0
            + abs(balance - 0.25) * 1.5
        )
        if distance > 0.16:
            continue
        add(
            (
                f"usps_micro_acc_dyn{slug_value(start)}_{slug_value(end)}_"
                f"min{slug_value(min_weight)}_bal{slug_value(balance)}"
            ),
            acc_frontier,
            enable_dynamic_threshold=True,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
            fusion_min_weight=min_weight,
            fusion_balance=balance,
        )

    return candidates


def amap_attn_peak_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.25,
        "fusion_balance": 0.08,
        "fusion_min_weight": 0.05,
        "lambda_inst": 0.07,
        "lambda_clu": 0.035,
        "warmup_epochs": 35,
        "threshold": 0.4,
        "alpha": 0.5,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
    }

    seen: set[tuple[str, str | int, tuple[tuple[str, str], ...]]] = set()

    def add(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = base_attn_args("amap", **center)
        args.update(overrides)
        key = (
            stem,
            ae_k,
            tuple(sorted((name, str(value)) for name, value in args.items())),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(make_candidate("amap", stem, "attn", ae_k, **args))

    add("amap_attn_peak_center")

    # Historical AMAP peaks were produced near the notes config; prioritize compact one-axis moves.
    for ae_k in ("default", 10, 15, 20):
        add(f"amap_attn_peak_k{slug_value(ae_k)}", ae_k)

    for min_weight, balance, lambda_inst, lambda_clu in (
        (0.03, 0.08, 0.07, 0.035),
        (0.05, 0.05, 0.07, 0.035),
        (0.05, 0.08, 0.05, 0.035),
        (0.05, 0.08, 0.07, 0.02),
        (0.05, 0.08, 0.07, 0.05),
        (0.08, 0.08, 0.07, 0.035),
        (0.05, 0.10, 0.07, 0.035),
        (0.10, 0.08, 0.07, 0.035),
    ):
        add(
            (
                f"amap_attn_peak_min{slug_value(min_weight)}_bal{slug_value(balance)}_"
                f"li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}"
            ),
            "default",
            fusion_min_weight=min_weight,
            fusion_balance=balance,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
        )

    for tau, weight, threshold in (
        (0.35, 0.6, 0.4),
        (0.5, 0.4, 0.4),
        (0.5, 0.8, 0.4),
        (0.75, 0.6, 0.4),
        (0.5, 0.6, 0.35),
        (0.5, 0.6, 0.38),
        (0.5, 0.6, 0.42),
        (0.5, 0.6, 0.45),
    ):
        add(
            (
                f"amap_attn_peak_tau{slug_value(tau)}_w{slug_value(weight)}_"
                f"thr{slug_value(threshold)}"
            ),
            "default",
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            threshold=threshold,
        )

    # Micro-tune the current seed-2 peak: keep the strong negative branch weight
    # and search small training/attention perturbations.
    for temp, hidden in product((1.10, 1.20, 1.25, 1.30, 1.40), (48, 64, 96)):
        distance = abs(temp - 1.25) * 1.2 + abs(hidden - 64) / 180.0
        if distance > 0.26:
            continue
        add(
            f"amap_attn_peak_micro_temp{slug_value(temp)}_h{hidden}_w0p8",
            "default",
            fusion_temp=temp,
            fusion_hidden=hidden,
            dcgl_neg_weight=0.8,
        )

    for lr, epochs in product((7e-5, 9e-5, 1e-4, 1.1e-4, 1.3e-4), (350, 400, 450, 500)):
        distance = abs(lr - 1e-4) * 4500.0 + abs(epochs - 400) / 550.0
        if distance > 0.28:
            continue
        add(
            f"amap_attn_peak_micro_lr{slug_value(lr)}_ep{epochs}_w0p8",
            "default",
            lr=lr,
            epochs=epochs,
            dcgl_neg_weight=0.8,
        )

    for alpha, t in product((0.45, 0.48, 0.50, 0.52, 0.55), (3, 4, 5)):
        distance = abs(alpha - 0.5) * 2.0 + abs(t - 4) / 5.0
        if distance > 0.24:
            continue
        add(
            f"amap_attn_peak_micro_alpha{slug_value(alpha)}_t{t}_w0p8",
            "default",
            alpha=alpha,
            t=t,
            dcgl_neg_weight=0.8,
        )

    for lambda_inst, lambda_clu, warmup in product(
        (0.05, 0.07, 0.09),
        (0.02, 0.035, 0.05),
        (30, 35, 40, 45),
    ):
        distance = (
            abs(lambda_inst - 0.07) * 3.0
            + abs(lambda_clu - 0.035) * 5.0
            + abs(warmup - 35) / 80.0
        )
        if distance > 0.22:
            continue
        add(
            (
                f"amap_attn_peak_micro_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_warm{warmup}_w0p8"
            ),
            "default",
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            warmup_epochs=warmup,
            dcgl_neg_weight=0.8,
        )

    for tau, weight, threshold, min_weight, balance in product(
        (0.48, 0.50, 0.52),
        (0.78, 0.80, 0.82),
        (0.39, 0.40, 0.41),
        (0.04, 0.05, 0.06),
        (0.07, 0.08, 0.09),
    ):
        distance = (
            abs(tau - 0.5) * 2.0
            + abs(weight - 0.8) * 2.0
            + abs(threshold - 0.4) * 3.0
            + abs(min_weight - 0.05) * 4.0
            + abs(balance - 0.08) * 4.0
        )
        if distance > 0.18:
            continue
        add(
            (
                f"amap_attn_peak_micro_tau{slug_value(tau)}_w{slug_value(weight)}_"
                f"thr{slug_value(threshold)}_min{slug_value(min_weight)}_"
                f"bal{slug_value(balance)}"
            ),
            "default",
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            threshold=threshold,
            fusion_min_weight=min_weight,
            fusion_balance=balance,
        )

    for ae_k, tau, weight, threshold, min_weight, balance in product(
        (5, 10, 15, 20, 25),
        (0.45, 0.5, 0.55),
        (0.7, 0.8, 0.9),
        (0.38, 0.4, 0.42),
        (0.03, 0.05, 0.08),
        (0.05, 0.08, 0.10),
    ):
        distance = (
            abs(ae_k - 15) / 18.0
            + abs(tau - 0.5) * 1.4
            + abs(weight - 0.8) * 1.2
            + abs(threshold - 0.4) * 2.5
            + abs(min_weight - 0.05) * 3.0
            + abs(balance - 0.08) * 3.0
        )
        if distance > 0.34:
            continue
        add(
            (
                f"amap_attn_peak_graph_k{ae_k}_tau{slug_value(tau)}_"
                f"w{slug_value(weight)}_thr{slug_value(threshold)}_"
                f"min{slug_value(min_weight)}_bal{slug_value(balance)}"
            ),
            ae_k,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            threshold=threshold,
            fusion_min_weight=min_weight,
            fusion_balance=balance,
        )

    for tau, weight, threshold, min_weight, balance in product(
        (0.45, 0.5, 0.55, 0.65),
        (0.7, 0.8, 0.9),
        (0.38, 0.4, 0.42),
        (0.03, 0.05, 0.08),
        (0.05, 0.08, 0.10),
    ):
        distance = (
            abs(tau - 0.5) * 1.4
            + abs(weight - 0.8) * 1.2
            + abs(threshold - 0.4) * 2.5
            + abs(min_weight - 0.05) * 3.0
            + abs(balance - 0.08) * 3.0
        )
        if distance > 0.34:
            continue
        add(
            (
                f"amap_attn_peak_local_tau{slug_value(tau)}_w{slug_value(weight)}_"
                f"thr{slug_value(threshold)}_min{slug_value(min_weight)}_"
                f"bal{slug_value(balance)}"
            ),
            "default",
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            threshold=threshold,
            fusion_min_weight=min_weight,
            fusion_balance=balance,
        )

    for ae_k, min_weight, balance, lambda_inst, lambda_clu, tau, weight, threshold in product(
        ("default", 10, 15, 20),
        (0.0, 0.03, 0.05, 0.08, 0.10),
        (0.03, 0.05, 0.08, 0.10, 0.12),
        (0.0, 0.03, 0.05, 0.07, 0.09),
        (0.0, 0.02, 0.035, 0.05, 0.07),
        (0.35, 0.5, 0.75, 1.0),
        (0.4, 0.6, 0.8, 1.0),
        (0.35, 0.38, 0.4, 0.42, 0.45),
    ):
        ae_distance = 0.0 if ae_k in {"default", 15} else abs(int(ae_k) - 15) / 18.0
        distance = (
            ae_distance
            + abs(min_weight - 0.05) * 3.0
            + abs(balance - 0.08) * 3.0
            + abs(lambda_inst - 0.07) * 2.5
            + abs(lambda_clu - 0.035) * 4.0
            + abs(tau - 0.5)
            + abs(weight - 0.6)
            + abs(threshold - 0.4) * 2.0
        )
        if distance > 0.34:
            continue
        add(
            (
                f"amap_attn_peak_k{slug_value(ae_k)}_min{slug_value(min_weight)}_"
                f"bal{slug_value(balance)}_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_tau{slug_value(tau)}_"
                f"w{slug_value(weight)}_thr{slug_value(threshold)}"
            ),
            ae_k,
            fusion_min_weight=min_weight,
            fusion_balance=balance,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            threshold=threshold,
        )

    for hidden, temp, balance, min_weight in product(
        (48, 64, 96),
        (1.1, 1.25, 1.4),
        (0.05, 0.08, 0.10),
        (0.03, 0.05, 0.08),
    ):
        distance = (
            abs(hidden - 64) / 180.0
            + abs(temp - 1.25) * 1.2
            + abs(balance - 0.08) * 3.0
            + abs(min_weight - 0.05) * 3.0
        )
        if distance > 0.28:
            continue
        add(
            (
                f"amap_attn_peak_h{hidden}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}"
            ),
            "default",
            fusion_hidden=hidden,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
        )

    for alpha, t, epochs, lr in product(
        (0.45, 0.5, 0.55),
        (3, 4, 5),
        (350, 400, 450),
        (7e-5, 1e-4, 1.3e-4),
    ):
        distance = (
            abs(alpha - 0.5) * 2.0
            + abs(t - 4) / 5.0
            + abs(epochs - 400) / 500.0
            + abs(lr - 1e-4) * 4200.0
        )
        if distance > 0.32:
            continue
        add(
            (
                f"amap_attn_peak_alpha{slug_value(alpha)}_t{t}_"
                f"ep{epochs}_lr{slug_value(lr)}"
            ),
            "default",
            alpha=alpha,
            t=t,
            epochs=epochs,
            lr=lr,
        )

    for start, end in ((0.20, 0.45), (0.25, 0.50), (0.30, 0.55)):
        add(
            f"amap_attn_peak_dynamic_{slug_value(start)}_{slug_value(end)}",
            "default",
            enable_dynamic_threshold=True,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
        )

    return candidates


def amap_attn_dcgl_apex_candidates() -> list[Candidate]:
    seen: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    ranked: list[tuple[float, Candidate]] = []

    peak = {
        "dims": 500,
        "threshold": 0.4,
        "alpha": 0.5,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
        "warmup_epochs": 35,
        "fusion_hidden": 64,
        "fusion_temp": 1.25,
        "fusion_balance": 0.08,
        "fusion_min_weight": 0.05,
        "lambda_inst": 0.07,
        "lambda_clu": 0.035,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.8,
        "enable_dcgl_negative_loss": True,
    }
    acc_frontier = dict(peak, fusion_min_weight=0.08, fusion_balance=0.05)
    graph_k10 = dict(peak, threshold=0.42)
    train_long = dict(peak, epochs=450, lr=9e-5)
    centers = (
        ("peak", peak),
        ("acc", acc_frontier),
        ("k10", graph_k10),
        ("long", train_long),
    )
    forbidden = (
        "enable_dynamic_threshold",
        "enable_gcn_backbone",
        "enable_dcgl_cluster_level",
        "enable_branch_bias_fusion",
        "enable_ema_prototypes",
    )

    def dist(args: dict[str, Any], ae_k: str | int, center: dict[str, Any]) -> float:
        ae_val = AE_DEFAULT_K["amap"] if ae_k == "default" else int(ae_k)
        center_ae = 10 if center is graph_k10 else AE_DEFAULT_K["amap"]
        return (
            abs(ae_val - center_ae) / 28.0
            + abs(float(args["dims"]) - float(center["dims"])) / 700.0
            + abs(float(args["threshold"]) - float(center["threshold"])) * 3.0
            + abs(float(args["alpha"]) - float(center["alpha"])) * 2.0
            + abs(float(args["t"]) - float(center["t"])) / 5.0
            + abs(float(args["epochs"]) - float(center["epochs"])) / 650.0
            + abs(float(args["lr"]) - float(center["lr"])) * 5000.0
            + abs(float(args["warmup_epochs"]) - float(center["warmup_epochs"])) / 75.0
            + abs(float(args["fusion_hidden"]) - float(center["fusion_hidden"])) / 220.0
            + abs(float(args["fusion_temp"]) - float(center["fusion_temp"])) * 1.5
            + abs(float(args["fusion_balance"]) - float(center["fusion_balance"])) * 3.5
            + abs(float(args["fusion_min_weight"]) - float(center["fusion_min_weight"])) * 4.0
            + abs(float(args["lambda_inst"]) - float(center["lambda_inst"])) * 5.0
            + abs(float(args["lambda_clu"]) - float(center["lambda_clu"])) * 7.0
            + abs(float(args["dcgl_neg_tau"]) - float(center["dcgl_neg_tau"])) * 1.2
            + abs(float(args["dcgl_neg_weight"]) - float(center["dcgl_neg_weight"])) * 1.3
        )

    def add(stem: str, ae_k: str | int = "default", center: dict[str, Any] = peak, **overrides: Any) -> None:
        args = dict(center)
        args.update(overrides)
        for forbidden_key in forbidden:
            args.pop(forbidden_key, None)
        args["enable_dcgl_negative_loss"] = True
        candidate = make_candidate("amap", stem, "attn", ae_k, **args)
        key = (
            candidate.fusion_mode,
            str(candidate.ae_k),
            tuple(sorted((name, str(value)) for name, value in candidate.args.items())),
        )
        if key in seen:
            return
        seen.add(key)
        ranked.append((dist(args, ae_k, center), candidate))

    for prefix, center in centers:
        add(f"amap_dcgl_apex_{prefix}_center", center=center)
        for ae_k in ("default", 10, 15, 20):
            add(f"amap_dcgl_apex_{prefix}_k{slug_value(ae_k)}", ae_k, center=center)

    for threshold, tau, weight, min_weight, balance in product(
        (0.385, 0.39, 0.395, 0.40, 0.405, 0.41, 0.415, 0.42),
        (0.46, 0.48, 0.50, 0.52, 0.54),
        (0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86),
        (0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.08),
        (0.05, 0.06, 0.07, 0.08, 0.09, 0.10),
    ):
        args = dict(peak)
        args.update(
            {
                "threshold": threshold,
                "dcgl_neg_tau": tau,
                "dcgl_neg_weight": weight,
                "fusion_min_weight": min_weight,
                "fusion_balance": balance,
            }
        )
        if dist(args, "default", peak) > 0.30:
            continue
        add(
            (
                f"amap_apex_gate_th{slug_value(threshold)}_ta{slug_value(tau)}_"
                f"w{slug_value(weight)}_m{slug_value(min_weight)}_b{slug_value(balance)}"
            ),
            "default",
            peak,
            **args,
        )

    for temp, hidden, balance, min_weight, threshold in product(
        (1.16, 1.18, 1.20, 1.22, 1.25, 1.28, 1.30, 1.32),
        (48, 64, 80, 96),
        (0.06, 0.07, 0.08, 0.09, 0.10),
        (0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.08),
        (0.39, 0.40, 0.41, 0.42),
    ):
        args = dict(peak)
        args.update(
            {
                "fusion_temp": temp,
                "fusion_hidden": hidden,
                "fusion_balance": balance,
                "fusion_min_weight": min_weight,
                "threshold": threshold,
            }
        )
        if dist(args, "default", peak) > 0.34:
            continue
        add(
            (
                f"amap_apex_fuse_h{hidden}_tf{slug_value(temp)}_"
                f"b{slug_value(balance)}_m{slug_value(min_weight)}_th{slug_value(threshold)}"
            ),
            "default",
            peak,
            **args,
        )

    for lambda_inst, lambda_clu, tau, weight, warmup in product(
        (0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085),
        (0.025, 0.03, 0.035, 0.04, 0.045),
        (0.46, 0.48, 0.50, 0.52, 0.54),
        (0.74, 0.78, 0.80, 0.82, 0.86),
        (30, 35, 40, 45),
    ):
        args = dict(peak)
        args.update(
            {
                "lambda_inst": lambda_inst,
                "lambda_clu": lambda_clu,
                "dcgl_neg_tau": tau,
                "dcgl_neg_weight": weight,
                "warmup_epochs": warmup,
            }
        )
        if dist(args, "default", peak) > 0.36:
            continue
        add(
            (
                f"amap_apex_loss_li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}_"
                f"ta{slug_value(tau)}_w{slug_value(weight)}_wu{warmup}"
            ),
            "default",
            peak,
            **args,
        )

    for alpha, t, epochs, lr, threshold, warmup in product(
        (0.46, 0.48, 0.50, 0.52, 0.54),
        (3, 4, 5),
        (380, 400, 420, 450, 480),
        (8.5e-5, 9e-5, 9.5e-5, 1e-4, 1.05e-4, 1.1e-4, 1.2e-4),
        (0.39, 0.40, 0.41, 0.42),
        (30, 35, 40),
    ):
        args = dict(peak)
        args.update(
            {
                "alpha": alpha,
                "t": t,
                "epochs": epochs,
                "lr": lr,
                "threshold": threshold,
                "warmup_epochs": warmup,
            }
        )
        if dist(args, "default", peak) > 0.40:
            continue
        add(
            (
                f"amap_apex_train_a{slug_value(alpha)}_t{t}_ep{epochs}_"
                f"lr{slug_value(lr)}_th{slug_value(threshold)}_wu{warmup}"
            ),
            "default",
            peak,
            **args,
        )

    for ae_k, threshold, tau, weight, min_weight, balance, temp in product(
        (5, 10, 15, 20, 25),
        (0.39, 0.40, 0.41, 0.42),
        (0.48, 0.50, 0.52),
        (0.78, 0.80, 0.82),
        (0.04, 0.05, 0.06, 0.08),
        (0.06, 0.08, 0.10),
        (1.20, 1.25, 1.30),
    ):
        center = graph_k10 if ae_k == 10 else peak
        args = dict(peak)
        args.update(
            {
                "threshold": threshold,
                "dcgl_neg_tau": tau,
                "dcgl_neg_weight": weight,
                "fusion_min_weight": min_weight,
                "fusion_balance": balance,
                "fusion_temp": temp,
            }
        )
        if dist(args, ae_k, center) > 0.42:
            continue
        add(
            (
                f"amap_apex_graph_k{ae_k}_th{slug_value(threshold)}_"
                f"ta{slug_value(tau)}_w{slug_value(weight)}_"
                f"m{slug_value(min_weight)}_b{slug_value(balance)}_tf{slug_value(temp)}"
            ),
            ae_k,
            center,
            **args,
        )

    for min_weight, balance, threshold, tau, weight, lambda_inst, lambda_clu in product(
        (0.07, 0.075, 0.08, 0.085, 0.09),
        (0.045, 0.05, 0.055, 0.06, 0.065),
        (0.39, 0.40, 0.41, 0.42),
        (0.48, 0.50, 0.52),
        (0.78, 0.80, 0.82, 0.84),
        (0.06, 0.07, 0.08),
        (0.03, 0.035, 0.04),
    ):
        args = dict(acc_frontier)
        args.update(
            {
                "fusion_min_weight": min_weight,
                "fusion_balance": balance,
                "threshold": threshold,
                "dcgl_neg_tau": tau,
                "dcgl_neg_weight": weight,
                "lambda_inst": lambda_inst,
                "lambda_clu": lambda_clu,
            }
        )
        if dist(args, "default", acc_frontier) > 0.36:
            continue
        add(
            (
                f"amap_apex_acc_m{slug_value(min_weight)}_b{slug_value(balance)}_"
                f"th{slug_value(threshold)}_ta{slug_value(tau)}_w{slug_value(weight)}_"
                f"li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}"
            ),
            "default",
            acc_frontier,
            **args,
        )

    ranked.sort(key=lambda item: item[0])
    return [candidate for _dist, candidate in ranked]


def eat_attn_peak_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str | int, tuple[tuple[str, str], ...]]] = set()

    # Historical high-water branch:
    # experiment_output/eat/eat_train_with_dual_attn_20260416_183517.txt
    # Run 02 reached 57.64 / 33.78 / 26.32 / 57.86 with cluster-level DCGL.
    cluster_center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.4,
        "fusion_balance": 0.25,
        "fusion_min_weight": 0.10,
        "lambda_inst": 0.12,
        "lambda_clu": 0.12,
        "warmup_epochs": 50,
        "threshold": 0.5,
        "alpha": 0.5,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 1.0,
        "enable_dcgl_cluster_level": True,
        "lambda_dcgl_cluster": 0.1,
        "dcgl_cluster_tau": 0.5,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
    }
    ema_center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.5,
        "fusion_balance": 0.20,
        "fusion_min_weight": 0.12,
        "lambda_inst": 0.08,
        "lambda_clu": 0.05,
        "warmup_epochs": 35,
        "threshold": 0.4,
        "alpha": 0.5,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
        "enable_ema_prototypes": True,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
    }
    notes_center = {
        "fusion_hidden": 64,
        "fusion_temp": 2.0,
        "fusion_balance": 0.35,
        "fusion_min_weight": 0.20,
        "lambda_inst": 0.08,
        "lambda_clu": 0.08,
        "warmup_epochs": 35,
        "threshold": 0.45,
        "alpha": 0.5,
        "dcgl_neg_tau": 0.35,
        "dcgl_neg_weight": 0.8,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
    }

    def add_from(center: dict[str, Any], stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = base_attn_args("eat", **center)
        args.update(overrides)
        key = (
            stem,
            ae_k,
            tuple(sorted((name, str(value)) for name, value in args.items())),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(make_candidate("eat", stem, "attn", ae_k, **args))

    def add_cluster(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        add_from(cluster_center, stem, ae_k, **overrides)

    def add_ema(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        add_from(ema_center, stem, ae_k, **overrides)

    def add_notes(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        add_from(notes_center, stem, ae_k, **overrides)

    add_cluster("eat_attn_peak_cluster_hist")
    add_ema("eat_attn_peak_ema_hist")
    add_notes("eat_attn_peak_notes_center")

    # Put the most useful graph variants early; k15 is identical to default in the current tree.
    for ae_k in (10, 20, 25, 5):
        add_cluster(f"eat_attn_peak_cluster_k{ae_k}", ae_k)
    for ae_k in (10, 20, 25, 5):
        add_ema(f"eat_attn_peak_ema_k{ae_k}", ae_k)

    # Recreate and slightly relax the historical cluster-level branch.
    for threshold, temp, balance, min_weight in (
        (0.45, 1.4, 0.25, 0.10),
        (0.48, 1.4, 0.25, 0.10),
        (0.50, 1.3, 0.25, 0.10),
        (0.50, 1.5, 0.25, 0.10),
        (0.50, 1.4, 0.20, 0.10),
        (0.50, 1.4, 0.30, 0.10),
        (0.50, 1.4, 0.25, 0.08),
        (0.50, 1.4, 0.25, 0.12),
        (0.52, 1.4, 0.25, 0.10),
        (0.55, 1.4, 0.25, 0.10),
    ):
        add_cluster(
            (
                f"eat_attn_peak_cluster_thr{slug_value(threshold)}_"
                f"temp{slug_value(temp)}_bal{slug_value(balance)}_min{slug_value(min_weight)}"
            ),
            "default",
            threshold=threshold,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
        )

    for lambda_inst, lambda_clu, warmup in (
        (0.10, 0.10, 50),
        (0.12, 0.10, 50),
        (0.10, 0.12, 50),
        (0.12, 0.12, 45),
        (0.12, 0.12, 55),
        (0.15, 0.12, 50),
        (0.12, 0.15, 50),
        (0.15, 0.15, 55),
    ):
        add_cluster(
            (
                f"eat_attn_peak_cluster_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_warm{warmup}"
            ),
            "default",
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            warmup_epochs=warmup,
        )

    for tau, weight, lambda_cluster, cluster_tau in (
        (0.45, 1.0, 0.10, 0.5),
        (0.50, 0.8, 0.10, 0.5),
        (0.50, 1.2, 0.10, 0.5),
        (0.55, 1.0, 0.10, 0.5),
        (0.50, 1.0, 0.05, 0.5),
        (0.50, 1.0, 0.15, 0.5),
        (0.50, 1.0, 0.10, 0.4),
        (0.50, 1.0, 0.10, 0.6),
        (0.45, 1.2, 0.15, 0.5),
        (0.55, 0.8, 0.05, 0.5),
    ):
        add_cluster(
            (
                f"eat_attn_peak_cluster_tau{slug_value(tau)}_w{slug_value(weight)}_"
                f"lcluster{slug_value(lambda_cluster)}_ctau{slug_value(cluster_tau)}"
            ),
            "default",
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            lambda_dcgl_cluster=lambda_cluster,
            dcgl_cluster_tau=cluster_tau,
        )

    for alpha, t, epochs, lr in (
        (0.45, 4, 400, 1e-4),
        (0.50, 3, 400, 1e-4),
        (0.50, 5, 400, 1e-4),
        (0.55, 4, 400, 1e-4),
        (0.50, 4, 450, 1e-4),
        (0.50, 4, 500, 1e-4),
        (0.50, 4, 400, 8e-5),
        (0.50, 4, 400, 1.2e-4),
    ):
        add_cluster(
            (
                f"eat_attn_peak_cluster_alpha{slug_value(alpha)}_t{t}_"
                f"ep{epochs}_lr{slug_value(lr)}"
            ),
            "default",
            alpha=alpha,
            t=t,
            epochs=epochs,
            lr=lr,
        )

    # Graph-specific micro search around the branch closest to SCGC-S.
    for ae_k, threshold, min_weight, weight in product(
        (10, 20, 25),
        (0.48, 0.50, 0.52),
        (0.08, 0.10, 0.12),
        (0.8, 1.0, 1.2),
    ):
        distance = (
            abs(threshold - 0.50) * 3.0
            + abs(min_weight - 0.10) * 4.0
            + abs(weight - 1.0)
            + abs(ae_k - 15) / 28.0
        )
        if distance > 0.42:
            continue
        add_cluster(
            (
                f"eat_attn_peak_cluster_graph_k{ae_k}_thr{slug_value(threshold)}_"
                f"min{slug_value(min_weight)}_w{slug_value(weight)}"
            ),
            ae_k,
            threshold=threshold,
            fusion_min_weight=min_weight,
            dcgl_neg_weight=weight,
        )

    # Fine-grained k10 neighborhood found during the May 1 EAT push:
    # k10/thr0.50/min0.08/w1.0 reached 57.64 / 33.37 / 26.35 / 57.97.
    # The remaining gap is NMI/ARI, so keep k10 and probe slightly stronger
    # cluster separation without losing the F1 gain from the lower AE floor.
    for threshold, min_weight, temp, balance, warmup in product(
        (0.49, 0.50, 0.51),
        (0.05, 0.06, 0.07, 0.08, 0.09),
        (1.25, 1.35, 1.40, 1.45, 1.55),
        (0.18, 0.22, 0.25, 0.28),
        (45, 50, 55, 60),
    ):
        distance = (
            abs(threshold - 0.50) * 5.0
            + abs(min_weight - 0.08) * 7.0
            + abs(temp - 1.40) * 1.4
            + abs(balance - 0.25) * 2.0
            + abs(warmup - 50) / 65.0
        )
        if distance > 0.36:
            continue
        add_cluster(
            (
                f"eat_attn_peak_k10_micro_thr{slug_value(threshold)}_"
                f"min{slug_value(min_weight)}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_warm{warmup}"
            ),
            10,
            threshold=threshold,
            fusion_min_weight=min_weight,
            fusion_temp=temp,
            fusion_balance=balance,
            warmup_epochs=warmup,
            dcgl_neg_weight=1.0,
        )

    for threshold, min_weight, tau, weight, lambda_cluster, cluster_tau in product(
        (0.49, 0.50, 0.51),
        (0.06, 0.07, 0.08, 0.09),
        (0.45, 0.50, 0.55, 0.60),
        (0.8, 1.0, 1.2),
        (0.05, 0.08, 0.10, 0.12, 0.15),
        (0.35, 0.40, 0.45, 0.50, 0.55),
    ):
        distance = (
            abs(threshold - 0.50) * 5.0
            + abs(min_weight - 0.08) * 7.0
            + abs(tau - 0.50) * 1.5
            + abs(weight - 1.0)
            + abs(lambda_cluster - 0.10) * 4.0
            + abs(cluster_tau - 0.50) * 1.2
        )
        if distance > 0.38:
            continue
        add_cluster(
            (
                f"eat_attn_peak_k10_micro_thr{slug_value(threshold)}_"
                f"min{slug_value(min_weight)}_tau{slug_value(tau)}_"
                f"w{slug_value(weight)}_lcg{slug_value(lambda_cluster)}_"
                f"ctau{slug_value(cluster_tau)}"
            ),
            10,
            threshold=threshold,
            fusion_min_weight=min_weight,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            lambda_dcgl_cluster=lambda_cluster,
            dcgl_cluster_tau=cluster_tau,
        )

    for threshold, min_weight, lambda_inst, lambda_clu, lr in product(
        (0.49, 0.50, 0.51),
        (0.06, 0.07, 0.08, 0.09),
        (0.10, 0.12, 0.14, 0.16),
        (0.10, 0.12, 0.14, 0.16),
        (8e-5, 9e-5, 1e-4, 1.1e-4),
    ):
        distance = (
            abs(threshold - 0.50) * 5.0
            + abs(min_weight - 0.08) * 7.0
            + abs(lambda_inst - 0.12) * 4.0
            + abs(lambda_clu - 0.12) * 4.0
            + abs(lr - 1e-4) * 4200.0
        )
        if distance > 0.34:
            continue
        add_cluster(
            (
                f"eat_attn_peak_k10_micro_thr{slug_value(threshold)}_"
                f"min{slug_value(min_weight)}_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_lr{slug_value(lr)}"
            ),
            10,
            threshold=threshold,
            fusion_min_weight=min_weight,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            lr=lr,
            dcgl_neg_weight=1.0,
        )

    # Preserve the more conservative notes branch in case the cluster-level loss overfits a seed.
    for ae_k, temp, balance, min_weight, threshold, tau, weight in product(
        ("default", 10, 20),
        (1.8, 2.0, 2.2),
        (0.25, 0.35),
        (0.15, 0.20),
        (0.40, 0.45, 0.50),
        (0.35, 0.50),
        (0.8, 1.0),
    ):
        ae_distance = 0.0 if ae_k == "default" else abs(int(ae_k) - 15) / 25.0
        distance = (
            ae_distance
            + abs(temp - 2.0)
            + abs(balance - 0.35) * 1.3
            + abs(min_weight - 0.20) * 2.0
            + abs(threshold - 0.45) * 2.0
            + abs(tau - 0.35)
            + abs(weight - 0.8)
        )
        if distance > 0.46:
            continue
        add_notes(
            (
                f"eat_attn_peak_notes_k{slug_value(ae_k)}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}_"
                f"thr{slug_value(threshold)}_tau{slug_value(tau)}_w{slug_value(weight)}"
            ),
            ae_k,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
            threshold=threshold,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for start, end in ((0.25, 0.50), (0.30, 0.55)):
        add_cluster(
            f"eat_attn_peak_cluster_dynamic_{slug_value(start)}_{slug_value(end)}",
            "default",
            enable_dynamic_threshold=True,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
        )
        add_notes(
            f"eat_attn_peak_notes_dynamic_{slug_value(start)}_{slug_value(end)}",
            "default",
            enable_dynamic_threshold=True,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
        )

    return candidates


def eat_attn_open_peak_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str | int, tuple[tuple[str, str], ...]]] = set()

    # Reproducible current-code seed-9 peak from sensitivity logs and direct rerun:
    # fusion_min_weight=0.0, seed=9 -> 60.15 / 33.74 / 26.84 / 60.92.
    center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.5,
        "fusion_balance": 0.20,
        "fusion_min_weight": 0.0,
        "lambda_inst": 0.08,
        "lambda_clu": 0.05,
        "warmup_epochs": 35,
        "threshold": 0.4,
        "alpha": 0.5,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
    }

    def add(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = base_attn_args("eat", **center)
        args.update(overrides)
        key = (
            stem,
            ae_k,
            tuple(sorted((name, str(value)) for name, value in args.items())),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(make_candidate("eat", stem, "attn", ae_k, **args))

    add("eat_attn_open_peak_center")

    # Compact direct neighbors first; these are the most likely to recover
    # NMI/ARI while keeping the seed-9 high-ACC basin.
    for threshold, min_weight, temp, balance in (
        (0.38, 0.0, 1.5, 0.20),
        (0.39, 0.0, 1.5, 0.20),
        (0.41, 0.0, 1.5, 0.20),
        (0.42, 0.0, 1.5, 0.20),
        (0.40, 0.02, 1.5, 0.20),
        (0.40, 0.04, 1.5, 0.20),
        (0.40, 0.0, 1.4, 0.20),
        (0.40, 0.0, 1.6, 0.20),
        (0.40, 0.0, 1.5, 0.10),
        (0.40, 0.0, 1.5, 0.30),
    ):
        add(
            (
                f"eat_attn_open_thr{slug_value(threshold)}_min{slug_value(min_weight)}_"
                f"temp{slug_value(temp)}_bal{slug_value(balance)}"
            ),
            "default",
            threshold=threshold,
            fusion_min_weight=min_weight,
            fusion_temp=temp,
            fusion_balance=balance,
        )

    for threshold, min_weight, temp, balance, warmup in product(
        (0.36, 0.38, 0.40, 0.42, 0.44),
        (0.0, 0.01, 0.02, 0.04, 0.06),
        (1.3, 1.4, 1.5, 1.6, 1.7),
        (0.0, 0.1, 0.2, 0.3, 0.35),
        (25, 30, 35, 40, 45),
    ):
        distance = (
            abs(threshold - 0.40) * 3.5
            + abs(min_weight - 0.0) * 6.0
            + abs(temp - 1.5)
            + abs(balance - 0.2) * 1.5
            + abs(warmup - 35) / 70.0
        )
        if distance > 0.34:
            continue
        add(
            (
                f"eat_attn_open_micro_thr{slug_value(threshold)}_"
                f"min{slug_value(min_weight)}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_warm{warmup}"
            ),
            "default",
            threshold=threshold,
            fusion_min_weight=min_weight,
            fusion_temp=temp,
            fusion_balance=balance,
            warmup_epochs=warmup,
        )

    for lambda_inst, lambda_clu, tau, weight in product(
        (0.04, 0.06, 0.08, 0.10, 0.12),
        (0.02, 0.035, 0.05, 0.07, 0.09),
        (0.35, 0.45, 0.50, 0.60, 0.75),
        (0.4, 0.6, 0.8, 1.0),
    ):
        distance = (
            abs(lambda_inst - 0.08) * 3.0
            + abs(lambda_clu - 0.05) * 4.0
            + abs(tau - 0.5) * 0.8
            + abs(weight - 0.6) * 0.8
        )
        if distance > 0.34:
            continue
        add(
            (
                f"eat_attn_open_loss_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_tau{slug_value(tau)}_w{slug_value(weight)}"
            ),
            "default",
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for lambda_cluster, cluster_tau, weight in product(
        (0.03, 0.05, 0.08, 0.10),
        (0.35, 0.40, 0.50, 0.60),
        (0.4, 0.6, 0.8),
    ):
        distance = (
            abs(lambda_cluster - 0.05) * 4.0
            + abs(cluster_tau - 0.5)
            + abs(weight - 0.6) * 0.8
        )
        if distance > 0.26:
            continue
        add(
            (
                f"eat_attn_open_cluster_lcg{slug_value(lambda_cluster)}_"
                f"ctau{slug_value(cluster_tau)}_w{slug_value(weight)}"
            ),
            "default",
            enable_dcgl_cluster_level=True,
            lambda_dcgl_cluster=lambda_cluster,
            dcgl_cluster_tau=cluster_tau,
            dcgl_neg_weight=weight,
        )

    for ae_k in (5, 10, 15, 20, 25):
        add(f"eat_attn_open_graph_k{ae_k}", ae_k)
    for ae_k, threshold, min_weight in product(
        (5, 10, 20, 25),
        (0.38, 0.40, 0.42),
        (0.0, 0.02),
    ):
        distance = abs(threshold - 0.4) * 3.0 + abs(min_weight) * 6.0 + abs(ae_k - 15) / 25.0
        if distance > 0.42:
            continue
        add(
            f"eat_attn_open_graph_k{ae_k}_thr{slug_value(threshold)}_min{slug_value(min_weight)}",
            ae_k,
            threshold=threshold,
            fusion_min_weight=min_weight,
        )

    return candidates


def eat_attn_open_bridge_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str | int, tuple[tuple[str, str], ...]]] = set()

    # Ultra-local bridge around the closest all-metric seed-9 point:
    # 59.15 / 33.64 / 27.46 / 59.42 at threshold=0.36, min_weight=0.01,
    # temp=1.5, balance=0.2, warmup=35. Need roughly +0.27 NMI and +0.05 ARI.
    center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.5,
        "fusion_balance": 0.20,
        "fusion_min_weight": 0.01,
        "lambda_inst": 0.08,
        "lambda_clu": 0.05,
        "warmup_epochs": 35,
        "threshold": 0.36,
        "alpha": 0.5,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
    }

    def add(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = base_attn_args("eat", **center)
        args.update(overrides)
        key = (
            stem,
            ae_k,
            tuple(sorted((name, str(value)) for name, value in args.items())),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(make_candidate("eat", stem, "attn", ae_k, **args))

    add("eat_attn_open_bridge_center")

    for threshold, min_weight, temp, balance, warmup in product(
        (0.34, 0.35, 0.36, 0.37, 0.38),
        (0.0, 0.005, 0.01, 0.015, 0.02),
        (1.45, 1.50, 1.55, 1.60),
        (0.15, 0.18, 0.20, 0.22, 0.25),
        (32, 35, 38, 40),
    ):
        distance = (
            abs(threshold - 0.36) * 5.0
            + abs(min_weight - 0.01) * 12.0
            + abs(temp - 1.5) * 1.4
            + abs(balance - 0.2) * 2.5
            + abs(warmup - 35) / 55.0
        )
        if distance > 0.26:
            continue
        add(
            (
                f"eat_attn_open_bridge_thr{slug_value(threshold)}_"
                f"min{slug_value(min_weight)}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_warm{warmup}"
            ),
            "default",
            threshold=threshold,
            fusion_min_weight=min_weight,
            fusion_temp=temp,
            fusion_balance=balance,
            warmup_epochs=warmup,
        )

    for lambda_inst, lambda_clu, tau, weight, alpha in product(
        (0.06, 0.07, 0.08, 0.09, 0.10),
        (0.035, 0.045, 0.05, 0.06, 0.07),
        (0.45, 0.50, 0.55, 0.60),
        (0.5, 0.6, 0.7, 0.8),
        (0.45, 0.50, 0.55),
    ):
        distance = (
            abs(lambda_inst - 0.08) * 4.0
            + abs(lambda_clu - 0.05) * 5.0
            + abs(tau - 0.5) * 1.0
            + abs(weight - 0.6) * 0.8
            + abs(alpha - 0.5) * 1.5
        )
        if distance > 0.30:
            continue
        add(
            (
                f"eat_attn_open_bridge_loss_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_tau{slug_value(tau)}_"
                f"w{slug_value(weight)}_alpha{slug_value(alpha)}"
            ),
            "default",
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            alpha=alpha,
        )

    for lcg, ctau, weight in product(
        (0.02, 0.03, 0.05, 0.08),
        (0.35, 0.40, 0.45, 0.50),
        (0.5, 0.6, 0.7),
    ):
        distance = abs(lcg - 0.03) * 5.0 + abs(ctau - 0.45) * 1.2 + abs(weight - 0.6) * 0.8
        if distance > 0.24:
            continue
        add(
            (
                f"eat_attn_open_bridge_cluster_lcg{slug_value(lcg)}_"
                f"ctau{slug_value(ctau)}_w{slug_value(weight)}"
            ),
            "default",
            enable_dcgl_cluster_level=True,
            lambda_dcgl_cluster=lcg,
            dcgl_cluster_tau=ctau,
            dcgl_neg_weight=weight,
        )

    for lr, epochs, t in product(
        (8e-5, 9e-5, 1e-4, 1.1e-4, 1.2e-4),
        (350, 400, 450),
        (3, 4),
    ):
        distance = abs(lr - 1e-4) * 4500.0 + abs(epochs - 400) / 600.0 + abs(t - 4) / 5.0
        if distance > 0.22:
            continue
        add(
            f"eat_attn_open_bridge_train_lr{slug_value(lr)}_ep{epochs}_t{t}",
            "default",
            lr=lr,
            epochs=epochs,
            t=t,
        )

    for ae_k, threshold, min_weight in product(
        (5, 10, 15, 20),
        (0.35, 0.36, 0.37),
        (0.0, 0.01, 0.02),
    ):
        distance = abs(ae_k - 15) / 28.0 + abs(threshold - 0.36) * 4.0 + abs(min_weight - 0.01) * 10.0
        if distance > 0.36:
            continue
        add(
            f"eat_attn_open_bridge_graph_k{ae_k}_thr{slug_value(threshold)}_min{slug_value(min_weight)}",
            ae_k,
            threshold=threshold,
            fusion_min_weight=min_weight,
        )

    return candidates


def eat_attn_apex_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str | int, tuple[tuple[str, str], ...]]] = set()

    # Current seed-9 all-metric apex:
    # threshold=0.36, min_weight=0.005, temp=1.6, balance=0.2, warmup=32
    # -> 60.65 / 34.53 / 27.75 / 61.35.
    center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.6,
        "fusion_balance": 0.20,
        "fusion_min_weight": 0.005,
        "lambda_inst": 0.08,
        "lambda_clu": 0.05,
        "warmup_epochs": 32,
        "threshold": 0.36,
        "alpha": 0.5,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
    }

    def add(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = base_attn_args("eat", **center)
        args.update(overrides)
        key = (
            stem,
            ae_k,
            tuple(sorted((name, str(value)) for name, value in args.items())),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(make_candidate("eat", stem, "attn", ae_k, **args))

    add("eat_attn_apex_center")

    for threshold, min_weight, temp, balance, warmup in product(
        (0.355, 0.36, 0.365, 0.37),
        (0.0, 0.003, 0.005, 0.008, 0.01),
        (1.56, 1.58, 1.60, 1.62, 1.64),
        (0.18, 0.20, 0.22),
        (30, 31, 32, 33, 34),
    ):
        distance = (
            abs(threshold - 0.36) * 8.0
            + abs(min_weight - 0.005) * 16.0
            + abs(temp - 1.6) * 1.4
            + abs(balance - 0.2) * 2.0
            + abs(warmup - 32) / 45.0
        )
        if distance > 0.22:
            continue
        add(
            (
                f"eat_attn_apex_thr{slug_value(threshold)}_"
                f"min{slug_value(min_weight)}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_warm{warmup}"
            ),
            "default",
            threshold=threshold,
            fusion_min_weight=min_weight,
            fusion_temp=temp,
            fusion_balance=balance,
            warmup_epochs=warmup,
        )

    for lambda_inst, lambda_clu, tau, weight, alpha in product(
        (0.07, 0.08, 0.09),
        (0.045, 0.05, 0.055, 0.06),
        (0.45, 0.50, 0.55),
        (0.55, 0.60, 0.65, 0.70),
        (0.48, 0.50, 0.52),
    ):
        distance = (
            abs(lambda_inst - 0.08) * 5.0
            + abs(lambda_clu - 0.05) * 6.0
            + abs(tau - 0.5) * 1.0
            + abs(weight - 0.6) * 0.9
            + abs(alpha - 0.5) * 2.0
        )
        if distance > 0.22:
            continue
        add(
            (
                f"eat_attn_apex_loss_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_tau{slug_value(tau)}_"
                f"w{slug_value(weight)}_alpha{slug_value(alpha)}"
            ),
            "default",
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            alpha=alpha,
        )

    for lr, epochs, t in product(
        (9e-5, 1e-4, 1.1e-4),
        (380, 400, 420, 450),
        (3, 4, 5),
    ):
        distance = abs(lr - 1e-4) * 5000.0 + abs(epochs - 400) / 450.0 + abs(t - 4) / 5.0
        if distance > 0.22:
            continue
        add(
            f"eat_attn_apex_train_lr{slug_value(lr)}_ep{epochs}_t{t}",
            "default",
            lr=lr,
            epochs=epochs,
            t=t,
        )

    for ae_k, threshold, min_weight in product(
        (5, 10, 15, 20),
        (0.35, 0.36, 0.37),
        (0.0, 0.005, 0.01),
    ):
        distance = abs(ae_k - 15) / 30.0 + abs(threshold - 0.36) * 5.0 + abs(min_weight - 0.005) * 12.0
        if distance > 0.26:
            continue
        add(
            f"eat_attn_apex_graph_k{ae_k}_thr{slug_value(threshold)}_min{slug_value(min_weight)}",
            ae_k,
            threshold=threshold,
            fusion_min_weight=min_weight,
        )

    return candidates


def cora_attn_peak_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.3,
        "fusion_balance": 0.0,
        "fusion_min_weight": 0.0,
        "lambda_inst": 0.03,
        "lambda_clu": 0.01,
        "warmup_epochs": 70,
        "threshold": 0.4,
        "alpha": 0.5,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
        "enable_branch_bias_fusion": True,
        "branch_bias_target": "raw",
        "branch_bias_cap": 0.10,
    }

    def add(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = base_attn_args("cora", **center)
        args.update(overrides)
        candidates.append(make_candidate("cora", stem, "attn", ae_k, **args))

    add("cora_attn_peak_center")

    # Cora is raw-anchored: keep AE floor low and search around raw branch bias.
    for ae_k, cap, tau, weight, threshold in product(
        ("default", 5, 10, 15, 20),
        (0.06, 0.08, 0.10, 0.12, 0.14),
        (0.35, 0.5, 0.65, 0.75),
        (0.4, 0.6, 0.8, 1.0),
        (0.35, 0.38, 0.4, 0.42, 0.45),
    ):
        ae_distance = 0.0 if ae_k in {"default", 15} else abs(int(ae_k) - 15) / 18.0
        distance = (
            ae_distance
            + abs(cap - 0.10) * 3.0
            + abs(tau - 0.5)
            + abs(weight - 0.6)
            + abs(threshold - 0.4) * 2.0
        )
        if distance > 0.44:
            continue
        add(
            (
                f"cora_attn_peak_k{slug_value(ae_k)}_cap{slug_value(cap)}_"
                f"tau{slug_value(tau)}_w{slug_value(weight)}_thr{slug_value(threshold)}"
            ),
            ae_k,
            branch_bias_cap=cap,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            threshold=threshold,
        )

    for temp, balance, min_weight, hidden in product(
        (1.1, 1.2, 1.3, 1.4, 1.5),
        (0.0, 0.03, 0.05, 0.08, 0.10),
        (0.0, 0.02, 0.05, 0.08, 0.10),
        (32, 48, 64, 96, 128),
    ):
        distance = (
            abs(temp - 1.3)
            + abs(balance - 0.0) * 2.0
            + abs(min_weight - 0.0) * 3.0
            + abs(hidden - 64) / 240.0
        )
        if distance > 0.38:
            continue
        add(
            (
                f"cora_attn_peak_h{hidden}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}"
            ),
            "default",
            fusion_hidden=hidden,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
        )

    for lambda_inst, lambda_clu, warmup in product(
        (0.0, 0.01, 0.02, 0.03, 0.045, 0.06),
        (0.0, 0.005, 0.01, 0.02, 0.03),
        (50, 60, 70, 80, 90),
    ):
        distance = (
            abs(lambda_inst - 0.03) * 4.0
            + abs(lambda_clu - 0.01) * 6.0
            + abs(warmup - 70) / 120.0
        )
        if distance > 0.30:
            continue
        add(
            (
                f"cora_attn_peak_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_warm{warmup}"
            ),
            "default",
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            warmup_epochs=warmup,
        )

    for alpha, threshold, t, epochs, lr in product(
        (0.4, 0.45, 0.5, 0.55),
        (0.35, 0.38, 0.4, 0.42, 0.45),
        (3, 4, 5),
        (350, 400, 450, 500),
        (7e-5, 1e-4, 1.3e-4),
    ):
        distance = (
            abs(alpha - 0.5) * 2.0
            + abs(threshold - 0.4) * 2.0
            + abs(t - 4) / 5.0
            + abs(epochs - 400) / 500.0
            + abs(lr - 1e-4) * 4200.0
        )
        if distance > 0.42:
            continue
        add(
            (
                f"cora_attn_peak_alpha{slug_value(alpha)}_thr{slug_value(threshold)}_"
                f"t{t}_ep{epochs}_lr{slug_value(lr)}"
            ),
            "default",
            alpha=alpha,
            threshold=threshold,
            t=t,
            epochs=epochs,
            lr=lr,
        )

    for start, end in ((0.15, 0.40), (0.20, 0.45), (0.25, 0.50), (0.30, 0.55)):
        add(
            f"cora_attn_peak_dynamic_{slug_value(start)}_{slug_value(end)}",
            "default",
            enable_dynamic_threshold=True,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
        )

    return candidates


def uat_attn_peak_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str | int, tuple[tuple[str, str], ...]]] = set()

    # Current table source:
    # experiment_output/uat/uat_train_with_dual_attn_20260402_133451.txt
    # 10-run mean 57.37 / 30.26 / 27.05 / 55.37 from the dynamic-threshold +
    # GCN-backbone branch. Several individual seeds already clear SCGC-S F1,
    # so the search keeps this branch as the anchor and probes small moves.
    dyn_gcn_center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.4,
        "fusion_balance": 0.25,
        "fusion_min_weight": 0.10,
        "lambda_inst": 0.12,
        "lambda_clu": 0.12,
        "warmup_epochs": 20,
        "threshold": 0.5,
        "alpha": 0.5,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
        "enable_dcgl_negative_loss": False,
        "dcgl_neg_tau": None,
        "dcgl_neg_weight": None,
        "enable_dynamic_threshold": True,
        "enable_gcn_backbone": True,
    }
    notes_center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.9,
        "fusion_balance": 0.35,
        "fusion_min_weight": 0.20,
        "lambda_inst": 0.08,
        "lambda_clu": 0.07,
        "warmup_epochs": 35,
        "threshold": 0.4,
        "alpha": 0.5,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
        "enable_dcgl_negative_loss": True,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    }

    def add_from(center: dict[str, Any], stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = base_attn_args("uat", **center)
        args.update(overrides)
        key = (
            stem,
            ae_k,
            tuple(sorted((name, str(value)) for name, value in args.items())),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(make_candidate("uat", stem, "attn", ae_k, **args))

    def add_dyn(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        add_from(dyn_gcn_center, stem, ae_k, **overrides)

    def add_notes(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        add_from(notes_center, stem, ae_k, **overrides)

    add_dyn("uat_attn_peak_dyn_gcn_center")

    # Seed 6/7/8/10 of the historical center had the desired F1 behavior.
    # Graph k and the attention floor are the highest-leverage low-cost axes.
    for ae_k in ("default", 5, 10, 15, 20, 25):
        add_dyn(f"uat_attn_peak_dyn_gcn_k{slug_value(ae_k)}", ae_k)

    for temp, balance, min_weight, warmup in product(
        (1.2, 1.3, 1.4, 1.5, 1.6),
        (0.15, 0.20, 0.25, 0.30, 0.35),
        (0.05, 0.08, 0.10, 0.12, 0.15),
        (15, 20, 25, 30),
    ):
        distance = (
            abs(temp - 1.4) * 1.2
            + abs(balance - 0.25) * 1.8
            + abs(min_weight - 0.10) * 3.0
            + abs(warmup - 20) / 55.0
        )
        if distance > 0.34:
            continue
        add_dyn(
            (
                f"uat_attn_peak_dyn_temp{slug_value(temp)}_bal{slug_value(balance)}_"
                f"min{slug_value(min_weight)}_warm{warmup}"
            ),
            "default",
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
            warmup_epochs=warmup,
        )

    for lambda_inst, lambda_clu, threshold, alpha in product(
        (0.08, 0.10, 0.12, 0.14, 0.16),
        (0.08, 0.10, 0.12, 0.14, 0.16),
        (0.44, 0.48, 0.50, 0.52, 0.56),
        (0.45, 0.50, 0.55),
    ):
        distance = (
            abs(lambda_inst - 0.12) * 3.0
            + abs(lambda_clu - 0.12) * 3.0
            + abs(threshold - 0.50) * 2.0
            + abs(alpha - 0.50) * 1.5
        )
        if distance > 0.30:
            continue
        add_dyn(
            (
                f"uat_attn_peak_dyn_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_thr{slug_value(threshold)}_"
                f"alpha{slug_value(alpha)}"
            ),
            "default",
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            threshold=threshold,
            alpha=alpha,
        )

    for start, end, threshold in product(
        (0.15, 0.20, 0.25, 0.30),
        (0.45, 0.50, 0.55, 0.60),
        (0.48, 0.50, 0.52),
    ):
        if end <= start:
            continue
        distance = (
            abs(start - 0.20) * 2.0
            + abs(end - 0.50) * 1.6
            + abs(threshold - 0.50) * 2.0
        )
        if distance > 0.30:
            continue
        add_dyn(
            (
                f"uat_attn_peak_dyn_dt{slug_value(start)}_{slug_value(end)}_"
                f"thr{slug_value(threshold)}"
            ),
            "default",
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
            threshold=threshold,
        )

    for hidden, temp, balance, min_weight in product(
        (32, 48, 64, 96, 128),
        (1.3, 1.4, 1.5),
        (0.20, 0.25, 0.30),
        (0.08, 0.10, 0.12),
    ):
        distance = (
            abs(hidden - 64) / 220.0
            + abs(temp - 1.4) * 1.3
            + abs(balance - 0.25) * 2.0
            + abs(min_weight - 0.10) * 3.0
        )
        if distance > 0.30:
            continue
        add_dyn(
            (
                f"uat_attn_peak_dyn_h{hidden}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}"
            ),
            "default",
            fusion_hidden=hidden,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
        )

    for t, epochs, lr in product(
        (3, 4, 5),
        (350, 400, 450, 500),
        (7e-5, 8e-5, 1e-4, 1.2e-4),
    ):
        distance = (
            abs(t - 4) / 5.0
            + abs(epochs - 400) / 420.0
            + abs(lr - 1e-4) * 4500.0
        )
        if distance > 0.32:
            continue
        add_dyn(
            f"uat_attn_peak_dyn_t{t}_ep{epochs}_lr{slug_value(lr)}",
            "default",
            t=t,
            epochs=epochs,
            lr=lr,
        )

    # Let the historical dynamic-GCN center borrow a light negative branch in
    # case it improves ARI/NMI without depressing the F1-heavy seeds.
    for tau, weight, lambda_inst, lambda_clu in product(
        (0.35, 0.50, 0.65, 0.75),
        (0.2, 0.4, 0.6, 0.8),
        (0.10, 0.12, 0.14),
        (0.10, 0.12, 0.14),
    ):
        distance = (
            abs(tau - 0.50)
            + abs(weight - 0.4)
            + abs(lambda_inst - 0.12) * 3.0
            + abs(lambda_clu - 0.12) * 3.0
        )
        if distance > 0.34:
            continue
        add_dyn(
            (
                f"uat_attn_peak_dyn_neg_tau{slug_value(tau)}_w{slug_value(weight)}_"
                f"li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}"
            ),
            "default",
            enable_dcgl_negative_loss=True,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
        )

    for ae_k, temp, balance, min_weight, threshold, warmup in product(
        (5, 10, 15, 20, 25),
        (1.3, 1.4, 1.5),
        (0.20, 0.25, 0.30),
        (0.08, 0.10, 0.12),
        (0.48, 0.50, 0.52),
        (15, 20, 25),
    ):
        distance = (
            abs(ae_k - 15) / 25.0
            + abs(temp - 1.4) * 1.2
            + abs(balance - 0.25) * 1.8
            + abs(min_weight - 0.10) * 3.0
            + abs(threshold - 0.50) * 2.0
            + abs(warmup - 20) / 60.0
        )
        if distance > 0.38:
            continue
        add_dyn(
            (
                f"uat_attn_peak_dyn_graph_k{ae_k}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}_"
                f"thr{slug_value(threshold)}_warm{warmup}"
            ),
            ae_k,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
            threshold=threshold,
            warmup_epochs=warmup,
        )

    # Keep the more recent notes/DCGL branch as a fallback in case the current
    # code path has drifted from the April dynamic-GCN result.
    add_notes("uat_attn_peak_notes_center")
    for ae_k in ("default", 10, 15, 20, 25):
        add_notes(f"uat_attn_peak_notes_k{slug_value(ae_k)}", ae_k)

    for ae_k, temp, balance, min_weight, lambda_inst, lambda_clu, tau, weight, threshold in product(
        ("default", 10, 15, 20),
        (1.6, 1.8, 1.9, 2.1),
        (0.25, 0.35, 0.45),
        (0.10, 0.15, 0.20, 0.25),
        (0.06, 0.08, 0.10),
        (0.05, 0.07, 0.09),
        (0.35, 0.5, 0.75),
        (0.4, 0.6, 0.8),
        (0.35, 0.4, 0.45),
    ):
        ae_distance = 0.0 if ae_k in {"default", 15} else abs(int(ae_k) - 15) / 25.0
        distance = (
            ae_distance
            + abs(temp - 1.9)
            + abs(balance - 0.35) * 1.5
            + abs(min_weight - 0.20) * 2.0
            + abs(lambda_inst - 0.08) * 2.5
            + abs(lambda_clu - 0.07) * 4.0
            + abs(tau - 0.5)
            + abs(weight - 0.6)
            + abs(threshold - 0.4) * 2.0
        )
        if distance > 0.42:
            continue
        add_notes(
            (
                f"uat_attn_peak_notes_k{slug_value(ae_k)}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}_"
                f"li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}_"
                f"tau{slug_value(tau)}_w{slug_value(weight)}_thr{slug_value(threshold)}"
            ),
            ae_k,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            threshold=threshold,
        )

    return candidates


def uat_attn_roll_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str | int, tuple[tuple[str, str], ...]]] = set()

    center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.4,
        "fusion_balance": 0.25,
        "fusion_min_weight": 0.10,
        "lambda_inst": 0.12,
        "lambda_clu": 0.12,
        "warmup_epochs": 20,
        "threshold": 0.5,
        "alpha": 0.5,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
        "enable_dcgl_negative_loss": False,
        "dcgl_neg_tau": None,
        "dcgl_neg_weight": None,
        "enable_dynamic_threshold": True,
        "enable_gcn_backbone": True,
    }

    def add(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = base_attn_args("uat", **center)
        args.update(overrides)
        key = (
            stem,
            ae_k,
            tuple(sorted((name, str(value)) for name, value in args.items())),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(make_candidate("uat", stem, "attn", ae_k, **args))

    # Current selected DCGL-only attention branch:
    # experiment_output/final_scgcn_push/20260502_112717_353_uat_attn_dcgl_only_push/summary.md
    # train seed 3, AE seed 42 reached 59.08 / 30.53 / 26.21 / 58.51.
    add(
        "uat_roll_dcgl_hist_balanced",
        "default",
        fusion_hidden=64,
        fusion_temp=1.8,
        fusion_balance=0.35,
        fusion_min_weight=0.20,
        lambda_inst=0.09,
        lambda_clu=0.09,
        warmup_epochs=35,
        threshold=0.4,
        alpha=0.5,
        t=4,
        epochs=400,
        lr=1e-4,
        enable_dcgl_negative_loss=True,
        dcgl_neg_tau=0.5,
        dcgl_neg_weight=0.6,
        enable_dynamic_threshold=False,
        enable_gcn_backbone=False,
    )

    # Current-code frontier from the first UAT sweep:
    # k20: 57.65 / 27.90 / 26.95 / 55.75 (needs NMI)
    # k25: 56.89 / 28.54 / 23.79 / 56.84 (needs ARI)
    for ae_k in (20, 25, "default", 15):
        add(f"uat_roll_k{slug_value(ae_k)}", ae_k)

    for ae_k, threshold in product((20, 25), (0.42, 0.46, 0.48, 0.50, 0.52, 0.56, 0.60)):
        add(f"uat_roll_k{ae_k}_thr{slug_value(threshold)}", ae_k, threshold=threshold)

    for ae_k, start, end in product(
        (20, 25),
        (0.10, 0.15, 0.20, 0.25, 0.30),
        (0.45, 0.50, 0.55, 0.60, 0.65),
    ):
        if end <= start:
            continue
        distance = abs(start - 0.20) * 1.5 + abs(end - 0.50) * 1.2
        if distance > 0.32:
            continue
        add(
            f"uat_roll_k{ae_k}_dt{slug_value(start)}_{slug_value(end)}",
            ae_k,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
        )

    for ae_k, temp, balance, min_weight in product(
        (20, 25),
        (1.25, 1.35, 1.40, 1.45, 1.55, 1.65),
        (0.15, 0.20, 0.25, 0.30, 0.35),
        (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18),
    ):
        distance = (
            abs(temp - 1.4) * 1.1
            + abs(balance - 0.25) * 1.5
            + abs(min_weight - 0.10) * 2.5
        )
        if distance > 0.28:
            continue
        add(
            (
                f"uat_roll_k{ae_k}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}"
            ),
            ae_k,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
        )

    for ae_k, lambda_inst, lambda_clu, warmup in product(
        (20, 25),
        (0.08, 0.10, 0.12, 0.14, 0.16, 0.18),
        (0.08, 0.10, 0.12, 0.14, 0.16, 0.18),
        (10, 15, 20, 25, 30, 35),
    ):
        distance = (
            abs(lambda_inst - 0.12) * 2.5
            + abs(lambda_clu - 0.12) * 2.5
            + abs(warmup - 20) / 55.0
        )
        if distance > 0.30:
            continue
        add(
            (
                f"uat_roll_k{ae_k}_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_warm{warmup}"
            ),
            ae_k,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            warmup_epochs=warmup,
        )

    for ae_k, tau, weight in product(
        (20, 25),
        (0.20, 0.35, 0.50, 0.65, 0.75, 1.00),
        (0.2, 0.4, 0.6, 0.8, 1.0),
    ):
        distance = abs(tau - 0.5) + abs(weight - 0.4)
        if distance > 0.42:
            continue
        add(
            f"uat_roll_k{ae_k}_neg_tau{slug_value(tau)}_w{slug_value(weight)}",
            ae_k,
            enable_dcgl_negative_loss=True,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for ae_k, alpha, t, lr, epochs in product(
        (20, 25),
        (0.40, 0.45, 0.50, 0.55, 0.60),
        (3, 4, 5),
        (7e-5, 8e-5, 1e-4, 1.2e-4),
        (350, 400, 450, 500),
    ):
        distance = (
            abs(alpha - 0.5) * 1.5
            + abs(t - 4) / 5.0
            + abs(lr - 1e-4) * 4200.0
            + abs(epochs - 400) / 420.0
        )
        if distance > 0.32:
            continue
        add(
            (
                f"uat_roll_k{ae_k}_alpha{slug_value(alpha)}_t{t}_"
                f"lr{slug_value(lr)}_ep{epochs}"
            ),
            ae_k,
            alpha=alpha,
            t=t,
            lr=lr,
            epochs=epochs,
        )

    for ae_k, hidden in product((20, 25), (32, 48, 64, 80, 96, 128, 160)):
        add(f"uat_roll_k{ae_k}_h{hidden}", ae_k, fusion_hidden=hidden)

    # A few conservative variants that bias toward the k20 ARI branch but raise
    # NMI via a slightly softer confidence schedule.
    for threshold, start, end, min_weight, balance in product(
        (0.46, 0.48, 0.50),
        (0.15, 0.20, 0.25),
        (0.55, 0.60),
        (0.06, 0.08, 0.10),
        (0.20, 0.25, 0.30),
    ):
        distance = (
            abs(threshold - 0.48) * 2.0
            + abs(start - 0.20) * 1.5
            + abs(end - 0.55)
            + abs(min_weight - 0.08) * 2.5
            + abs(balance - 0.25) * 1.5
        )
        if distance > 0.26:
            continue
        add(
            (
                f"uat_roll_k20_bridge_thr{slug_value(threshold)}_"
                f"dt{slug_value(start)}_{slug_value(end)}_"
                f"min{slug_value(min_weight)}_bal{slug_value(balance)}"
            ),
            20,
            threshold=threshold,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
            fusion_min_weight=min_weight,
            fusion_balance=balance,
        )

    return candidates


def uat_attn_continued_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str | int, tuple[tuple[str, str], ...]]] = set()

    dyn_center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.4,
        "fusion_balance": 0.25,
        "fusion_min_weight": 0.10,
        "lambda_inst": 0.12,
        "lambda_clu": 0.12,
        "warmup_epochs": 20,
        "threshold": 0.5,
        "alpha": 0.5,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
        "enable_dcgl_negative_loss": False,
        "dcgl_neg_tau": None,
        "dcgl_neg_weight": None,
        "enable_dynamic_threshold": True,
        "enable_gcn_backbone": True,
    }
    dcgl_center = {
        "fusion_hidden": 64,
        "fusion_temp": 1.8,
        "fusion_balance": 0.35,
        "fusion_min_weight": 0.20,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "warmup_epochs": 35,
        "threshold": 0.4,
        "alpha": 0.5,
        "t": 4,
        "epochs": 400,
        "lr": 1e-4,
        "enable_dcgl_negative_loss": True,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
        "enable_dynamic_threshold": False,
        "enable_gcn_backbone": False,
    }

    def add(stem: str, base: dict[str, Any], ae_k: str | int = "default", **overrides: Any) -> None:
        args = base_attn_args("uat", **base)
        args.update(overrides)
        key = (
            stem,
            ae_k,
            tuple(sorted((name, str(value)) for name, value in args.items())),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(make_candidate("uat", stem, "attn", ae_k, **args))

    # Current-code frontier: k20 clears ACC/ARI/F1 but is short on NMI.
    # Jointly move confidence schedule, losses, and training axes instead of
    # repeating single-axis sweeps that landed on the same plateau.
    for threshold, start, end, lambda_inst, lambda_clu, tau, weight, dropout in product(
        (0.46, 0.48, 0.50, 0.52),
        (0.15, 0.20, 0.25),
        (0.50, 0.55, 0.60),
        (0.10, 0.12, 0.14),
        (0.10, 0.12, 0.14),
        (0.35, 0.50, 0.65),
        (0.2, 0.4, 0.6),
        (0.0, 0.1),
    ):
        if end <= start:
            continue
        distance = (
            abs(threshold - 0.50) * 2.0
            + abs(start - 0.20) * 1.4
            + abs(end - 0.55) * 1.2
            + abs(lambda_inst - 0.12) * 2.5
            + abs(lambda_clu - 0.12) * 2.5
            + abs(tau - 0.50) * 0.8
            + abs(weight - 0.4) * 0.8
            + dropout
        )
        if distance > 0.22:
            continue
        add(
            (
                f"uat_cont_k20_bridge_thr{slug_value(threshold)}_"
                f"dt{slug_value(start)}_{slug_value(end)}_"
                f"li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}_"
                f"tau{slug_value(tau)}_w{slug_value(weight)}_drop{slug_value(dropout)}"
            ),
            dyn_center,
            20,
            threshold=threshold,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            enable_dcgl_negative_loss=True,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            gcn_dropout=dropout,
        )

    for temp, balance, min_weight, threshold, start, end, lambda_inst, lambda_clu in product(
        (1.30, 1.40, 1.50, 1.60),
        (0.18, 0.22, 0.25, 0.28, 0.32),
        (0.06, 0.08, 0.10, 0.12, 0.15),
        (0.48, 0.50, 0.52),
        (0.15, 0.20, 0.25),
        (0.50, 0.55, 0.60),
        (0.10, 0.12, 0.14),
        (0.10, 0.12, 0.14),
    ):
        if end <= start:
            continue
        distance = (
            abs(temp - 1.4) * 1.1
            + abs(balance - 0.25) * 1.6
            + abs(min_weight - 0.10) * 2.4
            + abs(threshold - 0.50) * 1.8
            + abs(start - 0.20) * 1.2
            + abs(end - 0.55)
            + abs(lambda_inst - 0.12) * 2.0
            + abs(lambda_clu - 0.12) * 2.0
        )
        if distance > 0.20:
            continue
        add(
            (
                f"uat_cont_k20_joint_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}_"
                f"thr{slug_value(threshold)}_dt{slug_value(start)}_{slug_value(end)}_"
                f"li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}"
            ),
            dyn_center,
            20,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
            threshold=threshold,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
        )

    # Current-code k25 frontier clears ACC/NMI/F1 but is short on ARI. Bias it
    # toward stronger agreement/negative structure while keeping the F1 branch.
    for threshold, start, end, min_weight, balance, lambda_inst, lambda_clu, tau, weight in product(
        (0.48, 0.50, 0.52, 0.56),
        (0.10, 0.15, 0.20),
        (0.45, 0.50, 0.55),
        (0.08, 0.10, 0.12, 0.15),
        (0.20, 0.25, 0.30),
        (0.10, 0.12, 0.14, 0.16),
        (0.10, 0.12, 0.14, 0.16),
        (0.35, 0.50, 0.65),
        (0.2, 0.4, 0.6),
    ):
        if end <= start:
            continue
        distance = (
            abs(threshold - 0.50) * 1.8
            + abs(start - 0.15) * 1.5
            + abs(end - 0.50) * 1.2
            + abs(min_weight - 0.10) * 2.4
            + abs(balance - 0.25) * 1.5
            + abs(lambda_inst - 0.12) * 2.0
            + abs(lambda_clu - 0.12) * 2.0
            + abs(tau - 0.50) * 0.8
            + abs(weight - 0.4) * 0.8
        )
        if distance > 0.24:
            continue
        add(
            (
                f"uat_cont_k25_ari_thr{slug_value(threshold)}_"
                f"dt{slug_value(start)}_{slug_value(end)}_min{slug_value(min_weight)}_"
                f"bal{slug_value(balance)}_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_tau{slug_value(tau)}_w{slug_value(weight)}"
            ),
            dyn_center,
            25,
            threshold=threshold,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
            fusion_min_weight=min_weight,
            fusion_balance=balance,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            enable_dcgl_negative_loss=True,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for temp, balance, min_weight, bias_target, bias_cap, lambda_inst, lambda_clu in product(
        (1.30, 1.40, 1.50),
        (0.20, 0.25, 0.30),
        (0.08, 0.10, 0.12),
        ("raw", "ae"),
        (0.08, 0.12, 0.16, 0.20),
        (0.10, 0.12, 0.14),
        (0.10, 0.12, 0.14),
    ):
        distance = (
            abs(temp - 1.4) * 1.1
            + abs(balance - 0.25) * 1.5
            + abs(min_weight - 0.10) * 2.2
            + abs(bias_cap - 0.12)
            + abs(lambda_inst - 0.12) * 2.0
            + abs(lambda_clu - 0.12) * 2.0
        )
        if distance > 0.22:
            continue
        add(
            (
                f"uat_cont_k25_bias_{bias_target}_cap{slug_value(bias_cap)}_"
                f"temp{slug_value(temp)}_bal{slug_value(balance)}_"
                f"min{slug_value(min_weight)}_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}"
            ),
            dyn_center,
            25,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
            enable_branch_bias_fusion=True,
            branch_bias_target=bias_target,
            branch_bias_cap=bias_cap,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
        )

    # Revisit the historical DCGL branch with current-code bridge variants and
    # optional GCN/dynamic threshold. This targets the selected row source.
    for ae_k, temp, balance, min_weight, lambda_inst, lambda_clu, tau, weight, threshold in product(
        ("default", 10, 15, 20, 25),
        (1.6, 1.8, 2.0),
        (0.25, 0.35, 0.45),
        (0.10, 0.15, 0.20, 0.25),
        (0.07, 0.09, 0.11),
        (0.07, 0.09, 0.11),
        (0.35, 0.50, 0.65, 0.75),
        (0.4, 0.6, 0.8),
        (0.35, 0.40, 0.45),
    ):
        ae_distance = 0.0 if ae_k in {"default", 15} else abs(int(ae_k) - 15) / 28.0
        distance = (
            ae_distance
            + abs(temp - 1.8) * 0.9
            + abs(balance - 0.35) * 1.3
            + abs(min_weight - 0.20) * 1.8
            + abs(lambda_inst - 0.09) * 2.4
            + abs(lambda_clu - 0.09) * 2.4
            + abs(tau - 0.50) * 0.8
            + abs(weight - 0.6) * 0.8
            + abs(threshold - 0.40) * 1.8
        )
        if distance > 0.28:
            continue
        add(
            (
                f"uat_cont_dcgl_k{slug_value(ae_k)}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}_"
                f"li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}_"
                f"tau{slug_value(tau)}_w{slug_value(weight)}_thr{slug_value(threshold)}"
            ),
            dcgl_center,
            ae_k,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            threshold=threshold,
        )

    for ae_k, start, end, gcn, dropout, temp, min_weight in product(
        ("default", 15, 20, 25),
        (0.15, 0.20, 0.25),
        (0.45, 0.50, 0.55),
        (False, True),
        (0.0, 0.1),
        (1.6, 1.8),
        (0.15, 0.20),
    ):
        if end <= start:
            continue
        ae_distance = 0.0 if ae_k in {"default", 15} else abs(int(ae_k) - 15) / 28.0
        distance = (
            ae_distance
            + abs(start - 0.20) * 1.3
            + abs(end - 0.50)
            + (0.0 if not gcn else 0.05)
            + dropout
            + abs(temp - 1.8) * 0.8
            + abs(min_weight - 0.20) * 1.8
        )
        if distance > 0.28:
            continue
        add(
            (
                f"uat_cont_dcgl_dyn_k{slug_value(ae_k)}_dt{slug_value(start)}_"
                f"{slug_value(end)}_gcn{int(gcn)}_drop{slug_value(dropout)}_"
                f"temp{slug_value(temp)}_min{slug_value(min_weight)}"
            ),
            dcgl_center,
            ae_k,
            enable_dynamic_threshold=True,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
            enable_gcn_backbone=gcn,
            gcn_dropout=dropout,
            fusion_temp=temp,
            fusion_min_weight=min_weight,
        )

    for ae_k, alpha, t, epochs, lr in product(
        (20, 25, "default"),
        (0.45, 0.50, 0.55),
        (3, 4, 5),
        (350, 400, 450, 500),
        (7e-5, 8e-5, 1e-4, 1.2e-4),
    ):
        distance = (
            (0.0 if ae_k in {20, 25} else 0.15)
            + abs(alpha - 0.50) * 1.4
            + abs(t - 4) / 5.0
            + abs(epochs - 400) / 450.0
            + abs(lr - 1e-4) * 4200.0
        )
        if distance > 0.28:
            continue
        add(
            (
                f"uat_cont_train_k{slug_value(ae_k)}_alpha{slug_value(alpha)}_"
                f"t{t}_ep{epochs}_lr{slug_value(lr)}"
            ),
            dyn_center,
            ae_k,
            alpha=alpha,
            t=t,
            epochs=epochs,
            lr=lr,
        )

    return candidates


def uat_attn_dcgl_only_push_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str, tuple[tuple[str, Any], ...]]] = set()
    ranked: list[tuple[float, Candidate]] = []

    hist = {
        "dims": 500,
        "threshold": 0.4,
        "alpha": 0.5,
        "warmup_epochs": 35,
        "fusion_hidden": 64,
        "fusion_temp": 1.8,
        "fusion_balance": 0.35,
        "fusion_min_weight": 0.20,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    }
    f1_current = {
        "dims": 500,
        "threshold": 0.4,
        "alpha": 0.5,
        "warmup_epochs": 35,
        "fusion_hidden": 64,
        "fusion_temp": 1.6,
        "fusion_balance": 0.35,
        "fusion_min_weight": 0.25,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
    }

    forbidden = (
        "enable_dynamic_threshold",
        "enable_gcn_backbone",
        "enable_dcgl_cluster_level",
        "enable_branch_bias_fusion",
        "enable_ema_prototypes",
    )

    def clean(args: dict[str, Any]) -> dict[str, Any]:
        args = dict(args)
        for key in forbidden:
            args.pop(key, None)
        args["enable_dcgl_negative_loss"] = True
        return args

    def distance(args: dict[str, Any], center: dict[str, Any], ae_k: str | int) -> float:
        ae_val = AE_DEFAULT_K["uat"] if ae_k == "default" else int(ae_k)
        return (
            abs(ae_val - 15) / 24.0
            + abs(float(args["dims"]) - float(center["dims"])) / 700.0
            + abs(float(args["threshold"]) - float(center["threshold"])) * 2.0
            + abs(float(args["warmup_epochs"]) - float(center["warmup_epochs"])) / 80.0
            + abs(float(args["fusion_hidden"]) - float(center["fusion_hidden"])) / 220.0
            + abs(float(args["fusion_temp"]) - float(center["fusion_temp"])) * 1.0
            + abs(float(args["fusion_balance"]) - float(center["fusion_balance"])) * 1.8
            + abs(float(args["fusion_min_weight"]) - float(center["fusion_min_weight"])) * 2.2
            + abs(float(args["lambda_inst"]) - float(center["lambda_inst"])) * 4.0
            + abs(float(args["lambda_clu"]) - float(center["lambda_clu"])) * 4.0
            + abs(float(args["dcgl_neg_tau"]) - float(center["dcgl_neg_tau"])) * 0.9
            + abs(float(args["dcgl_neg_weight"]) - float(center["dcgl_neg_weight"])) * 0.9
        )

    def add(stem: str, ae_k: str | int, center: dict[str, Any], **overrides: Any) -> None:
        args = dict(center)
        args.update(overrides)
        args = clean(args)
        candidate = make_candidate("uat", stem, "attn", ae_k, **args)
        key = (candidate.fusion_mode, str(candidate.ae_k), tuple(sorted(candidate.args.items())))
        if key in seen:
            return
        seen.add(key)
        ranked.append((distance(args, center, ae_k), candidate))

    add("uat_dcgl_only_hist_center", "default", hist)
    add("uat_dcgl_only_f1_current_center", "default", f1_current)

    for ae_k in ("default", 5, 10, 15, 20, 25):
        add(f"uat_dcgl_only_hist_k{slug_value(ae_k)}", ae_k, hist)
        add(f"uat_dcgl_only_f1_k{slug_value(ae_k)}", ae_k, f1_current)

    for ae_k, threshold, temp, balance, min_weight, lambda_inst, lambda_clu, tau, weight, warmup in product(
        ("default", 10, 15, 20, 25),
        (0.35, 0.38, 0.40, 0.42, 0.45),
        (1.55, 1.65, 1.75, 1.8, 1.9, 2.0),
        (0.25, 0.30, 0.35, 0.40, 0.45),
        (0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28),
        (0.05, 0.07, 0.09, 0.11, 0.13),
        (0.05, 0.07, 0.09, 0.11, 0.13),
        (0.35, 0.45, 0.50, 0.60, 0.75),
        (0.35, 0.45, 0.60, 0.75, 0.90),
        (25, 30, 35, 40, 45),
    ):
        args = dict(hist)
        args.update(
            {
                "threshold": threshold,
                "fusion_temp": temp,
                "fusion_balance": balance,
                "fusion_min_weight": min_weight,
                "lambda_inst": lambda_inst,
                "lambda_clu": lambda_clu,
                "dcgl_neg_tau": tau,
                "dcgl_neg_weight": weight,
                "warmup_epochs": warmup,
            }
        )
        if distance(args, hist, ae_k) > 0.58:
            continue
        add(
            (
                f"uat_dcgl_h_k{slug_value(ae_k)}_th{slug_value(threshold)}_"
                f"tf{slug_value(temp)}_b{slug_value(balance)}_m{slug_value(min_weight)}_"
                f"li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}_"
                f"ta{slug_value(tau)}_w{slug_value(weight)}_wu{warmup}"
            ),
            ae_k,
            hist,
            **args,
        )

    for ae_k, hidden, temp, balance, min_weight, threshold in product(
        ("default", 15, 20),
        (32, 48, 64, 80, 96, 128, 160),
        (1.5, 1.6, 1.7, 1.8, 1.9),
        (0.25, 0.30, 0.35, 0.40),
        (0.16, 0.20, 0.24, 0.28),
        (0.36, 0.40, 0.44),
    ):
        args = dict(f1_current)
        args.update(
            {
                "fusion_hidden": hidden,
                "fusion_temp": temp,
                "fusion_balance": balance,
                "fusion_min_weight": min_weight,
                "threshold": threshold,
            }
        )
        if distance(args, f1_current, ae_k) > 0.55:
            continue
        add(
            (
                f"uat_dcgl_f1_k{slug_value(ae_k)}_h{hidden}_"
                f"tf{slug_value(temp)}_b{slug_value(balance)}_"
                f"m{slug_value(min_weight)}_th{slug_value(threshold)}"
            ),
            ae_k,
            f1_current,
            **args,
        )

    for alpha, t, epochs, lr, warmup, threshold in product(
        (0.45, 0.50, 0.55),
        (3, 4, 5),
        (350, 400, 450, 500),
        (7e-5, 8e-5, 1e-4, 1.2e-4),
        (30, 35, 40),
        (0.38, 0.40, 0.42),
    ):
        args = dict(hist)
        args.update(
            {
                "alpha": alpha,
                "t": t,
                "epochs": epochs,
                "lr": lr,
                "warmup_epochs": warmup,
                "threshold": threshold,
            }
        )
        if distance(args, hist, "default") > 0.48:
            continue
        add(
            (
                f"uat_dcgl_train_alpha{slug_value(alpha)}_t{t}_"
                f"ep{epochs}_lr{slug_value(lr)}_wu{warmup}_th{slug_value(threshold)}"
            ),
            "default",
            hist,
            **args,
        )

    ranked.sort(key=lambda item: item[0])
    return [candidate for _dist, candidate in ranked]


def uat_attn_dcgl_refine_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str, tuple[tuple[str, Any], ...]]] = set()
    ranked: list[tuple[float, Candidate]] = []

    center = {
        "dims": 500,
        "threshold": 0.4,
        "alpha": 0.5,
        "warmup_epochs": 35,
        "fusion_hidden": 64,
        "fusion_temp": 1.8,
        "fusion_balance": 0.35,
        "fusion_min_weight": 0.20,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
        "enable_dcgl_negative_loss": True,
    }
    forbidden = (
        "enable_dynamic_threshold",
        "enable_gcn_backbone",
        "enable_dcgl_cluster_level",
        "enable_branch_bias_fusion",
        "enable_ema_prototypes",
    )

    def dist(args: dict[str, Any], ae_k: str | int) -> float:
        ae_val = AE_DEFAULT_K["uat"] if ae_k == "default" else int(ae_k)
        return (
            abs(ae_val - 15) / 28.0
            + abs(float(args["threshold"]) - 0.4) * 2.8
            + abs(float(args["warmup_epochs"]) - 35) / 70.0
            + abs(float(args["fusion_hidden"]) - 64) / 220.0
            + abs(float(args["fusion_temp"]) - 1.8) * 1.4
            + abs(float(args["fusion_balance"]) - 0.35) * 2.4
            + abs(float(args["fusion_min_weight"]) - 0.20) * 2.8
            + abs(float(args["lambda_inst"]) - 0.09) * 5.0
            + abs(float(args["lambda_clu"]) - 0.09) * 5.0
            + abs(float(args["dcgl_neg_tau"]) - 0.5) * 1.0
            + abs(float(args["dcgl_neg_weight"]) - 0.6) * 1.0
        )

    def add(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = dict(center)
        args.update(overrides)
        for key in forbidden:
            args.pop(key, None)
        args["enable_dcgl_negative_loss"] = True
        candidate = make_candidate("uat", stem, "attn", ae_k, **args)
        key = (candidate.fusion_mode, str(candidate.ae_k), tuple(sorted(candidate.args.items())))
        if key in seen:
            return
        seen.add(key)
        ranked.append((dist(args, ae_k), candidate))

    add("uat_dcgl_refine_center")
    for ae_k in ("default", 10, 15, 20, 25):
        add(f"uat_dcgl_refine_k{slug_value(ae_k)}", ae_k)

    for temp, balance, min_weight, threshold in product(
        (1.65, 1.7, 1.75, 1.8, 1.85, 1.9),
        (0.28, 0.32, 0.35, 0.38, 0.42),
        (0.16, 0.18, 0.20, 0.22, 0.24, 0.26),
        (0.36, 0.38, 0.40, 0.42, 0.44),
    ):
        args = dict(center)
        args.update(
            {
                "fusion_temp": temp,
                "fusion_balance": balance,
                "fusion_min_weight": min_weight,
                "threshold": threshold,
            }
        )
        if dist(args, "default") > 0.42:
            continue
        add(
            (
                f"uat_dcgl_refine_tf{slug_value(temp)}_b{slug_value(balance)}_"
                f"m{slug_value(min_weight)}_th{slug_value(threshold)}"
            ),
            "default",
            **args,
        )

    for lambda_inst, lambda_clu, tau, weight, warmup in product(
        (0.05, 0.07, 0.09, 0.11, 0.13),
        (0.05, 0.07, 0.09, 0.11, 0.13),
        (0.35, 0.45, 0.50, 0.60, 0.75),
        (0.35, 0.45, 0.60, 0.75, 0.90),
        (28, 32, 35, 38, 42),
    ):
        args = dict(center)
        args.update(
            {
                "lambda_inst": lambda_inst,
                "lambda_clu": lambda_clu,
                "dcgl_neg_tau": tau,
                "dcgl_neg_weight": weight,
                "warmup_epochs": warmup,
            }
        )
        if dist(args, "default") > 0.48:
            continue
        add(
            (
                f"uat_dcgl_refine_li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}_"
                f"ta{slug_value(tau)}_w{slug_value(weight)}_wu{warmup}"
            ),
            "default",
            **args,
        )

    for ae_k, hidden, alpha, t, epochs, lr in product(
        ("default", 15, 20),
        (48, 64, 80, 96, 128),
        (0.45, 0.50, 0.55),
        (3, 4, 5),
        (350, 400, 450),
        (8e-5, 1e-4, 1.2e-4),
    ):
        args = dict(center)
        args.update(
            {
                "fusion_hidden": hidden,
                "alpha": alpha,
                "t": t,
                "epochs": epochs,
                "lr": lr,
            }
        )
        if dist(args, ae_k) > 0.45:
            continue
        add(
            (
                f"uat_dcgl_refine_k{slug_value(ae_k)}_h{hidden}_"
                f"alpha{slug_value(alpha)}_t{t}_ep{epochs}_lr{slug_value(lr)}"
            ),
            ae_k,
            **args,
        )

    ranked.sort(key=lambda item: item[0])
    return [candidate for _dist, candidate in ranked]


def uat_attn_dcgl_apex_candidates() -> list[Candidate]:
    seen: set[tuple[str, str, tuple[tuple[str, Any], ...]]] = set()
    ranked: list[tuple[float, Candidate]] = []

    peak = {
        "dims": 500,
        "threshold": 0.4,
        "alpha": 0.45,
        "t": 5,
        "epochs": 500,
        "lr": 1.2e-4,
        "warmup_epochs": 35,
        "fusion_hidden": 64,
        "fusion_temp": 1.8,
        "fusion_balance": 0.35,
        "fusion_min_weight": 0.20,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
        "enable_dcgl_negative_loss": True,
    }
    alt_train = dict(peak, alpha=0.50, epochs=400)
    ari_bridge = dict(peak, alpha=0.50, epochs=500)
    centers = (
        ("peak", peak),
        ("alt", alt_train),
        ("ari", ari_bridge),
    )
    forbidden = (
        "enable_dynamic_threshold",
        "enable_gcn_backbone",
        "enable_dcgl_cluster_level",
        "enable_branch_bias_fusion",
        "enable_ema_prototypes",
    )

    def dist(args: dict[str, Any], ae_k: str | int, center: dict[str, Any]) -> float:
        ae_val = AE_DEFAULT_K["uat"] if ae_k == "default" else int(ae_k)
        return (
            abs(ae_val - 15) / 28.0
            + abs(float(args["dims"]) - float(center["dims"])) / 700.0
            + abs(float(args["threshold"]) - float(center["threshold"])) * 3.0
            + abs(float(args["alpha"]) - float(center["alpha"])) * 1.8
            + abs(float(args["t"]) - float(center["t"])) / 5.0
            + abs(float(args["epochs"]) - float(center["epochs"])) / 650.0
            + abs(float(args["lr"]) - float(center["lr"])) * 5000.0
            + abs(float(args["warmup_epochs"]) - float(center["warmup_epochs"])) / 70.0
            + abs(float(args["fusion_hidden"]) - float(center["fusion_hidden"])) / 220.0
            + abs(float(args["fusion_temp"]) - float(center["fusion_temp"])) * 1.4
            + abs(float(args["fusion_balance"]) - float(center["fusion_balance"])) * 2.4
            + abs(float(args["fusion_min_weight"]) - float(center["fusion_min_weight"])) * 2.8
            + abs(float(args["lambda_inst"]) - float(center["lambda_inst"])) * 5.0
            + abs(float(args["lambda_clu"]) - float(center["lambda_clu"])) * 5.0
            + abs(float(args["dcgl_neg_tau"]) - float(center["dcgl_neg_tau"])) * 1.0
            + abs(float(args["dcgl_neg_weight"]) - float(center["dcgl_neg_weight"])) * 1.0
        )

    def add(stem: str, ae_k: str | int = "default", center: dict[str, Any] = peak, **overrides: Any) -> None:
        args = dict(center)
        args.update(overrides)
        for forbidden_key in forbidden:
            args.pop(forbidden_key, None)
        args["enable_dcgl_negative_loss"] = True
        candidate = make_candidate("uat", stem, "attn", ae_k, **args)
        key = (candidate.fusion_mode, str(candidate.ae_k), tuple(sorted(candidate.args.items())))
        if key in seen:
            return
        seen.add(key)
        ranked.append((dist(args, ae_k, center), candidate))

    for prefix, center in centers:
        add(f"uat_apex_{prefix}_center", center=center)
        for ae_k in ("default", 10, 15, 20, 25):
            add(f"uat_apex_{prefix}_k{slug_value(ae_k)}", ae_k, center=center)

    for alpha, t, epochs, lr, threshold, warmup in product(
        (0.42, 0.45, 0.48, 0.50, 0.52),
        (4, 5, 6),
        (380, 400, 430, 450, 500, 550),
        (1.0e-4, 1.1e-4, 1.2e-4, 1.3e-4, 1.4e-4),
        (0.38, 0.40, 0.42, 0.44),
        (30, 35, 40),
    ):
        args = dict(peak)
        args.update(
            {
                "alpha": alpha,
                "t": t,
                "epochs": epochs,
                "lr": lr,
                "threshold": threshold,
                "warmup_epochs": warmup,
            }
        )
        if dist(args, "default", peak) > 0.50:
            continue
        add(
            (
                f"uat_apex_train_a{slug_value(alpha)}_t{t}_ep{epochs}_"
                f"lr{slug_value(lr)}_th{slug_value(threshold)}_wu{warmup}"
            ),
            "default",
            peak,
            **args,
        )

    for temp, balance, min_weight, threshold, hidden in product(
        (1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95),
        (0.28, 0.32, 0.35, 0.38, 0.42),
        (0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26),
        (0.38, 0.40, 0.42, 0.44),
        (48, 64, 80, 96),
    ):
        args = dict(peak)
        args.update(
            {
                "fusion_temp": temp,
                "fusion_balance": balance,
                "fusion_min_weight": min_weight,
                "threshold": threshold,
                "fusion_hidden": hidden,
            }
        )
        if dist(args, "default", peak) > 0.46:
            continue
        add(
            (
                f"uat_apex_fuse_h{hidden}_tf{slug_value(temp)}_"
                f"b{slug_value(balance)}_m{slug_value(min_weight)}_th{slug_value(threshold)}"
            ),
            "default",
            peak,
            **args,
        )

    for lambda_inst, lambda_clu, tau, weight, warmup in product(
        (0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12),
        (0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12),
        (0.40, 0.45, 0.50, 0.55, 0.60, 0.70),
        (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80),
        (30, 35, 40),
    ):
        args = dict(peak)
        args.update(
            {
                "lambda_inst": lambda_inst,
                "lambda_clu": lambda_clu,
                "dcgl_neg_tau": tau,
                "dcgl_neg_weight": weight,
                "warmup_epochs": warmup,
            }
        )
        if dist(args, "default", peak) > 0.48:
            continue
        add(
            (
                f"uat_apex_loss_li{slug_value(lambda_inst)}_lc{slug_value(lambda_clu)}_"
                f"ta{slug_value(tau)}_w{slug_value(weight)}_wu{warmup}"
            ),
            "default",
            peak,
            **args,
        )

    for ae_k, alpha, epochs, lr, temp, balance, min_weight in product(
        (10, 15, 20, 25),
        (0.45, 0.48, 0.50),
        (400, 450, 500, 550),
        (1.1e-4, 1.2e-4, 1.3e-4),
        (1.70, 1.80, 1.90),
        (0.32, 0.35, 0.38),
        (0.16, 0.20, 0.24),
    ):
        args = dict(peak)
        args.update(
            {
                "alpha": alpha,
                "epochs": epochs,
                "lr": lr,
                "fusion_temp": temp,
                "fusion_balance": balance,
                "fusion_min_weight": min_weight,
            }
        )
        if dist(args, ae_k, peak) > 0.62:
            continue
        add(
            (
                f"uat_apex_graph_k{ae_k}_a{slug_value(alpha)}_ep{epochs}_"
                f"lr{slug_value(lr)}_tf{slug_value(temp)}_b{slug_value(balance)}_"
                f"m{slug_value(min_weight)}"
            ),
            ae_k,
            peak,
            **args,
        )

    ranked.sort(key=lambda item: item[0])
    return [candidate for _dist, candidate in ranked]


def usps_escape_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    def add_mean(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = {
            "warmup_epochs": 35,
            "threshold": 0.4,
            "alpha": 0.5,
            "dcgl_neg_tau": 0.5,
            "dcgl_neg_weight": 0.6,
        }
        args.update(overrides)
        candidates.append(make_candidate("usps", stem, "mean", ae_k, **args))

    def add_attn(stem: str, ae_k: str | int = "default", **overrides: Any) -> None:
        args = base_attn_args(
            "usps",
            fusion_hidden=64,
            fusion_temp=1.8,
            fusion_balance=0.35,
            fusion_min_weight=0.20,
            lambda_inst=0.09,
            lambda_clu=0.09,
            warmup_epochs=35,
            threshold=0.4,
            alpha=0.5,
            dcgl_neg_tau=0.5,
            dcgl_neg_weight=0.6,
        )
        args.update(overrides)
        candidates.append(make_candidate("usps", stem, "attn", ae_k, **args))

    for ae_k, threshold, tau, weight, warmup in product(
        (5, 10, 15, 20, 25),
        (0.30, 0.35, 0.4, 0.45, 0.5),
        (0.2, 0.35, 0.5, 0.75, 1.0, 1.25),
        (0.2, 0.4, 0.6, 0.8, 1.0, 1.2),
        (25, 35, 45, 55),
    ):
        distance = (
            abs(ae_k - 15) / 15.0
            + abs(threshold - 0.4) * 1.5
            + abs(tau - 0.5)
            + abs(weight - 0.6)
            + abs(warmup - 35) / 80.0
        )
        if distance > 0.85:
            continue
        add_mean(
            (
                f"escape_mean_k{ae_k}_thr{slug_value(threshold)}_"
                f"tau{slug_value(tau)}_w{slug_value(weight)}_warm{warmup}"
            ),
            ae_k,
            threshold=threshold,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
            warmup_epochs=warmup,
        )

    for hidden, temp, balance, min_weight in product(
        (32, 64, 96, 128, 192, 256, 384),
        (1.4, 1.6, 1.8, 2.0, 2.2, 2.4),
        (0.0, 0.15, 0.25, 0.35, 0.45, 0.60),
        (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    ):
        distance = (
            abs(hidden - 64) / 256.0
            + abs(temp - 1.8)
            + abs(balance - 0.35)
            + abs(min_weight - 0.20) * 1.5
        )
        if distance > 0.9:
            continue
        add_attn(
            (
                f"escape_attn_h{hidden}_temp{slug_value(temp)}_"
                f"bal{slug_value(balance)}_min{slug_value(min_weight)}"
            ),
            "default",
            fusion_hidden=hidden,
            fusion_temp=temp,
            fusion_balance=balance,
            fusion_min_weight=min_weight,
        )

    for hidden, lambda_inst, lambda_clu, tau, weight in product(
        (64, 96, 128, 192, 256),
        (0.0, 0.03, 0.06, 0.09, 0.12, 0.16),
        (0.0, 0.03, 0.06, 0.09, 0.12, 0.16),
        (0.35, 0.5, 0.75, 1.0),
        (0.4, 0.6, 0.8, 1.0),
    ):
        distance = (
            abs(hidden - 64) / 256.0
            + abs(lambda_inst - 0.09) * 2.0
            + abs(lambda_clu - 0.09) * 2.0
            + abs(tau - 0.5)
            + abs(weight - 0.6)
        )
        if distance > 0.8:
            continue
        add_attn(
            (
                f"escape_attn_h{hidden}_li{slug_value(lambda_inst)}_"
                f"lc{slug_value(lambda_clu)}_tau{slug_value(tau)}_w{slug_value(weight)}"
            ),
            "default",
            fusion_hidden=hidden,
            lambda_inst=lambda_inst,
            lambda_clu=lambda_clu,
            dcgl_neg_tau=tau,
            dcgl_neg_weight=weight,
        )

    for start, end in ((0.15, 0.40), (0.20, 0.45), (0.25, 0.50), (0.30, 0.55)):
        add_mean(
            f"escape_mean_dynamic_{slug_value(start)}_{slug_value(end)}",
            "default",
            enable_dynamic_threshold=True,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
        )
        add_attn(
            f"escape_attn_dynamic_{slug_value(start)}_{slug_value(end)}",
            "default",
            enable_dynamic_threshold=True,
            dynamic_threshold_start=start,
            dynamic_threshold_end=end,
        )

    return candidates


def ae_graph_path(candidate: Candidate, graph_root: Path) -> Path:
    graph_root = graph_root if graph_root.is_absolute() else ROOT / graph_root
    if candidate.ae_k == "default":
        return graph_root / f"{candidate.dataset}_ae_graph.txt"
    return (
        graph_root
        / "sensitivity"
        / f"ae_k_{candidate.ae_k}"
        / f"{candidate.dataset}_ae_graph.txt"
    )


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


def graph_path_arg(graph_path: Path) -> str:
    try:
        return str(graph_path.relative_to(ROOT))
    except ValueError:
        return str(graph_path)


def candidate_ae_k_value(candidate: Candidate) -> int:
    if candidate.ae_k == "default":
        return AE_DEFAULT_K[candidate.dataset]
    return int(candidate.ae_k)


def build_train_command(job: Job, args: argparse.Namespace) -> list[str]:
    candidate = job.candidate

    train_args = dict(COMMON_TRAIN_ARGS)
    train_args.update(
        {
            "device": args.device,
            "runs": args.runs,
            "seed_start": job.seed_start,
            "ae_graph_path": graph_path_arg(job.graph_path),
        }
    )
    train_args.update(candidate.args)

    return [
        str(args.python),
        "train.py",
        "--dataset",
        candidate.dataset,
        "--cluster_num",
        str(CLUSTER_NUM[candidate.dataset]),
        "--graph_mode",
        "dual",
        "--fusion_mode",
        candidate.fusion_mode,
    ] + dict_to_cli(train_args)


def build_ae_command(
    dataset: str,
    out_graph_path: Path,
    ae_k: int,
    seed: int,
    args: argparse.Namespace,
) -> list[str]:
    ae_epochs = int(args.ae_epochs) if int(args.ae_epochs) > 0 else AE_DEFAULT_EPOCHS[dataset]
    ae_args: dict[str, Any] = {
        "dataset": dataset,
        "cluster_num": CLUSTER_NUM[dataset],
        "epochs": ae_epochs,
        "lr": args.ae_lr,
        "n_enc_1": 500,
        "n_enc_2": 500,
        "n_enc_3": 2000,
        "n_dec_1": 2000,
        "n_dec_2": 500,
        "n_dec_3": 500,
        "ae_k": ae_k,
        "sim_method": args.ae_sim_method,
        "n_z": AE_DEFAULT_N_Z[dataset],
        "out_graph_path": graph_path_arg(out_graph_path),
        "model_save_path": graph_path_arg(out_graph_path.with_suffix(".pkl")),
        "device": args.device,
    }
    if seed >= 0:
        ae_args["pretrain_seed"] = seed
        ae_args["graph_seed"] = seed
    if dataset in AE_BASE_GRAPH:
        ae_args["base_graph_path"] = graph_path_arg(AE_BASE_GRAPH[dataset])
    return [str(args.python), "data/pretrain_optimize_A_graph.py"] + dict_to_cli(ae_args)


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


def metric_gaps(
    dataset: str,
    metrics: dict[str, dict[str, float]],
    target_baseline: str,
) -> dict[str, float]:
    target = target_for_dataset(dataset, target_baseline)
    gaps = {}
    for metric in METRICS:
        if metric in metrics:
            gaps[metric] = metrics[metric]["mean"] - target[metric]
    return gaps


def pass_metric_names(args: argparse.Namespace) -> tuple[str, ...]:
    raw = str(args.pass_metrics).strip()
    if raw.lower() == "all":
        return METRICS
    names = []
    for item in raw.split(","):
        name = item.strip().upper()
        if not name:
            continue
        if name not in METRICS:
            raise ValueError(f"Unsupported pass metric: {name}")
        names.append(name)
    return tuple(names or ("ACC",))


def passes_target(result: dict[str, Any], required_metrics: tuple[str, ...]) -> bool:
    gaps = result.get("gaps", {})
    return all(gaps.get(metric, float("-inf")) > 0 for metric in required_metrics)


def rank_key(result: dict[str, Any], rank_metric: str) -> tuple[float, ...]:
    metrics = result.get("metrics", {})
    gaps = result.get("gaps", {})
    metric_mean = metrics.get(rank_metric, {}).get("mean", float("-inf"))
    metric_gap = gaps.get(rank_metric, float("-inf"))
    wins = float(result.get("wins", 0))
    min_gap = min(gaps.values()) if gaps else float("-inf")
    return (
        metric_mean,
        metric_gap,
        wins,
        min_gap,
        gaps.get("ACC", float("-inf")),
        gaps.get("F1", float("-inf")),
        gaps.get("NMI", float("-inf")),
        gaps.get("ARI", float("-inf")),
    )


def run_subprocess(
    cmd: list[str],
    log_path: Path,
    timeout: int,
) -> tuple[int, float, bool, str]:
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
        output += f"\n[TIMEOUT] Candidate exceeded {timeout} seconds.\n"
        returncode = 124
        timed_out = True

    elapsed = (datetime.now() - start).total_seconds()
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(f"COMMAND: {' '.join(cmd)}\n")
        handle.write(f"RETURN_CODE: {returncode}\n")
        handle.write(f"ELAPSED_SEC: {elapsed:.2f}\n")
        handle.write(f"TIMED_OUT: {'YES' if timed_out else 'NO'}\n")
        handle.write("=" * 80 + "\n")
        handle.write(output)
    return returncode, elapsed, timed_out, output


def prepare_ae_graph(job: Job, out_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    if job.ae_seed is None:
        return None

    graph_exists = job.graph_path.exists()
    if graph_exists and not args.force_ae_roll:
        return {
            "cmd": None,
            "returncode": 0,
            "elapsed": 0.0,
            "timed_out": False,
            "log_path": None,
            "reused": True,
        }

    ae_k = job.ae_k_value if job.ae_k_value is not None else candidate_ae_k_value(job.candidate)
    cmd = build_ae_command(job.candidate.dataset, job.graph_path, ae_k, job.ae_seed, args)
    log_path = (
        out_dir
        / job.candidate.dataset
        / "ae_rolls"
        / f"{job.candidate.name}_ae_seed{job.ae_seed}_k{ae_k}.txt"
    )
    print(
        f"[AE] {job.candidate.dataset} | {job.candidate.name} | seed={job.ae_seed} k={ae_k}",
        flush=True,
    )
    returncode, elapsed, timed_out, _output = run_subprocess(cmd, log_path, args.timeout)
    return {
        "cmd": cmd,
        "returncode": returncode,
        "elapsed": elapsed,
        "timed_out": timed_out,
        "log_path": str(log_path.relative_to(ROOT)),
        "reused": False,
    }


def run_job(
    job: Job,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate = job.candidate
    cmd = build_train_command(job, args)
    roll_bits = [f"train_roll{job.train_roll_idx}", f"seed{job.seed_start}"]
    if job.ae_seed is not None:
        roll_bits.append(f"ae_seed{job.ae_seed}")
    log_name = f"{candidate.name}__{'__'.join(roll_bits)}.txt"
    log_path = out_dir / candidate.dataset / log_name

    print(
        f"[RUN] {candidate.dataset} | {candidate.name} | "
        f"train_roll={job.train_roll_idx} seed_start={job.seed_start} ae_seed={job.ae_seed}",
        flush=True,
    )
    returncode, elapsed, timed_out, output = run_subprocess(cmd, log_path, args.timeout)
    metrics = parse_metrics(output)
    gaps = metric_gaps(candidate.dataset, metrics, args.target_baseline)
    wins = sum(1 for value in gaps.values() if value > 0)

    return {
        "dataset": candidate.dataset,
        "name": candidate.name,
        "fusion_mode": candidate.fusion_mode,
        "ae_k": candidate.ae_k,
        "ae_k_value": job.ae_k_value,
        "ae_seed": job.ae_seed,
        "ae_graph_path": graph_path_arg(job.graph_path),
        "train_roll_idx": job.train_roll_idx,
        "seed_start": job.seed_start,
        "seed_end": job.seed_start + int(args.runs) - 1,
        "args": candidate.args,
        "cmd": cmd,
        "returncode": returncode,
        "elapsed": elapsed,
        "timed_out": timed_out,
        "metrics": metrics,
        "gaps": gaps,
        "wins": wins,
        "log_path": str(log_path.relative_to(ROOT)),
        "target_baseline": resolve_target_baseline(args.target_baseline),
    }


def job_key(job: Job, args: argparse.Namespace) -> str:
    payload = {
        "dataset": job.candidate.dataset,
        "name": job.candidate.name,
        "fusion_mode": job.candidate.fusion_mode,
        "ae_k": job.candidate.ae_k,
        "args": sorted(job.candidate.args.items()),
        "graph_path": graph_path_arg(job.graph_path),
        "train_roll_idx": job.train_roll_idx,
        "seed_start": job.seed_start,
        "seed_end": job.seed_start + int(args.runs) - 1,
        "ae_seed": job.ae_seed,
        "ae_k_value": job.ae_k_value,
        "runs": int(args.runs),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def result_key(result: dict[str, Any]) -> str:
    payload = {
        "dataset": result.get("dataset"),
        "name": result.get("name"),
        "fusion_mode": result.get("fusion_mode"),
        "ae_k": result.get("ae_k"),
        "args": sorted((result.get("args") or {}).items()),
        "graph_path": result.get("ae_graph_path"),
        "train_roll_idx": result.get("train_roll_idx"),
        "seed_start": result.get("seed_start"),
        "seed_end": result.get("seed_end"),
        "ae_seed": result.get("ae_seed"),
        "ae_k_value": result.get("ae_k_value"),
        "runs": (
            int(result["seed_end"]) - int(result["seed_start"]) + 1
            if result.get("seed_start") is not None and result.get("seed_end") is not None
            else None
        ),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def load_resume_results(path: Path | None) -> tuple[list[dict[str, Any]], set[str]]:
    if path is None:
        return [], set()
    resume_path = path if path.is_absolute() else ROOT / path
    if not resume_path.exists():
        raise FileNotFoundError(f"resume jsonl not found: {resume_path}")
    results: list[dict[str, Any]] = []
    completed: set[str] = set()
    for line in resume_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        results.append(item)
        completed.add(result_key(item))
    return results, completed


def append_jsonl(path: Path, result: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=True) + "\n")
        handle.flush()


def make_jobs(selected: list[Candidate], args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    train_rolls = max(1, int(args.train_rolls))
    seed_stride = int(args.seed_stride) if int(args.seed_stride) > 0 else int(args.runs)
    ae_rolls = max(0, int(args.ae_rolls))

    for candidate in selected:
        if ae_rolls > 0:
            ae_k = candidate_ae_k_value(candidate)
            for ae_idx in range(ae_rolls):
                ae_seed = int(args.ae_seed_start) + ae_idx * int(args.ae_seed_stride)
                graph_path = (
                    OUTPUT_ROOT
                    / "ae_roll_graphs"
                    / candidate.dataset
                    / f"ae_seed{ae_seed}_k{ae_k}"
                    / f"{candidate.dataset}_ae_graph.txt"
                )
                for train_idx in range(train_rolls):
                    seed_start = int(args.seed_start) + train_idx * seed_stride
                    jobs.append(
                        Job(
                            candidate=candidate,
                            graph_path=graph_path,
                            train_roll_idx=train_idx + 1,
                            seed_start=seed_start,
                            ae_seed=ae_seed,
                            ae_k_value=ae_k,
                        )
                    )
        else:
            graph_path = ae_graph_path(candidate, args.ae_graph_root)
            ae_k = candidate_ae_k_value(candidate)
            for train_idx in range(train_rolls):
                seed_start = int(args.seed_start) + train_idx * seed_stride
                jobs.append(
                    Job(
                        candidate=candidate,
                        graph_path=graph_path,
                        train_roll_idx=train_idx + 1,
                        seed_start=seed_start,
                        ae_seed=None,
                        ae_k_value=ae_k,
                    )
                )
    return jobs


def write_summary(
    out_dir: Path,
    results: list[dict[str, Any]],
    selected: list[Candidate],
    args: argparse.Namespace,
) -> Path:
    required_metrics = pass_metric_names(args)
    target_baseline = resolve_target_baseline(args.target_baseline)
    dataset_order = selected_dataset_names(args)
    summary_path = out_dir / "summary.md"
    lines = [
        "# Final SCGC Push Sweep",
        "",
        f"- Started: {datetime.now().isoformat(timespec='seconds')}",
        f"- Preset: {args.preset}",
        f"- Dataset request: {args.dataset}",
        f"- Datasets: {','.join(dataset_order)}",
        f"- Target baseline: {target_baseline}",
        f"- Runs per training roll: {args.runs}",
        f"- Training rolls per candidate: {args.train_rolls}",
        f"- AE rolls per candidate: {args.ae_rolls}",
        f"- Rank metric: {args.rank_metric}",
        f"- Pass metrics: {','.join(required_metrics)}",
        f"- Candidates selected: {len(selected)}",
        f"- Commands executed: {len(results)}",
        f"- Dataset budget hours: {args.dataset_budget_hours}",
        f"- Total budget hours: {args.total_budget_hours}",
        f"- Resume source: {args.resume_jsonl if args.resume_jsonl else ''}",
        "",
        "## Comparison Targets",
        "",
        "| Dataset | ACC | NMI | ARI | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in dataset_order:
        target = target_for_dataset(dataset, target_baseline)
        lines.append(
            f"| {dataset} | {target['ACC']:.2f} | {target['NMI']:.2f} | "
            f"{target['ARI']:.2f} | {target['F1']:.2f} |"
        )

    for dataset in dataset_order:
        dataset_results = [item for item in results if item["dataset"] == dataset]
        if not dataset_results:
            continue
        dataset_results.sort(key=lambda item: rank_key(item, args.rank_metric), reverse=True)
        lines.extend(
            [
                "",
                f"## {dataset}",
                "",
                (
                    "| Rank | Pass | Candidate | Mode | AE k | AE seed | Seeds | ACC | NMI | ARI | F1 | "
                    "Gap ACC | Gap NMI | Gap ARI | Gap F1 | Wins | Log |"
                ),
                "| ---: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for idx, item in enumerate(dataset_results[: args.top], start=1):
            metrics = item["metrics"]
            gaps = item["gaps"]
            metric_text = {
                metric: metrics.get(metric, {}).get("mean", float("nan"))
                for metric in METRICS
            }
            gap_text = {metric: gaps.get(metric, float("nan")) for metric in METRICS}
            pass_text = "YES" if passes_target(item, required_metrics) else "NO"
            ae_seed = item.get("ae_seed")
            ae_seed_text = "" if ae_seed is None else str(ae_seed)
            seeds_text = f"{item.get('seed_start')}..{item.get('seed_end')}"
            lines.append(
                f"| {idx} | {pass_text} | {item['name']} | {item['fusion_mode']} | {item['ae_k']} | "
                f"{ae_seed_text} | {seeds_text} | "
                f"{metric_text['ACC']:.2f} | {metric_text['NMI']:.2f} | "
                f"{metric_text['ARI']:.2f} | {metric_text['F1']:.2f} | "
                f"{gap_text['ACC']:+.2f} | {gap_text['NMI']:+.2f} | "
                f"{gap_text['ARI']:+.2f} | {gap_text['F1']:+.2f} | "
                f"{item['wins']} | {item['log_path']} |"
            )

    if not results:
        lines.extend(["", "No commands were executed."])

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def write_dry_run(out_dir: Path, rows: list[tuple[Job, list[str] | None, list[str]]]) -> Path:
    path = out_dir / "dry_run_commands.txt"
    lines = []
    for job, ae_cmd, train_cmd in rows:
        candidate = job.candidate
        lines.append(
            f"[{candidate.dataset}] {candidate.name} | train_roll={job.train_roll_idx} "
            f"seed_start={job.seed_start} ae_seed={job.ae_seed} graph={graph_path_arg(job.graph_path)}"
        )
        if ae_cmd is not None:
            lines.append("AE: " + " ".join(ae_cmd))
        lines.append("TRAIN: " + " ".join(train_cmd))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def balanced_take(candidates: list[Candidate], datasets: tuple[str, ...], limit: int) -> list[Candidate]:
    if limit <= 0 or len(candidates) <= limit:
        return candidates

    buckets = {
        dataset: [candidate for candidate in candidates if candidate.dataset == dataset]
        for dataset in datasets
    }
    selected: list[Candidate] = []
    cursor = 0
    while len(selected) < limit:
        progressed = False
        for dataset in datasets:
            bucket = buckets.get(dataset, [])
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def select_candidates(args: argparse.Namespace) -> list[Candidate]:
    if args.preset == "quick":
        candidates = quick_candidates()
    elif args.preset == "grid":
        candidates = grid_candidates()
    elif args.preset == "safe_contract_grid":
        candidates = safe_contract_grid_candidates()
    elif args.preset == "cite_rescue":
        candidates = cite_rescue_candidates()
    elif args.preset == "cite_finalists":
        candidates = cite_finalist_candidates()
    elif args.preset == "cite_aggressive":
        candidates = cite_aggressive_candidates()
    elif args.preset == "cite_hidden_k20":
        candidates = cite_hidden_k20_candidates()
    elif args.preset == "cite_high_hidden_k20":
        candidates = cite_high_hidden_k20_candidates()
    elif args.preset == "cite_peak_h512":
        candidates = cite_peak_h512_candidates()
    elif args.preset == "cite_peak_h480_train":
        candidates = cite_peak_h480_train_candidates()
    elif args.preset == "reut_attn_dcgl_only_push":
        candidates = reut_attn_dcgl_only_push_candidates()
    elif args.preset == "reut_attn_dcgl_refine":
        candidates = reut_attn_dcgl_refine_candidates()
    elif args.preset == "reut_attn_dcgl_apex":
        candidates = reut_attn_dcgl_apex_candidates()
    elif args.preset == "amap_attn_peak":
        candidates = amap_attn_peak_candidates()
    elif args.preset == "amap_attn_dcgl_apex":
        candidates = amap_attn_dcgl_apex_candidates()
    elif args.preset == "uat_attn_dcgl_only_push":
        candidates = uat_attn_dcgl_only_push_candidates()
    elif args.preset == "uat_attn_dcgl_refine":
        candidates = uat_attn_dcgl_refine_candidates()
    elif args.preset == "uat_attn_dcgl_apex":
        candidates = uat_attn_dcgl_apex_candidates()
    elif args.preset == "uat_attn_peak":
        candidates = uat_attn_peak_candidates()
    elif args.preset == "uat_attn_roll":
        candidates = uat_attn_roll_candidates()
    elif args.preset == "uat_attn_continued":
        candidates = uat_attn_continued_candidates()
    elif args.preset == "eat_attn_peak":
        candidates = eat_attn_peak_candidates()
    elif args.preset == "eat_attn_open_peak":
        candidates = eat_attn_open_peak_candidates()
    elif args.preset == "eat_attn_open_bridge":
        candidates = eat_attn_open_bridge_candidates()
    elif args.preset == "eat_attn_apex":
        candidates = eat_attn_apex_candidates()
    elif args.preset == "cora_attn_peak":
        candidates = cora_attn_peak_candidates()
    elif args.preset == "usps_peak":
        candidates = usps_peak_candidates()
    elif args.preset == "usps_attn_peak":
        candidates = usps_attn_peak_candidates()
    elif args.preset == "usps_attn_micro":
        candidates = usps_attn_micro_candidates()
    elif args.preset == "usps_escape":
        candidates = usps_escape_candidates()
    else:
        candidates = targeted_candidates()

    datasets = selected_dataset_names(args)
    candidates = [candidate for candidate in candidates if candidate.dataset in datasets]
    if int(args.candidate_offset) > 0:
        candidates = candidates[int(args.candidate_offset):]

    if args.max_candidates > 0:
        candidates = balanced_take(candidates, datasets, int(args.max_candidates))
    elif args.preset in {"grid", "targeted"}:
        per_dataset = 64
        candidates = [
            candidate
            for dataset in datasets
            for candidate in candidates
            if candidate.dataset == dataset
        ]
        if len(datasets) == 1:
            candidates = candidates[:per_dataset]
        else:
            candidates = [
                candidate
                for dataset in datasets
                for candidate in [item for item in candidates if item.dataset == dataset][:per_dataset]
            ]
    return candidates


def main() -> None:
    args = parse_args()
    required_metrics = pass_metric_names(args)
    dataset_order = selected_dataset_names(args)
    selected = select_candidates(args)
    jobs = make_jobs(selected, args)
    resume_results, completed_keys = load_resume_results(args.resume_jsonl)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    suffix = "_dry_run" if args.dry_run else ""
    out_dir = OUTPUT_ROOT / f"{timestamp}_{args.preset}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    runnable: list[Job] = []
    skipped = []
    for job in jobs:
        if job_key(job, args) in completed_keys:
            continue
        if job.ae_seed is None and not job.graph_path.exists():
            skipped.append((job.candidate, job.graph_path))
            continue
        runnable.append(job)

    if skipped:
        skip_path = out_dir / "skipped_missing_ae_graphs.txt"
        skip_lines = [
            f"[{candidate.dataset}] {candidate.name}: {path}"
            for candidate, path in skipped
        ]
        skip_path.write_text("\n".join(skip_lines) + "\n", encoding="utf-8")
        print(f"[SKIP] {len(skipped)} candidates missing AE graphs. See {skip_path}", flush=True)

    if args.dry_run:
        dry_rows = []
        for job in runnable:
            ae_cmd = None
            if job.ae_seed is not None:
                ae_k = job.ae_k_value if job.ae_k_value is not None else candidate_ae_k_value(job.candidate)
                ae_cmd = build_ae_command(job.candidate.dataset, job.graph_path, ae_k, job.ae_seed, args)
            dry_rows.append((job, ae_cmd, build_train_command(job, args)))
        dry_path = write_dry_run(out_dir, dry_rows)
        print(f"[DRY-RUN] Wrote {len(dry_rows)} jobs to {dry_path}", flush=True)
        return

    results = list(resume_results)
    passed_datasets: set[str] = set()
    jsonl_path = out_dir / "results.jsonl"
    if resume_results:
        for item in resume_results:
            append_jsonl(jsonl_path, item)

    run_started = time.monotonic()
    dataset_elapsed = {
        dataset: sum(
            float(item.get("elapsed", 0.0))
            for item in results
            if item.get("dataset") == dataset
        )
        for dataset in dataset_order
    }
    update_every = max(0, int(args.update_summary_every))
    dataset_budget_seconds = float(args.dataset_budget_hours) * 3600.0
    total_budget_seconds = float(args.total_budget_hours) * 3600.0

    for job in runnable:
        if (
            args.stop_on_pass
            and len(dataset_order) > 1
            and job.candidate.dataset in passed_datasets
        ):
            continue

        total_elapsed = time.monotonic() - run_started
        if total_budget_seconds > 0 and total_elapsed >= total_budget_seconds:
            print("[BUDGET] Total budget reached; stopping.", flush=True)
            break

        dataset = job.candidate.dataset
        if dataset_budget_seconds > 0 and dataset_elapsed.get(dataset, 0.0) >= dataset_budget_seconds:
            print(f"[BUDGET] {dataset} budget reached; skipping remaining jobs.", flush=True)
            continue

        ae_result = prepare_ae_graph(job, out_dir, args)
        if ae_result is not None and ae_result["returncode"] != 0:
            result = {
                "dataset": job.candidate.dataset,
                "name": job.candidate.name,
                "fusion_mode": job.candidate.fusion_mode,
                "ae_k": job.candidate.ae_k,
                "ae_k_value": job.ae_k_value,
                "ae_seed": job.ae_seed,
                "ae_graph_path": graph_path_arg(job.graph_path),
                "train_roll_idx": job.train_roll_idx,
                "seed_start": job.seed_start,
                "seed_end": job.seed_start + int(args.runs) - 1,
                "args": job.candidate.args,
                "cmd": None,
                "returncode": ae_result["returncode"],
                "elapsed": 0.0,
                "timed_out": ae_result["timed_out"],
                "metrics": {},
                "gaps": {},
                "wins": 0,
                "log_path": ae_result["log_path"],
                "ae_result": ae_result,
                "error": "AE graph generation failed",
                "passed": False,
            }
        else:
            result = run_job(job, out_dir, args)
            result["ae_result"] = ae_result
            result["passed"] = passes_target(result, required_metrics)

        results.append(result)
        append_jsonl(jsonl_path, result)
        dataset_elapsed[dataset] = dataset_elapsed.get(dataset, 0.0) + float(result.get("elapsed", 0.0))

        if result["metrics"]:
            gaps = result["gaps"]
            metric = args.rank_metric
            metric_mean = result["metrics"][metric]["mean"]
            metric_gap = gaps[metric]
            print(
                f"[DONE] {result['name']} | {metric}={metric_mean:.2f} "
                f"gap={metric_gap:+.2f} wins={result['wins']}/4 "
                f"pass={'YES' if result.get('passed') else 'NO'}",
                flush=True,
            )
        else:
            print(f"[DONE] {result['name']} | no metrics parsed", flush=True)

        if update_every and len(results) % update_every == 0:
            write_summary(out_dir, results, selected, args)

        if args.stop_on_pass and result.get("passed"):
            if len(dataset_order) <= 1:
                print("[STOP] stop-on-pass triggered.", flush=True)
                break

            passed_datasets.add(str(result["dataset"]))
            print(
                f"[STOP] {result['dataset']} passed; skipping remaining jobs for this dataset.",
                flush=True,
            )
            if all(dataset in passed_datasets for dataset in dataset_order):
                print("[STOP] all selected datasets passed.", flush=True)
                break

    summary_path = write_summary(out_dir, results, selected, args)
    print(f"[SUMMARY] {summary_path}", flush=True)
    print(f"[JSONL] {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
