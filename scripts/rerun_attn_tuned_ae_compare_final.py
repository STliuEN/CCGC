from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "backup" / "final" / "attn_tuned_10run_ae_compare"
SCGC_ENV_PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
DEFAULT_PYTHON = SCGC_ENV_PYTHON if SCGC_ENV_PYTHON.exists() else Path(sys.executable)
METRICS = ("ACC", "NMI", "ARI", "F1")


@dataclass(frozen=True)
class AeVariant:
    key: str
    label: str
    graph_relpath: str
    pkl_relpath: str | None
    note: str


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    source: str
    note: str
    args: dict[str, Any]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    cluster_num: int
    candidates: tuple[Candidate, ...]
    tuned_ae: AeVariant
    current_ae: AeVariant


def _candidate(
    key: str,
    label: str,
    source: str,
    note: str,
    **args: Any,
) -> Candidate:
    return Candidate(key=key, label=label, source=source, note=note, args=args)


def _ae_variant(
    key: str,
    label: str,
    graph_relpath: str,
    pkl_relpath: str | None,
    note: str,
) -> AeVariant:
    return AeVariant(
        key=key,
        label=label,
        graph_relpath=graph_relpath,
        pkl_relpath=pkl_relpath,
        note=note,
    )


DATASET_SPECS: dict[str, DatasetSpec] = {
    "reut": DatasetSpec(
        key="reut",
        label="Reuters",
        cluster_num=4,
        candidates=(
            _candidate(
                key="apex_seed42_history",
                label="Historical apex center",
                source="experiment_output/reut_push_tuning/20260502_reut_attn_dcgl_only/README.md",
                note="Highest strict DCGL-negative-only single-run tuning peak.",
                t=4,
                linlayers=1,
                epochs=400,
                dims=500,
                lr=0.0001,
                threshold=0.4,
                alpha=0.5,
                knn_k=5,
                fusion_hidden=64,
                fusion_temp=1.75,
                fusion_balance=0.40,
                fusion_min_weight=0.20,
                lambda_inst=0.09,
                lambda_clu=0.09,
                warmup_epochs=35,
                dcgl_neg_tau=0.5,
                dcgl_neg_weight=0.6,
            ),
        ),
        tuned_ae=_ae_variant(
            key="tuned",
            label="Historical tuned AE",
            graph_relpath="experiment_output/final_scgcn_push/ae_roll_graphs/reut/ae_seed42_k15/reut_ae_graph.txt",
            pkl_relpath="experiment_output/final_scgcn_push/ae_roll_graphs/reut/ae_seed42_k15/reut_ae_graph.pkl",
            note="AE seed 42 / k15 asset used by the historical Reuters apex sweep.",
        ),
        current_ae=_ae_variant(
            key="current",
            label="Current project AE",
            graph_relpath="data/ae_graph/reut_ae_graph.txt",
            pkl_relpath="pretrain_graph/reut_ae_pretrain.pkl",
            note="Current restored main-project Reuters AE asset.",
        ),
    ),
    "uat": DatasetSpec(
        key="uat",
        label="UAT",
        cluster_num=4,
        candidates=(
            _candidate(
                key="dcgl_seed42_peak",
                label="Historical tuned peak center",
                source="experiment_output/uat_push_tuning/20260502_uat_attn_dcgl_only/README.md",
                note="Best strict DCGL-negative-only tuned result from the UAT push.",
                t=5,
                linlayers=1,
                epochs=500,
                dims=500,
                lr=0.00012,
                threshold=0.4,
                alpha=0.45,
                knn_k=5,
                fusion_hidden=64,
                fusion_temp=1.8,
                fusion_balance=0.35,
                fusion_min_weight=0.20,
                lambda_inst=0.09,
                lambda_clu=0.09,
                warmup_epochs=35,
                dcgl_neg_tau=0.5,
                dcgl_neg_weight=0.6,
            ),
        ),
        tuned_ae=_ae_variant(
            key="tuned",
            label="Historical tuned AE",
            graph_relpath="experiment_output/final_scgcn_push/ae_roll_graphs/uat/ae_seed42_k15/uat_ae_graph.txt",
            pkl_relpath="experiment_output/final_scgcn_push/ae_roll_graphs/uat/ae_seed42_k15/uat_ae_graph.pkl",
            note="AE seed 42 / k15 asset used by the UAT strict-tuning peak.",
        ),
        current_ae=_ae_variant(
            key="current",
            label="Current project AE",
            graph_relpath="data/ae_graph/uat_ae_graph.txt",
            pkl_relpath="pretrain_graph/uat_ae_pretrain.pkl",
            note="Current restored main-project UAT AE asset.",
        ),
    ),
    "amap": DatasetSpec(
        key="amap",
        label="AMAP",
        cluster_num=8,
        candidates=(
            _candidate(
                key="peak_tau05_w08",
                label="Historical peak center",
                source="experiment_output/amap_push_tuning/README.md",
                note="Highest AMAP tuning peak from the attention push history.",
                t=4,
                linlayers=1,
                epochs=400,
                dims=500,
                lr=0.0001,
                threshold=0.4,
                alpha=0.5,
                knn_k=5,
                fusion_hidden=64,
                fusion_temp=1.25,
                fusion_balance=0.08,
                fusion_min_weight=0.05,
                lambda_inst=0.07,
                lambda_clu=0.035,
                warmup_epochs=35,
                dcgl_neg_tau=0.5,
                dcgl_neg_weight=0.8,
            ),
        ),
        tuned_ae=_ae_variant(
            key="tuned",
            label="Historical tuned AE",
            graph_relpath="data/ae_graph/amap_ae_graph.txt",
            pkl_relpath="pretrain_graph/amap_ae_pretrain.pkl",
            note="Historical AMAP peak already used the current main-project AE graph.",
        ),
        current_ae=_ae_variant(
            key="current",
            label="Current project AE",
            graph_relpath="data/ae_graph/amap_ae_graph.txt",
            pkl_relpath="pretrain_graph/amap_ae_pretrain.pkl",
            note="Current main-project AMAP AE asset.",
        ),
    ),
    "usps": DatasetSpec(
        key="usps",
        label="USPS",
        cluster_num=10,
        candidates=(
            _candidate(
                key="manual_peak_center",
                label="Historical tuned peak center",
                source="experiment_output/usps_push_tuning/20260505_usps_manual_attn_dcgl_only/README.md",
                note="Best USPS strict DCGL-negative-only attention center from manual tuning.",
                t=4,
                linlayers=1,
                epochs=400,
                dims=500,
                lr=0.0001,
                threshold=0.4,
                alpha=0.5,
                knn_k=5,
                fusion_hidden=64,
                fusion_temp=1.55,
                fusion_balance=0.15,
                fusion_min_weight=0.03,
                lambda_inst=0.09,
                lambda_clu=0.09,
                warmup_epochs=35,
                dcgl_neg_tau=0.5,
                dcgl_neg_weight=0.35,
            ),
        ),
        tuned_ae=_ae_variant(
            key="tuned",
            label="Historical tuned AE",
            graph_relpath="experiment_output/final_scgcn_push/ae_roll_graphs/usps/ae_seed-7_k15/usps_ae_graph.txt",
            pkl_relpath="experiment_output/final_scgcn_push/ae_roll_graphs/usps/ae_seed-7_k15/usps_ae_graph.pkl",
            note="Selected USPS engineering AE asset from the attention peak line.",
        ),
        current_ae=_ae_variant(
            key="current",
            label="Current project AE",
            graph_relpath="data/ae_graph/usps_ae_graph.txt",
            pkl_relpath="pretrain_graph/usps_ae_pretrain.pkl",
            note="Current main-project USPS AE asset.",
        ),
    ),
    "eat": DatasetSpec(
        key="eat",
        label="EAT",
        cluster_num=4,
        candidates=(
            _candidate(
                key="k15_plateau_nmi_f1",
                label="k15 NMI/F1 plateau",
                source="experiment_output/eat_push_tuning/20260505_eat_manual_attn_dcgl_only/README.md",
                note="Best reproduced EAT all-metric peak on the historical k15 AE graph.",
                t=4,
                linlayers=1,
                epochs=400,
                dims=500,
                lr=0.0001,
                threshold=0.359,
                alpha=0.5,
                knn_k=5,
                fusion_hidden=64,
                fusion_temp=1.6,
                fusion_balance=0.205,
                fusion_min_weight=0.005,
                lambda_inst=0.08,
                lambda_clu=0.05,
                warmup_epochs=32,
                dcgl_neg_tau=1.0,
                dcgl_neg_weight=0.55,
            ),
            _candidate(
                key="k15_plateau_high_ari",
                label="k15 high-ARI branch",
                source="experiment_output/eat_push_tuning/20260505_eat_manual_attn_dcgl_only/README.md",
                note="Alternate tied EAT branch with slightly higher ARI at the same ACC.",
                t=4,
                linlayers=1,
                epochs=400,
                dims=500,
                lr=0.0001,
                threshold=0.359,
                alpha=0.5,
                knn_k=5,
                fusion_hidden=64,
                fusion_temp=1.6,
                fusion_balance=0.2035,
                fusion_min_weight=0.005,
                lambda_inst=0.08,
                lambda_clu=0.05,
                warmup_epochs=32,
                dcgl_neg_tau=1.0,
                dcgl_neg_weight=0.55,
            ),
        ),
        tuned_ae=_ae_variant(
            key="tuned",
            label="Historical tuned AE",
            graph_relpath="data/ae_graph/sensitivity/ae_k_15/eat_ae_graph.txt",
            pkl_relpath=None,
            note="Historical EAT k15 AE graph that reproduces the old apex; no dedicated sensitivity pretrain checkpoint was found.",
        ),
        current_ae=_ae_variant(
            key="current",
            label="Current project AE",
            graph_relpath="data/ae_graph/eat_ae_graph.txt",
            pkl_relpath="pretrain_graph/eat_ae_pretrain.pkl",
            note="Current main-project EAT AE asset.",
        ),
    ),
    "cora": DatasetSpec(
        key="cora",
        label="Cora",
        cluster_num=7,
        candidates=(
            _candidate(
                key="aek5_all_metric_anchor",
                label="k5 all-metric anchor",
                source="experiment_output/cora_push_tuning/20260505_cora_manual_attn_dcgl_only/README.md",
                note="Best all-metric strict DCGL-negative-only Cora branch from manual tuning.",
                t=4,
                linlayers=1,
                epochs=400,
                dims=500,
                lr=0.0001,
                threshold=0.4,
                alpha=0.5,
                knn_k=5,
                fusion_hidden=64,
                fusion_temp=2.0,
                fusion_balance=0.25,
                fusion_min_weight=0.15,
                lambda_inst=0.08,
                lambda_clu=0.08,
                warmup_epochs=35,
                dcgl_neg_tau=0.5,
                dcgl_neg_weight=0.6,
            ),
            _candidate(
                key="aek5_acc_f1_branch",
                label="k5 ACC/F1 branch",
                source="experiment_output/cora_push_tuning/20260505_cora_manual_attn_dcgl_only/README.md",
                note="Best ACC/F1 strict DCGL-negative-only Cora branch from manual tuning.",
                t=4,
                linlayers=1,
                epochs=400,
                dims=500,
                lr=0.0001,
                threshold=0.4,
                alpha=0.5,
                knn_k=5,
                fusion_hidden=64,
                fusion_temp=2.0,
                fusion_balance=0.25,
                fusion_min_weight=0.145,
                lambda_inst=0.08,
                lambda_clu=0.08,
                warmup_epochs=35,
                dcgl_neg_tau=0.5,
                dcgl_neg_weight=0.6,
            ),
        ),
        tuned_ae=_ae_variant(
            key="tuned",
            label="Historical tuned AE",
            graph_relpath="data/ae_graph/sensitivity/ae_k_5/cora_ae_graph.txt",
            pkl_relpath=None,
            note="Historical Cora k5 AE graph from strict manual tuning; no dedicated sensitivity pretrain checkpoint was found.",
        ),
        current_ae=_ae_variant(
            key="current",
            label="Current project AE",
            graph_relpath="data/ae_graph/cora_ae_graph.txt",
            pkl_relpath="pretrain_graph/cora_ae_pretrain.pkl",
            note="Current main-project Cora AE asset.",
        ),
    ),
    "cite": DatasetSpec(
        key="cite",
        label="Citeseer",
        cluster_num=6,
        candidates=(
            _candidate(
                key="backup_midstrict_tau035",
                label="Backup mid-strict tau=0.35",
                source="experiment_output/cite_push_tuning/20260505_cite_manual_attn_dcgl_only/README.md",
                note="Best strict DCGL-negative-only CiteSeer point from the recovered backup AE lane.",
                t=4,
                linlayers=1,
                epochs=400,
                dims=500,
                lr=0.0001,
                threshold=0.4,
                alpha=0.5,
                knn_k=5,
                fusion_hidden=64,
                fusion_temp=2.25,
                fusion_balance=0.50,
                fusion_min_weight=0.26,
                lambda_inst=0.065,
                lambda_clu=0.035,
                warmup_epochs=35,
                dcgl_neg_tau=0.35,
                dcgl_neg_weight=0.6,
            ),
            _candidate(
                key="backup_midstrict_tau040",
                label="Backup mid-strict tau=0.40",
                source="experiment_output/cite_push_tuning/20260505_cite_manual_attn_dcgl_only/README.md",
                note="Tied neighboring CiteSeer tau plateau point from the recovered backup AE lane.",
                t=4,
                linlayers=1,
                epochs=400,
                dims=500,
                lr=0.0001,
                threshold=0.4,
                alpha=0.5,
                knn_k=5,
                fusion_hidden=64,
                fusion_temp=2.25,
                fusion_balance=0.50,
                fusion_min_weight=0.26,
                lambda_inst=0.065,
                lambda_clu=0.035,
                warmup_epochs=35,
                dcgl_neg_tau=0.40,
                dcgl_neg_weight=0.6,
            ),
        ),
        tuned_ae=_ae_variant(
            key="tuned",
            label="Historical tuned AE",
            graph_relpath="backup/backup_asset1/ae_graph/cite_ae_graph.txt",
            pkl_relpath="backup/backup_asset1/pretrain_graph/cite_ae_pretrain.pkl",
            note="Recovered backup CiteSeer AE asset that restores the strict tuning lane.",
        ),
        current_ae=_ae_variant(
            key="current",
            label="Current project AE",
            graph_relpath="data/ae_graph/cite_ae_graph.txt",
            pkl_relpath="pretrain_graph/cite_ae_pretrain.pkl",
            note="Current main-project CiteSeer AE asset.",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce 10-run stability for the seven tuned attn/DCGL-negative-only datasets "
            "using historical-best AE assets versus the current project AE assets, then archive "
            "the selected best reusable assets under backup/final."
        )
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset key, comma-separated list, or 'all'.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of standard training runs per scenario.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First training seed for the standard rerun.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to train.py.",
    )
    parser.add_argument(
        "--python",
        default=str(DEFAULT_PYTHON),
        help="Python executable used to launch train.py.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Root directory for logs, summaries, and selected assets.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse any completed scenario JSON files already present under the output root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the manifest and commands without running train.py.",
    )
    return parser.parse_args()


def resolve_datasets(dataset_arg: str) -> list[str]:
    raw = dataset_arg.strip().lower()
    if raw == "all":
        return list(DATASET_SPECS.keys())
    datasets = [item.strip().lower() for item in raw.split(",") if item.strip()]
    bad = [item for item in datasets if item not in DATASET_SPECS]
    if bad:
        raise ValueError(f"Unknown dataset keys: {', '.join(bad)}")
    return datasets


def _path(relpath: str | None) -> Path | None:
    if not relpath:
        return None
    return ROOT / relpath


def sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def parse_final_metrics(text: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        head = line.split("|", 1)
        if len(head) != 2:
            continue
        metric = head[0].strip()
        if metric not in METRICS:
            continue
        nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", head[1])
        if len(nums) >= 2:
            metrics[metric] = {"mean": float(nums[0]), "std": float(nums[1])}
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


def compute_metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    metrics: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = [float(row["metrics"][metric]) for row in rows]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        metrics[metric] = {"mean": mean, "std": var ** 0.5}
    return metrics


def build_command(
    python_exe: str,
    dataset: str,
    cluster_num: int,
    candidate: Candidate,
    ae_graph_path: Path,
    runs: int,
    seed_start: int,
    device: str,
) -> list[str]:
    args = dict(candidate.args)
    args.update(
        {
            "dataset": dataset,
            "cluster_num": cluster_num,
            "graph_mode": "dual",
            "fusion_mode": "attn",
            "runs": runs,
            "seed_start": seed_start,
            "device": device,
            "ae_graph_path": str(ae_graph_path),
            "enable_dcgl_negative_loss": True,
        }
    )
    return [python_exe, "train.py"] + dict_to_cli(args)


def scenario_json_path(output_root: Path, dataset: str, candidate_key: str, ae_key: str) -> Path:
    return output_root / "results" / dataset / f"{candidate_key}__{ae_key}.json"


def scenario_log_path(output_root: Path, dataset: str, candidate_key: str, ae_key: str) -> Path:
    return output_root / "logs" / dataset / f"{candidate_key}__{ae_key}.txt"


def run_scenario(
    output_root: Path,
    python_exe: str,
    spec: DatasetSpec,
    candidate: Candidate,
    ae_variant: AeVariant,
    runs: int,
    seed_start: int,
    device: str,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    graph_path = _path(ae_variant.graph_relpath)
    if graph_path is None or not graph_path.exists():
        raise FileNotFoundError(f"Missing AE graph: {ae_variant.graph_relpath}")

    json_path = scenario_json_path(output_root, spec.key, candidate.key, ae_variant.key)
    log_path = scenario_log_path(output_root, spec.key, candidate.key, ae_variant.key)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if resume and json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))

    cmd = build_command(
        python_exe=python_exe,
        dataset=spec.key,
        cluster_num=spec.cluster_num,
        candidate=candidate,
        ae_graph_path=graph_path,
        runs=runs,
        seed_start=seed_start,
        device=device,
    )
    payload: dict[str, Any] = {
        "dataset": spec.key,
        "dataset_label": spec.label,
        "candidate_key": candidate.key,
        "candidate_label": candidate.label,
        "candidate_source": candidate.source,
        "candidate_note": candidate.note,
        "ae_key": ae_variant.key,
        "ae_label": ae_variant.label,
        "ae_graph_relpath": ae_variant.graph_relpath,
        "ae_pkl_relpath": ae_variant.pkl_relpath,
        "ae_note": ae_variant.note,
        "ae_graph_hash": sha256_of(graph_path),
        "command": cmd,
        "runs": runs,
        "seed_start": seed_start,
        "status": "dry-run" if dry_run else "pending",
    }

    if dry_run:
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log_path.write_text("DRY RUN\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return payload

    started_at = datetime.now().isoformat()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    finished_at = datetime.now().isoformat()

    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""
    parsed_rows = parse_per_seed(stdout_text)
    parsed_metrics = parse_final_metrics(stdout_text)
    if parsed_rows:
        parsed_metrics = compute_metrics_from_rows(parsed_rows)

    payload.update(
        {
            "status": "ok" if proc.returncode == 0 and parsed_metrics else "failed",
            "returncode": proc.returncode,
            "started_at": started_at,
            "finished_at": finished_at,
            "metrics": parsed_metrics,
            "per_seed": parsed_rows,
        }
    )

    log_text = [
        f"COMMAND: {' '.join(cmd)}",
        f"DATASET: {spec.key}",
        f"CANDIDATE: {candidate.key}",
        f"AE_VARIANT: {ae_variant.key}",
        f"RETURN_CODE: {proc.returncode}",
        "=" * 80,
        "[STDOUT]",
        stdout_text,
        "=" * 80,
        "[STDERR]",
        stderr_text,
    ]
    log_path.write_text("\n".join(log_text), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_metrics = left.get("metrics", {})
    right_metrics = right.get("metrics", {})
    if not left_metrics or not right_metrics:
        return False
    ge_all = True
    gt_any = False
    for metric in METRICS:
        left_mean = float(left_metrics.get(metric, {}).get("mean", float("-inf")))
        right_mean = float(right_metrics.get(metric, {}).get("mean", float("-inf")))
        if left_mean < right_mean - 1e-9:
            ge_all = False
            break
        if left_mean > right_mean + 1e-9:
            gt_any = True
    return ge_all and gt_any


def select_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [result for result in results if result.get("status") == "ok" and result.get("metrics")]
    if not ok_results:
        raise RuntimeError("No successful results available for selection.")

    dominant = [
        candidate
        for candidate in ok_results
        if all(candidate is other or dominates(candidate, other) or not other.get("metrics") for other in ok_results)
    ]
    if dominant:
        dominant.sort(
            key=lambda row: tuple(float(row["metrics"][metric]["mean"]) for metric in METRICS),
            reverse=True,
        )
        chosen = dict(dominant[0])
        chosen["selection_rule"] = "dominates_all"
        return chosen

    ranks: dict[int, dict[str, int]] = {}
    for metric in METRICS:
        ordered = sorted(
            enumerate(ok_results),
            key=lambda item: float(item[1]["metrics"][metric]["mean"]),
            reverse=True,
        )
        for rank_idx, (result_idx, _) in enumerate(ordered, start=1):
            ranks.setdefault(result_idx, {})[metric] = rank_idx

    scored: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for idx, result in enumerate(ok_results):
        rank_sum = sum(ranks.get(idx, {}).get(metric, len(ok_results) + 1) for metric in METRICS)
        means = result["metrics"]
        score = (
            -float(rank_sum),
            float(means["ACC"]["mean"]),
            float(means["NMI"]["mean"]),
            float(means["ARI"]["mean"]),
            float(means["F1"]["mean"]),
        )
        scored.append((score, result))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = dict(scored[0][1])
    chosen["selection_rule"] = "rank_sum_then_acc_nmi_ari_f1"
    return chosen


def copy_variant_asset(output_root: Path, dataset: str, prefix: str, variant: AeVariant) -> dict[str, Any]:
    graph_src = _path(variant.graph_relpath)
    pkl_src = _path(variant.pkl_relpath)
    if graph_src is None or not graph_src.exists():
        raise FileNotFoundError(f"Missing graph asset: {variant.graph_relpath}")
    asset_dir = output_root / "selected_assets" / dataset
    asset_dir.mkdir(parents=True, exist_ok=True)

    graph_dst = asset_dir / f"{prefix}_ae_graph.txt"
    shutil.copy2(graph_src, graph_dst)

    copied: dict[str, Any] = {
        "graph_src": str(graph_src.relative_to(ROOT)).replace("\\", "/"),
        "graph_dst": str(graph_dst.relative_to(ROOT)).replace("\\", "/"),
    }
    if pkl_src is not None and pkl_src.exists():
        pkl_dst = asset_dir / f"{prefix}_ae_graph.pkl"
        shutil.copy2(pkl_src, pkl_dst)
        copied["pkl_src"] = str(pkl_src.relative_to(ROOT)).replace("\\", "/")
        copied["pkl_dst"] = str(pkl_dst.relative_to(ROOT)).replace("\\", "/")
    return copied


def write_dataset_readme(
    output_root: Path,
    spec: DatasetSpec,
    result_rows: list[dict[str, Any]],
    chosen: dict[str, Any],
) -> None:
    asset_dir = output_root / "selected_assets" / spec.key
    asset_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {spec.label} 10-run AE Comparison",
        "",
        f"- Dataset key: `{spec.key}`",
        f"- Selection rule: `{chosen.get('selection_rule', 'unknown')}`",
        f"- Selected candidate: `{chosen['candidate_key']}`",
        f"- Selected AE variant: `{chosen['ae_key']}`",
        f"- Selected AE graph hash: `{chosen['ae_graph_hash']}`",
        "",
        "## Selected 10-run Metrics",
        "",
    ]
    for metric in METRICS:
        row = chosen["metrics"][metric]
        lines.append(f"- {metric}: `{row['mean']:.2f} +- {row['std']:.2f}`")
    lines.extend(
        [
            "",
            "## Scenario Summary",
            "",
            "| Candidate | AE | ACC | NMI | ARI | F1 | Source |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result_rows:
        metrics = row.get("metrics", {})
        if not metrics:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row["candidate_key"],
                    row["ae_key"],
                    f"{metrics['ACC']['mean']:.2f} +- {metrics['ACC']['std']:.2f}",
                    f"{metrics['NMI']['mean']:.2f} +- {metrics['NMI']['std']:.2f}",
                    f"{metrics['ARI']['mean']:.2f} +- {metrics['ARI']['std']:.2f}",
                    f"{metrics['F1']['mean']:.2f} +- {metrics['F1']['std']:.2f}",
                    row["candidate_source"],
                ]
            )
            + " |"
        )
    (asset_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_overall_summary(
    output_root: Path,
    run_stamp: str,
    manifest: dict[str, Any],
    chosen_rows: dict[str, dict[str, Any]],
    all_results: dict[str, list[dict[str, Any]]],
) -> Path:
    lines = [
        "# Final Attn 10-run AE Comparison Summary",
        "",
        f"- Generated at: `{datetime.now().isoformat()}`",
        f"- Run stamp: `{run_stamp}`",
        f"- Manifest: `backup/final/attn_tuned_10run_ae_compare/manifests/manifest_{run_stamp}.json`",
        "",
        "## Final Selection",
        "",
        "| Dataset | Candidate | AE | ACC | NMI | ARI | F1 | Rule |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for dataset in manifest["datasets"]:
        row = chosen_rows[dataset]
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    manifest["specs"][dataset]["label"],
                    row["candidate_key"],
                    row["ae_key"],
                    f"{metrics['ACC']['mean']:.2f} +- {metrics['ACC']['std']:.2f}",
                    f"{metrics['NMI']['mean']:.2f} +- {metrics['NMI']['std']:.2f}",
                    f"{metrics['ARI']['mean']:.2f} +- {metrics['ARI']['std']:.2f}",
                    f"{metrics['F1']['mean']:.2f} +- {metrics['F1']['std']:.2f}",
                    row.get("selection_rule", ""),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Dataset Details",
            "",
        ]
    )
    for dataset in manifest["datasets"]:
        spec_info = manifest["specs"][dataset]
        lines.append(f"### {spec_info['label']}")
        lines.append("")
        lines.append(f"- Tuned AE graph: `{spec_info['tuned_ae']['graph_relpath']}`")
        lines.append(f"- Current AE graph: `{spec_info['current_ae']['graph_relpath']}`")
        lines.append(f"- Same hash: `{spec_info['same_ae_hash']}`")
        chosen = chosen_rows[dataset]
        lines.append(f"- Selected candidate: `{chosen['candidate_key']}`")
        lines.append(f"- Selected AE variant: `{chosen['ae_key']}`")
        lines.append("")
        lines.append("| Candidate | AE | ACC | NMI | ARI | F1 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for row in all_results[dataset]:
            metrics = row.get("metrics", {})
            if not metrics:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["candidate_key"],
                        row["ae_key"],
                        f"{metrics['ACC']['mean']:.2f} +- {metrics['ACC']['std']:.2f}",
                        f"{metrics['NMI']['mean']:.2f} +- {metrics['NMI']['std']:.2f}",
                        f"{metrics['ARI']['mean']:.2f} +- {metrics['ARI']['std']:.2f}",
                        f"{metrics['F1']['mean']:.2f} +- {metrics['F1']['std']:.2f}",
                    ]
                )
                + " |"
            )
        lines.append("")

    summary_path = output_root / f"summary_{run_stamp}.md"
    summary_latest = output_root / "summary_latest.md"
    text = "\n".join(lines) + "\n"
    summary_path.write_text(text, encoding="utf-8")
    summary_latest.write_text(text, encoding="utf-8")
    return summary_path


def build_manifest(datasets: list[str], runs: int, seed_start: int, device: str, python_exe: str) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "runs": runs,
        "seed_start": seed_start,
        "device": device,
        "python": python_exe,
        "datasets": datasets,
        "specs": {},
    }
    for dataset in datasets:
        spec = DATASET_SPECS[dataset]
        tuned_graph = _path(spec.tuned_ae.graph_relpath)
        current_graph = _path(spec.current_ae.graph_relpath)
        if tuned_graph is None or current_graph is None:
            raise FileNotFoundError(f"Missing AE graph path for {dataset}")
        if not tuned_graph.exists():
            raise FileNotFoundError(f"Missing tuned AE graph for {dataset}: {tuned_graph}")
        if not current_graph.exists():
            raise FileNotFoundError(f"Missing current AE graph for {dataset}: {current_graph}")

        tuned_hash = sha256_of(tuned_graph)
        current_hash = sha256_of(current_graph)
        manifest["specs"][dataset] = {
            "label": spec.label,
            "cluster_num": spec.cluster_num,
            "same_ae_hash": tuned_hash == current_hash,
            "tuned_ae": {
                "graph_relpath": spec.tuned_ae.graph_relpath,
                "pkl_relpath": spec.tuned_ae.pkl_relpath,
                "note": spec.tuned_ae.note,
                "sha256": tuned_hash,
            },
            "current_ae": {
                "graph_relpath": spec.current_ae.graph_relpath,
                "pkl_relpath": spec.current_ae.pkl_relpath,
                "note": spec.current_ae.note,
                "sha256": current_hash,
            },
            "candidates": [
                {
                    "key": candidate.key,
                    "label": candidate.label,
                    "source": candidate.source,
                    "note": candidate.note,
                    "args": candidate.args,
                }
                for candidate in spec.candidates
            ],
        }
    return manifest


def main() -> int:
    args = parse_args()
    datasets = resolve_datasets(args.dataset)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest = build_manifest(
        datasets=datasets,
        runs=args.runs,
        seed_start=args.seed_start,
        device=args.device,
        python_exe=args.python,
    )
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"manifest_{run_stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "manifest_latest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        for dataset in datasets:
            spec = DATASET_SPECS[dataset]
            seen_hashes: set[str] = set()
            for ae_variant in (spec.tuned_ae, spec.current_ae):
                graph_path = _path(ae_variant.graph_relpath)
                if graph_path is None:
                    continue
                graph_hash = sha256_of(graph_path)
                if graph_hash in seen_hashes:
                    continue
                seen_hashes.add(graph_hash)
                for candidate in spec.candidates:
                    cmd = build_command(
                        python_exe=args.python,
                        dataset=spec.key,
                        cluster_num=spec.cluster_num,
                        candidate=candidate,
                        ae_graph_path=graph_path,
                        runs=args.runs,
                        seed_start=args.seed_start,
                        device=args.device,
                    )
                    print(" ".join(cmd))
        return 0

    all_results: dict[str, list[dict[str, Any]]] = {}
    chosen_rows: dict[str, dict[str, Any]] = {}

    for dataset in datasets:
        spec = DATASET_SPECS[dataset]
        results: list[dict[str, Any]] = []

        variant_order = [spec.tuned_ae, spec.current_ae]
        hash_owner: dict[str, dict[str, Any]] = {}
        for ae_variant in variant_order:
            graph_path = _path(ae_variant.graph_relpath)
            if graph_path is None:
                continue
            graph_hash = sha256_of(graph_path)
            if graph_hash in hash_owner:
                reused = hash_owner[graph_hash]
                for candidate in spec.candidates:
                    clone = dict(reused[candidate.key])
                    clone["ae_key"] = ae_variant.key
                    clone["ae_label"] = ae_variant.label
                    clone["ae_graph_relpath"] = ae_variant.graph_relpath
                    clone["ae_pkl_relpath"] = ae_variant.pkl_relpath
                    clone["ae_note"] = ae_variant.note
                    results.append(clone)
                continue

            per_candidate: dict[str, dict[str, Any]] = {}
            for candidate in spec.candidates:
                row = run_scenario(
                    output_root=output_root,
                    python_exe=args.python,
                    spec=spec,
                    candidate=candidate,
                    ae_variant=ae_variant,
                    runs=args.runs,
                    seed_start=args.seed_start,
                    device=args.device,
                    resume=args.resume,
                    dry_run=False,
                )
                results.append(row)
                per_candidate[candidate.key] = row
            hash_owner[graph_hash] = per_candidate

        chosen = select_best_result(results)
        chosen_rows[dataset] = chosen
        all_results[dataset] = results

        copy_variant_asset(output_root, dataset, "tuned", spec.tuned_ae)
        copy_variant_asset(output_root, dataset, "current", spec.current_ae)
        selected_variant = spec.tuned_ae if chosen["ae_key"] == spec.tuned_ae.key else spec.current_ae
        copy_variant_asset(output_root, dataset, "selected", selected_variant)
        write_dataset_readme(output_root, spec, results, chosen)

    summary_path = write_overall_summary(
        output_root=output_root,
        run_stamp=run_stamp,
        manifest=manifest,
        chosen_rows=chosen_rows,
        all_results=all_results,
    )
    print(f"[DONE] manifest={manifest_path}")
    print(f"[DONE] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
