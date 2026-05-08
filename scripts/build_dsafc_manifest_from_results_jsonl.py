from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dsafc_dual_structure_ablation import (
    ABLATION_ORDER,
    compute_structure_diag,
    load_experiment_config,
    parse_dataset_list,
    parse_variant_list,
    resolve_ae_graph_and_model,
    safe_json,
    write_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a complete DSAFC ablation manifest/summary from an accumulated results.jsonl."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--reuse-current-ae-assets", action="store_true")
    parser.add_argument("--export-fusion-artifacts", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[(str(row["dataset"]), str(row["variant"]))] = row
    return rows


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    results_jsonl = run_dir / "results.jsonl"
    if not results_jsonl.exists():
        raise FileNotFoundError(f"Missing results file: {results_jsonl}")

    config = load_experiment_config()
    datasets = parse_dataset_list(args.dataset, config)
    variants = parse_variant_list(args.variants)
    rows_by_key = load_jsonl(results_jsonl)

    missing: list[str] = []
    dataset_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        profile = config["dataset_profiles"][dataset]
        knn_k = int(profile.get("train_args", {}).get("knn_k", config.get("train_common_args", {}).get("knn_k", 5)))
        ae_graph_path, ae_model_path = resolve_ae_graph_and_model(
            dataset,
            config,
            profile,
            args.reuse_current_ae_assets,
        )
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
            row = rows_by_key.get((dataset, variant.key))
            if row is None:
                missing.append(f"{dataset}/{variant.key}")
                continue
            dataset_row["variants"][variant.key] = row
        dataset_rows.append(dataset_row)

    if missing:
        raise ValueError("Missing results:\n" + "\n".join(f"- {item}" for item in missing))

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
    (run_dir / "manifest.json").write_text(json.dumps(safe_json(manifest), ensure_ascii=False, indent=2), encoding="utf-8")

    # Keep the same table writer as the original ablation script so downstream
    # update/render scripts see an ordinary unified run.
    write_summary(run_dir, dataset_rows, datasets=datasets, variants=variants, args=args)
    print(f"[DONE] rebuilt manifest={run_dir / 'manifest.json'}")
    print(f"[DONE] rebuilt summary={run_dir / 'summary.md'}")
    print(f"[INFO] variants_order={','.join(ABLATION_ORDER)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
