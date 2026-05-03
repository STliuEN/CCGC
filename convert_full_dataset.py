import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert full_dataset npy format to CCGC txt/edge-list format."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="data/full_dataset",
        help="Root directory containing per-dataset folders with *_feat.npy/*_label.npy/*_adj.npy",
    )
    parser.add_argument(
        "--target_data_dir",
        type=str,
        default="data/data",
        help="Output directory for <dataset>.txt and <dataset>_label.txt",
    )
    parser.add_argument(
        "--target_graph_dir",
        type=str,
        default="data/graph",
        help="Output directory for <dataset>_graph.txt",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset names. Empty means all subfolders in source_dir.",
    )
    parser.add_argument(
        "--write_graph",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to export graph edge-list from *_adj.npy",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to overwrite existing target files.",
    )
    parser.add_argument(
        "--dry_run",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, only print planned actions without writing files.",
    )
    return parser.parse_args()


def _discover_triplets(dataset_dir: Path):
    buckets = defaultdict(dict)
    for npy_path in dataset_dir.rglob("*.npy"):
        name = npy_path.name
        for key in ("feat", "label", "adj"):
            suffix = f"_{key}.npy"
            if name.endswith(suffix):
                prefix = name[: -len(suffix)]
                buckets[prefix][key] = npy_path
                break

    complete = {k: v for k, v in buckets.items() if {"feat", "label", "adj"}.issubset(v.keys())}
    return complete


def _choose_triplet_key(dataset_name: str, complete_triplets):
    if not complete_triplets:
        return None
    if dataset_name in complete_triplets:
        return dataset_name

    keys = sorted(complete_triplets.keys())
    for key in keys:
        if key.lower() == dataset_name.lower():
            return key
    for key in keys:
        if key.lower().startswith(dataset_name.lower()):
            return key
    return keys[0]


def _prepare_label(label_array: np.ndarray):
    label = np.asarray(label_array)
    if label.ndim == 2:
        if label.shape[1] == 1:
            label = label[:, 0]
        else:
            label = np.argmax(label, axis=1)
    label = np.asarray(label).reshape(-1)
    return label.astype(np.int64)


def _write_edge_list_from_adj(adj: np.ndarray, out_path: Path):
    n = adj.shape[0]
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            cols = np.flatnonzero(adj[i] != 0)
            if cols.size == 0:
                continue
            cols = cols[cols != i]
            if cols.size == 0:
                continue
            for j in cols:
                f.write(f"{i} {int(j)}\n")


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    target_data_dir = Path(args.target_data_dir)
    target_graph_dir = Path(args.target_graph_dir)
    dry_run = bool(args.dry_run)
    overwrite = bool(args.overwrite)
    write_graph = bool(args.write_graph)

    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")

    if args.datasets.strip():
        dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    else:
        dataset_names = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])

    if not dry_run:
        target_data_dir.mkdir(parents=True, exist_ok=True)
        if write_graph:
            target_graph_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    print(f"Source: {source_dir}")
    print(f"Target data: {target_data_dir}")
    if write_graph:
        print(f"Target graph: {target_graph_dir}")
    print(f"Datasets: {dataset_names}")
    print("-" * 72)

    for dataset_name in dataset_names:
        dataset_dir = source_dir / dataset_name
        if not dataset_dir.exists():
            print(f"[SKIP] {dataset_name}: folder not found -> {dataset_dir}")
            skipped += 1
            continue

        complete_triplets = _discover_triplets(dataset_dir)
        chosen_key = _choose_triplet_key(dataset_name, complete_triplets)
        if chosen_key is None:
            print(f"[SKIP] {dataset_name}: no complete *_feat/*_label/*_adj npy triplet found.")
            skipped += 1
            continue

        paths = complete_triplets[chosen_key]
        feat_path = paths["feat"]
        label_path = paths["label"]
        adj_path = paths["adj"]

        out_feat = target_data_dir / f"{dataset_name}.txt"
        out_label = target_data_dir / f"{dataset_name}_label.txt"
        out_graph = target_graph_dir / f"{dataset_name}_graph.txt"

        if not overwrite:
            existing = [p for p in [out_feat, out_label] + ([out_graph] if write_graph else []) if p.exists()]
            if existing:
                print(f"[SKIP] {dataset_name}: target exists -> {[str(p) for p in existing]}")
                skipped += 1
                continue

        feat = np.load(feat_path, allow_pickle=True)
        label = np.load(label_path, allow_pickle=True)
        adj = np.load(adj_path, mmap_mode="r", allow_pickle=True)

        feat = np.asarray(feat, dtype=np.float64)
        label = _prepare_label(label)
        adj = np.asarray(adj)

        if feat.ndim != 2:
            print(f"[SKIP] {dataset_name}: feat ndim != 2 ({feat.ndim})")
            skipped += 1
            continue
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            print(f"[SKIP] {dataset_name}: adj is not square 2D ({adj.shape})")
            skipped += 1
            continue
        if feat.shape[0] != label.shape[0] or feat.shape[0] != adj.shape[0]:
            print(
                f"[SKIP] {dataset_name}: row mismatch feat={feat.shape[0]}, label={label.shape[0]}, adj={adj.shape[0]}"
            )
            skipped += 1
            continue

        print(
            f"[OK] {dataset_name}: feat={feat.shape}, label={label.shape}, adj={adj.shape}, "
            f"from={chosen_key}"
        )
        print(f"     -> {out_feat}")
        print(f"     -> {out_label}")
        if write_graph:
            print(f"     -> {out_graph}")

        if dry_run:
            converted += 1
            continue

        np.savetxt(out_feat, feat, fmt="%.10g")
        np.savetxt(out_label, label, fmt="%d")
        if write_graph:
            _write_edge_list_from_adj(adj, out_graph)
        converted += 1

    print("-" * 72)
    print(f"Done. converted={converted}, skipped={skipped}, dry_run={dry_run}")


if __name__ == "__main__":
    main()
