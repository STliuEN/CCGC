import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Temporarily convert Citeseer raw content/cites files into feat/label/adj npy triplet."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/full_dataset/cite",
        help="Directory containing citeseer.content and citeseer.cites",
    )
    parser.add_argument(
        "--content_file",
        type=str,
        default="citeseer.content",
        help="Citeseer content filename",
    )
    parser.add_argument(
        "--cites_file",
        type=str,
        default="citeseer.cites",
        help="Citeseer cites filename",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="cite",
        help="Prefix used when writing <prefix>_feat.npy / <prefix>_label.npy / <prefix>_adj.npy",
    )
    parser.add_argument(
        "--undirected",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to symmetrize citation edges when building adjacency",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=0,
        choices=[0, 1],
        help="Overwrite existing npy files if they already exist",
    )
    return parser.parse_args()


def _read_content(content_path):
    paper_ids = []
    features = []
    raw_labels = []

    with open(content_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 3:
                raise ValueError(f"Invalid content line at {content_path}:{lineno}")

            paper_ids.append(parts[0])
            features.append([float(x) for x in parts[1:-1]])
            raw_labels.append(parts[-1])

    if not paper_ids:
        raise ValueError(f"No valid samples found in {content_path}")

    label_names = sorted(set(raw_labels))
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    labels = np.asarray([label_to_idx[name] for name in raw_labels], dtype=np.int64)
    feature_array = np.asarray(features, dtype=np.float32)
    id_to_idx = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}
    return id_to_idx, feature_array, labels


def _read_cites(cites_path, id_to_idx, num_nodes, undirected):
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    skipped_missing = 0
    skipped_self_loop = 0

    with open(cites_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(f"Invalid cites line at {cites_path}:{lineno}")

            src_id, dst_id = parts[0], parts[1]
            if src_id not in id_to_idx or dst_id not in id_to_idx:
                skipped_missing += 1
                continue

            src = id_to_idx[src_id]
            dst = id_to_idx[dst_id]
            if src == dst:
                skipped_self_loop += 1
                continue

            adj[src, dst] = 1.0
            if undirected:
                adj[dst, src] = 1.0

    return adj, skipped_missing, skipped_self_loop


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    content_path = input_dir / args.content_file
    cites_path = input_dir / args.cites_file

    if not content_path.exists():
        raise FileNotFoundError(f"Cannot find content file: {content_path}")
    if not cites_path.exists():
        raise FileNotFoundError(f"Cannot find cites file: {cites_path}")

    out_feat = input_dir / f"{args.output_prefix}_feat.npy"
    out_label = input_dir / f"{args.output_prefix}_label.npy"
    out_adj = input_dir / f"{args.output_prefix}_adj.npy"

    if not bool(args.overwrite):
        existing = [p for p in (out_feat, out_label, out_adj) if p.exists()]
        if existing:
            raise FileExistsError(f"Output files already exist: {[str(p) for p in existing]}")

    id_to_idx, feat, label = _read_content(content_path)
    adj, skipped_missing, skipped_self_loop = _read_cites(
        cites_path,
        id_to_idx=id_to_idx,
        num_nodes=feat.shape[0],
        undirected=bool(args.undirected),
    )

    np.save(out_feat, feat)
    np.save(out_label, label)
    np.save(out_adj, adj)

    print(f"Saved feature npy: {out_feat}")
    print(f"Saved label npy  : {out_label}")
    print(f"Saved adj npy    : {out_adj}")
    print(f"Nodes            : {feat.shape[0]}")
    print(f"Features         : {feat.shape[1]}")
    print(f"Classes          : {int(label.max()) + 1}")
    print(f"Edges(nonzero)   : {int(np.count_nonzero(adj))}")
    print(f"Skipped missing  : {skipped_missing}")
    print(f"Skipped self-loop: {skipped_self_loop}")


if __name__ == "__main__":
    main()
