from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_dsafc_paper_figures import CLUSTER_COLORS, style_framed_axes

PYTHON = Path(r"C:\Users\stern\anaconda3\envs\SCGC_1\python.exe")
EMBED_DIR = ROOT / "assets" / "tsne_embeddings"
OUT_DIR = ROOT / "assets"
FONT_SCALE = 0.70

DATASETS = ("reut", "uat")
METHODS = (
    ("DAEGC", "daegc"),
    ("SDCN", "sdcn"),
    ("DFCN", "dfcn"),
    ("CCGC", "ccgc"),
    ("SCGC-N", "scgc-n"),
    ("SCGC-S", "scgc-s"),
    ("Ours", "ours"),
)

CLUSTERS = {"reut": 4, "uat": 4}
MAIN_TABLE_ACC = {
    "reut": {"DAEGC": 65.50, "SDCN": 77.15, "DFCN": 77.70, "CCGC": 70.78, "SCGC-N": 80.32, "SCGC-S": 76.67, "Ours": 83.20},
    "uat": {"DAEGC": 52.29, "SDCN": 52.25, "DFCN": 33.61, "CCGC": 56.34, "SCGC-N": 52.02, "SCGC-S": 56.58, "Ours": 56.24},
}

OURS_PROFILE = {
    "reut": {
        "fusion_temp": 1.6,
        "fusion_balance": 0.25,
        "lambda_inst": 0.08,
        "lambda_clu": 0.06,
        "warmup_epochs": 35,
    },
    "uat": {
        "fusion_temp": 1.9,
        "fusion_balance": 0.35,
        "lambda_inst": 0.08,
        "lambda_clu": 0.07,
        "warmup_epochs": 35,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run methods and draw Reuters/UAT t-SNE comparison.")
    parser.add_argument("--skip-runs", action="store_true", help="Only draw from existing npz embeddings.")
    parser.add_argument("--skip-daegc", action="store_true")
    parser.add_argument("--skip-sdcn", action="store_true")
    parser.add_argument("--skip-dfcn", action="store_true")
    parser.add_argument("--skip-ccgc", action="store_true")
    parser.add_argument("--skip-scgcn", action="store_true")
    parser.add_argument("--skip-scgcs", action="store_true")
    parser.add_argument("--skip-ours", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Override train epochs for all run-capable methods.")
    parser.add_argument("--baseline-epochs", type=int, default=None, help="Override DAEGC/SDCN/DFCN epochs.")
    parser.add_argument("--ours-epochs", type=int, default=None, help="Override CCGC/Ours epochs.")
    parser.add_argument("--scgc-epochs", type=int, default=None, help="Override SCGC-N/SCGC-S epochs.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for CCGC/Ours train.py.")
    parser.add_argument("--sample-size", type=int, default=2000, help="Max points per dataset row in the figure.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "DSAFC_tsne_reuters_uat.png")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}. See {log_path}")


def ensure_scgcn_local_data() -> None:
    local = ROOT / "other_projects" / "SCGC-N" / "local_data"
    for sub in ("data", "graph"):
        (local / sub).mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        for suffix in (".txt", "_label.txt", ".pkl"):
            src = ROOT / "data" / "data" / f"{dataset}{suffix}"
            dst = local / "data" / f"{dataset}{suffix}"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    graph_map = {
        "reut": ("reut3_graph.txt", "reut5_graph.txt", "reut_graph.txt"),
        "uat": ("uat_graph.txt", "uat_graph.txt", "uat_graph.txt"),
    }
    for dataset, (needed, source, fallback) in graph_map.items():
        dst = local / "graph" / needed
        if dst.exists():
            continue
        src = ROOT / "data" / "graph" / source
        if not src.exists():
            src = ROOT / "data" / "graph" / fallback
        shutil.copy2(src, dst)


def ensure_scgcs_uat_data() -> None:
    dataset_root = ROOT / "other_projects" / "SCGC-S" / "dataset"
    uat_dir = dataset_root / "uat"
    if (uat_dir / "uat_feat.npy").exists():
        return
    zip_path = dataset_root / "uat.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dataset_root)
        return
    uat_dir.mkdir(parents=True, exist_ok=True)
    np.save(uat_dir / "uat_feat.npy", np.load(ROOT / "data" / "full_dataset" / "uat" / "uat_feat.npy").astype(np.float32))
    np.save(uat_dir / "uat_label.npy", np.load(ROOT / "data" / "full_dataset" / "uat" / "uat_label.npy").astype(np.int64))
    np.save(uat_dir / "uat_adj.npy", np.load(ROOT / "data" / "full_dataset" / "uat" / "uat_adj.npy").astype(np.float32))


def train_command_for_ours(dataset: str, method: str, args: argparse.Namespace) -> list[str]:
    epochs = args.ours_epochs or args.epochs or 400
    out_path = EMBED_DIR / dataset / f"{method.lower()}.npz"
    cmd = [
        str(args.python),
        "train.py",
        "--dataset",
        dataset,
        "--cluster_num",
        str(CLUSTERS[dataset]),
        "--epochs",
        str(epochs),
        "--device",
        args.device,
        "--threshold",
        "0.4",
        "--alpha",
        "0.5",
        "--lr",
        "1e-4",
        "--save_embedding_path",
        str(out_path),
        "--save_embedding_method",
        "Ours" if method == "ours" else "CCGC",
        "--runs",
        str(args.runs),
    ]
    if method == "ccgc":
        cmd.extend(["--graph_mode", "raw", "--knn_k", "5"])
        return cmd

    profile = OURS_PROFILE[dataset]
    cmd.extend(
        [
            "--graph_mode",
            "dual",
            "--fusion_mode",
            "attn",
            "--fusion_hidden",
            "64",
            "--fusion_temp",
            str(profile["fusion_temp"]),
            "--fusion_balance",
            str(profile["fusion_balance"]),
            "--lambda_inst",
            str(profile["lambda_inst"]),
            "--lambda_clu",
            str(profile["lambda_clu"]),
            "--warmup_epochs",
            str(profile["warmup_epochs"]),
            "--fusion_min_weight",
            "0.10",
            "--enable_dcgl_negative_loss",
            "--dcgl_neg_tau",
            "0.5",
            "--dcgl_neg_weight",
            "0.6",
        ]
    )
    return cmd


def run_all(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "experiment_output" / "cluster_tsne_runs" / timestamp

    baseline_epochs = args.baseline_epochs or args.epochs
    if not args.skip_daegc:
        cmd = [str(args.python), "run_reproduce.py", "--datasets", *DATASETS, "--device", args.device, "--seed", str(args.seed)]
        if baseline_epochs:
            cmd.extend(["--train-epochs", str(baseline_epochs)])
        run(cmd, ROOT / "other_projects" / "DAEGC", log_dir / "daegc.log")

    if not args.skip_sdcn:
        cmd = [str(args.python), "run_reproduce.py", "--datasets", *DATASETS, "--device", args.device, "--seed", str(args.seed)]
        if baseline_epochs:
            cmd.extend(["--epoch", str(baseline_epochs)])
        run(cmd, ROOT / "other_projects" / "SDCN", log_dir / "sdcn.log")

    if not args.skip_dfcn:
        cmd = [str(args.python), "run_reproduce.py", "--datasets", *DATASETS, "--device", args.device, "--seed", str(args.seed)]
        if baseline_epochs:
            cmd.extend(["--epoch", str(baseline_epochs)])
        run(cmd, ROOT / "other_projects" / "DFCN", log_dir / "dfcn.log")

    if not args.skip_ccgc:
        for dataset in DATASETS:
            run(train_command_for_ours(dataset, "ccgc", args), ROOT, log_dir / f"ccgc_{dataset}.log")

    if not args.skip_scgcn:
        ensure_scgcn_local_data()
        scgc_epochs = args.scgc_epochs or args.epochs or 200
        for dataset in DATASETS:
            out_path = EMBED_DIR / dataset / "scgc-n.npz"
            cmd = [
                str(args.python),
                "train.py",
                "--name",
                dataset,
                "--data_path",
                str(ROOT / "other_projects" / "SCGC-N" / "local_data"),
                "--iterations",
                "1",
                "--epochs",
                str(scgc_epochs),
                "--model",
                "SCGC",
                "--verbosity",
                "0",
                "--alpha",
                "1.0",
                "--beta",
                "0.1",
                "--order",
                "2",
                "--tau",
                "0.25",
                "--lr",
                "1e-3",
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--save_embedding_path",
                str(out_path),
                "--save_embedding_method",
                "SCGC-N",
            ]
            run(cmd, ROOT / "other_projects" / "SCGC-N", log_dir / f"scgc-n_{dataset}.log")

    if not args.skip_scgcs:
        ensure_scgcs_uat_data()
        scgc_epochs = args.scgc_epochs or args.epochs or 400
        for dataset in DATASETS:
            cmd = [
                str(args.python),
                "run_single_dataset.py",
                "--dataset",
                dataset,
                "--epochs",
                str(scgc_epochs),
                "--seeds",
                "1",
                "--seed-start",
                str(args.seed),
                "--device",
                args.device,
            ]
            run(cmd, ROOT / "other_projects" / "SCGC-S", log_dir / f"scgc-s_{dataset}.log")

    if not args.skip_ours:
        for dataset in DATASETS:
            run(train_command_for_ours(dataset, "ours", args), ROOT, log_dir / f"ours_{dataset}.log")

    manifest = {
        "timestamp": timestamp,
        "logs": str(log_dir.relative_to(ROOT)),
        "datasets": DATASETS,
        "methods": [label for label, _ in METHODS],
    }
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def choose_indices(labels: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if labels.shape[0] <= sample_size:
        return np.arange(labels.shape[0])
    rng = np.random.default_rng(seed)
    selected = []
    classes = np.unique(labels)
    per_class = max(1, sample_size // len(classes))
    for cls in classes:
        idx = np.flatnonzero(labels == cls)
        take = min(per_class, idx.shape[0])
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    remaining = sample_size - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(labels.shape[0]), np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(pool, size=min(remaining, pool.shape[0]), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def reduce_for_plot(embedding: np.ndarray, labels: np.ndarray, sample_idx: np.ndarray, seed: int) -> np.ndarray:
    x = np.asarray(embedding, dtype=np.float32)[sample_idx]
    x = StandardScaler().fit_transform(x)
    if x.shape[1] > 50:
        x = PCA(n_components=50, random_state=seed).fit_transform(x)
    perplexity = min(35, max(5, (x.shape[0] - 1) // 3))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=1000,
        metric="euclidean",
    ).fit_transform(x)
    coords = coords.astype(np.float32)
    coords -= coords.mean(axis=0, keepdims=True)
    scale = np.abs(coords).max()
    if scale > 0:
        coords /= scale
    return coords


def draw(args: argparse.Namespace) -> None:
    rows = len(DATASETS)
    cols = len(METHODS)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(rows, cols, figsize=(19.5, 5.8), constrained_layout=False)
    palette = list(CLUSTER_COLORS)

    metadata = {}
    for row, dataset in enumerate(DATASETS):
        label_ref = np.load(EMBED_DIR / dataset / "ours.npz")["labels"]
        sample_idx = choose_indices(label_ref, args.sample_size, args.seed + row)
        row_labels = label_ref[sample_idx]
        for col, (title, key) in enumerate(METHODS):
            ax = axes[row, col]
            path = EMBED_DIR / dataset / f"{key}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            data = np.load(path, allow_pickle=True)
            coords = reduce_for_plot(data["embedding"], data["labels"], sample_idx, args.seed + row * 17 + col)
            labels = data["labels"][sample_idx]
            metrics = data["metrics"] if "metrics" in data.files else np.full(4, np.nan, dtype=np.float32)
            for cls_idx, cls in enumerate(np.unique(row_labels)):
                mask = labels == cls
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=5.5 if dataset == "reut" else 10,
                    c=palette[cls_idx % len(palette)],
                    alpha=0.78,
                    linewidths=0,
                    rasterized=True,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            style_framed_axes(ax, grid_axis=None)
            table_acc = MAIN_TABLE_ACC[dataset][title]
            metadata[f"{dataset}/{key}"] = {
                "file": str(path.relative_to(ROOT)),
                "embedding_shape": list(data["embedding"].shape),
                "saved_metrics": [float(v) for v in metrics],
                "table_acc": table_acc,
            }
            if row == rows - 1:
                ax.set_xlabel(title, fontsize=18 * FONT_SCALE, fontfamily="serif", fontweight="bold", labelpad=14)
        axes[row, 0].set_ylabel("Reuters" if dataset == "reut" else "UAT", fontsize=14 * FONT_SCALE, fontweight="bold", labelpad=16)

    fig.subplots_adjust(left=0.045, right=0.995, top=0.96, bottom=0.13, wspace=0.03, hspace=0.10)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)

    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps({"sample_size": args.sample_size, "seed": args.seed, "methods": metadata}, indent=2), encoding="utf-8")
    print(f"Saved figure: {args.output}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved metadata: {meta_path}")


def main() -> int:
    args = parse_args()
    if not args.python.exists():
        raise FileNotFoundError(args.python)
    if not args.skip_runs:
        run_all(args)
    draw(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
