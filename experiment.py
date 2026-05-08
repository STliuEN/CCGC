import os
import re
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path


### <--- [MODIFIED] ---------------------------------------
# Temporary runtime monitor. Default OFF so test.py batch ablations stay quiet.
SHOW_STATUS = "--show_status" in sys.argv
if SHOW_STATUS:
    sys.argv = [arg for arg in sys.argv if arg != "--show_status"]
### ---------------------------------------


### <--- [MODIFIED] ---------------------------------------
# This runner is intentionally NOT independent.
# It delegates all core training / graph-generation logic to:
# 1) train.py
# 2) data/pretrain_optimize_A_graph.py
#
# You only need to edit this CONFIG block for dataset switching / ablations.
CONFIG = {
    "output_dir": "experiment_output",
    # Save command logs by dataset subfolder: experiment_output/<dataset>/*.txt
    "log_by_dataset": True,
    # 1) Fast switch: edit only this list.
    # Example: ["acm"], ["usps"], ["acm", "usps"], ["acm","dblp","usps","reut","hhar","amap", "bat", "cite", "cora", "eat", "pubmed", "uat"]，corafull is too slow for quick tests, so it's not in the default list.

    "active_datasets": ["reut", "uat", "amap", "usps", "cora", "cite"],

    # 2) Compare modes: baseline / ae / dual(shared A+AE)
    "run_baseline": False,
    "run_ae": False,
    # Backward-compatible switch: if True, maps to dual_mean.
    "run_dual": False,
    "run_dual_mean": False,
    "run_dual_attn": True,
    # Final-paper default: reuse fixed/precomputed AE artifacts.
    # This keeps the reported 10-run variance tied to main-training seeds only
    # and prevents experiment.py from overwriting released AE graph assets.
    # Set this to False only when deliberately rebuilding AE preprocessing files.
    "reuse_existing_ae_results": True,
    # 2.5) Optional CCGC improvement modules (decoupled switches).
    # Keep default OFF to preserve legacy training behavior.
    "enable_dynamic_threshold_module": False,
    "dynamic_threshold_args": {
        # Example when enabled:
        # "dynamic_threshold_start": 0.2,
        # "dynamic_threshold_end": 0.5,
    },
    "enable_ema_prototypes_module": False,
    "ema_prototypes_args": {
        # Example when enabled:
        # "ema_proto_momentum": 0.9,
    },
    # Legacy compatibility (old single-switch config).
    "enable_improved_module": False,
    "improved_module_args": {},
    # 2.6) Optional DCGL-inspired modules (all default OFF).
    "enable_dcgl_negative_module": True,
    "dcgl_negative_args": {
        "dcgl_neg_tau": 0.5,
        "dcgl_neg_weight": 0.6,
        # Reliability-gated negative separation is the default internal
        # iteration. Dataset profiles can set
        # `disable_dcgl_neg_reliability_gate=True` to reproduce the earlier
        # row-weighted DCGL-negative frontier when that is empirically stronger.
        "dcgl_neg_gate_threshold": 0.55,
        "dcgl_neg_gate_power": 2.0,
        "dcgl_neg_gate_min": 0.0,
    },
    "enable_dcgl_cluster_module": False,
        "dcgl_cluster_args": {

        # Example when enabled:
        # "lambda_dcgl_cluster": 0.1,
        # "dcgl_cluster_tau": 0.5,
    },
    # 2.7) Optional encoder backbone switch (default keeps original MLP).
    "enable_gcn_backbone_module": False,
    "gcn_backbone_args": {
        # Example when enabled:
        # "gcn_dropout": 0.0,
    },
    # 3) Dataset-specific profile (for reproducible comparison settings).
    # `train_args` applies to train.py (baseline / AE-train / dual-train).
    # `ae_args` applies to data/pretrain_optimize_A_graph.py.
    # Optional `dual_attn_args` applies only to train.py dual-attn runs for
    # the current dataset, and overrides CONFIG["dual_attn_args"] key by key.
    # Later per-dataset args override the shared CONFIG args above.
    # `safe_tuning_grid` stores fairness-safe push candidates only. It is not
    # consumed by main() by default, so current paper-run defaults stay stable.
    "safe_tuning_contract": {
        "enable_dcgl_negative_module": True,
        "enable_dynamic_threshold_module": False,
        "enable_ema_prototypes_module": False,
        "enable_dcgl_cluster_module": False,
        "enable_gcn_backbone_module": False,
        "note": "Use these grids only for dual-attn + DCGL-negative-only tuning.",
    },
    "dataset_profiles": {
        "acm": {
            "cluster_num": 3,
            "train_args": {
                ### <--- [MODIFIED] ---------------------------------------
                # Keep ACM on the earlier stable main-train setting.
                "lr": 1e-4,
                ### ---------------------------------------
            },
            "ae_args": {
                "base_graph_path": "data/graph/acm_graph.txt",
                ### <--- [MODIFIED] ---------------------------------------
                # SIRM/GCRM-style AE setting:
                # lr=1e-3 for ACM in the reference setting, latent dim follows cluster count K.
                "lr": 1e-3,
                "n_z": 3,
                "pretrain_seed": 42,
                "graph_seed": 42,
                ### ---------------------------------------
            },
            "dual_attn_args": {
                "fusion_hidden": 64,
                "fusion_temp": 1.8,
                "fusion_balance": 0.25,
                "lambda_inst": 0.12,
                "lambda_clu": 0.12,
                "warmup_epochs": 35,
                "fusion_min_weight": 0.10,

                "enable_branch_bias_fusion": True,
                "branch_bias_target": "raw",
                "branch_bias_cap": 0.20,   # acm: historical best charging setup, raw-anchored with restrained AE correction
            },
        },
        "dblp": {
            "cluster_num": 4,
            "train_args": {
                ### <--- [MODIFIED] ---------------------------------------
                # Main training suggestion: keep the default stable setting.
                "lr": 1e-4,
                ### ---------------------------------------
            },
            "ae_args": { 
                "base_graph_path": "data/graph/dblp_graph.txt",
                ### <--- [MODIFIED] ---------------------------------------
                # SIRM/GCRM-style AE setting:
                # lr=1e-3 for DBLP in the reference setting, latent dim follows cluster count K.
                "lr": 1e-3,
                "n_z": 4,
                "pretrain_seed": 42,
                "graph_seed": 42,
                ### ---------------------------------------
            },
            "dual_attn_args": {
                "fusion_hidden": 64,
                "fusion_temp": 1.9,
                "fusion_balance": 0.35,
                "lambda_inst": 0.08,
                "lambda_clu": 0.07,
                "warmup_epochs": 35,
                "fusion_min_weight": 0.10,
            },
        },
        "usps": {
            "cluster_num": 10,
            "train_args": {
                ### <--- [MODIFIED] ---------------------------------------
                # Main-table USPS center keeps the standard 1e-4 train step.
                "lr": 1e-4,
                ### ---------------------------------------
            },
            "ae_args": {
                "base_graph_path": "data/graph/usps5_graph.txt",
                ### <--- [MODIFIED] ---------------------------------------
                # Main-table USPS row used the default project AE graph asset.
                "epochs": 30,
                "lr": 1e-3,
                "n_z": 10,
                "pretrain_seed": 42,
                "graph_seed": 42,
                ### ---------------------------------------
            },
            "dual_attn_args": {
                "fusion_hidden": 64,
                "fusion_temp": 1.8,
                "fusion_balance": 0.35,
                "lambda_inst": 0.09,
                "lambda_clu": 0.09,
                "warmup_epochs": 35,
                "fusion_min_weight": 0.20,
            },
            "dcgl_negative_args": {
                "dcgl_neg_tau": 0.5,
                "dcgl_neg_weight": 0.6,
                "disable_dcgl_neg_reliability_gate": True,
            },
            "safe_tuning_grid": {
                "train_args": {
                    "threshold": [0.35, 0.4, 0.45],
                },
                "dual_attn_args": {
                    "fusion_temp": [1.8, 2.0, 2.2],
                    "fusion_balance": [0.30, 0.35, 0.45],
                    "fusion_min_weight": [0.15, 0.20, 0.25],
                    "lambda_inst": [0.07, 0.09, 0.11],
                    "lambda_clu": [0.07, 0.09, 0.11],
                    "warmup_epochs": [25, 35, 45],
                },
                "dcgl_negative_args": {
                    "dcgl_neg_tau": [0.5, 0.75, 1.0],
                    "dcgl_neg_weight": [0.3, 0.4, 0.5, 0.6, 0.8],
                },
            },
        },
        "reut": {
            "cluster_num": 4,
            "train_args": {
                ### <--- [MODIFIED] ---------------------------------------
                # Main training suggestion: high-dimensional text data is more stable with a smaller lr.
                "lr": 1e-4,
                ### ---------------------------------------
            },
            "ae_args": {
                "base_graph_path": "data/graph/reut5_graph.txt",
                ### <--- [MODIFIED] ---------------------------------------
                # Reuters regressed sharply in the latest trial.
                # Restore the historically stabler AE-pretrain setting under the
                # unified 30-epoch horizon: default 1e-3 step and compact latent dim n_z=3.
                "epochs": 30,
                "lr": 1e-3,
                "n_z": 3,
                "pretrain_seed": 42,
                "graph_seed": 42,
                ### ---------------------------------------
            },
            "dual_attn_args": {
                "fusion_hidden": 64,
                "fusion_temp": 1.6,
                "fusion_balance": 0.25,
                "lambda_inst": 0.06,
                "lambda_clu": 0.06,
                "warmup_epochs": 35,
                "fusion_min_weight": 0.15,
            },
            "dcgl_negative_args": {
                "dcgl_neg_tau": 0.5,
                "dcgl_neg_weight": 0.3,
                "disable_dcgl_neg_reliability_gate": True,
            },
            "safe_tuning_grid": {
                "train_args": {
                    "threshold": [0.35, 0.4, 0.45],
                },
                "dual_attn_args": {
                    "fusion_balance": [0.20, 0.25, 0.30],
                    "fusion_min_weight": [0.10, 0.15, 0.20],
                    "lambda_inst": [0.06, 0.08, 0.10],
                    "lambda_clu": [0.04, 0.06, 0.08],
                    "warmup_epochs": [25, 35, 45],
                },
                "dcgl_negative_args": {
                    "dcgl_neg_tau": [0.5, 0.75, 1.0],
                    "dcgl_neg_weight": [0.3, 0.4, 0.5, 0.6],
                },
            },
        },
        "hhar": {
            "cluster_num": 6,
            "train_args": {
                ### <--- [MODIFIED] ---------------------------------------
                # Main training suggestion: use a smaller lr for the more volatile sensor setting.
                "lr": 5e-5,
                ### ---------------------------------------
            },
            "ae_args": {
                "base_graph_path": "data/graph/hhar5_graph.txt",
                ### <--- [MODIFIED] ---------------------------------------
                # SIRM/GCRM-style AE setting:
                # lr=1e-4 for HHAR in the reference setting, latent dim follows cluster count K.
                "lr": 1e-4,
                "n_z": 6,
                "pretrain_seed": 42,
                "graph_seed": 42,
                ### ---------------------------------------
            },
            "dual_attn_args": {
                "fusion_hidden": 64,
                "fusion_temp": 1.6,
                "fusion_balance": 0.25,
                "lambda_inst": 0.08,
                "lambda_clu": 0.06,
                "warmup_epochs": 35,
                "fusion_min_weight": 0.10,
            },
        },
        "amap": {
            "cluster_num": 8,
            "train_args": {
                ### <--- [MODIFIED] ---------------------------------------
                # CCGC original dataset: keep the paper-aligned main-train lr.
                "lr": 1e-4,
                ### ---------------------------------------
            },
            "ae_args": {
                "base_graph_path": "data/graph/amap_graph.txt",
                ### <--- [MODIFIED] ---------------------------------------
                # AE suggestion: medium step size for this denser feature scale.
                "lr": 5e-4,
                "n_z": 8,
                "pretrain_seed": 42,
                "graph_seed": 42,
                ### ---------------------------------------
            },
            "dual_attn_args": {
                "fusion_hidden": 64,
                "fusion_temp": 1.25,
                "fusion_balance": 0.08,
                "lambda_inst": 0.0,
                "lambda_clu": 0.035,
                "warmup_epochs": 35,
                "fusion_min_weight": 0.05,
            },
            "dcgl_negative_args": {
                "dcgl_neg_tau": 0.5,
                "dcgl_neg_weight": 0.6,
                "disable_dcgl_neg_reliability_gate": True,
            },
            "safe_tuning_grid": {
                "train_args": {
                    "threshold": [0.35, 0.4, 0.45],
                },
                "dual_attn_args": {
                    "fusion_balance": [0.05, 0.08, 0.10],
                    "lambda_inst": [0.0, 0.03, 0.07],
                    "fusion_min_weight": [0.0, 0.05, 0.10],
                    "lambda_clu": [0.02, 0.035, 0.05],
                    "warmup_epochs": [25, 35, 45],
                },
                "dcgl_negative_args": {
                    "dcgl_neg_tau": [0.5, 1.0],
                    "dcgl_neg_weight": [0.6, 0.8, 1.0],
                },
            },
        },
        "bat": {
            "cluster_num": 4,
            "train_args": {
                ### <--- [MODIFIED] ---------------------------------------
                # CCGC original dataset: keep the paper-aligned main-train lr.
                "lr": 1e-4,
                ### ---------------------------------------
            },
            "ae_args": {
                "base_graph_path": "data/graph/bat_graph.txt",
                ### <--- [MODIFIED] ---------------------------------------
                # BAT is sensitive to the newer AE-pretrain setting.
                # Roll back to the earlier more robust compact AE configuration.
                "epochs": 30,
                "lr": 1e-3,
                "n_z": 3,
                "pretrain_seed": 42,
                "graph_seed": 42,
                ### ---------------------------------------
            },
            "dual_attn_args": {
                "fusion_hidden": 64,
                "fusion_temp": 1.3,
                "fusion_balance": 0.12,
                "lambda_inst": 0.08,
                "lambda_clu": 0.04,
                "warmup_epochs": 35,
                "fusion_min_weight": 0.10,
            },
        },
        "cite": {
            "cluster_num": 6,
            "train_args": {
                ### <--- [MODIFIED] ---------------------------------------
                # CCGC original dataset: keep the paper-aligned main-train lr.
                "lr": 1e-4,
                ### ---------------------------------------
            },
            "ae_args": {
                "base_graph_path": "data/graph/cite_graph.txt",
                ### <--- [MODIFIED] ---------------------------------------
                # CiteSeer dropped the most under the newer AE-pretrain override.
                # Restore the earlier effective setting under the unified 30-epoch
                # AE horizon: lr=1e-3 and n_z=3.
                "epochs": 30,
                "lr": 1e-3,
                "n_z": 3,
                "pretrain_seed": 42,
                "graph_seed": 42,
                ### ---------------------------------------
            },
            "dual_attn_args": {
                "fusion_hidden": 64,
                "fusion_temp": 1.8,
                "fusion_balance": 0.15,
                "lambda_inst": 0.045,
                "lambda_clu": 0.02,
                "warmup_epochs": 55,
                "fusion_min_weight": 0.10,

                "enable_branch_bias_fusion": True,
                "branch_bias_target": "raw",
                "branch_bias_cap": 0.15,   # cite: keep moderate AE correction

                # Optional citation-safe fusion. Keep commented unless enabled.
                # "enable_branch_bias_fusion": True,
                # "branch_bias_target": "raw",
                # "branch_bias_cap": 0.15,
            },
            "dcgl_negative_args": {
                "dcgl_neg_tau": 0.5,
                "dcgl_neg_weight": 0.6,
                "disable_dcgl_neg_reliability_gate": True,
            },
            "safe_tuning_grid": {
                "train_args": {
                    "threshold": [0.35, 0.4, 0.45],
                },
                "dual_attn_args": {
                    "fusion_temp": [1.6, 1.8, 2.0],
                    "fusion_balance": [0.10, 0.15, 0.25, 0.35],
                    "fusion_min_weight": [0.05, 0.10, 0.15, 0.20],
                    "lambda_inst": [0.03, 0.045, 0.06],
                    "lambda_clu": [0.01, 0.02, 0.03],
                    "warmup_epochs": [45, 55, 65],
                    "branch_bias_cap": [0.12, 0.15, 0.18],
                },
                "dcgl_negative_args": {
                    "dcgl_neg_tau": [0.35, 0.5, 0.75, 1.0],
                    "dcgl_neg_weight": [0.3, 0.4, 0.6, 0.8, 1.0],
                },
            },
        },
        "cora": {
            "cluster_num": 7,
            "train_args": {
                ### <--- [MODIFIED] ---------------------------------------
                # CCGC original dataset: keep the paper-aligned main-train lr.
                "lr": 1e-4,
                ### ---------------------------------------
            },
            "ae_args": {
                "base_graph_path": "data/graph/cora_graph.txt",
                ### <--- [MODIFIED] ---------------------------------------
                # AE suggestion: moderate lr for sparse citation features.
                "lr": 3e-4,
                "n_z": 7,
                "pretrain_seed": 42,
                "graph_seed": 42,
                ### ---------------------------------------
            },
            "dual_attn_args": {
                "fusion_hidden": 64,
                "fusion_temp": 1.3,
                "fusion_balance": 0.0,
                "lambda_inst": 0.03,
                "lambda_clu": 0.01,
                "warmup_epochs": 70,
                "fusion_min_weight": 0.0,

                "enable_branch_bias_fusion": True,
                "branch_bias_target": "raw",
                "branch_bias_cap": 0.10,   # cora: stronger citation-safe correction from the better run


                # Optional citation-safe fusion. Keep commented unless enabled.
                # "enable_branch_bias_fusion": True,
                # "branch_bias_target": "raw",
                # "branch_bias_cap": 0.10,
            },
            "dcgl_negative_args": {
                "dcgl_neg_tau": 0.5,
                "dcgl_neg_weight": 0.6,
            },
            "safe_tuning_grid": {
                "train_args": {
                    "threshold": [0.35, 0.4, 0.45],
                },
                "dual_attn_args": {
                    "fusion_balance": [0.0, 0.05, 0.10],
                    "fusion_min_weight": [0.0, 0.05, 0.10],
                    "lambda_inst": [0.02, 0.03, 0.04],
                    "lambda_clu": [0.005, 0.01, 0.02],
                    "warmup_epochs": [55, 70, 85],
                    "branch_bias_cap": [0.08, 0.10, 0.12],
                },
                "dcgl_negative_args": {
                    "dcgl_neg_tau": [0.35, 0.5, 0.75],
                    "dcgl_neg_weight": [0.3, 0.4, 0.6, 0.8, 1.0],
                },
            },
        },
        "corafull": {
            "cluster_num": 70,
            "train_args": {
                ### <--- [MODIFIED] ---------------------------------------
                # Main training suggestion: very large / high-dimensional setting, use a smaller lr.
                "lr": 5e-5,
                ### ---------------------------------------
            },
            "ae_args": {
                "base_graph_path": "data/graph/corafull_graph.txt",
                ### <--- [MODIFIED] ---------------------------------------
                # AE suggestion: very high-dimensional input, so start conservatively.
                "lr": 1e-4,
                "n_z": 70,
                "pretrain_seed": 42,
                "graph_seed": 42,
                ### ---------------------------------------
            },
            "dual_attn_args": {
                "fusion_hidden": 64,
                "fusion_temp": 1.8,
                "fusion_balance": 0.35,
                "lambda_inst": 0.09,
                "lambda_clu": 0.09,
                "warmup_epochs": 35,
                "fusion_min_weight": 0.10,
            },
        },
        "eat": {
            "cluster_num": 4,
            "train_args": {
                ### <--- [MODIFIED] ---------------------------------------
                # Final selected stable combo keeps the standard train center
                # but uses a slightly lower threshold for the dual-attn run.
                "lr": 1e-4,
                "threshold": 0.36,
                ### ---------------------------------------
            },
            "ae_args": {
                "base_graph_path": "data/graph/eat_graph.txt",
                ### <--- [MODIFIED] ---------------------------------------
                # EAT also regressed after switching to the newer AE-pretrain override.
                # Use the earlier compact AE setting that was locally more stable.
                "epochs": 30,
                "lr": 1e-3,
                "n_z": 3,
                "pretrain_seed": 42,
                "graph_seed": 42,
                ### ---------------------------------------
            },
            "dual_attn_args": {
                "fusion_hidden": 64,
                "fusion_temp": 1.6,
                "fusion_balance": 0.22,
                "lambda_inst": 0.08,
                "lambda_clu": 0.05,
                "warmup_epochs": 32,
                "fusion_min_weight": 0.005,
            },
            "dcgl_negative_args": {
                "dcgl_neg_tau": 0.5,
                "dcgl_neg_weight": 0.6,
            },
            "safe_tuning_grid": {
                "train_args": {
                    "threshold": [0.35, 0.4, 0.45, 0.5],
                },
                "dual_attn_args": {
                    "fusion_temp": [1.8, 2.0, 2.2],
                    "fusion_balance": [0.25, 0.35],
                    "fusion_min_weight": [0.15, 0.20],
                    "lambda_inst": [0.06, 0.08, 0.10],
                    "lambda_clu": [0.06, 0.08, 0.10],
                    "warmup_epochs": [25, 35, 45],
                },
                "dcgl_negative_args": {
                    "dcgl_neg_tau": [0.35, 0.5, 0.75],
                    "dcgl_neg_weight": [0.3, 0.4, 0.5, 0.6, 0.8],
                },
            },
        },
        "pubmed": {
            "cluster_num": 3,
            "train_args": {
                ### <--- [MODIFIED] ---------------------------------------
                # Main training suggestion: keep the default stable setting.
                "lr": 1e-4,
                ### ---------------------------------------
            },
            "ae_args": {
                "base_graph_path": "data/graph/pubmed_graph.txt",
                ### <--- [MODIFIED] ---------------------------------------
                # AE suggestion: large graph but low-variance sparse features, use a medium lr.
                "lr": 5e-4,
                "n_z": 3,
                "pretrain_seed": 42,
                "graph_seed": 42,
                ### ---------------------------------------
            },
            "dual_attn_args": {
                "fusion_hidden": 64,
                "fusion_temp": 1.6,
                "fusion_balance": 0.25,
                "lambda_inst": 0.08,
                "lambda_clu": 0.06,
                "warmup_epochs": 35,
                "fusion_min_weight": 0.10,
            },
        },
        "uat": {
            "cluster_num": 4,
            "train_args": {
                ### <--- [MODIFIED] ---------------------------------------
                # Final selected stable combo uses a slightly longer horizon,
                # higher propagation depth, and a mild alpha reduction.
                "t": 5,
                "epochs": 500,
                "lr": 1.2e-4,
                "alpha": 0.45,
                ### ---------------------------------------
            },
            "ae_args": {
                "base_graph_path": "data/graph/uat_graph.txt",
                ### <--- [MODIFIED] ---------------------------------------
                # AE suggestion: moderate lr for the smaller low-variance dataset.
                "lr": 3e-4,
                "n_z": 4,
                "pretrain_seed": 42,
                "graph_seed": 42,
                ### ---------------------------------------
            },
            "dual_attn_args": {
                "fusion_hidden": 64,
                "fusion_temp": 1.8,
                "fusion_balance": 0.35,
                "lambda_inst": 0.09,
                "lambda_clu": 0.09,
                "warmup_epochs": 35,
                "fusion_min_weight": 0.20,
            },
            "dcgl_negative_args": {
                "dcgl_neg_tau": 0.5,
                "dcgl_neg_weight": 0.6,
                "disable_dcgl_neg_reliability_gate": True,
            },
            "safe_tuning_grid": {
                "train_args": {
                    "threshold": [0.35, 0.4, 0.45],
                },
                "dual_attn_args": {
                    "fusion_temp": [1.8, 1.9, 2.0, 2.1],
                    "fusion_balance": [0.30, 0.35, 0.40, 0.45],
                    "lambda_inst": [0.06, 0.08, 0.10],
                    "lambda_clu": [0.07, 0.075],
                    "fusion_min_weight": [0.20, 0.22, 0.25],
                    "warmup_epochs": [25, 35, 45],
                },
                "dcgl_negative_args": {
                    "dcgl_neg_tau": [0.5, 0.75, 1.0],
                    "dcgl_neg_weight": [0.3, 0.4, 0.5, 0.6],
                },
            },
        },
    },
    # Shared train args for baseline and ae-train
    "train_common_args": {
        ### <--- [MODIFIED] ---------------------------------------
        # Original CCGC main-train defaults from train.py / paper-aligned setup.
        # Dataset-specific `train_args` below can still override these values.
        "t": 4,
        "linlayers": 1,
        "epochs": 400,
        "dims": 500,
        "lr": 1e-4,
        "device": "cuda",
        "threshold": 0.4,
        "alpha": 0.5,
        "warmup_epochs": 35,
        ### ---------------------------------------
    },

    "baseline_args": {
        "knn_k": 5,
    },
    # Shared dual-train args (graph_mode=dual)
    "dual_args": {
        # Example:
        # "warmup_epochs": 50,
        # "lambda_inst": 0.2,
        # "lambda_clu": 0.2,
        # "dist_tau": 0.5,
    },
    # Extra args for dual-mean branch.
    "dual_mean_args": {
        # Example:
        # "fusion_balance": 0.0,
    },
    # Extra args for dual-attn branch.
    "dual_attn_args": {
        "fusion_hidden": 64,
        "fusion_temp": 1.8,
        "fusion_balance": 0.35,
        "lambda_inst": 0.09,
        "lambda_clu": 0.09,
        "warmup_epochs": 35,
        "fusion_min_weight": 0.10,
        # Example:
        # "fusion_hidden": 128,
        # "fusion_temp": 1.0,
        # "fusion_balance": 0.0,
        # Optional branch-biased attn fusion. Default OFF.
        # When not explicitly passed, train.py keeps the current attn path unchanged.
        # "enable_branch_bias_fusion": True,
        # "branch_bias_target": "raw",  # or "ae"
        # "branch_bias_cap": 0.15,

        #"fusion_hidden": 32,
        #"fusion_temp": 1.8,
        #"fusion_balance": 1,
        #"lambda_inst": 0.15,
        #"lambda_clu": 0.15,
        #"warmup_epochs": 20,
        #fusion_min_weight": 0.1,
        
    },
    # Shared ae-pretrain args
    "ae_args": {
        ### <--- [MODIFIED] ---------------------------------------
        # SIRM/GCRM-style common AE setting from the reference:=
        # pretrain epochs = 30, lr = 1e-3, AE dims = input-500-500-2000-K.
        # Dataset-specific ae_args below only override the fields that need to differ.
        # Set either seed to None to recover the original random behavior.
        "epochs": 30,
        "lr": 1e-3,
        "n_enc_1": 500,
        "n_enc_2": 500,
        "n_enc_3": 2000,
        "n_dec_1": 2000,
        "n_dec_2": 500,
        "n_dec_3": 500,
        "ae_k": 15,
        "sim_method": "cos",
        #"pretrain_seed": 59,
        #"graph_seed": 59,
        ### ---------------------------------------
        # Optional overrides for pretrain_optimize_A_graph.py:
        # "epochs": 30,
        # "n_z": 3,
        # best seed is 59.
    },
}
### ---------------------------------------


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


### <--- [MODIFIED] ---------------------------------------
def _status(message):
    if SHOW_STATUS:
        print(f"[STATUS] {message}", flush=True)
### ---------------------------------------


def _dict_to_cli(args):
    cli = []
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


def _merge_args(*arg_dicts):
    merged = {}
    for d in arg_dicts:
        if not d:
            continue
        merged.update(d)
    return merged


def _resolve_ae_graph_path(root_dir, dataset_name, ae_args):
    graph_relpath = str(ae_args.get("out_graph_path", "")).strip()
    if graph_relpath:
        return root_dir / graph_relpath
    return root_dir / "data" / "ae_graph" / f"{dataset_name}_ae_graph.txt"


def _resolve_ae_model_path(root_dir, dataset_name, ae_args):
    model_relpath = str(ae_args.get("model_save_path", "")).strip()
    if model_relpath:
        return root_dir / model_relpath
    return root_dir / "pretrain_graph" / f"{dataset_name}_ae_pretrain.pkl"


def _seed_summary_text():
    seed_info = CONFIG.get("run_seed_info", {})
    ae_args = CONFIG.get("ae_args", {})
    pretrain_seed = seed_info.get("pretrain_seed", ae_args.get("pretrain_seed", None))
    graph_seed = seed_info.get("graph_seed", ae_args.get("graph_seed", None))
    return f"pretrain_seed={pretrain_seed}, graph_seed={graph_seed}"


def _extract_metrics(text):
    metrics = {}
    for metric in ("ACC", "NMI", "ARI", "F1"):
        pattern = re.compile(rf"^\s*{metric}\s*\|(.+)$", re.MULTILINE)
        m = pattern.search(text)
        if not m:
            continue
        nums = re.findall(r"\d+(?:\.\d+)?", m.group(1))
        if len(nums) >= 2:
            metrics[metric] = {"mean": float(nums[0]), "std": float(nums[1])}
    return metrics


### <--- [MODIFIED] ---------------------------------------
def _append_metrics(summary_lines, metrics):
    if not metrics:
        return
    for m, val in metrics.items():
        summary_lines.append(f"    {m}: mean={val['mean']:.2f} std={val['std']:.2f}")


def _append_delta(summary_lines, title, left_metrics, right_metrics):
    if not left_metrics or not right_metrics:
        return
    summary_lines.append(f"  Delta ({title}):")
    for m in ("ACC", "NMI", "ARI", "F1"):
        if m in left_metrics and m in right_metrics:
            delta = left_metrics[m]["mean"] - right_metrics[m]["mean"]
            summary_lines.append(f"    {m}: {delta:+.2f}")
### ---------------------------------------


def _has_npy_triplet(root_dir, dataset_name):
    dataset_dir = root_dir / "data" / "full_dataset" / dataset_name
    if not dataset_dir.exists():
        return False

    buckets = {}
    for npy_path in dataset_dir.rglob("*.npy"):
        name = npy_path.name
        for key in ("feat", "label", "adj"):
            suffix = f"_{key}.npy"
            if name.endswith(suffix):
                prefix = name[: -len(suffix)]
                buckets.setdefault(prefix, set()).add(key)
                break

    preferred = [dataset_name]
    if dataset_name == "cite":
        preferred.append("citeseer")
    preferred.extend(sorted(buckets.keys()))

    for key in preferred:
        if buckets.get(key) == {"feat", "label", "adj"}:
            return True
    return False


def _check_dataset_files(root_dir, dataset_name, need_base_graph, base_graph_relpath=None):
    missing = []
    has_npy = _has_npy_triplet(root_dir, dataset_name)
    feat = root_dir / "data" / "data" / f"{dataset_name}.txt"
    label = root_dir / "data" / "data" / f"{dataset_name}_label.txt"
    if not has_npy:
        if not feat.exists():
            missing.append(str(feat))
        if not label.exists():
            missing.append(str(label))

    if base_graph_relpath:
        base_graph = root_dir / base_graph_relpath
        if need_base_graph and not base_graph.exists():
            missing.append(str(base_graph))
    else:
        base_graph = root_dir / "data" / "graph" / f"{dataset_name}_graph.txt"
        if need_base_graph and not base_graph.exists() and not has_npy:
            missing.append(str(base_graph))
    return missing, base_graph


def _run_and_log(
    name,
    cmd,
    workdir,
    output_dir,
    improved_module_enabled=False,
    dynamic_threshold_enabled=False,
    ema_prototypes_enabled=False,
    dcgl_negative_enabled=False,
    dcgl_cluster_enabled=False,
    gcn_backbone_enabled=False,
    improved_module_args=None
):
    ### <--- [MODIFIED] ---------------------------------------
    _status(f"{name} started")
    ### ---------------------------------------
    start = time.time()
    seed_text = _seed_summary_text()
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - start
    ### <--- [MODIFIED] ---------------------------------------
    _status(f"{name} done rc={proc.returncode} elapsed={elapsed:.2f}s")
    ### ---------------------------------------

    ### <--- [MODIFIED] ---------------------------------------
    if CONFIG.get("log_by_dataset", True):
        dataset_tag = name.split("_")[0] if "_" in name else "misc"
        log_dir = output_dir / dataset_tag
    else:
        log_dir = output_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{_timestamp()}.txt"
    ### ---------------------------------------
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"COMMAND: {' '.join(cmd)}\n")
        f.write(f"SEEDS: {seed_text}\n")
        ### <--- [MODIFIED] ---------------------------------------
        f.write(f"IMPROVED_MODULE: {'ON' if improved_module_enabled else 'OFF'}\n")
        f.write(f"DYNAMIC_THRESHOLD: {'ON' if dynamic_threshold_enabled else 'OFF'}\n")
        f.write(f"EMA_PROTOTYPES: {'ON' if ema_prototypes_enabled else 'OFF'}\n")
        f.write(f"DCGL_NEGATIVE: {'ON' if dcgl_negative_enabled else 'OFF'}\n")
        f.write(f"DCGL_CLUSTER: {'ON' if dcgl_cluster_enabled else 'OFF'}\n")
        f.write(f"GCN_BACKBONE: {'ON' if gcn_backbone_enabled else 'OFF'}\n")
        if improved_module_args:
            arg_text = " ".join([f"--{k}={v}" for k, v in improved_module_args.items()])
            f.write(f"IMPROVED_MODULE_ARGS: {arg_text}\n")
        ### ---------------------------------------
        f.write(f"RETURN_CODE: {proc.returncode}\n")
        f.write(f"ELAPSED_SEC: {elapsed:.2f}\n")
        f.write("=" * 80 + "\n[STDOUT]\n")
        f.write(proc.stdout if proc.stdout else "")
        f.write("\n" + "=" * 80 + "\n[STDERR]\n")
        f.write(proc.stderr if proc.stderr else "")

    return {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed": elapsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "log_path": log_path,
        "metrics": _extract_metrics(proc.stdout),
    }


### <--- [MODIFIED] ---------------------------------------
def _log_ref(log_path, output_dir):
    try:
        return str(log_path.relative_to(output_dir)).replace("\\", "/")
    except Exception:
        return log_path.name


def _write_note_log(name, output_dir, lines):
    if CONFIG.get("log_by_dataset", True):
        dataset_tag = name.split("_")[0] if "_" in name else "misc"
        log_dir = output_dir / dataset_tag
    else:
        log_dir = output_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{_timestamp()}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
    return log_path
### ---------------------------------------


def main():
    root_dir = Path(__file__).resolve().parent
    output_dir = root_dir / CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    ### <--- [MODIFIED] ---------------------------------------
    _status(f"Experiment started | output_dir={output_dir}")
    ### ---------------------------------------

    summary_lines = []
    ### <--- [MODIFIED] ---------------------------------------
    legacy_enable = CONFIG.get("enable_improved_module", False)
    dynamic_threshold_enabled = CONFIG.get("enable_dynamic_threshold_module", legacy_enable)
    ema_prototypes_enabled = CONFIG.get("enable_ema_prototypes_module", legacy_enable)
    dcgl_negative_enabled = CONFIG.get("enable_dcgl_negative_module", False)
    dcgl_cluster_enabled = CONFIG.get("enable_dcgl_cluster_module", False)
    gcn_backbone_enabled = CONFIG.get("enable_gcn_backbone_module", False)
    reuse_existing_ae_results = CONFIG.get("reuse_existing_ae_results", False)
    improved_module_enabled = bool(
        dynamic_threshold_enabled
        or ema_prototypes_enabled
        or dcgl_negative_enabled
        or dcgl_cluster_enabled
        or gcn_backbone_enabled
    )

    legacy_args = CONFIG.get("improved_module_args", {})
    improved_module_args = {}
    if dynamic_threshold_enabled:
        improved_module_args["enable_dynamic_threshold"] = True
        improved_module_args.update(CONFIG.get("dynamic_threshold_args", {}))
        # legacy compatibility for old arg container
        for key in ("dynamic_threshold_start", "dynamic_threshold_end"):
            if key in legacy_args and key not in improved_module_args:
                improved_module_args[key] = legacy_args[key]
    if ema_prototypes_enabled:
        improved_module_args["enable_ema_prototypes"] = True
        improved_module_args.update(CONFIG.get("ema_prototypes_args", {}))
        # legacy compatibility for old arg container
        if "ema_proto_momentum" in legacy_args and "ema_proto_momentum" not in improved_module_args:
            improved_module_args["ema_proto_momentum"] = legacy_args["ema_proto_momentum"]
    if dcgl_negative_enabled:
        improved_module_args["enable_dcgl_negative_loss"] = True
        improved_module_args.update(CONFIG.get("dcgl_negative_args", {}))
    if dcgl_cluster_enabled:
        improved_module_args["enable_dcgl_cluster_level"] = True
        improved_module_args.update(CONFIG.get("dcgl_cluster_args", {}))
    if gcn_backbone_enabled:
        improved_module_args["enable_gcn_backbone"] = True
        improved_module_args.update(CONFIG.get("gcn_backbone_args", {}))
    ### ---------------------------------------
    summary_lines.append(f"Experiment started at: {datetime.now().isoformat()}")
    summary_lines.append(f"Project root: {root_dir}")
    summary_lines.append(f"Seeds: {_seed_summary_text()}")
    ### <--- [MODIFIED] ---------------------------------------
    summary_lines.append(f"Improved Module: {'ON' if improved_module_enabled else 'OFF'}")
    summary_lines.append(f"Dynamic Threshold: {'ON' if dynamic_threshold_enabled else 'OFF'}")
    summary_lines.append(f"EMA Prototypes: {'ON' if ema_prototypes_enabled else 'OFF'}")
    summary_lines.append(f"DCGL Negative: {'ON' if dcgl_negative_enabled else 'OFF'}")
    summary_lines.append(f"DCGL Cluster: {'ON' if dcgl_cluster_enabled else 'OFF'}")
    summary_lines.append(f"GCN Backbone: {'ON' if gcn_backbone_enabled else 'OFF'}")
    summary_lines.append(f"Reuse Existing AE Results: {'ON' if reuse_existing_ae_results else 'OFF'}")
    if improved_module_args:
        arg_text = " ".join([f"--{k}={v}" for k, v in improved_module_args.items()])
        summary_lines.append(f"Improved Module Args: {arg_text}")
    ### ---------------------------------------
    summary_lines.append("")

    python_exe = sys.executable

    for dataset in CONFIG["active_datasets"]:
        ### <--- [MODIFIED] ---------------------------------------
        _status(f"Dataset {dataset} started")
        ### ---------------------------------------
        ### <--- [MODIFIED] ---------------------------------------
        run_dual_mean = CONFIG.get("run_dual_mean", False) or CONFIG.get("run_dual", False)
        run_dual_attn = CONFIG.get("run_dual_attn", False)
        ### ---------------------------------------
        profile = CONFIG["dataset_profiles"].get(dataset, {})
        if not profile:
            summary_lines.append(f"[Dataset] {dataset}")
            summary_lines.append("  Status: SKIPPED (missing dataset profile in CONFIG['dataset_profiles'])")
            summary_lines.append("")
            continue

        cluster_num = profile["cluster_num"]
        ### <--- [MODIFIED] ---------------------------------------
        dataset_train_args = profile.get("train_args", {})
        merged_train_args = _merge_args(CONFIG["train_common_args"], dataset_train_args)
        dataset_dual_attn_args = profile.get("dual_attn_args", {})
        merged_dual_attn_args = _merge_args(CONFIG.get("dual_attn_args", {}), dataset_dual_attn_args)
        dataset_dcgl_negative_args = profile.get("dcgl_negative_args", {})
        dataset_improved_module_args = dict(improved_module_args)
        if dcgl_negative_enabled:
            dataset_improved_module_args["enable_dcgl_negative_loss"] = True
            dataset_improved_module_args.update(CONFIG.get("dcgl_negative_args", {}))
            dataset_improved_module_args.update(dataset_dcgl_negative_args)
        if dcgl_cluster_enabled:
            dataset_improved_module_args["enable_dcgl_cluster_level"] = True
            dataset_improved_module_args.update(CONFIG.get("dcgl_cluster_args", {}))
        if gcn_backbone_enabled:
            dataset_improved_module_args["enable_gcn_backbone"] = True
            dataset_improved_module_args.update(CONFIG.get("gcn_backbone_args", {}))
        ### ---------------------------------------
        dataset_ae_args = profile.get("ae_args", {})
        ### <--- [MODIFIED] ---------------------------------------
        # Prefer npy adj as AE-pretrain structural prior when the original npy triplet exists.
        # Keep explicit edge-list base graphs for datasets without npy triplets (e.g. usps/reut/hhar).
        has_npy_triplet = _has_npy_triplet(root_dir, dataset)
        base_graph_relpath = None if has_npy_triplet else dataset_ae_args.get("base_graph_path", None)
        ae_base_graph_source = "npy_adj" if has_npy_triplet else (
            dataset_ae_args.get("base_graph_path", "default_edge_list")
        )
        ### ---------------------------------------
        summary_lines.append(f"[Dataset] {dataset} (cluster_num={cluster_num})")
        ### <--- [MODIFIED] ---------------------------------------
        summary_lines.append(f"  AE Base Graph Source: {ae_base_graph_source}")
        ### ---------------------------------------

        missing, _ = _check_dataset_files(
            root_dir,
            dataset,
            need_base_graph=(CONFIG["run_ae"] or run_dual_mean or run_dual_attn),
            base_graph_relpath=base_graph_relpath
        )
        if missing:
            summary_lines.append("  Status: SKIPPED (missing files)")
            for item in missing:
                summary_lines.append(f"  Missing: {item}")
            summary_lines.append("")
            continue

        baseline_result = None
        ae_result = None
        dual_mean_result = None
        dual_attn_result = None

        # Baseline: original graph A if available, otherwise KNN fallback
        if CONFIG["run_baseline"]:
            baseline_cmd = [
                python_exe,
                "train.py",
                "--dataset", dataset,
                "--cluster_num", str(cluster_num),
                "--graph_mode", "raw",
            ] + _dict_to_cli(CONFIG["baseline_args"]) + _dict_to_cli(merged_train_args) + _dict_to_cli(dataset_improved_module_args)

            baseline_result = _run_and_log(
                name=f"{dataset}_baseline_raw",
                cmd=baseline_cmd,
                workdir=root_dir,
                output_dir=output_dir,
                improved_module_enabled=improved_module_enabled,
                dynamic_threshold_enabled=dynamic_threshold_enabled,
                ema_prototypes_enabled=ema_prototypes_enabled,
                dcgl_negative_enabled=dcgl_negative_enabled,
                dcgl_cluster_enabled=dcgl_cluster_enabled,
                gcn_backbone_enabled=gcn_backbone_enabled,
                improved_module_args=dataset_improved_module_args,
            )
            summary_lines.append(
                f"  Baseline: rc={baseline_result['returncode']} elapsed={baseline_result['elapsed']:.2f}s "
                f"log={_log_ref(baseline_result['log_path'], output_dir)}"
            )
            _append_metrics(summary_lines, baseline_result["metrics"])

        # Improved: generate Ae graph offline once, then train with graph_mode=ae and/or dual
        if CONFIG["run_ae"] or run_dual_mean or run_dual_attn:
            merged_ae_args = _merge_args(CONFIG["ae_args"], dataset_ae_args)
            ### <--- [MODIFIED] ---------------------------------------
            # Let pretrain_optimize_A_graph.py use adj.npy directly when available.
            if has_npy_triplet:
                merged_ae_args.pop("base_graph_path", None)
            ### ---------------------------------------
            ae_graph_path = _resolve_ae_graph_path(root_dir, dataset, merged_ae_args)
            ae_model_path = _resolve_ae_model_path(root_dir, dataset, merged_ae_args)

            if reuse_existing_ae_results:
                reuse_log_path = _write_note_log(
                    name=f"{dataset}_ae_pretrain_reuse",
                    output_dir=output_dir,
                    lines=[
                        "MODE: REUSE_EXISTING_AE_RESULTS",
                        f"SEEDS: {_seed_summary_text()}",
                        f"AE_GRAPH_PATH: {ae_graph_path}",
                        f"AE_MODEL_PATH: {ae_model_path}",
                        "AE_PRETRAIN_EXECUTION: SKIPPED",
                        "NOTE: experiment.py reused fixed AE preprocessing assets and did not run data/pretrain_optimize_A_graph.py",
                    ],
                )
                summary_lines.append(
                    f"  AE Pretrain: rc=0 elapsed=0.00s log={_log_ref(reuse_log_path, output_dir)} "
                    f"(reused existing Ae graph)"
                )
            else:
                pretrain_cmd = [
                    python_exe,
                    "data/pretrain_optimize_A_graph.py",
                    "--dataset", dataset,
                    "--cluster_num", str(cluster_num),
                ] + _dict_to_cli(merged_ae_args)

                pre_result = _run_and_log(
                    name=f"{dataset}_ae_pretrain",
                    cmd=pretrain_cmd,
                    workdir=root_dir,
                    output_dir=output_dir,
                    improved_module_enabled=improved_module_enabled,
                    dynamic_threshold_enabled=dynamic_threshold_enabled,
                    ema_prototypes_enabled=ema_prototypes_enabled,
                    dcgl_negative_enabled=dcgl_negative_enabled,
                    dcgl_cluster_enabled=dcgl_cluster_enabled,
                    gcn_backbone_enabled=gcn_backbone_enabled,
                    improved_module_args=dataset_improved_module_args,
                )
                summary_lines.append(
                    f"  AE Pretrain: rc={pre_result['returncode']} elapsed={pre_result['elapsed']:.2f}s "
                    f"log={_log_ref(pre_result['log_path'], output_dir)}"
                )

                if pre_result["returncode"] != 0:
                    if CONFIG["run_ae"]:
                        summary_lines.append("  AE Train: SKIPPED (pretrain failed)")
                    if run_dual_mean:
                        summary_lines.append("  Dual Mean Train: SKIPPED (pretrain failed)")
                    if run_dual_attn:
                        summary_lines.append("  Dual Attn Train: SKIPPED (pretrain failed)")
                    summary_lines.append("")
                    continue

            if not ae_graph_path.exists():
                if CONFIG["run_ae"]:
                    summary_lines.append(
                        f"  AE Train: SKIPPED (missing Ae graph: {ae_graph_path})"
                    )
                if run_dual_mean:
                    summary_lines.append(
                        f"  Dual Mean Train: SKIPPED (missing Ae graph: {ae_graph_path})"
                    )
                if run_dual_attn:
                    summary_lines.append(
                        f"  Dual Attn Train: SKIPPED (missing Ae graph: {ae_graph_path})"
                    )
                summary_lines.append("")
                continue

            if CONFIG["run_ae"]:
                ae_train_cmd = [
                    python_exe,
                    "train.py",
                    "--dataset", dataset,
                    "--cluster_num", str(cluster_num),
                    "--graph_mode", "ae",
                    "--ae_graph_path", str(ae_graph_path),
                ] + _dict_to_cli(merged_train_args) + _dict_to_cli(dataset_improved_module_args)

                ae_result = _run_and_log(
                    name=f"{dataset}_train_with_ae",
                    cmd=ae_train_cmd,
                    workdir=root_dir,
                    output_dir=output_dir,
                    improved_module_enabled=improved_module_enabled,
                    dynamic_threshold_enabled=dynamic_threshold_enabled,
                    ema_prototypes_enabled=ema_prototypes_enabled,
                    dcgl_negative_enabled=dcgl_negative_enabled,
                    dcgl_cluster_enabled=dcgl_cluster_enabled,
                    gcn_backbone_enabled=gcn_backbone_enabled,
                    improved_module_args=dataset_improved_module_args,
                )
                summary_lines.append(
                    f"  AE Train: rc={ae_result['returncode']} elapsed={ae_result['elapsed']:.2f}s "
                    f"log={_log_ref(ae_result['log_path'], output_dir)}"
                )
                _append_metrics(summary_lines, ae_result["metrics"])

            if run_dual_mean:
                dual_mean_cmd = [
                    python_exe,
                    "train.py",
                    "--dataset", dataset,
                    "--cluster_num", str(cluster_num),
                    "--graph_mode", "dual",
                    "--ae_graph_path", str(ae_graph_path),
                    "--fusion_mode", "mean",
                ] + _dict_to_cli(CONFIG["baseline_args"]) + _dict_to_cli(merged_train_args) + _dict_to_cli(CONFIG["dual_args"]) + _dict_to_cli(CONFIG.get("dual_mean_args", {})) + _dict_to_cli(dataset_improved_module_args)

                dual_mean_result = _run_and_log(
                    name=f"{dataset}_train_with_dual_mean",
                    cmd=dual_mean_cmd,
                    workdir=root_dir,
                    output_dir=output_dir,
                    improved_module_enabled=improved_module_enabled,
                    dynamic_threshold_enabled=dynamic_threshold_enabled,
                    ema_prototypes_enabled=ema_prototypes_enabled,
                    dcgl_negative_enabled=dcgl_negative_enabled,
                    dcgl_cluster_enabled=dcgl_cluster_enabled,
                    gcn_backbone_enabled=gcn_backbone_enabled,
                    improved_module_args=dataset_improved_module_args,
                )
                summary_lines.append(
                    f"  Dual Mean Train: rc={dual_mean_result['returncode']} elapsed={dual_mean_result['elapsed']:.2f}s "
                    f"log={_log_ref(dual_mean_result['log_path'], output_dir)}"
                )
                _append_metrics(summary_lines, dual_mean_result["metrics"])

            if run_dual_attn:
                dual_attn_cmd = [
                    python_exe,
                    "train.py",
                    "--dataset", dataset,
                    "--cluster_num", str(cluster_num),
                    "--graph_mode", "dual",
                    "--ae_graph_path", str(ae_graph_path),
                    "--fusion_mode", "attn",
                ] + _dict_to_cli(CONFIG["baseline_args"]) + _dict_to_cli(merged_train_args) + _dict_to_cli(CONFIG["dual_args"]) + _dict_to_cli(merged_dual_attn_args) + _dict_to_cli(dataset_improved_module_args)

                dual_attn_result = _run_and_log(
                    name=f"{dataset}_train_with_dual_attn",
                    cmd=dual_attn_cmd,
                    workdir=root_dir,
                    output_dir=output_dir,
                    improved_module_enabled=improved_module_enabled,
                    dynamic_threshold_enabled=dynamic_threshold_enabled,
                    ema_prototypes_enabled=ema_prototypes_enabled,
                    dcgl_negative_enabled=dcgl_negative_enabled,
                    dcgl_cluster_enabled=dcgl_cluster_enabled,
                    gcn_backbone_enabled=gcn_backbone_enabled,
                    improved_module_args=dataset_improved_module_args,
                )
                summary_lines.append(
                    f"  Dual Attn Train: rc={dual_attn_result['returncode']} elapsed={dual_attn_result['elapsed']:.2f}s "
                    f"log={_log_ref(dual_attn_result['log_path'], output_dir)}"
                )
                _append_metrics(summary_lines, dual_attn_result["metrics"])

        # Delta summary for quick reproduction comparison
        if baseline_result and baseline_result["metrics"]:
            if ae_result and ae_result["metrics"]:
                _append_delta(summary_lines, "AE - Baseline", ae_result["metrics"], baseline_result["metrics"])
            if dual_mean_result and dual_mean_result["metrics"]:
                _append_delta(summary_lines, "Dual Mean - Baseline", dual_mean_result["metrics"], baseline_result["metrics"])
            if dual_attn_result and dual_attn_result["metrics"]:
                _append_delta(summary_lines, "Dual Attn - Baseline", dual_attn_result["metrics"], baseline_result["metrics"])
            if dual_mean_result and dual_mean_result["metrics"] and ae_result and ae_result["metrics"]:
                _append_delta(summary_lines, "Dual Mean - AE", dual_mean_result["metrics"], ae_result["metrics"])
            if dual_attn_result and dual_attn_result["metrics"] and ae_result and ae_result["metrics"]:
                _append_delta(summary_lines, "Dual Attn - AE", dual_attn_result["metrics"], ae_result["metrics"])
            if dual_attn_result and dual_attn_result["metrics"] and dual_mean_result and dual_mean_result["metrics"]:
                _append_delta(summary_lines, "Dual Attn - Dual Mean", dual_attn_result["metrics"], dual_mean_result["metrics"])

        summary_lines.append("")

    summary_lines.append(f"Experiment finished at: {datetime.now().isoformat()}")
    summary_path = output_dir / f"summary_{_timestamp()}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    ### <--- [MODIFIED] ---------------------------------------
    _status(f"Experiment finished | summary={summary_path}")
    ### ---------------------------------------
    print(f"Experiment done. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
