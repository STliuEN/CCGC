import json
from itertools import product
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import threading
import time 
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import experiment


MP_CONTEXT = mp.get_context("spawn")


### <--- [MODIFIED] ---------------------------------------
# Parallel batch runner:
# - uses batch config structs to organize repeated experiments
# - each struct mainly controls experiment.CONFIG boolean switch groups
# - ACTIVE_BATCH_CONFIG_COUNT controls how many structs join this batch run
# - repeat_times means one full experiment.main() pass, matching experiment.py behavior
# - default mode is full module ablation over all optional switches
# - use a configurable number of worker threads at the orchestration layer
# - each experiment still runs through experiment.main() unchanged in its own process
# - print real-time CPU / RAM / GPU usage snapshots while jobs are running

MAX_WORKER_THREADS = 1
MONITOR_INTERVAL_SEC = 5.0
PROCESS_POLL_INTERVAL_SEC = 1.0
CHILD_RESULT_GRACE_SEC = 15.0
CHILD_JOIN_TIMEOUT_SEC = 5.0
CHILD_KILL_TIMEOUT_SEC = 5.0
RUN_OUTPUT_SUBDIR = "paper_main_ablation"
STATE_FILE_NAME = "test_ablation_state_paper_main.json"
SENSITIVITY_OUTPUT_SUBDIR = "paper_param_sensitivity_all_datasets"
SENSITIVITY_STATE_FILE_NAME = "test_sensitivity_state_all_datasets.json"

# Optional non-switch overrides applied to every run.
# For switch-group oriented batch experiments, leave this empty.
COMMON_CONFIG_OVERRIDES = {
    # Paper main ablation datasets:
    # Reuters / UAT / AMAP / USPS / EAT / Cora / Citeseer.
    "active_datasets": ["reut", "uat", "amap", "usps", "eat", "cora", "cite"],
}

# Seed control for AE pretrain / graph generation runs driven by experiment.py.
# - mode="fixed": every run uses the same base seed
# - mode="random": each run samples a new seed from RNG_SEED
# - mode="sequence": each run uses base_seed + run_index - 1
# Set apply_to_experiment=False to disable seed injection from this batch runner.
SEED_CONTROL = {
    "apply_to_experiment": True,
    "mode": "fixed",  # fixed | random | sequence
    "base_seed": 42,
    "rng_seed": 20260420,
}

# Control how many batch config structs participate in this run.
# None means all enabled structs below will participate.
ACTIVE_BATCH_CONFIG_COUNT = None

# Boolean switches from experiment.CONFIG that this batch runner manages directly.
CONFIG_SWITCH_DEFAULTS = {
    "log_by_dataset": True,
    "run_baseline": False,
    "run_ae": False,
    "run_dual": False,
    "run_dual_mean": False,
    "run_dual_attn": False,
    "enable_dynamic_threshold_module": False,
    "enable_ema_prototypes_module": False,
    "enable_dcgl_negative_module": False,
    "enable_dcgl_cluster_module": False,
    "enable_gcn_backbone_module": False,
}

MODULE_SWITCH_KEYS = [
    "enable_dynamic_threshold_module",
    "enable_ema_prototypes_module",
    "enable_dcgl_negative_module",
    "enable_dcgl_cluster_module",
    "enable_gcn_backbone_module",
]

# Batch build mode:
# - "full_module_ablation": generate all 2^N on/off combinations for module switches
# - "manual": use MANUAL_BATCH_CONFIGS as-is
# - "parameter_sensitivity": generate runs from SENSITIVITY_SPECS
BATCH_BUILD_MODE = "parameter_sensitivity"

FULL_ABLATION_REPEAT_TIMES = 1
FULL_ABLATION_COMMON_SWITCHES = {
    "run_baseline": True,
    "run_ae": True,
    "run_dual": False,
    "run_dual_mean": True,
    "run_dual_attn": True,
}
FULL_ABLATION_MODULE_SPECS = [
    ("enable_dynamic_threshold_module", "dynamic"),
    ("enable_ema_prototypes_module", "ema"),
    ("enable_dcgl_negative_module", "dcgl_neg"),
    ("enable_dcgl_cluster_module", "dcgl_clu"),
    ("enable_gcn_backbone_module", "gcn"),
]

# Parameter sensitivity mode.
# To use it, set BATCH_BUILD_MODE = "parameter_sensitivity".
# The generated runs use the paper full setting by default:
# dual-attention fusion + cluster-level negative separation.
# MAX_WORKER_THREADS=1 is conservative for full-dataset sensitivity runs.
# ae_k runs write separate AE graph paths under data/ae_graph/sensitivity/.
SENSITIVITY_REPEAT_TIMES = 1
SENSITIVITY_ACTIVE_DATASETS = ["reut", "uat", "amap", "usps", "eat", "cora", "cite"]
SENSITIVITY_COMMON_CONFIG_OVERRIDES = {
    "active_datasets": SENSITIVITY_ACTIVE_DATASETS,
    # Keep False unless the corresponding AE graph already exists for every run.
    # For ae_k sensitivity this must stay False, otherwise the changed graph
    # construction parameter will not take effect.
    "reuse_existing_ae_results": False,
}
SENSITIVITY_BASE_SWITCHES = {
    "run_baseline": False,
    "run_ae": False,
    "run_dual": False,
    "run_dual_mean": False,
    "run_dual_attn": True,
    "enable_dynamic_threshold_module": False,
    "enable_ema_prototypes_module": False,
    "enable_dcgl_negative_module": True,
    "enable_dcgl_cluster_module": False,
    "enable_gcn_backbone_module": False,
}
SENSITIVITY_SPECS = [
    {
        "name": "fusion_temp",
        "enabled": True,
        "values": [1.0, 1.3, 1.6, 1.9, 2.2],
        "paths": ["dataset_profiles.{dataset}.dual_attn_args.fusion_temp"],
    },
    {
        "name": "fusion_min_weight",
        "enabled": True,
        "values": [0.0, 0.05, 0.10, 0.15, 0.20],
        "paths": ["dataset_profiles.{dataset}.dual_attn_args.fusion_min_weight"],
    },
    {
        "name": "ae_k",
        "enabled": True,
        "values": [5, 10, 15, 20, 25],
        "paths": ["ae_args.ae_k"],
    },
    {
        "name": "dcgl_neg_weight",
        "enabled": True,
        "values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "paths": ["dcgl_negative_args.dcgl_neg_weight"],
    },
    {
        "name": "fusion_balance",
        "enabled": True,
        "values": [0.0, 0.05, 0.10, 0.20, 0.35],
        "paths": ["dataset_profiles.{dataset}.dual_attn_args.fusion_balance"],
    },
    {
        "name": "lambda_inst",
        "enabled": True,
        "values": [0.0, 0.03, 0.06, 0.08, 0.10],
        "paths": ["dataset_profiles.{dataset}.dual_attn_args.lambda_inst"],
    },
    {
        "name": "lambda_clu",
        "enabled": True,
        "values": [0.0, 0.01, 0.03, 0.05, 0.07],
        "paths": ["dataset_profiles.{dataset}.dual_attn_args.lambda_clu"],
    },
    {
        "name": "dcgl_neg_tau",
        "enabled": True,
        "values": [0.2, 0.35, 0.5, 0.75, 1.0],
        "paths": ["dcgl_negative_args.dcgl_neg_tau"],
    },
]

MANUAL_BATCH_CONFIGS = [
    {
        "name": "01_A_only",
        "enabled": True,
        "repeat_times": 1,
        "switches": {
            "run_baseline": True,
            "run_ae": False,
            "run_dual": False,
            "run_dual_mean": False,
            "run_dual_attn": False,
            "enable_dynamic_threshold_module": False,
            "enable_ema_prototypes_module": False,
            "enable_dcgl_negative_module": False,
            "enable_dcgl_cluster_module": False,
            "enable_gcn_backbone_module": False,
        },
        "config_overrides": {},
    },
    {
        "name": "02_AE_only",
        "enabled": True,
        "repeat_times": 1,
        "switches": {
            "run_baseline": False,
            "run_ae": True,
            "run_dual": False,
            "run_dual_mean": False,
            "run_dual_attn": False,
            "enable_dynamic_threshold_module": False,
            "enable_ema_prototypes_module": False,
            "enable_dcgl_negative_module": False,
            "enable_dcgl_cluster_module": False,
            "enable_gcn_backbone_module": False,
        },
        "config_overrides": {},
    },
    {
        "name": "03_Dual_Mean",
        "enabled": True,
        "repeat_times": 1,
        "switches": {
            "run_baseline": False,
            "run_ae": False,
            "run_dual": False,
            "run_dual_mean": True,
            "run_dual_attn": False,
            "enable_dynamic_threshold_module": False,
            "enable_ema_prototypes_module": False,
            "enable_dcgl_negative_module": False,
            "enable_dcgl_cluster_module": False,
            "enable_gcn_backbone_module": False,
        },
        "config_overrides": {},
    },
    {
        "name": "04_Dual_Attn",
        "enabled": True,
        "repeat_times": 1,
        "switches": {
            "run_baseline": False,
            "run_ae": False,
            "run_dual": False,
            "run_dual_mean": False,
            "run_dual_attn": True,
            "enable_dynamic_threshold_module": False,
            "enable_ema_prototypes_module": False,
            "enable_dcgl_negative_module": False,
            "enable_dcgl_cluster_module": False,
            "enable_gcn_backbone_module": False,
        },
        "config_overrides": {},
    },
    {
        "name": "05_Full",
        "enabled": True,
        "repeat_times": 1,
        "switches": {
            "run_baseline": False,
            "run_ae": False,
            "run_dual": False,
            "run_dual_mean": False,
            "run_dual_attn": True,
            "enable_dynamic_threshold_module": False,
            "enable_ema_prototypes_module": False,
            "enable_dcgl_negative_module": True,
            "enable_dcgl_cluster_module": False,
            "enable_gcn_backbone_module": False,
        },
        "config_overrides": {},
    },
]


def _deep_merge_dict(base, override):
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _set_by_path(target, path, value):
    parts = str(path).split(".")
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"Invalid override path: {path!r}")
    cursor = target
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = deepcopy(value)


def _format_value_for_name(value):
    text = str(value).replace("-", "m").replace(".", "p")
    return text.replace(" ", "")


def _build_path_overrides(paths, value, datasets):
    overrides = {}
    for raw_path in paths:
        if "{dataset}" in raw_path:
            for dataset in datasets:
                _set_by_path(overrides, raw_path.format(dataset=dataset), value)
        else:
            _set_by_path(overrides, raw_path, value)
    return overrides


def _build_full_module_ablation_configs():
    configs = []
    for states in product([False, True], repeat=len(FULL_ABLATION_MODULE_SPECS)):
        module_switches = {
            key: state for (key, _label), state in zip(FULL_ABLATION_MODULE_SPECS, states)
        }
        name = "_".join(
            [
                f"{label}_{'on' if state else 'off'}"
                for (_key, label), state in zip(FULL_ABLATION_MODULE_SPECS, states)
            ]
        )
        configs.append(
            {
                "name": name,
                "enabled": True,
                "repeat_times": FULL_ABLATION_REPEAT_TIMES,
                "switches": {
                    **FULL_ABLATION_COMMON_SWITCHES,
                    **module_switches,
                },
                "config_overrides": {},
            }
        )
    return configs


def _build_parameter_sensitivity_configs():
    configs = []
    datasets = list(SENSITIVITY_ACTIVE_DATASETS)
    for spec in SENSITIVITY_SPECS:
        if not spec.get("enabled", True):
            continue
        spec_name = spec["name"]
        paths = spec.get("paths", [])
        values = spec.get("values", [])
        for value in values:
            value_name = _format_value_for_name(value)
            path_overrides = _build_path_overrides(paths, value, datasets)
            if spec_name == "ae_k":
                for dataset in datasets:
                    _set_by_path(
                        path_overrides,
                        f"dataset_profiles.{dataset}.ae_args.out_graph_path",
                        f"data/ae_graph/sensitivity/ae_k_{value_name}/{dataset}_ae_graph.txt",
                    )
            config_overrides = _deep_merge_dict(
                SENSITIVITY_COMMON_CONFIG_OVERRIDES,
                path_overrides,
            )
            configs.append(
                {
                    "name": f"{spec_name}_{value_name}",
                    "enabled": True,
                    "repeat_times": SENSITIVITY_REPEAT_TIMES,
                    "switches": deepcopy(SENSITIVITY_BASE_SWITCHES),
                    "config_overrides": config_overrides,
                    "sensitivity": {
                        "parameter": spec_name,
                        "value": value,
                        "paths": deepcopy(paths),
                    },
                }
            )
    return configs


if BATCH_BUILD_MODE == "full_module_ablation":
    BATCH_CONFIGS = _build_full_module_ablation_configs()
elif BATCH_BUILD_MODE == "manual":
    BATCH_CONFIGS = deepcopy(MANUAL_BATCH_CONFIGS)
elif BATCH_BUILD_MODE == "parameter_sensitivity":
    RUN_OUTPUT_SUBDIR = SENSITIVITY_OUTPUT_SUBDIR
    STATE_FILE_NAME = SENSITIVITY_STATE_FILE_NAME
    BATCH_CONFIGS = _build_parameter_sensitivity_configs()
else:
    raise ValueError(f"Unsupported BATCH_BUILD_MODE: {BATCH_BUILD_MODE}")

# Backward-compatible alias for older naming in this file.
ABLATION_GROUPS = BATCH_CONFIGS


def _normalize_group_switches(group):
    switches = deepcopy(CONFIG_SWITCH_DEFAULTS)
    switches.update(group.get("switches", {}))
    return switches


def _normalize_group_overrides(group):
    overrides = deepcopy(COMMON_CONFIG_OVERRIDES)
    overrides = _deep_merge_dict(overrides, group.get("config_overrides", {}))
    return overrides


def _select_active_groups(groups):
    enabled_groups = [deepcopy(group) for group in groups if group.get("enabled", True)]
    if ACTIVE_BATCH_CONFIG_COUNT is None:
        return enabled_groups
    active_count = max(0, int(ACTIVE_BATCH_CONFIG_COUNT))
    return enabled_groups[:active_count]


def _build_runs(groups):
    runs = []
    seed_cfg = deepcopy(SEED_CONTROL)
    apply_seed = bool(seed_cfg.get("apply_to_experiment", False))
    seed_mode = str(seed_cfg.get("mode", "random")).strip().lower()
    base_seed = int(seed_cfg.get("base_seed", 42))
    rng = random.Random(int(seed_cfg.get("rng_seed", base_seed)))
    for group_idx, group in enumerate(groups, start=1):
        repeat_total = max(1, int(group.get("repeat_times", 1)))
        switches = _normalize_group_switches(group)
        overrides = _normalize_group_overrides(group)
        for repeat_idx in range(1, repeat_total + 1):
            run_order_idx = len(runs) + 1
            if apply_seed:
                if seed_mode == "fixed":
                    run_seed = base_seed
                elif seed_mode == "sequence":
                    run_seed = base_seed + run_order_idx - 1
                else:
                    run_seed = rng.randint(0, 2**31 - 1)
                seed_info = {
                    "pretrain_seed": run_seed,
                    "graph_seed": run_seed,
                }
            else:
                seed_info = {
                    "pretrain_seed": None,
                    "graph_seed": None,
                }
            runs.append(
                {
                    "order_idx": run_order_idx,
                    "group_idx": group_idx,
                    "name": f"{group['name']}_run{repeat_idx}",
                    "group_name": group["name"],
                    "repeat_idx": repeat_idx,
                    "repeat_total": repeat_total,
                    "switches": deepcopy(switches),
                    "config_overrides": deepcopy(overrides),
                    "seed_info": seed_info,
                    "sensitivity": deepcopy(group.get("sensitivity")),
                }
            )
    return runs


def _group_config_text(group):
    switches = _normalize_group_switches(group)
    overrides = _normalize_group_overrides(group)
    switch_text = ", ".join([f"{key}={value}" for key, value in switches.items()])
    override_text = json.dumps(overrides, ensure_ascii=False, sort_keys=True) if overrides else "none"
    base_text = (
        f"{group.get('name', 'unnamed')}: enabled={group.get('enabled', True)} | "
        f"repeat_times={max(1, int(group.get('repeat_times', 1)))} | "
        f"switches[{switch_text}] | overrides[{override_text}]"
    )
    sensitivity = group.get("sensitivity")
    if sensitivity:
        base_text += (
            f" | sensitivity[{sensitivity.get('parameter')}="
            f"{sensitivity.get('value')} paths={sensitivity.get('paths')}]"
        )
    return base_text


ACTIVE_ABLATION_GROUPS = _select_active_groups(ABLATION_GROUPS)
RUNS = _build_runs(ACTIVE_ABLATION_GROUPS)
### ---------------------------------------

def _state_path(output_dir):
    return output_dir / STATE_FILE_NAME


def _load_state(output_dir):
    state_path = _state_path(output_dir)
    run_names = [run["name"] for run in RUNS]
    default_state = {"runs": run_names, "completed_runs": []}
    if not state_path.exists():
        return default_state
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return default_state

    if state.get("runs") != run_names:
        return default_state

    completed_runs = [name for name in state.get("completed_runs", []) if name in run_names]
    return {"runs": run_names, "completed_runs": completed_runs}


def _save_state(output_dir, completed_runs):
    state = {
        "runs": [run["name"] for run in RUNS],
        "completed_runs": sorted(completed_runs),
        "updated_at": datetime.now().isoformat(),
    }
    _state_path(output_dir).write_text(
        json.dumps(state, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _reset_config_switches(cfg):
    for key, default_value in CONFIG_SWITCH_DEFAULTS.items():
        cfg[key] = default_value
    cfg["enable_improved_module"] = False


def _apply_run(cfg, run_spec):
    _reset_config_switches(cfg)
    for key, value in run_spec.get("switches", {}).items():
        cfg[key] = value
    cfg = _deep_merge_dict(cfg, run_spec.get("config_overrides", {}))
    seed_info = run_spec.get("seed_info", {})
    ae_args = deepcopy(cfg.get("ae_args", {}))
    if "pretrain_seed" in seed_info:
        ae_args["pretrain_seed"] = seed_info.get("pretrain_seed")
    if "graph_seed" in seed_info:
        ae_args["graph_seed"] = seed_info.get("graph_seed")
    cfg["ae_args"] = ae_args
    cfg["run_seed_info"] = deepcopy(seed_info)
    cfg["enable_improved_module"] = any(bool(cfg.get(key, False)) for key in MODULE_SWITCH_KEYS)
    return cfg


def _module_flags_text(cfg):
    short_names = {
        "log_by_dataset": "log_by_dataset",
        "run_baseline": "baseline",
        "run_ae": "ae",
        "run_dual": "dual_legacy",
        "run_dual_mean": "dual_mean",
        "run_dual_attn": "dual_attn",
        "enable_dynamic_threshold_module": "dynamic",
        "enable_ema_prototypes_module": "ema",
        "enable_dcgl_negative_module": "dcgl_neg",
        "enable_dcgl_cluster_module": "dcgl_clu",
        "enable_gcn_backbone_module": "gcn",
    }
    return ", ".join(
        [f"{short_names.get(key, key)}={cfg.get(key, False)}" for key in CONFIG_SWITCH_DEFAULTS.keys()]
    )


def _seed_text(seed_info):
    if not seed_info:
        return "pretrain_seed=None, graph_seed=None"
    return ", ".join(
        [
            f"pretrain_seed={seed_info.get('pretrain_seed', None)}",
            f"graph_seed={seed_info.get('graph_seed', None)}",
        ]
    )


def _run_output_relpath(run_spec):
    return f"experiment_output/{RUN_OUTPUT_SUBDIR}/{run_spec['group_name']}/{run_spec['name']}"


def _child_run_experiment(cfg, result_conn):
    status = "OK"
    detail = ""
    try:
        experiment.CONFIG = cfg
        experiment.main()
    except Exception:
        status = "FAIL"
        detail = traceback.format_exc()
    finally:
        try:
            result_conn.send((status, detail))
        except Exception:
            pass
        try:
            result_conn.close()
        except Exception:
            pass


def _write_log(log_path, log_lines, log_lock):
    with log_lock:
        log_path.write_text("\n".join(log_lines), encoding="utf-8")


def _append_log(log_path, log_lines, log_lock, *new_lines):
    with log_lock:
        log_lines.extend(new_lines)
        log_path.write_text("\n".join(log_lines), encoding="utf-8")


def _format_elapsed(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def _read_cpu_times():
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            line = f.readline().strip()
    except Exception:
        return None
    parts = line.split()
    if len(parts) < 8 or parts[0] != "cpu":
        return None
    values = [int(x) for x in parts[1:8]]
    user, nice, system, idle, iowait, irq, softirq = values
    total = user + nice + system + idle + iowait + irq + softirq
    idle_total = idle + iowait
    return total, idle_total


class _CpuUsageSampler:
    def __init__(self):
        self._prev = _read_cpu_times()

    def sample(self):
        current = _read_cpu_times()
        if current is None:
            return None
        if self._prev is None:
            self._prev = current
            return None
        total_delta = current[0] - self._prev[0]
        idle_delta = current[1] - self._prev[1]
        self._prev = current
        if total_delta <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def _read_memory_usage_percent():
    mem_total = None
    mem_available = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1])
                if mem_total is not None and mem_available is not None:
                    break
    except Exception:
        return None
    if not mem_total or mem_available is None:
        return None
    used_ratio = 1.0 - (mem_available / mem_total)
    return max(0.0, min(100.0, 100.0 * used_ratio))


def _query_gpu_status():
    if shutil.which("nvidia-smi") is None:
        return []
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []

    gpus = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "util": parts[2],
                "mem_used": parts[3],
                "mem_total": parts[4],
            }
        )
    return gpus


def _monitor_resources(stop_event, active_runs, active_lock, completed_runs, completed_lock, total_runs):
    cpu_sampler = _CpuUsageSampler()
    while not stop_event.wait(MONITOR_INTERVAL_SEC):
        with active_lock:
            active_snapshot = [dict(info) for info in active_runs.values()]
        with completed_lock:
            finished_count = len(completed_runs)

        cpu_usage = cpu_sampler.sample()
        mem_usage = _read_memory_usage_percent()
        gpu_info = _query_gpu_status()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        running_count = len(active_snapshot)
        pending_count = max(0, total_runs - finished_count - running_count)

        cpu_text = f"{cpu_usage:.1f}%" if cpu_usage is not None else "N/A"
        mem_text = f"{mem_usage:.1f}%" if mem_usage is not None else "N/A"
        print("=" * 100)
        print(
            f"[Monitor {timestamp}] finished={finished_count}/{total_runs} "
            f"pending={pending_count} running={running_count} cpu={cpu_text} ram={mem_text}"
        )

        if gpu_info:
            for gpu in gpu_info:
                print(
                    f"[GPU {gpu['index']}] util={gpu['util']}% "
                    f"mem={gpu['mem_used']}MiB/{gpu['mem_total']}MiB name={gpu['name']}"
                )
        else:
            print("[GPU] nvidia-smi unavailable or GPU metrics not readable.")

        if active_snapshot:
            now = time.time()
            active_snapshot.sort(key=lambda item: item["run_order_idx"])
            for info in active_snapshot:
                elapsed = _format_elapsed(now - info["start_time"])
                print(
                    f"[Worker {info['worker_id']}] {info['run_name']} | "
                    f"elapsed={elapsed} | flags={info['flags_text']} | seeds={info.get('seed_text', 'N/A')}"
                )
        else:
            print("[Monitor] No active runs.")


def _worker_loop(
    worker_id,
    run_queue,
    base_config,
    root_output_dir,
    active_runs,
    active_lock,
    completed_runs,
    completed_lock,
    log_path,
    log_lines,
    log_lock,
):
    while True:
        try:
            run_spec = run_queue.get_nowait()
        except Empty:
            return

        run_name = run_spec["name"]
        cfg = deepcopy(base_config)
        cfg = _apply_run(cfg, run_spec)
        cfg["output_dir"] = _run_output_relpath(run_spec)
        flags_text = _module_flags_text(cfg)
        seed_text = _seed_text(run_spec.get("seed_info", {}))
        sensitivity = run_spec.get("sensitivity")
        sensitivity_text = (
            f"{sensitivity.get('parameter')}={sensitivity.get('value')} paths={sensitivity.get('paths')}"
            if sensitivity else "none"
        )
        run_output_dir = root_output_dir / RUN_OUTPUT_SUBDIR / run_spec["group_name"] / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        header = (
            f"[{run_spec['order_idx']}/{len(RUNS)}] {run_name} "
            f"(group={run_spec['group_name']}, repeat={run_spec['repeat_idx']}/{run_spec['repeat_total']}, worker={worker_id})"
        )
        _append_log(
            log_path,
            log_lines,
            log_lock,
            header,
            f"Flags: {flags_text}",
            f"Seeds: {seed_text}",
            f"Sensitivity: {sensitivity_text}",
            f"Output Dir: {run_output_dir}",
            f"Start: {datetime.now().isoformat()}",
        )
        print("=" * 100)
        print(header)
        print(f"Flags: {flags_text}")
        print(f"Seeds: {seed_text}")
        print(f"Sensitivity: {sensitivity_text}")
        print(f"Output Dir: {run_output_dir}")

        parent_conn, child_conn = MP_CONTEXT.Pipe(duplex=False)
        p = MP_CONTEXT.Process(target=_child_run_experiment, args=(cfg, child_conn))
        start = time.time()
        result_received = False
        cleanup_note = ""
        status = None
        detail = ""

        with active_lock:
            active_runs[run_name] = {
                "worker_id": worker_id,
                "run_name": run_name,
                "run_order_idx": run_spec["order_idx"],
                "start_time": start,
                "flags_text": flags_text,
                "seed_text": seed_text,
            }

        p.start()
        child_conn.close()

        result_received_at = None
        while True:
            if not result_received and parent_conn.poll(PROCESS_POLL_INTERVAL_SEC):
                try:
                    status, detail = parent_conn.recv()
                    result_received = True
                    result_received_at = time.time()
                except EOFError:
                    result_received = False

            if not p.is_alive():
                break

            if result_received and result_received_at is not None:
                if (time.time() - result_received_at) >= CHILD_RESULT_GRACE_SEC:
                    cleanup_note = (
                        f"Child process still alive {CHILD_RESULT_GRACE_SEC:.0f}s after sending result; terminating it to avoid orchestration hang."
                    )
                    p.terminate()
                    p.join(CHILD_JOIN_TIMEOUT_SEC)
                    if p.is_alive() and hasattr(p, "kill"):
                        cleanup_note += f" Kill attempted after waiting {CHILD_JOIN_TIMEOUT_SEC:.0f}s."
                        p.kill()
                        p.join(CHILD_KILL_TIMEOUT_SEC)
                    break

        p.join(CHILD_JOIN_TIMEOUT_SEC)
        try:
            parent_conn.close()
        except Exception:
            pass

        elapsed = time.time() - start
        if not result_received:
            status = "FAIL" if p.exitcode not in (0, None) else "OK"
            detail = f"No child result received. exitcode={p.exitcode}"
        elif cleanup_note:
            detail = cleanup_note if not detail else f"{detail}\n{cleanup_note}"

        with active_lock:
            active_runs.pop(run_name, None)
        with completed_lock:
            completed_runs.add(run_name)
            completed_snapshot = set(completed_runs)
        _save_state(root_output_dir, completed_snapshot)

        if status == "OK":
            print(f"[Worker {worker_id}] {run_name} finished OK | elapsed={_format_elapsed(elapsed)}")
            _append_log(
                log_path,
                log_lines,
                log_lock,
                f"Status: OK | run={run_name} | worker={worker_id} | elapsed_sec={elapsed:.2f}",
                detail if detail else "",
                "",
            )
        else:
            print(f"[Worker {worker_id}] {run_name} finished FAIL | elapsed={_format_elapsed(elapsed)}")
            _append_log(
                log_path,
                log_lines,
                log_lock,
                f"Status: FAIL | run={run_name} | worker={worker_id} | elapsed_sec={elapsed:.2f}",
                detail,
                "",
            )

        run_queue.task_done()


def main():
    root_dir = Path(__file__).resolve().parent
    output_dir = root_dir / "experiment_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = output_dir / f"test_ablation_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    state = _load_state(output_dir)
    completed_runs = set(state["completed_runs"])
    pending_runs = [run for run in RUNS if run["name"] not in completed_runs]
    worker_count = max(1, min(int(MAX_WORKER_THREADS), len(pending_runs) if pending_runs else len(RUNS)))

    log_lines = []
    log_lock = threading.Lock()
    log_lines.append(f"Start: {datetime.now().isoformat()}")
    log_lines.append(f"Project root: {root_dir}")
    log_lines.append(f"Configured batch struct count: {len(ABLATION_GROUPS)}")
    log_lines.append(f"Active batch struct count: {len(ACTIVE_ABLATION_GROUPS)}")
    log_lines.append(f"ACTIVE_BATCH_CONFIG_COUNT: {ACTIVE_BATCH_CONFIG_COUNT}")
    log_lines.append(f"BATCH_BUILD_MODE: {BATCH_BUILD_MODE}")
    log_lines.append(
        "Full ablation module specs: "
        + ", ".join([f"{key}:{label}" for key, label in FULL_ABLATION_MODULE_SPECS])
    )
    log_lines.append(f"Total run count: {len(RUNS)}")
    log_lines.append(f"Worker threads: {worker_count}")
    log_lines.append(f"Monitor interval sec: {MONITOR_INTERVAL_SEC}")
    log_lines.append(f"Per-run output root: {output_dir / RUN_OUTPUT_SUBDIR}")
    log_lines.append(f"Seed control: {SEED_CONTROL}")
    log_lines.append("Active batch structs:")
    for group in ACTIVE_ABLATION_GROUPS:
        log_lines.append(f"  - {_group_config_text(group)}")
    if completed_runs:
        log_lines.append(f"Already completed runs from state: {len(completed_runs)}")
    log_lines.append("")
    _write_log(run_log_path, log_lines, log_lock)

    print(f"[Config] worker_threads={worker_count}")
    print(f"[Config] monitor_interval_sec={MONITOR_INTERVAL_SEC}")
    print(f"[State] completed={len(completed_runs)} pending={len(pending_runs)} total={len(RUNS)}")
    print(f"[Log] {run_log_path}")

    if not pending_runs:
        print("[State] all runs in the current state file are already completed. Remove the state file to rerun them.")
        return

    base_config = deepcopy(experiment.CONFIG)
    total_start = time.time()

    run_queue = Queue()
    for run_spec in pending_runs:
        run_queue.put(run_spec)

    active_runs = {}
    active_lock = threading.Lock()
    completed_lock = threading.Lock()

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_resources,
        args=(stop_event, active_runs, active_lock, completed_runs, completed_lock, len(RUNS)),
        daemon=True,
    )
    monitor_thread.start()

    workers = []
    for worker_id in range(1, worker_count + 1):
        t = threading.Thread(
            target=_worker_loop,
            args=(
                worker_id,
                run_queue,
                base_config,
                output_dir,
                active_runs,
                active_lock,
                completed_runs,
                completed_lock,
                run_log_path,
                log_lines,
                log_lock,
            ),
            daemon=False,
        )
        workers.append(t)
        t.start()

    for t in workers:
        t.join()

    stop_event.set()
    monitor_thread.join(timeout=MONITOR_INTERVAL_SEC + 1.0)

    total_elapsed = time.time() - total_start
    _append_log(
        run_log_path,
        log_lines,
        log_lock,
        f"Finish: {datetime.now().isoformat()}",
        f"Total elapsed sec: {total_elapsed:.2f}",
        f"Completed runs in state: {len(completed_runs)}/{len(RUNS)}",
    )
    print("=" * 100)
    print(f"Done. Total elapsed: {total_elapsed / 3600:.2f} h")
    print(f"Run log saved to: {run_log_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
