import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


### <--- [MODIFIED] ---------------------------------------
METRICS = ("ACC", "NMI", "ARI", "F1")
SUMMARY_PATTERN = re.compile(r"^summary_(\d{8}_\d{6})\.txt$")
DATASET_HEADER_PATTERN = re.compile(
    r"^\[Dataset\]\s+([^\s]+)(?:\s+\(cluster_num=(\d+)\))?$"
)
MEAN_STD_PATTERN = re.compile(
    r"^(ACC|NMI|ARI|F1):\s*mean=([+-]?\d+(?:\.\d+)?)\s*std=([+-]?\d+(?:\.\d+)?)$"
)
DELTA_PATTERN = re.compile(r"^(ACC|NMI|ARI|F1):\s*([+-]?\d+(?:\.\d+)?)$")
### ---------------------------------------


def _summary_id_from_name(filename):
    match = SUMMARY_PATTERN.match(filename)
    if match:
        return match.group(1)
    return Path(filename).stem


### <--- [MODIFIED] ---------------------------------------
def _compute_delta_metrics(left_metrics, right_metrics):
    delta = {}
    for metric in METRICS:
        if metric in left_metrics and metric in right_metrics:
            delta[metric] = left_metrics[metric]["mean"] - right_metrics[metric]["mean"]
    return delta
### ---------------------------------------


### <--- [MODIFIED] ---------------------------------------
def _parse_on_off(value_text):
    raw = value_text.strip().upper()
    if raw.startswith("ON"):
        return True
    if raw.startswith("OFF"):
        return False
    return None


def _parse_module_flags(lines):
    improved = None
    dynamic = None
    ema = None
    dcgl_negative = None
    dcgl_cluster = None
    gcn_backbone = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Improved Module:"):
            improved = _parse_on_off(stripped.split(":", 1)[1])
        elif stripped.startswith("Dynamic Threshold:"):
            dynamic = _parse_on_off(stripped.split(":", 1)[1])
        elif stripped.startswith("EMA Prototypes:"):
            ema = _parse_on_off(stripped.split(":", 1)[1])
        elif stripped.startswith("DCGL Negative:"):
            dcgl_negative = _parse_on_off(stripped.split(":", 1)[1])
        elif stripped.startswith("DCGL Cluster:"):
            dcgl_cluster = _parse_on_off(stripped.split(":", 1)[1])
        elif stripped.startswith("GCN Backbone:"):
            gcn_backbone = _parse_on_off(stripped.split(":", 1)[1])

    # Backward compatibility for old summaries with only "Improved Module".
    if dynamic is None:
        dynamic = improved
    if ema is None:
        ema = improved

    # Derive combined flag when only decoupled flags are available.
    if improved is None:
        if (
            dynamic is None
            and ema is None
            and dcgl_negative is None
            and dcgl_cluster is None
            and gcn_backbone is None
        ):
            improved = None
        else:
            improved = bool(
                dynamic is True
                or ema is True
                or dcgl_negative is True
                or dcgl_cluster is True
                or gcn_backbone is True
            )

    return improved, dynamic, ema, dcgl_negative, dcgl_cluster, gcn_backbone
### ---------------------------------------


def _parse_summary_file(summary_path):
    records = []
    lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
    ### <--- [MODIFIED] ---------------------------------------
    improved_module_enabled, dynamic_threshold_enabled, ema_prototypes_enabled, dcgl_negative_enabled, dcgl_cluster_enabled, gcn_backbone_enabled = _parse_module_flags(lines)
    ### ---------------------------------------
    i = 0

    while i < len(lines):
        header_match = DATASET_HEADER_PATTERN.match(lines[i].strip())
        if not header_match:
            i += 1
            continue

        dataset = header_match.group(1)
        cluster_num = int(header_match.group(2)) if header_match.group(2) else None

        ### <--- [MODIFIED] ---------------------------------------
        block = {
            "dataset": dataset,
            "cluster_num": cluster_num,
            "summary_file": summary_path.name,
            "summary_id": _summary_id_from_name(summary_path.name),
            "improved_module_enabled": improved_module_enabled,
            "dynamic_threshold_enabled": dynamic_threshold_enabled,
            "ema_prototypes_enabled": ema_prototypes_enabled,
            "dcgl_negative_enabled": dcgl_negative_enabled,
            "dcgl_cluster_enabled": dcgl_cluster_enabled,
            "gcn_backbone_enabled": gcn_backbone_enabled,
            "status": None,
            "baseline_info": None,
            "ae_pretrain_info": None,
            "ae_train_info": None,
            "dual_mean_train_info": None,
            "dual_attn_train_info": None,
            "baseline_metrics": {},
            "ae_metrics": {},
            "dual_mean_metrics": {},
            "dual_attn_metrics": {},
            "delta_ae_baseline": {},
            "delta_dual_mean_baseline": {},
            "delta_dual_attn_baseline": {},
            "delta_dual_mean_ae": {},
            "delta_dual_attn_ae": {},
        }
        ### ---------------------------------------

        i += 1
        section = None
        while i < len(lines):
            stripped = lines[i].strip()

            if DATASET_HEADER_PATTERN.match(stripped):
                break
            if stripped.startswith("Experiment finished at:"):
                break

            if not stripped:
                i += 1
                continue

            if stripped.startswith("Status:"):
                block["status"] = stripped[len("Status:") :].strip()
                section = None
                i += 1
                continue

            if stripped.startswith("Baseline:"):
                block["baseline_info"] = stripped[len("Baseline:") :].strip()
                section = "baseline"
                i += 1
                continue

            if stripped.startswith("AE Pretrain:"):
                block["ae_pretrain_info"] = stripped[len("AE Pretrain:") :].strip()
                section = None
                i += 1
                continue

            if stripped.startswith("AE Train:"):
                block["ae_train_info"] = stripped[len("AE Train:") :].strip()
                section = "ae"
                i += 1
                continue

            if stripped.startswith("Dual Mean Train:"):
                block["dual_mean_train_info"] = stripped[len("Dual Mean Train:") :].strip()
                section = "dual_mean"
                i += 1
                continue
            if stripped.startswith("Dual Mean:"):
                block["dual_mean_train_info"] = stripped[len("Dual Mean:") :].strip()
                section = "dual_mean"
                i += 1
                continue

            if stripped.startswith("Dual Attn Train:"):
                block["dual_attn_train_info"] = stripped[len("Dual Attn Train:") :].strip()
                section = "dual_attn"
                i += 1
                continue
            if stripped.startswith("Dual Attn:"):
                block["dual_attn_train_info"] = stripped[len("Dual Attn:") :].strip()
                section = "dual_attn"
                i += 1
                continue

            # Legacy summary compatibility
            if stripped.startswith("Dual Train:"):
                block["dual_mean_train_info"] = stripped[len("Dual Train:") :].strip()
                section = "dual_mean"
                i += 1
                continue

            if stripped.startswith("Delta (AE - Baseline):"):
                section = "delta_ae_baseline"
                i += 1
                continue

            if stripped.startswith("Delta (Dual Mean - Baseline):"):
                section = "delta_dual_mean_baseline"
                i += 1
                continue

            if stripped.startswith("Delta (Dual Attn - Baseline):"):
                section = "delta_dual_attn_baseline"
                i += 1
                continue

            if stripped.startswith("Delta (Dual Mean - AE):"):
                section = "delta_dual_mean_ae"
                i += 1
                continue

            if stripped.startswith("Delta (Dual Attn - AE):"):
                section = "delta_dual_attn_ae"
                i += 1
                continue

            # Legacy summary compatibility
            if stripped.startswith("Delta (Dual - Baseline):"):
                section = "delta_dual_mean_baseline"
                i += 1
                continue

            if stripped.startswith("Delta (Dual - AE):"):
                section = "delta_dual_mean_ae"
                i += 1
                continue

            mean_std_match = MEAN_STD_PATTERN.match(stripped)
            if mean_std_match:
                metric = mean_std_match.group(1)
                mean = float(mean_std_match.group(2))
                std = float(mean_std_match.group(3))
                if section == "baseline":
                    block["baseline_metrics"][metric] = {"mean": mean, "std": std}
                elif section == "ae":
                    block["ae_metrics"][metric] = {"mean": mean, "std": std}
                elif section == "dual_mean":
                    block["dual_mean_metrics"][metric] = {"mean": mean, "std": std}
                elif section == "dual_attn":
                    block["dual_attn_metrics"][metric] = {"mean": mean, "std": std}
                i += 1
                continue

            delta_match = DELTA_PATTERN.match(stripped)
            if delta_match:
                metric = delta_match.group(1)
                value = float(delta_match.group(2))
                if section == "delta_ae_baseline":
                    block["delta_ae_baseline"][metric] = value
                elif section == "delta_dual_mean_baseline":
                    block["delta_dual_mean_baseline"][metric] = value
                elif section == "delta_dual_attn_baseline":
                    block["delta_dual_attn_baseline"][metric] = value
                elif section == "delta_dual_mean_ae":
                    block["delta_dual_mean_ae"][metric] = value
                elif section == "delta_dual_attn_ae":
                    block["delta_dual_attn_ae"][metric] = value
                i += 1
                continue

            i += 1

        ### <--- [MODIFIED] ---------------------------------------
        # Backfill deltas for older summaries without explicit Delta blocks.
        if not block["delta_ae_baseline"] and block["baseline_metrics"] and block["ae_metrics"]:
            block["delta_ae_baseline"] = _compute_delta_metrics(
                block["ae_metrics"], block["baseline_metrics"]
            )
        if not block["delta_dual_mean_baseline"] and block["baseline_metrics"] and block["dual_mean_metrics"]:
            block["delta_dual_mean_baseline"] = _compute_delta_metrics(
                block["dual_mean_metrics"], block["baseline_metrics"]
            )
        if not block["delta_dual_attn_baseline"] and block["baseline_metrics"] and block["dual_attn_metrics"]:
            block["delta_dual_attn_baseline"] = _compute_delta_metrics(
                block["dual_attn_metrics"], block["baseline_metrics"]
            )
        if not block["delta_dual_mean_ae"] and block["ae_metrics"] and block["dual_mean_metrics"]:
            block["delta_dual_mean_ae"] = _compute_delta_metrics(
                block["dual_mean_metrics"], block["ae_metrics"]
            )
        if not block["delta_dual_attn_ae"] and block["ae_metrics"] and block["dual_attn_metrics"]:
            block["delta_dual_attn_ae"] = _compute_delta_metrics(
                block["dual_attn_metrics"], block["ae_metrics"]
            )
        ### ---------------------------------------

        records.append(block)

    return records


### <--- [MODIFIED] ---------------------------------------
def _iter_candidates(record):
    if record["status"] and "SKIPPED" in record["status"].upper():
        return []
    if not record["baseline_metrics"]:
        return []

    candidates = []

    if record["ae_metrics"] and "ACC" in record["delta_ae_baseline"]:
        candidates.append(
            {
                "dataset": record["dataset"],
                "cluster_num": record["cluster_num"],
                "variant": "AE",
                "summary_file": record["summary_file"],
                "summary_id": record["summary_id"],
                "improved_module_enabled": record["improved_module_enabled"],
                "dynamic_threshold_enabled": record["dynamic_threshold_enabled"],
                "ema_prototypes_enabled": record["ema_prototypes_enabled"],
                "dcgl_negative_enabled": record["dcgl_negative_enabled"],
                "dcgl_cluster_enabled": record["dcgl_cluster_enabled"],
                "gcn_backbone_enabled": record["gcn_backbone_enabled"],
                "baseline_info": record["baseline_info"],
                "baseline_metrics": record["baseline_metrics"],
                "pretrain_info": record["ae_pretrain_info"],
                "train_info": record["ae_train_info"],
                "train_metrics": record["ae_metrics"],
                "delta_title": "AE - Baseline",
                "delta_metrics": record["delta_ae_baseline"],
            }
        )

    if record["dual_mean_metrics"] and "ACC" in record["delta_dual_mean_baseline"]:
        candidates.append(
            {
                "dataset": record["dataset"],
                "cluster_num": record["cluster_num"],
                "variant": "Dual Mean",
                "summary_file": record["summary_file"],
                "summary_id": record["summary_id"],
                "improved_module_enabled": record["improved_module_enabled"],
                "dynamic_threshold_enabled": record["dynamic_threshold_enabled"],
                "ema_prototypes_enabled": record["ema_prototypes_enabled"],
                "dcgl_negative_enabled": record["dcgl_negative_enabled"],
                "dcgl_cluster_enabled": record["dcgl_cluster_enabled"],
                "gcn_backbone_enabled": record["gcn_backbone_enabled"],
                "baseline_info": record["baseline_info"],
                "baseline_metrics": record["baseline_metrics"],
                "pretrain_info": record["ae_pretrain_info"],
                "train_info": record["dual_mean_train_info"],
                "train_metrics": record["dual_mean_metrics"],
                "delta_title": "Dual Mean - Baseline",
                "delta_metrics": record["delta_dual_mean_baseline"],
            }
        )

    if record["dual_attn_metrics"] and "ACC" in record["delta_dual_attn_baseline"]:
        candidates.append(
            {
                "dataset": record["dataset"],
                "cluster_num": record["cluster_num"],
                "variant": "Dual Attn",
                "summary_file": record["summary_file"],
                "summary_id": record["summary_id"],
                "improved_module_enabled": record["improved_module_enabled"],
                "dynamic_threshold_enabled": record["dynamic_threshold_enabled"],
                "ema_prototypes_enabled": record["ema_prototypes_enabled"],
                "dcgl_negative_enabled": record["dcgl_negative_enabled"],
                "dcgl_cluster_enabled": record["dcgl_cluster_enabled"],
                "gcn_backbone_enabled": record["gcn_backbone_enabled"],
                "baseline_info": record["baseline_info"],
                "baseline_metrics": record["baseline_metrics"],
                "pretrain_info": record["ae_pretrain_info"],
                "train_info": record["dual_attn_train_info"],
                "train_metrics": record["dual_attn_metrics"],
                "delta_title": "Dual Attn - Baseline",
                "delta_metrics": record["delta_dual_attn_baseline"],
            }
        )

    return candidates
### ---------------------------------------


def _format_metric_lines(metrics):
    lines = []
    for metric in METRICS:
        if metric in metrics:
            lines.append(
                f"    {metric}: mean={metrics[metric]['mean']:.2f} std={metrics[metric]['std']:.2f}"
            )
    return lines


def _format_delta_lines(delta_metrics):
    lines = []
    for metric in METRICS:
        if metric in delta_metrics:
            lines.append(f"    {metric}: {delta_metrics[metric]:+.2f}")
    return lines


### <--- [MODIFIED] ---------------------------------------
def _collect_summary_files(output_dir):
    """
    Support both legacy flat layout and folder-classified layout:
    1) experiment_output/summary_*.txt
    2) experiment_output/summary/summary_*.txt
    3) any nested location under experiment_output containing summary_*.txt
    """
    collected = {}
    for p in output_dir.glob("summary_*.txt"):
        collected[str(p.resolve())] = p
    summary_dir = output_dir / "summary"
    if summary_dir.exists():
        for p in summary_dir.glob("summary_*.txt"):
            collected[str(p.resolve())] = p
    for p in output_dir.rglob("summary_*.txt"):
        collected[str(p.resolve())] = p

    return sorted(collected.values(), key=lambda x: (x.name, str(x)))
### ---------------------------------------


### <--- [MODIFIED] ---------------------------------------
def _pick_best(candidates):
    if not candidates:
        return None
    return max(candidates, key=lambda x: (x["delta_metrics"]["ACC"], x["summary_id"]))


def _append_candidate_block(lines, title, candidate):
    lines.append(f"  {title}:")
    if candidate is None:
        lines.append("    N/A")
        return

    module_text = (
        "ON" if candidate["improved_module_enabled"] is True
        else "OFF" if candidate["improved_module_enabled"] is False
        else "UNKNOWN"
    )
    dynamic_text = (
        "ON" if candidate["dynamic_threshold_enabled"] is True
        else "OFF" if candidate["dynamic_threshold_enabled"] is False
        else "UNKNOWN"
    )
    ema_text = (
        "ON" if candidate["ema_prototypes_enabled"] is True
        else "OFF" if candidate["ema_prototypes_enabled"] is False
        else "UNKNOWN"
    )
    dcgl_negative_text = (
        "ON" if candidate["dcgl_negative_enabled"] is True
        else "OFF" if candidate["dcgl_negative_enabled"] is False
        else "UNKNOWN"
    )
    dcgl_cluster_text = (
        "ON" if candidate["dcgl_cluster_enabled"] is True
        else "OFF" if candidate["dcgl_cluster_enabled"] is False
        else "UNKNOWN"
    )
    gcn_backbone_text = (
        "ON" if candidate["gcn_backbone_enabled"] is True
        else "OFF" if candidate["gcn_backbone_enabled"] is False
        else "UNKNOWN"
    )
    lines.append(
        f"    Source Summary: {candidate['summary_file']} (id={candidate['summary_id']}, improved={module_text}, dynamic={dynamic_text}, ema={ema_text}, dcgl_neg={dcgl_negative_text}, dcgl_clu={dcgl_cluster_text}, gcn={gcn_backbone_text})"
    )
    lines.append(f"    Variant: {candidate['variant']}")
    lines.append(f"    Baseline: {candidate['baseline_info'] or 'N/A'}")
    lines.extend(_format_metric_lines(candidate["baseline_metrics"]))
    if candidate["pretrain_info"]:
        lines.append(f"    AE Pretrain: {candidate['pretrain_info']}")
    if candidate["variant"] == "AE":
        train_label = "AE Train"
    elif candidate["variant"] == "Dual Mean":
        train_label = "Dual Mean Train"
    else:
        train_label = "Dual Attn Train"
    lines.append(f"    {train_label}: {candidate['train_info'] or 'N/A'}")
    lines.extend(_format_metric_lines(candidate["train_metrics"]))
    lines.append(f"    Delta ({candidate['delta_title']}):")
    lines.extend(_format_delta_lines(candidate["delta_metrics"]))


def _append_on_off_compare(lines, title, best_off, best_on):
    lines.append(f"  {title}:")
    if best_off is None or best_on is None:
        lines.append("    N/A (missing ON or OFF records in scanned summaries)")
        return
    lines.append("    Delta-Baseline Gap (ON - OFF):")
    for metric in METRICS:
        if metric in best_on["delta_metrics"] and metric in best_off["delta_metrics"]:
            gap = best_on["delta_metrics"][metric] - best_off["delta_metrics"][metric]
            lines.append(f"    {metric}: {gap:+.2f}")
### ---------------------------------------


def main():
    root_dir = Path(__file__).resolve().parent
    output_dir = root_dir / "experiment_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_files = _collect_summary_files(output_dir)

    ### <--- [MODIFIED] ---------------------------------------
    grouped_candidates = defaultdict(list)
    for summary_file in summary_files:
        for record in _parse_summary_file(summary_file):
            for candidate in _iter_candidates(record):
                grouped_candidates[candidate["dataset"]].append(candidate)
    ### ---------------------------------------

    lines = []
    lines.append(f"Experiment started at: {datetime.now().isoformat()}")
    lines.append(f"Project root: {root_dir}")
    lines.append(f"Scanned summaries: {len(summary_files)}")
    lines.append("")

    ### <--- [MODIFIED] ---------------------------------------
    if not grouped_candidates:
        lines.append("No valid dataset records found for baseline-vs-improved comparison.")
        lines.append("")
    else:
        for dataset in sorted(grouped_candidates.keys()):
            dataset_candidates = grouped_candidates[dataset]
            best = _pick_best(dataset_candidates)
            best_off = _pick_best([
                c for c in dataset_candidates
                if c["dynamic_threshold_enabled"] is False and c["ema_prototypes_enabled"] is False
            ])
            best_on = _pick_best([
                c for c in dataset_candidates
                if c["dynamic_threshold_enabled"] is True or c["ema_prototypes_enabled"] is True
            ])
            best_dynamic_off = _pick_best([c for c in dataset_candidates if c["dynamic_threshold_enabled"] is False])
            best_dynamic_on = _pick_best([c for c in dataset_candidates if c["dynamic_threshold_enabled"] is True])
            best_ema_off = _pick_best([c for c in dataset_candidates if c["ema_prototypes_enabled"] is False])
            best_ema_on = _pick_best([c for c in dataset_candidates if c["ema_prototypes_enabled"] is True])
            best_dcgl_negative_off = _pick_best([c for c in dataset_candidates if c["dcgl_negative_enabled"] is False])
            best_dcgl_negative_on = _pick_best([c for c in dataset_candidates if c["dcgl_negative_enabled"] is True])
            best_dcgl_cluster_off = _pick_best([c for c in dataset_candidates if c["dcgl_cluster_enabled"] is False])
            best_dcgl_cluster_on = _pick_best([c for c in dataset_candidates if c["dcgl_cluster_enabled"] is True])
            best_gcn_backbone_off = _pick_best([c for c in dataset_candidates if c["gcn_backbone_enabled"] is False])
            best_gcn_backbone_on = _pick_best([c for c in dataset_candidates if c["gcn_backbone_enabled"] is True])

            cluster_text = (
                f" (cluster_num={best['cluster_num']})"
                if best["cluster_num"] is not None
                else ""
            )
            lines.append(f"[Dataset] {dataset}{cluster_text}")
            _append_candidate_block(lines, "Best Overall", best)
            _append_candidate_block(lines, "Best OFF (dynamic=OFF, ema=OFF)", best_off)
            _append_candidate_block(lines, "Best ON (dynamic/ema enabled)", best_on)
            _append_candidate_block(lines, "Best Dynamic OFF", best_dynamic_off)
            _append_candidate_block(lines, "Best Dynamic ON", best_dynamic_on)
            _append_candidate_block(lines, "Best EMA OFF", best_ema_off)
            _append_candidate_block(lines, "Best EMA ON", best_ema_on)
            _append_candidate_block(lines, "Best DCGL Negative OFF", best_dcgl_negative_off)
            _append_candidate_block(lines, "Best DCGL Negative ON", best_dcgl_negative_on)
            _append_candidate_block(lines, "Best DCGL Cluster OFF", best_dcgl_cluster_off)
            _append_candidate_block(lines, "Best DCGL Cluster ON", best_dcgl_cluster_on)
            _append_candidate_block(lines, "Best GCN Backbone OFF", best_gcn_backbone_off)
            _append_candidate_block(lines, "Best GCN Backbone ON", best_gcn_backbone_on)
            _append_on_off_compare(lines, "Best ON/OFF Compare", best_off, best_on)
            _append_on_off_compare(lines, "Dynamic ON/OFF Compare", best_dynamic_off, best_dynamic_on)
            _append_on_off_compare(lines, "EMA ON/OFF Compare", best_ema_off, best_ema_on)
            _append_on_off_compare(lines, "DCGL Negative ON/OFF Compare", best_dcgl_negative_off, best_dcgl_negative_on)
            _append_on_off_compare(lines, "DCGL Cluster ON/OFF Compare", best_dcgl_cluster_off, best_dcgl_cluster_on)
            _append_on_off_compare(lines, "GCN Backbone ON/OFF Compare", best_gcn_backbone_off, best_gcn_backbone_on)
            lines.append("")
    ### ---------------------------------------

    lines.append(f"Experiment finished at: {datetime.now().isoformat()}")

    out_path = output_dir / "best_data.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Best data summary saved to: {out_path}")


if __name__ == "__main__":
    main()
