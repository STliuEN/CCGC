import argparse
from pathlib import Path
import subprocess
import time
import torch
from utils import *
from tqdm import tqdm
from torch import optim
from model import Encoder_Net, GCNEncoder_Net
import torch.nn.functional as F
from attention_fusion import DualViewAttention, fuse_dual_views


### <--- [MODIFIED] ---------------------------------------
_DYNAMIC_HC_STATE = {}


def _try_process_monitor():
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        psutil.cpu_percent(None)
        proc.cpu_percent(None)
        return psutil, proc
    except Exception:
        return None, None


def _system_snapshot(psutil_mod=None, proc=None) -> dict[str, float | str]:
    snapshot: dict[str, float | str] = {}
    if psutil_mod is not None:
        try:
            vm = psutil_mod.virtual_memory()
            snapshot["ram_used_gb"] = float(vm.used) / (1024 ** 3)
            snapshot["ram_total_gb"] = float(vm.total) / (1024 ** 3)
            snapshot["ram_percent"] = float(vm.percent)
        except Exception:
            pass
        try:
            snapshot["cpu_percent"] = float(psutil_mod.cpu_percent(interval=None))
        except Exception:
            pass
    if proc is not None:
        try:
            mem = proc.memory_info()
            snapshot["process_rss_gb"] = float(mem.rss) / (1024 ** 3)
            snapshot["process_cpu_percent"] = float(proc.cpu_percent(None))
        except Exception:
            pass
    return snapshot


def _gpu_snapshot(device: str) -> dict[str, float | str]:
    snapshot: dict[str, float | str] = {}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            snapshot["torch_gpu_allocated_gb"] = float(torch.cuda.max_memory_allocated(idx)) / (1024 ** 3)
            snapshot["torch_gpu_reserved_gb"] = float(torch.cuda.max_memory_reserved(idx)) / (1024 ** 3)
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            first = out.strip().splitlines()[0].split(",")
            if len(first) >= 3:
                snapshot["gpu_memory_used_gb"] = float(first[0].strip()) / 1024.0
                snapshot["gpu_memory_total_gb"] = float(first[1].strip()) / 1024.0
                snapshot["gpu_util_percent"] = float(first[2].strip())
        except Exception:
            pass
    return snapshot


def _resource_summary(start_time: float, psutil_mod=None, proc=None, device: str = "cpu") -> dict[str, float | str]:
    summary: dict[str, float | str] = {"wall_time_sec": float(time.perf_counter() - start_time)}
    summary.update(_system_snapshot(psutil_mod, proc))
    summary.update(_gpu_snapshot(device))
    return summary


def _fmt_resource_value(value: float | str, unit: str = "") -> str:
    if isinstance(value, str):
        return value
    if unit == "sec":
        return f"{value:.2f}"
    if unit == "gb":
        return f"{value:.3f}"
    if unit == "pct":
        return f"{value:.1f}"
    return f"{value:.4f}"


def _print_resource_summary(summary: dict[str, float | str]) -> None:
    fields = [
        ("Wall time (sec)", "wall_time_sec", "sec"),
        ("CPU util (%)", "cpu_percent", "pct"),
        ("Process CPU (%)", "process_cpu_percent", "pct"),
        ("RAM used (GB)", "ram_used_gb", "gb"),
        ("RAM total (GB)", "ram_total_gb", "gb"),
        ("RAM util (%)", "ram_percent", "pct"),
        ("Process RSS (GB)", "process_rss_gb", "gb"),
        ("GPU util (%)", "gpu_util_percent", "pct"),
        ("GPU memory used (GB)", "gpu_memory_used_gb", "gb"),
        ("GPU memory total (GB)", "gpu_memory_total_gb", "gb"),
        ("Torch max allocated (GB)", "torch_gpu_allocated_gb", "gb"),
        ("Torch max reserved (GB)", "torch_gpu_reserved_gb", "gb"),
    ]
    print(f"\n{'='*20} RESOURCE SUMMARY {'='*20}")
    for label, key, unit in fields:
        if key in summary:
            print(f"RESOURCE | {label:<24} | {_fmt_resource_value(summary[key], unit)}")
    print(f"{'='*58}\n")


def _reset_dynamic_hc_state():
    _DYNAMIC_HC_STATE.clear()


def _update_dynamic_hc_state(dis, predict_labels_t):
    if dis is None or predict_labels_t is None:
        return

    with torch.no_grad():
        min_dis = torch.min(dis, dim=1).values.detach()
        if min_dis.numel() == 0:
            return

        sorted_dis = torch.sort(min_dis).values
        sample_count = sorted_dis.numel()
        q20_idx = min(sample_count - 1, max(0, int(0.20 * (sample_count - 1))))
        q50_idx = min(sample_count - 1, max(0, int(0.50 * (sample_count - 1))))
        q80_idx = min(sample_count - 1, max(0, int(0.80 * (sample_count - 1))))

        q20 = sorted_dis[q20_idx]
        q50 = sorted_dis[q50_idx]
        q80 = sorted_dis[q80_idx]
        spread = torch.clamp(q80 - q20, min=1e-6)

        # Steep distributions indicate a clearer confidence split, so we can
        # safely admit more samples into the high-confidence pool.
        steepness = torch.clamp((q80 - q50) / spread, min=0.0, max=1.0)
        concentration = torch.clamp(1.0 - (q50 / torch.clamp(q80, min=1e-6)), min=0.0, max=1.0)
        conf_score = 0.55 * concentration + 0.45 * steepness

        label_signature = predict_labels_t.detach()
        prev_labels = _DYNAMIC_HC_STATE.get("prev_labels")
        if prev_labels is None or prev_labels.shape != label_signature.shape:
            stability = torch.tensor(0.5, device=min_dis.device, dtype=min_dis.dtype)
        else:
            stability = torch.mean((prev_labels == label_signature).float())

        prev_conf = _DYNAMIC_HC_STATE.get("ema_conf")
        prev_stability = _DYNAMIC_HC_STATE.get("ema_stability")
        if prev_conf is None:
            ema_conf = conf_score
        else:
            ema_conf = 0.8 * prev_conf + 0.2 * conf_score
        if prev_stability is None:
            ema_stability = stability
        else:
            ema_stability = 0.8 * prev_stability + 0.2 * stability

        _DYNAMIC_HC_STATE["ema_conf"] = ema_conf.detach()
        _DYNAMIC_HC_STATE["ema_stability"] = ema_stability.detach()
        _DYNAMIC_HC_STATE["prev_labels"] = label_signature.clone()


def _smooth_with_adj(adj, features, t, device):
    """
    Build smoothed features from a given graph adjacency.
    """
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_norm_s = preprocess_graph(adj, t, norm='sym', renorm=True)
    smooth_fea = features.float().to(device)
    adj_norm_s_torch = [sparse_mx_to_torch_sparse_tensor(a).coalesce().to(device) for a in adj_norm_s]
    for a in adj_norm_s_torch:
        smooth_fea = torch.sparse.mm(a, smooth_fea)
    return smooth_fea


### <--- [MODIFIED] ---------------------------------------
def _build_gcn_adj_torch(adj, device):
    """
    Build a single normalized adjacency tensor for optional GCN backbone.
    """
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_norm = preprocess_graph(adj, 1, norm='sym', renorm=True)[0]
    return sparse_mx_to_torch_sparse_tensor(adj_norm).coalesce().to(device)
### ---------------------------------------


def _get_high_confidence_idx(dis, threshold_ratio):
    high_confidence = torch.min(dis, dim=1).values
    sorted_conf = torch.sort(high_confidence).values
    threshold_idx = min(len(sorted_conf) - 1, int(len(sorted_conf) * threshold_ratio))
    threshold = sorted_conf[threshold_idx]
    return torch.nonzero(high_confidence < threshold, as_tuple=False).squeeze(1)


def _resolve_hc_ratio(epoch, total_epochs, base_ratio, use_dynamic, start_ratio, end_ratio):
    if not use_dynamic or total_epochs <= 1:
        return base_ratio

    progress = float(epoch) / float(max(1, total_epochs - 1))
    ratio = start_ratio + (end_ratio - start_ratio) * progress

    # Adaptive refinement:
    # 1) early epochs stay conservative;
    # 2) clearer distance split => admit more samples;
    # 3) unstable pseudo labels => slow down expansion.
    ema_conf = _DYNAMIC_HC_STATE.get("ema_conf")
    ema_stability = _DYNAMIC_HC_STATE.get("ema_stability")
    if ema_conf is not None and ema_stability is not None:
        conf_val = float(torch.clamp(ema_conf, min=0.0, max=1.0).item())
        stability_val = float(torch.clamp(ema_stability, min=0.0, max=1.0).item())
        adaptive_gain = 0.45 * (conf_val - 0.5) + 0.55 * (stability_val - 0.5)
        adaptive_scale = 0.35 + 0.65 * progress
        ratio = ratio + (end_ratio - start_ratio) * adaptive_gain * adaptive_scale

        # If pseudo labels are still shaking, keep the threshold conservative.
        if stability_val < 0.55:
            ratio = min(ratio, start_ratio + (end_ratio - start_ratio) * (0.25 + 0.5 * progress))

    return min(0.999, max(1e-6, ratio))


### <--- [MODIFIED] ---------------------------------------
def _dcgl_center_contrastive_negative_loss(centers_1, centers_2, temperature, row_weights=None):
    """
    DCGL-style cluster-guided contrastive objective on cluster centers.
    """
    tau = max(1e-6, float(temperature))
    sim12 = torch.exp((centers_1 @ centers_2.T) / tau)
    sim21 = torch.exp((centers_2 @ centers_1.T) / tau)

    pos12 = torch.diagonal(sim12)
    pos21 = torch.diagonal(sim21)
    loss12 = -torch.log(pos12 / torch.clamp(torch.sum(sim12, dim=1), min=1e-12))
    loss21 = -torch.log(pos21 / torch.clamp(torch.sum(sim21, dim=1), min=1e-12))
    if row_weights is None:
        return 0.5 * (loss12.mean() + loss21.mean())

    row_weights = row_weights.to(loss12.dtype)
    row_weights = torch.clamp(row_weights, min=0.0, max=1.0)
    return 0.5 * (torch.mean(row_weights * loss12) + torch.mean(row_weights * loss21))


def _offdiag_center_mse_negative_loss(centers_1, centers_2):
    s = centers_1 @ centers_2.T
    s = s - torch.diag_embed(torch.diag(s))
    return F.mse_loss(s, torch.zeros_like(s))


def _smooth_reliability_gate(scores, threshold=0.55, power=2.0, min_gate=0.0):
    threshold_t = torch.tensor(float(threshold), dtype=scores.dtype, device=scores.device)
    denom = torch.clamp(1.0 - threshold_t, min=1e-6)
    gate = torch.clamp((scores - threshold_t) / denom, min=0.0, max=1.0)
    gate = gate.pow(max(1e-6, float(power)))
    min_gate_t = torch.tensor(float(min_gate), dtype=scores.dtype, device=scores.device)
    min_gate_t = torch.clamp(min_gate_t, min=0.0, max=1.0)
    return min_gate_t + (1.0 - min_gate_t) * gate


def _legacy_dcgl_negative_row_weights(reliability_scores):
    if reliability_scores.numel() > 1:
        reliability_std = torch.std(reliability_scores, unbiased=False)
    else:
        reliability_std = torch.tensor(0.0, dtype=reliability_scores.dtype, device=reliability_scores.device)

    conservative_floor = torch.minimum(
        torch.tensor(0.6, dtype=reliability_scores.dtype, device=reliability_scores.device),
        torch.mean(reliability_scores) - 0.5 * reliability_std
    )
    conservative_floor = torch.clamp(conservative_floor, min=0.35, max=0.6)
    badness = torch.relu(conservative_floor - reliability_scores) / torch.clamp(conservative_floor, min=1e-6)
    return (1.0 - 0.25 * badness.pow(2)).detach()
### ---------------------------------------


### <--- [MODIFIED] ---------------------------------------
def _compute_branch_reliability_score(hidden_emb, predict_labels_t, high_confidence_idx, cluster_num):
    """
    Unsupervised branch-quality proxy for delayed raw/AE selection.

    The score uses only current pseudo labels and branch embeddings:
    compact clusters, separated centers, and a non-collapsed cluster-size
    distribution are treated as more reliable. No ground-truth label or
    external metric is used.
    """
    if high_confidence_idx is None or high_confidence_idx.numel() == 0:
        return None

    z = F.normalize(hidden_emb[high_confidence_idx], dim=1, p=2)
    labels = predict_labels_t[high_confidence_idx]
    unique_labels = torch.unique(labels, sorted=True)
    if unique_labels.numel() < 2:
        return None

    centers = []
    sizes = []
    compact_scores = []
    for label in unique_labels:
        members = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        if members.numel() == 0:
            continue
        member_z = z[members]
        center = F.normalize(torch.mean(member_z, dim=0, keepdim=True), dim=1, p=2)
        compact = torch.mean(torch.clamp(torch.sum(member_z * center, dim=1), min=-1.0, max=1.0))
        centers.append(center.squeeze(0))
        sizes.append(float(members.numel()))
        compact_scores.append(0.5 * (compact + 1.0))

    if len(centers) < 2:
        return None

    centers_t = torch.stack(centers, dim=0)
    compact_score = torch.mean(torch.stack(compact_scores))

    center_sim = torch.clamp(centers_t @ centers_t.T, min=-1.0, max=1.0)
    offdiag = center_sim[~torch.eye(center_sim.shape[0], dtype=torch.bool, device=center_sim.device)]
    separation_score = torch.clamp(torch.mean(0.5 * (1.0 - offdiag)), min=0.0, max=1.0)

    size_t = torch.tensor(sizes, dtype=hidden_emb.dtype, device=hidden_emb.device)
    probs = size_t / torch.clamp(torch.sum(size_t), min=1.0)
    entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-8)))
    max_entropy = torch.log(torch.tensor(float(max(2, min(cluster_num, len(sizes)))), dtype=hidden_emb.dtype, device=hidden_emb.device))
    balance_score = torch.clamp(entropy / torch.clamp(max_entropy, min=1e-8), min=0.0, max=1.0)

    coverage_score = torch.clamp(
        torch.tensor(float(len(sizes)) / float(max(1, cluster_num)), dtype=hidden_emb.dtype, device=hidden_emb.device),
        min=0.0,
        max=1.0,
    )
    score = 0.45 * compact_score + 0.35 * separation_score + 0.15 * balance_score + 0.05 * coverage_score
    return torch.clamp(score.detach(), min=0.0, max=1.0)


def _fusion_reliability_feedback_loss(
    fusion_mean,
    adaptive_bias_state,
    raw_reliability,
    ae_reliability,
    args,
    branch_losses=None,
):
    target = None
    source = None
    dtype = fusion_mean.dtype
    device = fusion_mean.device

    prior_target = adaptive_bias_state.get("target")
    has_runtime = raw_reliability is not None and ae_reliability is not None
    if prior_target in ("raw", "ae") or (args.fusion_feedback_use_runtime_reliability and has_runtime):
        if has_runtime:
            scores = torch.stack([
                raw_reliability.to(device=device, dtype=dtype),
                ae_reliability.to(device=device, dtype=dtype),
            ])
            scores = scores - torch.mean(scores)
        else:
            scores = torch.zeros(2, dtype=dtype, device=device)

        if prior_target in ("raw", "ae"):
            prior_strength = max(0.0, float(args.fusion_feedback_prior_strength))
            if prior_target == "raw":
                scores = scores + torch.tensor([0.5 * prior_strength, -0.5 * prior_strength], dtype=dtype, device=device)
            else:
                scores = scores + torch.tensor([-0.5 * prior_strength, 0.5 * prior_strength], dtype=dtype, device=device)
            source = str(adaptive_bias_state.get("source", "adaptive"))
            if has_runtime:
                source = source + "+runtime"
        else:
            source = "runtime"

        target = F.softmax(scores / max(1e-6, float(args.fusion_feedback_temp)), dim=0)
        floor = min(0.49, max(0.0, float(args.fusion_feedback_min_weak_weight)))
        target = floor + (1.0 - 2.0 * floor) * target

    if target is None:
        return None, None

    target_detached = target.detach()
    if args.fusion_feedback_loss_type == "barrier":
        weights = torch.clamp(fusion_mean, min=1e-8, max=1.0)
        coeff = torch.ones_like(weights)
        if target is not None:
            coeff = target_detached / torch.clamp(torch.mean(target_detached), min=1e-8)
        if branch_losses is not None:
            branch_losses_t = torch.stack(branch_losses).to(device=device, dtype=dtype).detach()
            branch_losses_t = branch_losses_t / torch.clamp(torch.mean(branch_losses_t), min=1e-8)
            loss_pref = F.softmax(-branch_losses_t / max(1e-6, float(args.fusion_feedback_loss_temp)), dim=0)
            loss_pref = loss_pref / torch.clamp(torch.mean(loss_pref), min=1e-8)
            loss_blend = min(1.0, max(0.0, float(args.fusion_feedback_loss_blend)))
            coeff = (1.0 - loss_blend) * coeff + loss_blend * loss_pref
        coeff = torch.clamp(
            coeff,
            min=float(args.fusion_feedback_min_barrier_coeff),
            max=float(args.fusion_feedback_max_barrier_coeff),
        )
        loss = -torch.sum(coeff.detach() * torch.log(weights)) / torch.clamp(torch.sum(coeff.detach()), min=1e-8)
    elif args.fusion_feedback_loss_type == "selective_countergrad":
        # Selective counter-gradient feedback. The branch-weighted objective is
        # allowed to keep moving attention when its gradient agrees with the
        # reliability evidence. This term only pushes back when the branch-loss
        # shortcut would increase the unreliable branch weight.
        if branch_losses is None:
            return None, None
        weights = torch.clamp(fusion_mean, min=1e-8, max=1.0)
        trusted_idx = 0 if target_detached[0] >= target_detached[1] else 1
        wrong_idx = 1 - trusted_idx
        branch_losses_t = torch.stack(branch_losses).to(device=device, dtype=dtype).detach()
        shortcut_grad = branch_losses_t[wrong_idx] - branch_losses_t[trusted_idx]
        conflict_strength = torch.relu(-shortcut_grad)
        target_wrong = torch.clamp(
            target_detached[wrong_idx],
            min=max(1e-6, float(args.fusion_feedback_min_weak_weight)),
            max=0.49,
        )
        wrong_weight = weights[wrong_idx]
        curvature = max(0.0, float(args.fusion_feedback_countergrad_curvature))
        base = max(0.0, float(args.fusion_feedback_countergrad_base))
        boundary_excess = torch.relu(wrong_weight - target_wrong)
        loss = (
            conflict_strength * wrong_weight
            + 0.5 * curvature * conflict_strength * boundary_excess ** 2
            + base * F.mse_loss(weights, target_detached)
        )
        coeff = torch.zeros_like(weights)
        coeff[trusted_idx] = 1.0
        coeff[wrong_idx] = float(curvature)
    elif args.fusion_feedback_loss_type == "countergrad":
        # Counter-gradient feedback for the branch-loss shortcut. With two
        # normalized weights, the branch objective contributes
        # dL/dq = L_wrong - L_trusted to the weak/wrong branch weight q. This
        # term adds the opposite detached gradient and a small curvature around
        # the reliability target, so the loss has a real equilibrium instead
        # of letting q run to 0 or 1.
        if branch_losses is None:
            return None, None
        weights = torch.clamp(fusion_mean, min=1e-8, max=1.0)
        trusted_idx = 0 if target_detached[0] >= target_detached[1] else 1
        wrong_idx = 1 - trusted_idx
        branch_losses_t = torch.stack(branch_losses).to(device=device, dtype=dtype).detach()
        shortcut_grad = branch_losses_t[wrong_idx] - branch_losses_t[trusted_idx]
        target_wrong = torch.clamp(
            target_detached[wrong_idx],
            min=max(1e-6, float(args.fusion_feedback_min_weak_weight)),
            max=0.49,
        )
        wrong_weight = weights[wrong_idx]
        curvature = max(0.0, float(args.fusion_feedback_countergrad_curvature))
        base = max(0.0, float(args.fusion_feedback_countergrad_base))
        loss = (
            -shortcut_grad * wrong_weight
            + 0.5 * curvature * torch.abs(shortcut_grad) * (wrong_weight - target_wrong) ** 2
            + base * F.mse_loss(weights, target_detached)
        )
        coeff = torch.zeros_like(weights)
        coeff[trusted_idx] = 1.0
        coeff[wrong_idx] = float(curvature)
    elif args.fusion_feedback_loss_type == "shortcut":
        # Directly counter the branch-loss shortcut. If the branch with lower
        # training loss contradicts the reliability evidence, ordinary
        # branch-weighted loss pulls attention toward that branch. This
        # quadratic term adds a differentiable negative feedback whose
        # equilibrium is set by the reliability target, without clamping the
        # forward weights.
        if branch_losses is None:
            return None, None
        weights = torch.clamp(fusion_mean, min=1e-8, max=1.0)
        trusted_idx = 0 if target_detached[0] >= target_detached[1] else 1
        wrong_idx = 1 - trusted_idx
        branch_losses_t = torch.stack(branch_losses).to(device=device, dtype=dtype).detach()
        advantage = torch.relu(branch_losses_t[trusted_idx] - branch_losses_t[wrong_idx])
        target_wrong = torch.clamp(
            target_detached[wrong_idx],
            min=max(1e-6, float(args.fusion_feedback_min_weak_weight)),
            max=0.49,
        )
        wrong_weight = weights[wrong_idx]
        gain = max(0.0, float(args.fusion_feedback_shortcut_gain))
        loss = gain * advantage * (wrong_weight ** 2) / (2.0 * target_wrong)
        coeff = torch.zeros_like(weights)
        coeff[trusted_idx] = 1.0
        coeff[wrong_idx] = float(gain)
    elif args.fusion_feedback_loss_type == "controller":
        # Boundary-aware loss-level negative feedback. The ordinary branch
        # weighted loss can pull attention toward the currently easier view;
        # this term only becomes strong near collapse and pushes back through
        # gradients on the attention weights. It does not clamp the forward
        # weights or impose a fixed minimum.
        weights = torch.clamp(fusion_mean, min=1e-8, max=1.0)
        if branch_losses is not None:
            branch_losses_t = torch.stack(branch_losses).to(device=device, dtype=dtype).detach()
            branch_losses_t = branch_losses_t / torch.clamp(torch.mean(branch_losses_t), min=1e-8)
            loss_pref = F.softmax(-branch_losses_t / max(1e-6, float(args.fusion_feedback_loss_temp)), dim=0)
            loss_pref = loss_pref.detach()
            loss_gap = torch.abs(branch_losses_t[0] - branch_losses_t[1])
        else:
            loss_pref = torch.full_like(weights, 0.5)
            loss_gap = torch.tensor(0.0, dtype=dtype, device=device)

        prior_blend = min(1.0, max(0.0, float(args.fusion_feedback_prior_blend)))
        evidence = (1.0 - prior_blend) * loss_pref + prior_blend * target_detached
        evidence = evidence / torch.clamp(torch.sum(evidence), min=1e-8)

        # Evidence decides which side may dominate. The controller provides a
        # weak always-on reliability gradient, then amplifies it when the
        # attention crosses into the opposite-side majority. This is negative
        # feedback in the loss, not a forward clamp.
        coeff = evidence / torch.clamp(torch.mean(evidence), min=1e-8)
        coeff = torch.clamp(
            coeff,
            min=float(args.fusion_feedback_min_barrier_coeff),
            max=float(args.fusion_feedback_max_barrier_coeff),
        )
        target_ce = -torch.sum(evidence.detach() * torch.log(weights))
        raw_is_trusted = evidence[0] >= evidence[1]
        weak_weight = weights[1] if raw_is_trusted else weights[0]
        wrong_gate = torch.sigmoid(
            (weak_weight - float(args.fusion_feedback_wrong_branch_margin))
            * max(1e-6, float(args.fusion_feedback_boundary_sharpness))
        )
        barrier = -torch.sum(coeff.detach() * torch.log(weights)) / torch.clamp(torch.sum(coeff.detach()), min=1e-8)
        weak_violation = torch.relu(weak_weight - float(args.fusion_feedback_wrong_branch_margin)) ** 2
        adaptive_scale = 1.0 + min(10.0, max(0.0, float(args.fusion_feedback_loss_gap_gain))) * loss_gap.detach()
        loss = adaptive_scale * (
            float(args.fusion_feedback_controller_base) * target_ce
            + wrong_gate * float(args.fusion_feedback_controller_boost) * barrier
            + wrong_gate * weak_violation
        )
    elif args.fusion_feedback_loss_type == "violation":
        # Negative feedback only when attention assigns more mass than the
        # reliability controller allows. If it has already self-corrected past
        # the boundary, no fixed minimum is imposed on the weak branch.
        loss = torch.mean(torch.relu(fusion_mean - target_detached) ** 2)
    else:
        loss = F.mse_loss(fusion_mean, target_detached)
    state = {
        "source": source,
        "mode": args.fusion_feedback_loss_type,
        "target_raw": float(target[0].detach().cpu().item()),
        "target_ae": float(target[1].detach().cpu().item()),
        "loss": float(loss.detach().cpu().item()),
    }
    if branch_losses is not None:
        state["branch_loss_raw"] = float(branch_losses[0].detach().cpu().item())
        state["branch_loss_ae"] = float(branch_losses[1].detach().cpu().item())
    if args.fusion_feedback_loss_type in ("barrier", "controller"):
        state["coeff_raw"] = float(coeff[0].detach().cpu().item())
        state["coeff_ae"] = float(coeff[1].detach().cpu().item())
        state["weak_w"] = float(torch.min(fusion_mean).detach().cpu().item())
        if args.fusion_feedback_loss_type == "controller":
            state["collapse_start"] = float(args.fusion_feedback_wrong_branch_margin)
            state["collapse_gate"] = float(wrong_gate.detach().cpu().item())
            state["loss_gap"] = float(loss_gap.detach().cpu().item())
            state["adaptive_scale"] = float(adaptive_scale.detach().cpu().item())
    if args.fusion_feedback_loss_type == "shortcut":
        state["coeff_raw"] = float(coeff[0].detach().cpu().item())
        state["coeff_ae"] = float(coeff[1].detach().cpu().item())
        state["weak_w"] = float(torch.min(fusion_mean).detach().cpu().item())
        state["loss_gap"] = float(advantage.detach().cpu().item())
        state["adaptive_scale"] = float(args.fusion_feedback_shortcut_gain)
    if args.fusion_feedback_loss_type in ("countergrad", "selective_countergrad"):
        state["coeff_raw"] = float(coeff[0].detach().cpu().item())
        state["coeff_ae"] = float(coeff[1].detach().cpu().item())
        state["weak_w"] = float(torch.min(fusion_mean).detach().cpu().item())
        state["loss_gap"] = float(shortcut_grad.detach().cpu().item())
        state["adaptive_scale"] = float(args.fusion_feedback_countergrad_curvature)
    if has_runtime:
        state["raw_rel"] = float(raw_reliability.detach().cpu().item())
        state["ae_rel"] = float(ae_reliability.detach().cpu().item())
    return loss, state


def _edge_count(adj):
    return int(adj.nnz // 2)


def _edge_feature_cosine_mean(adj, features, max_edges=200000):
    adj_upper = sp.triu(adj, k=1).tocoo()
    if adj_upper.nnz == 0:
        return 0.0

    row = adj_upper.row
    col = adj_upper.col
    if adj_upper.nnz > max_edges:
        rng = np.random.default_rng(0)
        pick = rng.choice(adj_upper.nnz, size=max_edges, replace=False)
        row = row[pick]
        col = col[pick]

    if isinstance(features, torch.Tensor):
        feat_np = features.detach().cpu().numpy()
    else:
        feat_np = np.asarray(features)
    feat_np = np.asarray(feat_np, dtype=np.float32)
    norms = np.linalg.norm(feat_np, axis=1) + 1e-12
    sims = np.sum(feat_np[row] * feat_np[col], axis=1) / (norms[row] * norms[col])
    return float(np.mean(sims))


def _compute_adaptive_structure_prior(adj_a, adj_ae, features, args):
    """
    Conservative graph-level reliability prior for adaptive branch bias.

    This is intentionally a no-op for most datasets. It raw-anchors fusion only
    when graph diagnostics show that the refined graph is an unsafe correction:
    citation-like sparse features with many weak-gain AE edges, or dense features
    where the AE edges measurably reduce edge-feature consistency.
    """
    if getattr(args, "disable_adaptive_structure_prior", False):
        return None

    n_nodes = max(1, int(features.shape[0]))
    raw_edges = _edge_count(adj_a)
    ae_edges = _edge_count(adj_ae)
    raw_degree = 2.0 * raw_edges / n_nodes
    ae_degree = 2.0 * ae_edges / n_nodes
    degree_ratio = ae_edges / max(1, raw_edges)
    overlap = _edge_count(adj_a.multiply(adj_ae)) / max(1, raw_edges)
    new_edges = adj_ae - adj_ae.multiply(adj_a)
    new_edges.data[:] = 1
    new_edge_ratio = _edge_count(new_edges) / max(1, ae_edges)
    raw_feature_cos = _edge_feature_cosine_mean(adj_a, features)
    ae_feature_cos = _edge_feature_cosine_mean(adj_ae, features)
    feature_gain = ae_feature_cos - raw_feature_cos
    if isinstance(features, torch.Tensor):
        feat_np = features.detach().cpu().numpy()
    else:
        feat_np = np.asarray(features)
    feat_abs = np.abs(np.asarray(feat_np, dtype=np.float32))
    nonzero = feat_abs > 1e-12
    feature_density = float(np.mean(nonzero))
    nonzero_abs_mean = float(np.mean(feat_abs[nonzero])) if np.any(nonzero) else 0.0

    diag = {
        "raw_degree": raw_degree,
        "ae_degree": ae_degree,
        "degree_ratio": degree_ratio,
        "overlap": overlap,
        "new_edge_ratio": new_edge_ratio,
        "raw_feature_cos": raw_feature_cos,
        "ae_feature_cos": ae_feature_cos,
        "feature_gain": feature_gain,
        "feature_density": feature_density,
        "nonzero_abs_mean": nonzero_abs_mean,
    }

    citation_like_sparse_features = (
        feature_density <= float(args.adaptive_bias_feature_density_max)
        and nonzero_abs_mean <= float(args.adaptive_bias_nonzero_abs_mean_max)
    )
    dense_feature_degradation = (
        feature_density >= float(args.adaptive_bias_dense_feature_density_min)
        and feature_gain <= float(args.adaptive_bias_feature_loss_max)
    )
    sparse_raw = raw_degree <= float(args.adaptive_bias_sparse_raw_degree_max)
    ae_much_denser = degree_ratio >= float(args.adaptive_bias_ae_degree_ratio_min)
    mostly_new = new_edge_ratio >= float(args.adaptive_bias_new_edge_ratio_min)
    weak_feature_gain = feature_gain <= float(args.adaptive_bias_feature_gain_max)
    raw_not_feature_only = raw_feature_cos <= float(args.adaptive_bias_raw_feature_cos_max)
    if (
        citation_like_sparse_features
        and sparse_raw
        and ae_much_denser
        and mostly_new
        and weak_feature_gain
        and raw_not_feature_only
    ):
        return {
            "target": "raw",
            "cap": float(args.adaptive_bias_cap),
            "source": "structure",
            "diag": diag,
        }
    if (
        dense_feature_degradation
        and sparse_raw
        and ae_much_denser
        and mostly_new
    ):
        return {
            "target": "raw",
            "cap": float(args.adaptive_bias_cap),
            "source": "structure_feature_degradation",
            "diag": diag,
        }

    return {
        "target": None,
        "cap": 0.0,
        "source": "none",
        "diag": diag,
    }


def _is_structure_prior_source(state):
    return str(state.get("source", "")).startswith("structure")


def _resolve_adaptive_bias_start(args):
    configured = int(args.adaptive_bias_start_epoch)
    if configured >= 0:
        return configured
    return int(args.warmup_epochs)


def _update_adaptive_branch_bias_state(state, raw_score, ae_score, epoch, args):
    if raw_score is None or ae_score is None:
        return None, 0.0

    if epoch < _resolve_adaptive_bias_start(args):
        return None, 0.0

    margin = float((ae_score - raw_score).detach().cpu().item())
    ema_prev = state.get("ema_margin")
    if ema_prev is None:
        ema_margin = margin
    else:
        ema_margin = float(args.adaptive_bias_ema) * float(ema_prev) + (1.0 - float(args.adaptive_bias_ema)) * margin
    state["ema_margin"] = ema_margin

    threshold = float(args.adaptive_bias_margin)
    candidate = None
    if ema_margin >= threshold:
        candidate = "ae"
    elif ema_margin <= -threshold:
        candidate = "raw"

    if candidate is None:
        state["candidate"] = None
        state["candidate_count"] = 0
        if state.get("source") == "runtime":
            state["target"] = None
            state["activated_epoch"] = None
        return state.get("target"), _adaptive_branch_bias_cap(state, epoch, args)

    if state.get("candidate") == candidate:
        state["candidate_count"] = int(state.get("candidate_count", 0)) + 1
    else:
        state["candidate"] = candidate
        state["candidate_count"] = 1

    if int(state.get("candidate_count", 0)) >= int(args.adaptive_bias_patience):
        if state.get("target") != candidate:
            state["target"] = candidate
            state["activated_epoch"] = int(epoch)
            state["source"] = "runtime"

    return state.get("target"), _adaptive_branch_bias_cap(state, epoch, args)


def _adaptive_branch_bias_cap(state, epoch, args):
    target = state.get("target")
    if target not in ("raw", "ae"):
        return 0.0
    activated_epoch = int(state.get("activated_epoch", epoch))
    ramp_epochs = max(1, int(args.adaptive_bias_ramp_epochs))
    ramp = min(1.0, max(0.0, float(epoch - activated_epoch + 1) / float(ramp_epochs)))
    return ramp * float(args.adaptive_bias_cap)


def _update_fusion_collapse_guard_state(state, fusion_mean, epoch, args):
    if not args.enable_fusion_collapse_guard:
        return 0.0

    start_epoch = int(args.fusion_collapse_guard_start_epoch)
    if start_epoch < 0:
        start_epoch = int(args.warmup_epochs)
    end_epoch = int(args.fusion_collapse_guard_end_epoch)
    if end_epoch < 0:
        end_epoch = int(args.epochs)
    if epoch < start_epoch or epoch > end_epoch:
        state["candidate_count"] = 0
        state["floor"] = 0.0
        return 0.0

    weights = fusion_mean.detach()
    dominant = float(torch.max(weights).cpu().item())
    collapsed = dominant >= float(args.fusion_collapse_guard_threshold)
    if collapsed:
        state["candidate_count"] = int(state.get("candidate_count", 0)) + 1
    else:
        state["candidate_count"] = 0

    if int(state.get("candidate_count", 0)) < int(args.fusion_collapse_guard_patience):
        state["floor"] = 0.0
        return 0.0

    activated_epoch = state.get("activated_epoch")
    if activated_epoch is None:
        state["activated_epoch"] = int(epoch)
        activated_epoch = int(epoch)

    release_epochs = max(1, int(args.fusion_collapse_guard_release_epochs))
    age = max(0, int(epoch) - int(activated_epoch))
    release = max(0.0, 1.0 - float(age) / float(release_epochs))
    floor = float(args.fusion_collapse_guard_floor) * release
    state["floor"] = floor
    return floor


### ---------------------------------------


def _compute_cluster_reliability(z1, z2, centers_1, centers_2, cluster_members):
    if len(cluster_members) == 0:
        return None, None, None

    cluster_sizes = torch.tensor(
        [members.numel() for members in cluster_members],
        dtype=centers_1.dtype,
        device=centers_1.device
    )
    max_size = torch.clamp(torch.max(cluster_sizes), min=1.0)
    size_score = torch.sqrt(cluster_sizes / max_size)

    reliability_scores = []
    compactness_scores = []
    agreement_scores = []
    for idx, members in enumerate(cluster_members):
        c1 = centers_1[idx:idx + 1]
        c2 = centers_2[idx:idx + 1]
        z1_members = F.normalize(z1[members], dim=1, p=2)
        z2_members = F.normalize(z2[members], dim=1, p=2)

        compact_1 = torch.mean(torch.clamp(torch.sum(z1_members * c1, dim=1), min=-1.0, max=1.0))
        compact_2 = torch.mean(torch.clamp(torch.sum(z2_members * c2, dim=1), min=-1.0, max=1.0))
        agreement = torch.clamp(torch.sum(c1 * c2, dim=1), min=-1.0, max=1.0).squeeze(0)

        compact_score = 0.25 * (compact_1 + compact_2 + 2.0)
        agreement_score = 0.5 * (agreement + 1.0)
        reliability = 0.5 * agreement_score + 0.4 * compact_score + 0.1 * size_score[idx]

        reliability_scores.append(reliability)
        compactness_scores.append(compact_score)
        agreement_scores.append(agreement_score)

    reliability_scores = torch.stack(reliability_scores, dim=0)
    compactness_scores = torch.stack(compactness_scores, dim=0)
    agreement_scores = torch.stack(agreement_scores, dim=0)
    reliability_scores = torch.clamp(reliability_scores, min=0.0, max=1.0)
    compactness_scores = torch.clamp(compactness_scores, min=0.0, max=1.0)
    agreement_scores = torch.clamp(agreement_scores, min=0.0, max=1.0)
    return reliability_scores, compactness_scores, agreement_scores


def _ccgc_confidence_loss(
    z1,
    z2,
    predict_labels_t,
    high_confidence_idx,
    cluster_num,
    alpha,
    use_ema_proto=False,
    ema_state=None,
    ema_momentum=0.9,
    use_dcgl_negative=False,
    dcgl_neg_tau=0.5,
    dcgl_neg_weight=1.0,
    use_dcgl_reliability_gate=True,
    dcgl_neg_gate_threshold=0.55,
    dcgl_neg_gate_power=2.0,
    dcgl_neg_gate_min=0.0
):
    if high_confidence_idx.numel() == 0:
        return None

    y_sam = predict_labels_t[high_confidence_idx]
    unique_labels = torch.unique(y_sam, sorted=True)
    if unique_labels.numel() < 2:
        return None

    pos_contrastive = torch.tensor(0.0, device=z1.device)
    centers_1 = []
    centers_2 = []
    center_labels = []
    cluster_members = []

    for label in unique_labels:
        now = high_confidence_idx[y_sam == label]
        if now.numel() < 2:
            continue
        sample_size = max(1, int(now.numel() * 0.8))
        sample_idx = now[torch.randperm(now.numel(), device=z1.device)[:sample_size]]
        pos_embed_1 = z1[sample_idx]
        pos_embed_2 = z2[sample_idx]
        pos_contrastive += (2 - 2 * torch.sum(pos_embed_1 * pos_embed_2, dim=1)).sum()
        centers_1.append(torch.mean(z1[now], dim=0, keepdim=True))
        centers_2.append(torch.mean(z2[now], dim=0, keepdim=True))
        center_labels.append(int(label.item()))
        cluster_members.append(now)

    if len(centers_1) == 0:
        return None

    pos_contrastive = pos_contrastive / max(1, cluster_num)

    if len(centers_1) < 2:
        return pos_contrastive

    centers_1_t = torch.cat(centers_1, dim=0)
    centers_2_t = torch.cat(centers_2, dim=0)
    centers_1_now = F.normalize(centers_1_t, dim=1, p=2)
    centers_2_now = F.normalize(centers_2_t, dim=1, p=2)
    reliability_scores, compactness_scores, agreement_scores = _compute_cluster_reliability(
        z1,
        z2,
        centers_1_now,
        centers_2_now,
        cluster_members
    )
    ### <--- [MODIFIED] ---------------------------------------
    # Optional EMA prototypes: default OFF to keep original CCGC behavior unchanged.
    if use_ema_proto and ema_state is not None:
        with torch.no_grad():
            if "c1" not in ema_state:
                ema_state["c1"] = torch.zeros(cluster_num, centers_1_t.shape[1], device=z1.device)
                ema_state["c2"] = torch.zeros(cluster_num, centers_2_t.shape[1], device=z2.device)
                ema_state["valid"] = torch.zeros(cluster_num, dtype=torch.bool, device=z1.device)
                ema_state["steps"] = torch.zeros(cluster_num, dtype=torch.long, device=z1.device)
                ema_state["reliability"] = torch.zeros(cluster_num, dtype=centers_1_t.dtype, device=z1.device)

            ema_centers_1 = []
            ema_centers_2 = []
            for idx, label_idx in enumerate(center_labels):
                c1_now = centers_1_t[idx].detach()
                c2_now = centers_2_t[idx].detach()
                reliability = reliability_scores[idx].detach() if reliability_scores is not None else torch.tensor(
                    1.0, dtype=centers_1_t.dtype, device=z1.device
                )
                step_now = int(ema_state["steps"][label_idx].item())
                prev_rel = ema_state["reliability"][label_idx]
                ema_state["reliability"][label_idx] = 0.8 * prev_rel + 0.2 * reliability

                # Delay EMA takeover until the cluster has been observed enough
                # times and is sufficiently reliable; otherwise keep raw centers.
                use_ema_for_output = False
                if ema_state["valid"][label_idx]:
                    gate_ready = step_now >= 2 and float(ema_state["reliability"][label_idx].item()) >= 0.55
                    if gate_ready:
                        rel_scale = float(torch.clamp((ema_state["reliability"][label_idx] - 0.55) / 0.45, 0.0, 1.0).item())
                        adaptive_keep = ema_momentum + (1.0 - ema_momentum) * (0.35 * (1.0 - rel_scale))
                        adaptive_keep = min(0.995, max(0.0, adaptive_keep))
                        ema_state["c1"][label_idx] = adaptive_keep * ema_state["c1"][label_idx] + (1.0 - adaptive_keep) * c1_now
                        ema_state["c2"][label_idx] = adaptive_keep * ema_state["c2"][label_idx] + (1.0 - adaptive_keep) * c2_now
                        use_ema_for_output = True
                    else:
                        # Refresh the stored prototype, but do not force the
                        # downstream loss to consume EMA before the cluster stabilizes.
                        ema_state["c1"][label_idx] = c1_now
                        ema_state["c2"][label_idx] = c2_now
                else:
                    ema_state["c1"][label_idx] = c1_now
                    ema_state["c2"][label_idx] = c2_now
                    ema_state["valid"][label_idx] = True
                ema_state["steps"][label_idx] = step_now + 1
                if use_ema_for_output:
                    ema_centers_1.append(ema_state["c1"][label_idx].unsqueeze(0))
                    ema_centers_2.append(ema_state["c2"][label_idx].unsqueeze(0))
                else:
                    ema_centers_1.append(c1_now.unsqueeze(0))
                    ema_centers_2.append(c2_now.unsqueeze(0))

        centers_1_t = torch.cat(ema_centers_1, dim=0)
        centers_2_t = torch.cat(ema_centers_2, dim=0)
    ### ---------------------------------------
    centers_1 = F.normalize(centers_1_t, dim=1, p=2)
    centers_2 = F.normalize(centers_2_t, dim=1, p=2)
    ### <--- [MODIFIED] ---------------------------------------
    base_neg_contrastive = _offdiag_center_mse_negative_loss(centers_1, centers_2)
    neg_weight = 1.0
    if use_dcgl_negative:
        if reliability_scores is None:
            reliability_scores = torch.ones(centers_1.shape[0], dtype=centers_1.dtype, device=centers_1.device)
        if use_dcgl_reliability_gate:
            cluster_gate = _smooth_reliability_gate(
                reliability_scores,
                threshold=dcgl_neg_gate_threshold,
                power=dcgl_neg_gate_power,
                min_gate=dcgl_neg_gate_min,
            ).detach()
            global_gate = torch.clamp(torch.mean(cluster_gate), min=0.0, max=1.0)
            dcgl_neg_contrastive = _dcgl_center_contrastive_negative_loss(
                centers_1,
                centers_2,
                temperature=dcgl_neg_tau,
                row_weights=cluster_gate
            )
            # Reliability gate semantics:
            # low reliability should fall back to the A-DSF objective, not to a
            # weakened/replaced negative. DCGL-negative is therefore an extra
            # gated separation term on top of the base off-diagonal center loss.
            neg_contrastive = base_neg_contrastive + global_gate * float(dcgl_neg_weight) * dcgl_neg_contrastive
        else:
            legacy_weights = _legacy_dcgl_negative_row_weights(reliability_scores)
            neg_contrastive = _dcgl_center_contrastive_negative_loss(
                centers_1,
                centers_2,
                temperature=dcgl_neg_tau,
                row_weights=legacy_weights
            )
            neg_weight = float(dcgl_neg_weight)
    else:
        neg_contrastive = base_neg_contrastive
    return pos_contrastive + alpha * neg_weight * neg_contrastive
    ### ---------------------------------------


def _instance_align_loss(hidden_a, hidden_ae):
    return torch.mean(1 - F.cosine_similarity(hidden_a, hidden_ae, dim=1))


def _cluster_distribution_align_loss(hidden_a, hidden_ae, predict_labels_t, cluster_num, tau, active_idx=None):
    if active_idx is not None and active_idx.numel() > 0:
        hidden_a = hidden_a[active_idx]
        hidden_ae = hidden_ae[active_idx]
        predict_labels_t = predict_labels_t[active_idx]

    if hidden_a.shape[0] == 0:
        return torch.tensor(0.0, device=hidden_a.device)

    proto_a = []
    proto_ae = []
    global_a = torch.mean(hidden_a, dim=0, keepdim=True)
    global_ae = torch.mean(hidden_ae, dim=0, keepdim=True)

    for c in range(cluster_num):
        mask = (predict_labels_t == c)
        if mask.any():
            proto_a.append(torch.mean(hidden_a[mask], dim=0, keepdim=True))
            proto_ae.append(torch.mean(hidden_ae[mask], dim=0, keepdim=True))
        else:
            proto_a.append(global_a)
            proto_ae.append(global_ae)

    proto_shared = F.normalize((torch.cat(proto_a, dim=0) + torch.cat(proto_ae, dim=0)) / 2.0, dim=1, p=2)
    q_a = F.softmax((hidden_a @ proto_shared.T) / tau, dim=1)
    q_ae = F.softmax((hidden_ae @ proto_shared.T) / tau, dim=1)

    log_q_a = torch.log(torch.clamp(q_a, min=1e-8))
    log_q_ae = torch.log(torch.clamp(q_ae, min=1e-8))
    loss = 0.5 * (
        F.kl_div(log_q_a, q_ae, reduction='batchmean') +
        F.kl_div(log_q_ae, q_a, reduction='batchmean')
    )
    return loss


### <--- [MODIFIED] ---------------------------------------
def _cluster_level_contrastive_loss(hidden_a, hidden_ae, predict_labels_t, cluster_num, tau, active_idx=None):
    """
    DCGL-style cluster-level contrastive alignment for dual-view training.
    """
    if active_idx is not None and active_idx.numel() > 0:
        hidden_a = hidden_a[active_idx]
        hidden_ae = hidden_ae[active_idx]
        predict_labels_t = predict_labels_t[active_idx]

    if hidden_a.shape[0] == 0:
        return torch.tensor(0.0, device=hidden_a.device)

    proto_a = []
    proto_ae = []
    proto_weight = []
    hidden_a = F.normalize(hidden_a, dim=1, p=2)
    hidden_ae = F.normalize(hidden_ae, dim=1, p=2)
    global_a = torch.mean(hidden_a, dim=0, keepdim=True)
    global_ae = torch.mean(hidden_ae, dim=0, keepdim=True)
    for c in range(cluster_num):
        mask = (predict_labels_t == c)
        if mask.any():
            cluster_a = hidden_a[mask]
            cluster_ae = hidden_ae[mask]
            center_a = F.normalize(torch.mean(cluster_a, dim=0, keepdim=True), dim=1, p=2)
            center_ae = F.normalize(torch.mean(cluster_ae, dim=0, keepdim=True), dim=1, p=2)

            compact_a = torch.mean(torch.clamp(cluster_a @ center_a.T, min=-1.0, max=1.0))
            compact_ae = torch.mean(torch.clamp(cluster_ae @ center_ae.T, min=-1.0, max=1.0))
            agreement = torch.clamp(torch.sum(center_a * center_ae, dim=1), min=-1.0, max=1.0).squeeze(0)
            size_weight = min(1.0, float(mask.sum().item()) / max(2.0, float(hidden_a.shape[0]) / max(1, cluster_num)))

            compact_score = 0.25 * (compact_a + compact_ae + 2.0)
            agreement_score = 0.5 * (agreement + 1.0)
            reliability = 0.5 * agreement_score + 0.35 * compact_score + 0.15 * size_weight
            reliability = torch.clamp(reliability, min=0.0, max=1.0)

            # Reliability-aware interpolation:
            # unstable prototypes lean toward a shared cross-view anchor instead
            # of directly amplifying wrong pseudo-label assignments.
            shared_center = F.normalize((center_a + center_ae) / 2.0, dim=1, p=2)
            mixing = torch.clamp((reliability - 0.4) / 0.6, min=0.0, max=1.0)
            proto_a.append(F.normalize(mixing * center_a + (1.0 - mixing) * shared_center, dim=1, p=2))
            proto_ae.append(F.normalize(mixing * center_ae + (1.0 - mixing) * shared_center, dim=1, p=2))
            proto_weight.append(reliability.unsqueeze(0))
        else:
            proto_a.append(global_a)
            proto_ae.append(global_ae)
            proto_weight.append(torch.zeros(1, dtype=hidden_a.dtype, device=hidden_a.device))

    proto_a = F.normalize(torch.cat(proto_a, dim=0), dim=1, p=2)
    proto_ae = F.normalize(torch.cat(proto_ae, dim=0), dim=1, p=2)
    proto_weight = torch.cat(proto_weight, dim=0)

    tau = max(1e-6, float(tau))
    sim_cross1 = torch.exp((proto_a @ proto_ae.T) / tau)
    sim_cross2 = torch.exp((proto_ae @ proto_a.T) / tau)
    sim_same1 = torch.exp((proto_a @ proto_a.T) / tau)
    sim_same2 = torch.exp((proto_ae @ proto_ae.T) / tau)

    diag_cross1 = torch.diagonal(sim_cross1)
    diag_cross2 = torch.diagonal(sim_cross2)
    diag_same1 = torch.diagonal(sim_same1)
    diag_same2 = torch.diagonal(sim_same2)

    sep1 = (torch.sum(sim_same1, dim=1) - diag_same1) / max(1, cluster_num - 1)
    sep2 = (torch.sum(sim_same2, dim=1) - diag_same2) / max(1, cluster_num - 1)

    loss1 = -torch.log(diag_cross1 / torch.clamp(torch.sum(sim_cross1, dim=1), min=1e-12)) + sep1
    loss2 = -torch.log(diag_cross2 / torch.clamp(torch.sum(sim_cross2, dim=1), min=1e-12)) + sep2
    weights = torch.clamp(proto_weight, min=0.0, max=1.0)
    active_mask = weights > 1e-6
    if not torch.any(active_mask):
        return torch.tensor(0.0, device=hidden_a.device)

    # Suppress cluster-level pressure on low-confidence pseudo clusters while
    # keeping the old objective shape for reliable clusters.
    weights = torch.where(active_mask, 0.35 + 0.65 * weights, torch.zeros_like(weights))
    denom = torch.clamp(torch.sum(weights), min=1e-6)
    return 0.5 * (
        torch.sum(weights * loss1) / denom +
        torch.sum(weights * loss2) / denom
    )
### ---------------------------------------
### ---------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=4, help="Number of gnn layers")
parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=500, help='feature dim')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--dataset', type=str, default='cora', help='name of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='number of cluster.')
parser.add_argument('--device', type=str, default='cuda', help='the training device')
parser.add_argument('--threshold', type=float, default=0.5, help='the threshold of high-confidence')
parser.add_argument('--alpha', type=float, default=0.5, help='trade-off of loss')
### <--- [MODIFIED] ---------------------------------------
parser.add_argument('--enable_improved_ccgc', action='store_true',
                    help='[compat] enable both dynamic threshold and EMA prototypes')
parser.add_argument('--enable_dynamic_threshold', action='store_true',
                    help='enable dynamic high-confidence threshold schedule')
parser.add_argument('--enable_ema_prototypes', action='store_true',
                    help='enable EMA cluster prototypes for negative-sample centers')
parser.add_argument('--dynamic_threshold_start', type=float, default=0.2,
                    help='start ratio for dynamic high-confidence threshold when dynamic-threshold module is enabled')
parser.add_argument('--dynamic_threshold_end', type=float, default=0.5,
                    help='end ratio for dynamic high-confidence threshold when dynamic-threshold module is enabled')
parser.add_argument('--ema_proto_momentum', type=float, default=0.9,
                    help='EMA momentum for cluster prototypes when EMA-prototype module is enabled')
parser.add_argument('--enable_dcgl_negative_loss', action='store_true',
                    help='enable DCGL-style cluster-guided contrastive negatives in CCGC confidence loss')
parser.add_argument('--dcgl_neg_tau', type=float, default=0.5,
                    help='temperature for DCGL-style center-level negative contrast')
parser.add_argument('--dcgl_neg_weight', type=float, default=1.0,
                    help='weight multiplier for DCGL-style negative term')
parser.add_argument('--disable_dcgl_neg_reliability_gate', action='store_true',
                    help='disable reliability-gated DCGL negative and use the legacy conservative row-weighted DCGL negative')
parser.add_argument('--dcgl_neg_gate_threshold', type=float, default=0.55,
                    help='cluster reliability threshold for gradually enabling DCGL negative contrast')
parser.add_argument('--dcgl_neg_gate_power', type=float, default=2.0,
                    help='power of the smooth reliability gate for DCGL negative contrast')
parser.add_argument('--dcgl_neg_gate_min', type=float, default=0.0,
                    help='minimum gate value for the extra DCGL negative contrast; 0 makes low-reliability states use only the base negative')
parser.add_argument('--enable_dcgl_cluster_level', action='store_true',
                    help='enable DCGL-style cluster-level contrastive alignment for dual-view mode')
parser.add_argument('--lambda_dcgl_cluster', type=float, default=0.1,
                    help='weight of DCGL-style cluster-level contrastive alignment')
parser.add_argument('--dcgl_cluster_tau', type=float, default=0.5,
                    help='temperature for DCGL-style cluster-level contrastive alignment')
parser.add_argument('--enable_gcn_backbone', action='store_true',
                    help='enable optional GCN encoder backbone (default OFF keeps original MLP encoder)')
parser.add_argument('--gcn_dropout', type=float, default=0.0,
                    help='dropout used by GCN encoder backbone')
### ---------------------------------------
### <--- [MODIFIED] ---------------------------------------
parser.add_argument('--graph_mode', type=str, default='raw', choices=['raw', 'knn', 'ae', 'dual'],
                    help="graph source for project datasets: 'raw' (prefer npy/original graph, fallback to knn), 'knn', 'ae', or dual-view 'dual'")
parser.add_argument('--ae_graph_path', type=str, default='',
                    help='path to Ae edge-list file, used when --graph_mode ae')
parser.add_argument('--raw_graph_path', type=str, default='',
                    help='optional raw/original edge-list override for structural-uncertainty experiments')
parser.add_argument('--knn_k', type=int, default=5,
                    help='k for KNN graph construction when --graph_mode knn or when raw graph fallback is needed')
parser.add_argument('--warmup_epochs', type=int, default=50,
                    help='warmup epochs before confidence-guided objective')
parser.add_argument('--lambda_inst', type=float, default=0.2,
                    help='weight of instance-level alignment for dual mode')
parser.add_argument('--lambda_clu', type=float, default=0.2,
                    help='weight of cluster-distribution alignment for dual mode')
parser.add_argument('--dist_tau', type=float, default=0.5,
                    help='temperature for cluster-distribution alignment in dual mode')
parser.add_argument('--fusion_mode', type=str, default='mean', choices=['mean', 'fixed', 'attn'],
                    help="dual-view fusion strategy: fixed mean, explicit fixed raw/AE weight, or learnable attention")
parser.add_argument('--fixed_raw_weight', type=float, default=0.5,
                    help='raw-graph branch weight when --fusion_mode fixed; AE weight is 1-fixed_raw_weight')
parser.add_argument('--fusion_hidden', type=int, default=128,
                    help='hidden size of attention fusion MLP when fusion_mode=attn')
parser.add_argument('--fusion_temp', type=float, default=1.0,
                    help='softmax temperature for attention fusion')
parser.add_argument('--fusion_balance', type=float, default=0.0,
                    help='optional balance regularization weight for attention fusion')
parser.add_argument('--fusion_min_weight', type=float, default=0.0,
                    help='minimum per-branch fusion weight when fusion_mode=attn (0~0.49)')
parser.add_argument('--enable_fusion_reliability_feedback_loss', action='store_true',
                    help='decouple fusion-weight learning from branch-loss shortcut and train it with reliability feedback')
parser.add_argument('--fusion_feedback_weight', type=float, default=1.0,
                    help='weight for reliability feedback loss applied to attention fusion')
parser.add_argument('--fusion_feedback_temp', type=float, default=0.25,
                    help='temperature for runtime reliability feedback target')
parser.add_argument('--fusion_feedback_min_weak_weight', type=float, default=0.02,
                    help='minimum weak-branch target used by reliability feedback')
parser.add_argument('--fusion_feedback_prior_strength', type=float, default=1.0,
                    help='logit-scale structure-prior evidence used by reliability feedback')
parser.add_argument('--fusion_feedback_warmup_ramp_epochs', type=int, default=20,
                    help='epochs used to ramp reliability feedback during reconstruction warmup')
parser.add_argument('--fusion_feedback_use_runtime_reliability', action='store_true',
                    help='use embedding reliability scores as fusion feedback when no structure prior is active')
parser.add_argument('--fusion_feedback_loss_type', type=str, default='mse',
                    choices=['mse', 'violation', 'barrier', 'controller', 'shortcut', 'countergrad', 'selective_countergrad'],
                    help='reliability feedback loss type for attention weights')
parser.add_argument('--fusion_feedback_loss_temp', type=float, default=0.25,
                    help='temperature that converts branch losses into reliability evidence')
parser.add_argument('--fusion_feedback_loss_blend', type=float, default=0.35,
                    help='branch-loss evidence blend for barrier feedback')
parser.add_argument('--fusion_feedback_prior_blend', type=float, default=0.70,
                    help='structure-prior evidence blend for controller feedback')
parser.add_argument('--fusion_feedback_min_barrier_coeff', type=float, default=0.08,
                    help='minimum log-barrier coefficient for the weaker evidence branch')
parser.add_argument('--fusion_feedback_max_barrier_coeff', type=float, default=4.0,
                    help='maximum log-barrier coefficient for the stronger evidence branch')
parser.add_argument('--fusion_feedback_boundary_sharpness', type=float, default=24.0,
                    help='sharpness of controller activation near collapse boundary')
parser.add_argument('--fusion_feedback_wrong_branch_margin', type=float, default=0.50,
                    help='wrong-side branch weight that activates stronger controller feedback')
parser.add_argument('--fusion_feedback_controller_base', type=float, default=0.20,
                    help='always-on reliability controller strength inside feedback loss')
parser.add_argument('--fusion_feedback_controller_boost', type=float, default=1.50,
                    help='extra controller strength after wrong-side activation')
parser.add_argument('--fusion_feedback_loss_gap_gain', type=float, default=0.0,
                    help='increase controller strength when detached branch losses disagree strongly')
parser.add_argument('--fusion_feedback_shortcut_gain', type=float, default=1.0,
                    help='gain for shortcut-cancellation feedback loss')
parser.add_argument('--fusion_feedback_countergrad_curvature', type=float, default=1.0,
                    help='curvature around reliability target for counter-gradient feedback')
parser.add_argument('--fusion_feedback_countergrad_base', type=float, default=0.05,
                    help='small MSE anchor weight used by counter-gradient feedback')
parser.add_argument('--fusion_feedback_detach_branch_loss', action='store_true',
                    help='detach fusion weights from branch losses and train them only through reliability feedback')
parser.add_argument('--enable_learnable_boundary_gate', action='store_true',
                    help='enable learnable attention-boundary self-correction near extreme fusion weights')
parser.add_argument('--boundary_gate_max_floor', type=float, default=0.20,
                    help='maximum learnable symmetric correction floor for boundary gate')
parser.add_argument('--boundary_gate_init_floor', type=float, default=0.03,
                    help='initial correction floor for learnable boundary gate')
parser.add_argument('--boundary_gate_min_threshold', type=float, default=0.75,
                    help='lower bound for learned dominant-weight threshold')
parser.add_argument('--boundary_gate_max_threshold', type=float, default=0.98,
                    help='upper bound for learned dominant-weight threshold')
parser.add_argument('--boundary_gate_init_threshold', type=float, default=0.92,
                    help='initial dominant-weight threshold for learnable boundary gate')
parser.add_argument('--boundary_gate_sharpness', type=float, default=30.0,
                    help='smoothness of boundary-gate activation near the learned threshold')
parser.add_argument('--enable_branch_bias_fusion', action='store_true',
                    help='enable optional branch-biased attn fusion; default OFF keeps the current attn behavior unchanged')
parser.add_argument('--branch_bias_target', type=str, default='raw', choices=['raw', 'ae'],
                    help="which branch to anchor when branch-biased fusion is enabled: 'raw' or 'ae'")
parser.add_argument('--branch_bias_cap', type=float, default=0.2,
                    help='maximum correction weight for the non-anchored branch when branch-biased fusion is enabled')
parser.add_argument('--enable_adaptive_branch_bias', action='store_true',
                    help='enable delayed unsupervised raw/AE branch selection inside attention fusion')
parser.add_argument('--disable_adaptive_structure_prior', action='store_true',
                    help='disable conservative graph-level structure prior used by adaptive branch bias')
parser.add_argument('--enable_runtime_adaptive_branch_bias', action='store_true',
                    help='also enable embedding-level runtime branch selection after the structure prior')
parser.add_argument('--adaptive_bias_start_epoch', type=int, default=-1,
                    help='epoch to start adaptive branch selection; -1 uses warmup_epochs')
parser.add_argument('--adaptive_bias_margin', type=float, default=0.03,
                    help='EMA reliability margin required to choose raw or AE branch')
parser.add_argument('--adaptive_bias_patience', type=int, default=5,
                    help='number of consecutive checks before committing a branch target')
parser.add_argument('--adaptive_bias_cap', type=float, default=0.35,
                    help='maximum correction weight for the non-selected branch after adaptive selection')
parser.add_argument('--adaptive_bias_ramp_epochs', type=int, default=20,
                    help='epochs used to ramp the adaptive branch-bias cap after activation')
parser.add_argument('--adaptive_bias_ema', type=float, default=0.8,
                    help='EMA momentum for the raw-vs-AE branch reliability margin')
parser.add_argument('--adaptive_bias_mode', type=str, default='cap', choices=['boundary', 'cap'],
                    help='internal adaptive-fusion correction mode when a reliable branch is selected')
parser.add_argument('--adaptive_boundary_sharpness', type=float, default=24.0,
                    help='sharpness of the internal adaptive fusion boundary correction')
parser.add_argument('--adaptive_boundary_tighten', type=float, default=0.0,
                    help='extra weak-branch cap tightening after adaptive boundary activation')
parser.add_argument('--adaptive_bias_sparse_raw_degree_max', type=float, default=12.0,
                    help='raw average degree upper bound for conservative structure-prior raw anchoring')
parser.add_argument('--adaptive_bias_ae_degree_ratio_min', type=float, default=1.7,
                    help='minimum AE/raw edge ratio for conservative structure-prior raw anchoring')
parser.add_argument('--adaptive_bias_new_edge_ratio_min', type=float, default=0.75,
                    help='minimum AE new-edge ratio for conservative structure-prior raw anchoring')
parser.add_argument('--adaptive_bias_feature_gain_max', type=float, default=0.12,
                    help='maximum AE-minus-raw edge feature-cosine gain for conservative raw anchoring')
parser.add_argument('--adaptive_bias_raw_feature_cos_max', type=float, default=0.35,
                    help='maximum raw edge feature-cosine mean for conservative raw anchoring')
parser.add_argument('--adaptive_bias_feature_density_max', type=float, default=0.02,
                    help='maximum feature density for conservative citation-like raw anchoring')
parser.add_argument('--adaptive_bias_nonzero_abs_mean_max', type=float, default=1.05,
                    help='maximum mean absolute nonzero feature value for conservative citation-like raw anchoring')
parser.add_argument('--adaptive_bias_dense_feature_density_min', type=float, default=0.05,
                    help='minimum feature density for dense-feature AE degradation diagnostics')
parser.add_argument('--adaptive_bias_feature_loss_max', type=float, default=-0.02,
                    help='maximum AE-minus-raw edge feature-cosine gain that indicates AE degradation')
parser.add_argument('--enable_fusion_collapse_guard', action='store_true',
                    help='enable bounded self-correction when attention fusion collapses before reliability evidence is stable')
parser.add_argument('--fusion_collapse_guard_threshold', type=float, default=0.95,
                    help='dominant average fusion weight threshold that indicates premature single-view dominance')
parser.add_argument('--fusion_collapse_guard_patience', type=int, default=10,
                    help='number of consecutive epochs above the collapse threshold before the guard activates')
parser.add_argument('--fusion_collapse_guard_floor', type=float, default=0.05,
                    help='temporary symmetric floor applied by the collapse guard; permanent fusion_min_weight can stay 0')
parser.add_argument('--fusion_collapse_guard_start_epoch', type=int, default=-1,
                    help='epoch to start collapse monitoring; -1 uses warmup_epochs')
parser.add_argument('--fusion_collapse_guard_end_epoch', type=int, default=-1,
                    help='epoch to stop collapse monitoring; -1 uses total epochs')
parser.add_argument('--fusion_collapse_guard_release_epochs', type=int, default=80,
                    help='epochs over which the collapse guard floor decays after activation')
parser.add_argument('--runs', type=int, default=10,
                    help='number of independent training runs; default keeps the paper setting')
parser.add_argument('--seed_start', type=int, default=0,
                    help='first training seed; seeds are seed_start..seed_start+runs-1')
parser.add_argument('--save_embedding_path', type=str, default='',
                    help='optional npz path for saving the best embedding, labels, predictions, and metrics')
parser.add_argument('--save_embedding_method', type=str, default='',
                    help='method label stored in --save_embedding_path')
parser.add_argument('--save_fusion_weights_path', type=str, default='',
                    help='optional npz path for saving best-ACC dual-view fusion weights and branch embeddings')
### ---------------------------------------
args = parser.parse_args()

### <--- [MODIFIED] ---------------------------------------
if args.device == 'cuda' and not torch.cuda.is_available():
    print("CUDA is not available; fallback to CPU.")
    args.device = 'cpu'
### <--- [MODIFIED] ---------------------------------------
# Backward-compatible alias: old switch enables both decoupled modules.
if args.enable_improved_ccgc:
    args.enable_dynamic_threshold = True
    args.enable_ema_prototypes = True

args.fusion_min_weight = min(0.49, max(0.0, float(args.fusion_min_weight)))
args.fixed_raw_weight = min(1.0, max(0.0, float(args.fixed_raw_weight)))
args.fusion_feedback_weight = max(0.0, float(args.fusion_feedback_weight))
args.fusion_feedback_temp = max(1e-6, float(args.fusion_feedback_temp))
args.fusion_feedback_min_weak_weight = min(0.49, max(0.0, float(args.fusion_feedback_min_weak_weight)))
args.fusion_feedback_prior_strength = max(0.0, float(args.fusion_feedback_prior_strength))
args.fusion_feedback_warmup_ramp_epochs = max(1, int(args.fusion_feedback_warmup_ramp_epochs))
args.fusion_feedback_loss_temp = max(1e-6, float(args.fusion_feedback_loss_temp))
args.fusion_feedback_loss_blend = min(1.0, max(0.0, float(args.fusion_feedback_loss_blend)))
args.fusion_feedback_prior_blend = min(1.0, max(0.0, float(args.fusion_feedback_prior_blend)))
args.fusion_feedback_min_barrier_coeff = max(1e-6, float(args.fusion_feedback_min_barrier_coeff))
args.fusion_feedback_max_barrier_coeff = max(
    args.fusion_feedback_min_barrier_coeff,
    float(args.fusion_feedback_max_barrier_coeff),
)
args.fusion_feedback_boundary_sharpness = max(1e-6, float(args.fusion_feedback_boundary_sharpness))
args.fusion_feedback_wrong_branch_margin = min(0.99, max(0.01, float(args.fusion_feedback_wrong_branch_margin)))
args.fusion_feedback_controller_base = max(0.0, float(args.fusion_feedback_controller_base))
args.fusion_feedback_controller_boost = max(0.0, float(args.fusion_feedback_controller_boost))
args.fusion_feedback_loss_gap_gain = max(0.0, float(args.fusion_feedback_loss_gap_gain))
args.fusion_feedback_shortcut_gain = max(0.0, float(args.fusion_feedback_shortcut_gain))
args.fusion_feedback_countergrad_curvature = max(0.0, float(args.fusion_feedback_countergrad_curvature))
args.fusion_feedback_countergrad_base = max(0.0, float(args.fusion_feedback_countergrad_base))
args.boundary_gate_max_floor = min(0.49, max(0.0, float(args.boundary_gate_max_floor)))
args.boundary_gate_init_floor = min(args.boundary_gate_max_floor, max(0.0, float(args.boundary_gate_init_floor)))
args.boundary_gate_min_threshold = min(0.999, max(0.5, float(args.boundary_gate_min_threshold)))
args.boundary_gate_max_threshold = min(0.999, max(args.boundary_gate_min_threshold + 1e-6, float(args.boundary_gate_max_threshold)))
args.boundary_gate_init_threshold = min(args.boundary_gate_max_threshold, max(args.boundary_gate_min_threshold, float(args.boundary_gate_init_threshold)))
args.boundary_gate_sharpness = max(1e-6, float(args.boundary_gate_sharpness))
args.branch_bias_cap = min(0.49, max(0.0, float(args.branch_bias_cap)))
args.adaptive_bias_margin = max(0.0, float(args.adaptive_bias_margin))
args.adaptive_bias_patience = max(1, int(args.adaptive_bias_patience))
args.adaptive_bias_cap = min(0.49, max(0.0, float(args.adaptive_bias_cap)))
args.adaptive_bias_ramp_epochs = max(1, int(args.adaptive_bias_ramp_epochs))
args.adaptive_bias_ema = min(0.999, max(0.0, float(args.adaptive_bias_ema)))
args.adaptive_boundary_sharpness = max(1e-6, float(args.adaptive_boundary_sharpness))
args.adaptive_boundary_tighten = min(0.95, max(0.0, float(args.adaptive_boundary_tighten)))
args.adaptive_bias_sparse_raw_degree_max = max(0.0, float(args.adaptive_bias_sparse_raw_degree_max))
args.adaptive_bias_ae_degree_ratio_min = max(0.0, float(args.adaptive_bias_ae_degree_ratio_min))
args.adaptive_bias_new_edge_ratio_min = min(1.0, max(0.0, float(args.adaptive_bias_new_edge_ratio_min)))
args.adaptive_bias_feature_gain_max = float(args.adaptive_bias_feature_gain_max)
args.adaptive_bias_raw_feature_cos_max = min(1.0, max(-1.0, float(args.adaptive_bias_raw_feature_cos_max)))
args.adaptive_bias_feature_density_max = min(1.0, max(0.0, float(args.adaptive_bias_feature_density_max)))
args.adaptive_bias_nonzero_abs_mean_max = max(0.0, float(args.adaptive_bias_nonzero_abs_mean_max))
args.adaptive_bias_dense_feature_density_min = min(1.0, max(0.0, float(args.adaptive_bias_dense_feature_density_min)))
args.adaptive_bias_feature_loss_max = float(args.adaptive_bias_feature_loss_max)
args.fusion_collapse_guard_threshold = min(0.999, max(0.5, float(args.fusion_collapse_guard_threshold)))
args.fusion_collapse_guard_patience = max(1, int(args.fusion_collapse_guard_patience))
args.fusion_collapse_guard_floor = min(0.49, max(0.0, float(args.fusion_collapse_guard_floor)))
args.fusion_collapse_guard_release_epochs = max(1, int(args.fusion_collapse_guard_release_epochs))
args.dynamic_threshold_start = min(0.999, max(1e-6, float(args.dynamic_threshold_start)))
args.dynamic_threshold_end = min(0.999, max(1e-6, float(args.dynamic_threshold_end)))
args.ema_proto_momentum = min(0.999, max(0.0, float(args.ema_proto_momentum)))
args.dcgl_neg_tau = max(1e-6, float(args.dcgl_neg_tau))
args.dcgl_neg_gate_threshold = min(0.999, max(0.0, float(args.dcgl_neg_gate_threshold)))
args.dcgl_neg_gate_power = max(1e-6, float(args.dcgl_neg_gate_power))
args.dcgl_neg_gate_min = min(1.0, max(0.0, float(args.dcgl_neg_gate_min)))
args.dcgl_cluster_tau = max(1e-6, float(args.dcgl_cluster_tau))
args.gcn_dropout = min(0.99, max(0.0, float(args.gcn_dropout)))
### ---------------------------------------
### ---------------------------------------

_resource_start_time = time.perf_counter()
_psutil_mod, _process_monitor = _try_process_monitor()
if str(args.device).startswith("cuda") and torch.cuda.is_available():
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

#load data
### <--- [MODIFIED] ---------------------------------------
if args.graph_mode == 'dual':
    # View-1: original graph A (prefer raw npy/original graph, otherwise KNN fallback)
    adj_a, features, true_labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        graph_mode='raw',
        ae_graph_path=args.ae_graph_path,
        knn_k=args.knn_k,
        raw_graph_path=args.raw_graph_path,
    )
    # View-2: corrected graph AE
    adj_ae, features_ae, true_labels_ae, _, _, _ = load_data(
        args.dataset,
        graph_mode='ae',
        ae_graph_path=args.ae_graph_path,
        knn_k=args.knn_k
    )
    if features.shape != features_ae.shape:
        raise ValueError(f"Feature shape mismatch between A and AE: {features.shape} vs {features_ae.shape}")
    if len(true_labels) != len(true_labels_ae):
        raise ValueError(f"Label length mismatch between A and AE: {len(true_labels)} vs {len(true_labels_ae)}")

    smooth_fea_a = _smooth_with_adj(adj_a, features, args.t, args.device)
    smooth_fea_ae = _smooth_with_adj(adj_ae, features, args.t, args.device)
    # Fused view only for clustering initialization/evaluation.
    smooth_fea = (smooth_fea_a + smooth_fea_ae) / 2
    adaptive_structure_prior = (
        _compute_adaptive_structure_prior(adj_a, adj_ae, features, args)
        if args.enable_adaptive_branch_bias else None
    )
    if adaptive_structure_prior is not None:
        diag = adaptive_structure_prior.get("diag", {})
        print(
            "ADAPTIVE_STRUCTURE_PRIOR | "
            f"target={adaptive_structure_prior.get('target') or 'none'} | "
            f"source={adaptive_structure_prior.get('source', 'none')} | "
            f"raw_degree={diag.get('raw_degree', 0.0):.4f} | "
            f"ae_degree={diag.get('ae_degree', 0.0):.4f} | "
            f"degree_ratio={diag.get('degree_ratio', 0.0):.4f} | "
            f"new_edge_ratio={diag.get('new_edge_ratio', 0.0):.4f} | "
            f"feature_gain={diag.get('feature_gain', 0.0):.4f} | "
            f"feature_density={diag.get('feature_density', 0.0):.4f}"
        )
    ### <--- [MODIFIED] ---------------------------------------
    gcn_adj_a = _build_gcn_adj_torch(adj_a, args.device) if args.enable_gcn_backbone else None
    gcn_adj_ae = _build_gcn_adj_torch(adj_ae, args.device) if args.enable_gcn_backbone else None
    ### ---------------------------------------
else:
    adj, features, true_labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        graph_mode=args.graph_mode,
        ae_graph_path=args.ae_graph_path,
        knn_k=args.knn_k,
        raw_graph_path=args.raw_graph_path,
    )
    smooth_fea = _smooth_with_adj(adj, features, args.t, args.device)
    adaptive_structure_prior = None
    ### <--- [MODIFIED] ---------------------------------------
    gcn_adj_single = _build_gcn_adj_torch(adj, args.device) if args.enable_gcn_backbone else None
    ### ---------------------------------------
### ---------------------------------------

acc_list = []
nmi_list = []
ari_list = []
f1_list = []
best_export = None
best_fusion_export = None
last_fusion_export = None

for run_idx in range(args.runs):

    seed = int(args.seed_start) + run_idx
    setup_seed(seed)
    _reset_dynamic_hc_state()

    # init
    best_acc, best_nmi, best_ari, best_f1, predict_labels, dis= clustering(smooth_fea, true_labels, args.cluster_num)
    ### <--- [MODIFIED] ---------------------------------------
    predict_labels_t = torch.from_numpy(predict_labels).to(args.device)
    dis = dis.to(args.device)
    if args.enable_dynamic_threshold:
        _update_dynamic_hc_state(dis, predict_labels_t)
    ### ---------------------------------------

    # MLP / optional GCN backbone
    ### <--- [MODIFIED] ---------------------------------------
    if args.enable_gcn_backbone:
        model = GCNEncoder_Net(args.linlayers, [features.shape[1]] + [args.dims], dropout=args.gcn_dropout)
    else:
        model = Encoder_Net(args.linlayers, [features.shape[1]] + [args.dims])
    ### ---------------------------------------
    ### <--- [MODIFIED] ---------------------------------------
    if args.graph_mode == 'dual' and args.fusion_mode == 'attn':
        fusion_module = DualViewAttention(
            in_dim=args.dims,
            hidden_dim=args.fusion_hidden,
            temperature=args.fusion_temp,
            min_weight=args.fusion_min_weight,
            enable_branch_bias_fusion=(args.enable_branch_bias_fusion and not args.enable_adaptive_branch_bias),
            branch_bias_target=args.branch_bias_target,
            branch_bias_cap=args.branch_bias_cap,
            enable_learnable_boundary_gate=args.enable_learnable_boundary_gate,
            boundary_gate_max_floor=args.boundary_gate_max_floor,
            boundary_gate_init_floor=args.boundary_gate_init_floor,
            boundary_gate_min_threshold=args.boundary_gate_min_threshold,
            boundary_gate_max_threshold=args.boundary_gate_max_threshold,
            boundary_gate_init_threshold=args.boundary_gate_init_threshold,
            boundary_gate_sharpness=args.boundary_gate_sharpness,
            adaptive_bias_mode=args.adaptive_bias_mode,
            adaptive_boundary_sharpness=args.adaptive_boundary_sharpness,
            adaptive_boundary_tighten=args.adaptive_boundary_tighten,
        ).to(args.device)
        optimizer = optim.Adam(
            list(model.parameters()) + list(fusion_module.parameters()),
            lr=args.lr
        )
    else:
        fusion_module = None
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ### ---------------------------------------

    # GPU
    model.to(args.device)
    sample_size = features.shape[0]
    target = torch.eye(smooth_fea.shape[0]).to(args.device)
    ### <--- [MODIFIED] ---------------------------------------
    # Per-run EMA states. Keep None when improved module is OFF for exact legacy behavior.
    ema_state_single = {} if args.enable_ema_prototypes else None
    ema_state_a = {} if args.enable_ema_prototypes else None
    ema_state_ae = {} if args.enable_ema_prototypes else None
    adaptive_bias_state = {}
    collapse_guard_state = {}
    fusion_trace = []
    fusion_feedback_state = None
    if args.graph_mode == 'dual' and args.enable_adaptive_branch_bias and adaptive_structure_prior is not None:
        prior_target = adaptive_structure_prior.get("target")
        if prior_target in ("raw", "ae"):
            adaptive_bias_state["target"] = prior_target
            adaptive_bias_state["activated_epoch"] = 0
            adaptive_bias_state["source"] = adaptive_structure_prior.get("source", "structure")
    ### ---------------------------------------

    ### <--- [MODIFIED] ---------------------------------------
    pbar = tqdm(range(args.epochs), desc=f"Run {run_idx+1}/{args.runs} seed={seed}")
    for epoch in pbar:
    ### ---------------------------------------
        model.train()
        if fusion_module is not None:
            fusion_module.train()
        ### <--- [MODIFIED] ---------------------------------------
        optimizer.zero_grad(set_to_none=True)
        ### ---------------------------------------
        ### <--- [MODIFIED] ---------------------------------------
        hc_ratio = _resolve_hc_ratio(
            epoch=epoch,
            total_epochs=args.epochs,
            base_ratio=args.threshold,
            use_dynamic=args.enable_dynamic_threshold,
            start_ratio=args.dynamic_threshold_start,
            end_ratio=args.dynamic_threshold_end,
        )
        ### ---------------------------------------
        if args.graph_mode == 'dual':
            # Parallel dual-view forward with shared encoder parameters.
            ### <--- [MODIFIED] ---------------------------------------
            if args.enable_gcn_backbone:
                z1_a, z2_a = model(smooth_fea_a, gcn_adj_a)
                z1_ae, z2_ae = model(smooth_fea_ae, gcn_adj_ae)
            else:
                z1_a, z2_a = model(smooth_fea_a)
                z1_ae, z2_ae = model(smooth_fea_ae)
            ### ---------------------------------------
            hidden_emb_a = (z1_a + z2_a) / 2
            hidden_emb_ae = (z1_ae + z2_ae) / 2
            if fusion_module is not None:
                fusion_module.set_collapse_guard(0.0)
                if args.enable_adaptive_branch_bias:
                    if args.enable_fusion_reliability_feedback_loss:
                        # In feedback mode the structure prior supervises the
                        # fusion weights through loss-level reliability feedback
                        # instead of directly changing the forward attention.
                        fusion_module.set_adaptive_branch_bias(None, 0.0)
                    elif adaptive_bias_state.get("target") in ("raw", "ae") and _is_structure_prior_source(adaptive_bias_state):
                        fusion_module.set_adaptive_branch_bias(
                            adaptive_bias_state.get("target"),
                            _adaptive_branch_bias_cap(adaptive_bias_state, epoch, args)
                        )
                    elif not args.enable_runtime_adaptive_branch_bias:
                        fusion_module.set_adaptive_branch_bias(None, 0.0)
            _, fusion_weights = fuse_dual_views(
                hidden_emb_a,
                hidden_emb_ae,
                fusion_mode=args.fusion_mode,
                fusion_module=fusion_module,
                fixed_raw_weight=args.fixed_raw_weight,
            )
            fusion_mean = torch.mean(fusion_weights, dim=0)
            w_a = fusion_mean[0]
            w_ae = fusion_mean[1]
            high_confidence_idx = _get_high_confidence_idx(dis, hc_ratio)
            raw_reliability = None
            ae_reliability = None
            if fusion_module is not None and args.enable_adaptive_branch_bias:
                if (
                    not _is_structure_prior_source(adaptive_bias_state)
                    and args.enable_runtime_adaptive_branch_bias
                ):
                    raw_reliability = _compute_branch_reliability_score(
                        hidden_emb_a, predict_labels_t, high_confidence_idx, args.cluster_num
                    )
                    ae_reliability = _compute_branch_reliability_score(
                        hidden_emb_ae, predict_labels_t, high_confidence_idx, args.cluster_num
                    )
                    adaptive_target, adaptive_cap = _update_adaptive_branch_bias_state(
                        adaptive_bias_state, raw_reliability, ae_reliability, epoch, args
                    )
                    fusion_module.set_adaptive_branch_bias(adaptive_target, adaptive_cap)
                    if adaptive_target in ("raw", "ae") and adaptive_cap > 0.0:
                        _, fusion_weights = fuse_dual_views(
                            hidden_emb_a,
                            hidden_emb_ae,
                            fusion_mode=args.fusion_mode,
                            fusion_module=fusion_module,
                            fixed_raw_weight=args.fixed_raw_weight,
                        )
                        fusion_mean = torch.mean(fusion_weights, dim=0)
                        w_a = fusion_mean[0]
                        w_ae = fusion_mean[1]
            if (
                fusion_module is not None
                and args.enable_fusion_collapse_guard
                and adaptive_bias_state.get("target") not in ("raw", "ae")
            ):
                guard_floor = _update_fusion_collapse_guard_state(
                    collapse_guard_state, fusion_mean, epoch, args
                )
                if guard_floor > 0.0:
                    fusion_module.set_collapse_guard(guard_floor)
                    _, fusion_weights = fuse_dual_views(
                        hidden_emb_a,
                        hidden_emb_ae,
                        fusion_mode=args.fusion_mode,
                        fusion_module=fusion_module,
                        fixed_raw_weight=args.fixed_raw_weight,
                    )
                    fusion_mean = torch.mean(fusion_weights, dim=0)
                    w_a = fusion_mean[0]
                    w_ae = fusion_mean[1]

            if epoch > args.warmup_epochs:
                loss_a = _ccgc_confidence_loss(
                    z1_a, z2_a, predict_labels_t, high_confidence_idx, args.cluster_num, args.alpha,
                    use_ema_proto=args.enable_ema_prototypes,
                    ema_state=ema_state_a,
                    ema_momentum=args.ema_proto_momentum,
                    use_dcgl_negative=args.enable_dcgl_negative_loss,
                    dcgl_neg_tau=args.dcgl_neg_tau,
                    dcgl_neg_weight=args.dcgl_neg_weight,
                    use_dcgl_reliability_gate=not args.disable_dcgl_neg_reliability_gate,
                    dcgl_neg_gate_threshold=args.dcgl_neg_gate_threshold,
                    dcgl_neg_gate_power=args.dcgl_neg_gate_power,
                    dcgl_neg_gate_min=args.dcgl_neg_gate_min,
                )
                loss_ae = _ccgc_confidence_loss(
                    z1_ae, z2_ae, predict_labels_t, high_confidence_idx, args.cluster_num, args.alpha,
                    use_ema_proto=args.enable_ema_prototypes,
                    ema_state=ema_state_ae,
                    ema_momentum=args.ema_proto_momentum,
                    use_dcgl_negative=args.enable_dcgl_negative_loss,
                    dcgl_neg_tau=args.dcgl_neg_tau,
                    dcgl_neg_weight=args.dcgl_neg_weight,
                    use_dcgl_reliability_gate=not args.disable_dcgl_neg_reliability_gate,
                    dcgl_neg_gate_threshold=args.dcgl_neg_gate_threshold,
                    dcgl_neg_gate_power=args.dcgl_neg_gate_power,
                    dcgl_neg_gate_min=args.dcgl_neg_gate_min,
                )
                if loss_a is None or loss_ae is None:
                    continue

                ramp = min(
                    1.0,
                    float(epoch - args.warmup_epochs + 1) / max(1, args.epochs - args.warmup_epochs)
                )
                inst_align = _instance_align_loss(hidden_emb_a, hidden_emb_ae)
                clu_align = _cluster_distribution_align_loss(
                    hidden_emb_a, hidden_emb_ae, predict_labels_t, args.cluster_num, args.dist_tau, high_confidence_idx
                )
                if args.enable_fusion_reliability_feedback_loss and args.fusion_mode == 'attn':
                    if raw_reliability is None:
                        raw_reliability = _compute_branch_reliability_score(
                            hidden_emb_a, predict_labels_t, high_confidence_idx, args.cluster_num
                        )
                    if ae_reliability is None:
                        ae_reliability = _compute_branch_reliability_score(
                            hidden_emb_ae, predict_labels_t, high_confidence_idx, args.cluster_num
                        )
                if (
                    args.enable_fusion_reliability_feedback_loss
                    and args.fusion_feedback_detach_branch_loss
                    and args.fusion_mode == 'attn'
                ):
                    branch_w_a = w_a.detach()
                    branch_w_ae = w_ae.detach()
                else:
                    branch_w_a = w_a
                    branch_w_ae = w_ae

                loss = branch_w_a * loss_a + branch_w_ae * loss_ae + ramp * (
                    args.lambda_inst * inst_align + args.lambda_clu * clu_align
                )
                if (
                    args.enable_fusion_reliability_feedback_loss
                    and args.fusion_mode == 'attn'
                ):
                    feedback_loss, fusion_feedback_state = _fusion_reliability_feedback_loss(
                        fusion_mean,
                        adaptive_bias_state,
                        raw_reliability,
                        ae_reliability,
                        args,
                        branch_losses=(loss_a, loss_ae),
                    )
                    if feedback_loss is not None and args.fusion_feedback_weight > 0.0:
                        feedback_ramp = min(
                            1.0,
                            float(epoch + 1) / max(1, args.fusion_feedback_warmup_ramp_epochs),
                        )
                        loss = loss + feedback_ramp * args.fusion_feedback_weight * feedback_loss
                ### <--- [MODIFIED] ---------------------------------------
                if args.enable_dcgl_cluster_level:
                    dcgl_clu_loss = _cluster_level_contrastive_loss(
                        hidden_emb_a,
                        hidden_emb_ae,
                        predict_labels_t,
                        args.cluster_num,
                        args.dcgl_cluster_tau,
                        high_confidence_idx
                    )
                    loss = loss + ramp * args.lambda_dcgl_cluster * dcgl_clu_loss
                ### ---------------------------------------
                if args.fusion_mode == 'attn' and args.fusion_balance > 0:
                    balance_target = torch.tensor([0.5, 0.5], device=args.device)
                    balance_loss = F.mse_loss(fusion_mean, balance_target)
                    loss = loss + args.fusion_balance * balance_loss
            else:
                S_a = z1_a @ z2_a.T
                S_ae = z1_ae @ z2_ae.T
                recon_loss_a = F.mse_loss(S_a, target)
                recon_loss_ae = F.mse_loss(S_ae, target)
                if (
                    args.enable_fusion_reliability_feedback_loss
                    and args.fusion_feedback_detach_branch_loss
                    and args.fusion_mode == 'attn'
                ):
                    branch_w_a = w_a.detach()
                    branch_w_ae = w_ae.detach()
                else:
                    branch_w_a = w_a
                    branch_w_ae = w_ae
                loss = branch_w_a * recon_loss_a + branch_w_ae * recon_loss_ae
                if args.enable_fusion_reliability_feedback_loss and args.fusion_mode == 'attn':
                    feedback_loss, fusion_feedback_state = _fusion_reliability_feedback_loss(
                        fusion_mean,
                        adaptive_bias_state,
                        None,
                        None,
                        args,
                        branch_losses=(recon_loss_a, recon_loss_ae),
                    )
                    if feedback_loss is not None and args.fusion_feedback_weight > 0.0:
                        warmup_ramp = min(
                            1.0,
                            float(epoch + 1) / max(1, args.fusion_feedback_warmup_ramp_epochs),
                        )
                        loss = loss + warmup_ramp * args.fusion_feedback_weight * feedback_loss
        else:
            ### <--- [MODIFIED] ---------------------------------------
            if args.enable_gcn_backbone:
                z1, z2 = model(smooth_fea, gcn_adj_single)
            else:
                z1, z2 = model(smooth_fea)
            ### ---------------------------------------
            if epoch > args.warmup_epochs:
                high_confidence_idx = _get_high_confidence_idx(dis, hc_ratio)
                loss_single = _ccgc_confidence_loss(
                    z1, z2, predict_labels_t, high_confidence_idx, args.cluster_num, args.alpha,
                    use_ema_proto=args.enable_ema_prototypes,
                    ema_state=ema_state_single,
                    ema_momentum=args.ema_proto_momentum,
                    use_dcgl_negative=args.enable_dcgl_negative_loss,
                    dcgl_neg_tau=args.dcgl_neg_tau,
                    dcgl_neg_weight=args.dcgl_neg_weight,
                    use_dcgl_reliability_gate=not args.disable_dcgl_neg_reliability_gate,
                    dcgl_neg_gate_threshold=args.dcgl_neg_gate_threshold,
                    dcgl_neg_gate_power=args.dcgl_neg_gate_power,
                    dcgl_neg_gate_min=args.dcgl_neg_gate_min,
                )
                if loss_single is None:
                    continue
                loss = loss_single
            else:
                S = z1 @ z2.T
                loss = F.mse_loss(S, target)

        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            if fusion_module is not None:
                fusion_module.eval()
            if args.graph_mode == 'dual':
                ### <--- [MODIFIED] ---------------------------------------
                if args.enable_gcn_backbone:
                    z1_a, z2_a = model(smooth_fea_a, gcn_adj_a)
                    z1_ae, z2_ae = model(smooth_fea_ae, gcn_adj_ae)
                else:
                    z1_a, z2_a = model(smooth_fea_a)
                    z1_ae, z2_ae = model(smooth_fea_ae)
                ### ---------------------------------------
                hidden_emb_a = (z1_a + z2_a) / 2
                hidden_emb_ae = (z1_ae + z2_ae) / 2
                hidden_emb, fusion_weights_eval = fuse_dual_views(
                    hidden_emb_a,
                    hidden_emb_ae,
                    fusion_mode=args.fusion_mode,
                    fusion_module=fusion_module,
                    fixed_raw_weight=args.fixed_raw_weight,
                )
                fusion_mean_eval = torch.mean(fusion_weights_eval, dim=0)
                fusion_trace.append(
                    (
                        int(epoch),
                        float(fusion_mean_eval[0].detach().cpu().item()),
                        float(fusion_mean_eval[1].detach().cpu().item()),
                    )
                )
            else:
                ### <--- [MODIFIED] ---------------------------------------
                if args.enable_gcn_backbone:
                    z1, z2 = model(smooth_fea, gcn_adj_single)
                else:
                    z1, z2 = model(smooth_fea)
                ### ---------------------------------------
                hidden_emb = (z1 + z2) / 2

            acc, nmi, ari, f1, predict_labels, dis = clustering(hidden_emb, true_labels, args.cluster_num)
            if args.graph_mode == 'dual' and args.save_fusion_weights_path:
                last_fusion_export = {
                    "seed": seed,
                    "epoch": epoch,
                    "fusion_weights": fusion_weights_eval.detach().cpu().numpy(),
                    "fusion_mean": fusion_mean_eval.detach().cpu().numpy(),
                    "hidden_a": hidden_emb_a.detach().cpu().numpy(),
                    "hidden_ae": hidden_emb_ae.detach().cpu().numpy(),
                    "embedding": hidden_emb.detach().cpu().numpy(),
                    "labels": np.asarray(true_labels, dtype=np.int64),
                    "pred": np.asarray(predict_labels, dtype=np.int64),
                    "metrics": np.asarray([acc, nmi, ari, f1], dtype=np.float32) / 100.0,
                    "fusion_trace": np.asarray(fusion_trace, dtype=np.float32),
                    "graph_mode": np.asarray(args.graph_mode),
                    "fusion_mode": np.asarray(args.fusion_mode),
                    "adaptive_bias_target": np.asarray(str(adaptive_bias_state.get("target", "none"))),
                    "adaptive_bias_margin": np.asarray(float(adaptive_bias_state.get("ema_margin", 0.0)), dtype=np.float32),
                }
            ### <--- [MODIFIED] ---------------------------------------
            predict_labels_t = torch.from_numpy(predict_labels).to(args.device)
            dis = dis.to(args.device)
            if args.enable_dynamic_threshold:
                _update_dynamic_hc_state(dis, predict_labels_t)
            ### ---------------------------------------
            if acc >= best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1
                if args.save_embedding_path:
                    best_export = {
                        "seed": seed,
                        "epoch": epoch,
                        "embedding": hidden_emb.detach().cpu().numpy(),
                        "labels": np.asarray(true_labels, dtype=np.int64),
                        "pred": np.asarray(predict_labels, dtype=np.int64),
                        "metrics": np.asarray([acc, nmi, ari, f1], dtype=np.float32) / 100.0,
                    }
                if args.graph_mode == 'dual' and args.save_fusion_weights_path:
                    best_fusion_export = last_fusion_export
            ### <--- [MODIFIED] ---------------------------------------
            if args.graph_mode == 'dual':
                prior_label = str(adaptive_bias_state.get("target", "-")) if args.enable_adaptive_branch_bias else "-"
                prior_key = "prior" if args.enable_fusion_reliability_feedback_loss else "bias"
                pbar.set_postfix({
                    'Best ACC': f'{best_acc:.2f}',
                    'wA': f'{fusion_mean_eval[0].item():.2f}',
                    'wAE': f'{fusion_mean_eval[1].item():.2f}',
                    prior_key: prior_label,
                })
            else:
                pbar.set_postfix({'Best ACC': f'{best_acc:.2f}'})
            ### ---------------------------------------

    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)
    
    ### <--- [MODIFIED] ---------------------------------------
    bias_text = ""
    if args.graph_mode == 'dual' and args.enable_adaptive_branch_bias:
        if args.enable_fusion_reliability_feedback_loss:
            bias_text = f" | Prior: {adaptive_bias_state.get('target', 'none')}"
        else:
            bias_text = (
                f" | Bias: {adaptive_bias_state.get('target', 'none')}"
                f" | BiasMargin: {float(adaptive_bias_state.get('ema_margin', 0.0)):.4f}"
            )
    if args.graph_mode == 'dual' and args.enable_fusion_collapse_guard:
        bias_text += (
            f" | GuardFloor: {float(collapse_guard_state.get('floor', 0.0)):.4f}"
            f" | GuardCount: {int(collapse_guard_state.get('candidate_count', 0))}"
        )
    if (
        args.graph_mode == 'dual'
        and fusion_module is not None
        and args.enable_adaptive_branch_bias
    ):
        adaptive_prior_state = fusion_module.adaptive_prior_state()
        if adaptive_prior_state is not None:
            bias_text += (
                f" | PriorGate: {adaptive_prior_state.get('gate_mean', 0.0):.4f}"
                f" | PriorGateStd: {adaptive_prior_state.get('gate_std', 0.0):.4f}"
            )
    if args.graph_mode == 'dual' and args.enable_fusion_reliability_feedback_loss and fusion_feedback_state is not None:
        bias_text += (
            f" | FB:{fusion_feedback_state.get('mode', '-')}/{fusion_feedback_state.get('source', '-')}"
            f"({fusion_feedback_state.get('target_raw', 0.0):.3f},"
            f"{fusion_feedback_state.get('target_ae', 0.0):.3f})"
            f" | FBLoss:{fusion_feedback_state.get('loss', 0.0):.4f}"
        )
        if "coeff_raw" in fusion_feedback_state and "coeff_ae" in fusion_feedback_state:
            bias_text += (
                f" | FBCoeff:({fusion_feedback_state.get('coeff_raw', 0.0):.2f},"
                f"{fusion_feedback_state.get('coeff_ae', 0.0):.2f})"
                f" | FBWeakW:{fusion_feedback_state.get('weak_w', 0.0):.3f}"
            )
        if "collapse_gate" in fusion_feedback_state:
            bias_text += (
                f" | FBGate:{fusion_feedback_state.get('collapse_gate', 0.0):.3f}"
                f"@{fusion_feedback_state.get('collapse_start', 0.0):.3f}"
            )
        if "adaptive_scale" in fusion_feedback_state:
            bias_text += (
                f" | FBScale:{fusion_feedback_state.get('adaptive_scale', 0.0):.2f}"
                f"/Gap:{fusion_feedback_state.get('loss_gap', 0.0):.2f}"
            )
        if "branch_loss_raw" in fusion_feedback_state and "branch_loss_ae" in fusion_feedback_state:
            bias_text += (
                f" | BLoss:({fusion_feedback_state.get('branch_loss_raw', 0.0):.3f},"
                f"{fusion_feedback_state.get('branch_loss_ae', 0.0):.3f})"
            )
        if "raw_rel" in fusion_feedback_state and "ae_rel" in fusion_feedback_state:
            bias_text += (
                f" | Rel:({fusion_feedback_state.get('raw_rel', 0.0):.3f},"
                f"{fusion_feedback_state.get('ae_rel', 0.0):.3f})"
            )
    if (
        args.graph_mode == 'dual'
        and fusion_module is not None
        and args.enable_learnable_boundary_gate
    ):
        boundary_state = fusion_module.boundary_gate_state()
        if boundary_state is not None:
            boundary_threshold, boundary_floor = boundary_state
            bias_text += f" | BGateThr: {boundary_threshold:.4f} | BGateFloor: {boundary_floor:.4f}"
    tqdm.write(f"Run {run_idx+1:02d} Done | Seed: {seed} | ACC: {best_acc:.2f} | NMI: {best_nmi:.2f} | ARI: {best_ari:.2f} | F1: {best_f1:.2f}{bias_text}")
    if args.graph_mode == 'dual' and fusion_trace:
        trace_arr = np.asarray(fusion_trace, dtype=np.float32)
        dominant = np.max(trace_arr[:, 1:3], axis=1)
        tqdm.write(
            "FusionPath "
            f"| Seed: {seed} "
            f"| first=({trace_arr[0, 1]:.3f},{trace_arr[0, 2]:.3f}) "
            f"| mid=({trace_arr[len(trace_arr)//2, 1]:.3f},{trace_arr[len(trace_arr)//2, 2]:.3f}) "
            f"| last=({trace_arr[-1, 1]:.3f},{trace_arr[-1, 2]:.3f}) "
            f"| mean=({trace_arr[:, 1].mean():.3f},{trace_arr[:, 2].mean():.3f}) "
            f"| max_dom={dominant.max():.3f} "
            f"| end_dom={dominant[-1]:.3f}"
        )
    ### ---------------------------------------

acc_list = np.array(acc_list)
nmi_list = np.array(nmi_list)
ari_list = np.array(ari_list)
f1_list = np.array(f1_list)

### <--- [MODIFIED] ---------------------------------------
print(f"\n{'='*20} FINAL RESULTS {'='*20}")
print(f"Dataset : {args.dataset}")
print(f"Runs    : {args.runs}")
print(f"Seeds   : {int(args.seed_start)}..{int(args.seed_start) + int(args.runs) - 1}")
print(f"{'-'*55}")
print(f"Metric  |    Mean    ±    Std")
print(f"{'-'*55}")
print(f"ACC     |   {acc_list.mean():.2f}    ±   {acc_list.std():.2f}")
print(f"NMI     |   {nmi_list.mean():.2f}    ±   {nmi_list.std():.2f}")
print(f"ARI     |   {ari_list.mean():.2f}    ±   {ari_list.std():.2f}")
print(f"F1      |   {f1_list.mean():.2f}    ±   {f1_list.std():.2f}")
print(f"{'='*55}\n")
### ---------------------------------------

_resource_stats = _resource_summary(_resource_start_time, _psutil_mod, _process_monitor, args.device)
_print_resource_summary(_resource_stats)

if args.save_embedding_path:
    if best_export is None:
        best_export = {
            "seed": -1,
            "epoch": -1,
            "embedding": smooth_fea.detach().cpu().numpy() if isinstance(smooth_fea, torch.Tensor) else np.asarray(smooth_fea),
            "labels": np.asarray(true_labels, dtype=np.int64),
            "pred": np.asarray(predict_labels, dtype=np.int64),
            "metrics": np.asarray([acc_list.max(), nmi_list.max(), ari_list.max(), f1_list.max()], dtype=np.float32) / 100.0,
        }
    save_path = Path(args.save_embedding_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        embedding=best_export["embedding"].astype(np.float32),
        labels=best_export["labels"],
        pred=best_export["pred"],
        metrics=best_export["metrics"],
        method=np.asarray(args.save_embedding_method or ("Ours" if args.graph_mode == "dual" else "CCGC")),
        dataset=np.asarray(args.dataset),
        seed=np.asarray(best_export["seed"], dtype=np.int64),
        epoch=np.asarray(best_export["epoch"], dtype=np.int64),
        graph_mode=np.asarray(args.graph_mode),
        fusion_mode=np.asarray(args.fusion_mode),
    )
    print(f"Saved embedding: {save_path}")

if args.save_fusion_weights_path and args.graph_mode == 'dual':
    if best_fusion_export is None:
        best_fusion_export = last_fusion_export
    if best_fusion_export is None:
        raise RuntimeError("save_fusion_weights_path was requested, but no dual-view fusion state was captured.")
    save_path = Path(args.save_fusion_weights_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        fusion_weights=best_fusion_export["fusion_weights"].astype(np.float32),
        fusion_mean=best_fusion_export["fusion_mean"].astype(np.float32),
        hidden_a=best_fusion_export["hidden_a"].astype(np.float32),
        hidden_ae=best_fusion_export["hidden_ae"].astype(np.float32),
        embedding=best_fusion_export["embedding"].astype(np.float32),
        labels=best_fusion_export["labels"],
        pred=best_fusion_export["pred"],
        metrics=best_fusion_export["metrics"],
        fusion_trace=best_fusion_export.get("fusion_trace", np.empty((0, 3), dtype=np.float32)).astype(np.float32),
        dataset=np.asarray(args.dataset),
        seed=np.asarray(best_fusion_export["seed"], dtype=np.int64),
        epoch=np.asarray(best_fusion_export["epoch"], dtype=np.int64),
        graph_mode=best_fusion_export["graph_mode"],
        fusion_mode=best_fusion_export["fusion_mode"],
        adaptive_bias_target=best_fusion_export.get("adaptive_bias_target", np.asarray("none")),
        adaptive_bias_margin=best_fusion_export.get("adaptive_bias_margin", np.asarray(0.0, dtype=np.float32)),
    )
    print(f"Saved fusion weights: {save_path}")
