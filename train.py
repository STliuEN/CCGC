import argparse
from pathlib import Path
import torch
from utils import *
from tqdm import tqdm
from torch import optim
from model import Encoder_Net, GCNEncoder_Net
import torch.nn.functional as F
from attention_fusion import DualViewAttention, fuse_dual_views


### <--- [MODIFIED] ---------------------------------------
_DYNAMIC_HC_STATE = {}


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
parser.add_argument('--enable_branch_bias_fusion', action='store_true',
                    help='enable optional branch-biased attn fusion; default OFF keeps the current attn behavior unchanged')
parser.add_argument('--branch_bias_target', type=str, default='raw', choices=['raw', 'ae'],
                    help="which branch to anchor when branch-biased fusion is enabled: 'raw' or 'ae'")
parser.add_argument('--branch_bias_cap', type=float, default=0.2,
                    help='maximum correction weight for the non-anchored branch when branch-biased fusion is enabled')
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
args.branch_bias_cap = min(0.49, max(0.0, float(args.branch_bias_cap)))
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

#load data
### <--- [MODIFIED] ---------------------------------------
if args.graph_mode == 'dual':
    # View-1: original graph A (prefer raw npy/original graph, otherwise KNN fallback)
    adj_a, features, true_labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        graph_mode='raw',
        ae_graph_path=args.ae_graph_path,
        knn_k=args.knn_k
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
    ### <--- [MODIFIED] ---------------------------------------
    gcn_adj_a = _build_gcn_adj_torch(adj_a, args.device) if args.enable_gcn_backbone else None
    gcn_adj_ae = _build_gcn_adj_torch(adj_ae, args.device) if args.enable_gcn_backbone else None
    ### ---------------------------------------
else:
    adj, features, true_labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        graph_mode=args.graph_mode,
        ae_graph_path=args.ae_graph_path,
        knn_k=args.knn_k
    )
    smooth_fea = _smooth_with_adj(adj, features, args.t, args.device)
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
            enable_branch_bias_fusion=args.enable_branch_bias_fusion,
            branch_bias_target=args.branch_bias_target,
            branch_bias_cap=args.branch_bias_cap,
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
                loss = w_a * loss_a + w_ae * loss_ae + ramp * (
                    args.lambda_inst * inst_align + args.lambda_clu * clu_align
                )
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
                loss = w_a * F.mse_loss(S_a, target) + w_ae * F.mse_loss(S_ae, target)
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
                    "graph_mode": np.asarray(args.graph_mode),
                    "fusion_mode": np.asarray(args.fusion_mode),
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
                pbar.set_postfix({
                    'Best ACC': f'{best_acc:.2f}',
                    'wA': f'{fusion_mean_eval[0].item():.2f}',
                    'wAE': f'{fusion_mean_eval[1].item():.2f}'
                })
            else:
                pbar.set_postfix({'Best ACC': f'{best_acc:.2f}'})
            ### ---------------------------------------

    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)
    
    ### <--- [MODIFIED] ---------------------------------------
    tqdm.write(f"Run {run_idx+1:02d} Done | Seed: {seed} | ACC: {best_acc:.2f} | NMI: {best_nmi:.2f} | ARI: {best_ari:.2f} | F1: {best_f1:.2f}")
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
        dataset=np.asarray(args.dataset),
        seed=np.asarray(best_fusion_export["seed"], dtype=np.int64),
        epoch=np.asarray(best_fusion_export["epoch"], dtype=np.int64),
        graph_mode=best_fusion_export["graph_mode"],
        fusion_mode=best_fusion_export["fusion_mode"],
    )
    print(f"Saved fusion weights: {save_path}")
