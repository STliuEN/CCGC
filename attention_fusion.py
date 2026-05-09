import torch
import torch.nn as nn
import torch.nn.functional as F


### <--- [MODIFIED] ---------------------------------------
class DualViewAttention(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        temperature=1.0,
        min_weight=0.0,
        enable_branch_bias_fusion=False,
        branch_bias_target="raw",
        branch_bias_cap=0.2,
    ):
        super(DualViewAttention, self).__init__()
        self.temperature = max(1e-6, float(temperature))
        ### <--- [MODIFIED] ---------------------------------------
        # Per-branch minimum weight to avoid one branch being fully abandoned.
        self.min_weight = min(0.49, max(0.0, float(min_weight)))
        # Optional branch-biased fusion:
        # default OFF to preserve the exact current attention behavior.
        self.enable_branch_bias_fusion = bool(enable_branch_bias_fusion)
        if branch_bias_target not in ("raw", "ae"):
            raise ValueError("branch_bias_target must be either 'raw' or 'ae'.")
        self.branch_bias_target = branch_bias_target
        self.branch_bias_cap = min(0.49, max(0.0, float(branch_bias_cap)))
        self.adaptive_bias_target = None
        self.adaptive_bias_cap = 0.0
        ### ---------------------------------------
        # Reliability-aware attention:
        # enrich the logit input with directional discrepancy, shared content,
        # and lightweight branch-quality proxies, while keeping the external
        # behavior identical to the original attention interface.
        feature_dim = in_dim * 6 + 10
        self.attn_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def set_adaptive_branch_bias(self, target=None, cap=0.0):
        if target not in (None, "raw", "ae"):
            raise ValueError("adaptive branch bias target must be None, 'raw', or 'ae'.")
        self.adaptive_bias_target = target
        self.adaptive_bias_cap = min(0.49, max(0.0, float(cap)))

    def _apply_branch_bias_fusion(self, weights, target, cap):
        # Anchor one branch and only let the other branch act as a bounded
        # correction. This keeps the output as a convex two-view fusion so the
        # rest of the training logic can stay unchanged.
        cap = min(0.49, max(0.0, float(cap)))
        if cap <= 0.0 or target is None:
            return weights
        aux_floor = min(self.min_weight, cap)
        if target == "raw":
            beta = aux_floor + (cap - aux_floor) * weights[:, 1:2]
            return torch.cat([1.0 - beta, beta], dim=1)

        beta = aux_floor + (cap - aux_floor) * weights[:, 0:1]
        return torch.cat([beta, 1.0 - beta], dim=1)

    def forward(self, hidden_a, hidden_ae):
        raw_hidden_a = hidden_a
        raw_hidden_ae = hidden_ae
        hidden_a = F.normalize(raw_hidden_a, dim=1, p=2)
        hidden_ae = F.normalize(raw_hidden_ae, dim=1, p=2)
        signed_diff = hidden_a - hidden_ae
        abs_diff = torch.abs(signed_diff)
        shared_mean = 0.5 * (hidden_a + hidden_ae)

        # Agreement: high cross-view agreement means the two views can be fused
        # more freely; low agreement should make the module more conservative.
        agreement = torch.sum(hidden_a * hidden_ae, dim=1, keepdim=True)

        # Relative energy captures which branch currently carries a stronger
        # response for the same sample.
        energy_a = torch.norm(raw_hidden_a, p=2, dim=1, keepdim=True)
        energy_ae = torch.norm(raw_hidden_ae, p=2, dim=1, keepdim=True)
        energy_sum = torch.clamp(energy_a + energy_ae, min=1e-6)
        energy_gap = (energy_a - energy_ae) / energy_sum
        energy_ratio_a = energy_a / energy_sum
        energy_ratio_ae = energy_ae / energy_sum

        # Residual magnitude measures how much the two views disagree locally.
        residual = raw_hidden_a - raw_hidden_ae
        residual_norm = torch.norm(residual, p=2, dim=1, keepdim=True)
        norm_residual = residual_norm / energy_sum

        # Batch-level anchors provide a lightweight branch-quality proxy:
        # a sample that is more aligned with its own branch consensus is often
        # a safer candidate to trust during view fusion.
        with torch.no_grad():
            batch_center_a = F.normalize(torch.mean(hidden_a, dim=0, keepdim=True), dim=1, p=2)
            batch_center_ae = F.normalize(torch.mean(hidden_ae, dim=0, keepdim=True), dim=1, p=2)

        center_consistency_a = torch.sum(hidden_a * batch_center_a, dim=1, keepdim=True)
        center_consistency_ae = torch.sum(hidden_ae * batch_center_ae, dim=1, keepdim=True)
        cross_center_a = torch.sum(hidden_a * batch_center_ae, dim=1, keepdim=True)
        cross_center_ae = torch.sum(hidden_ae * batch_center_a, dim=1, keepdim=True)
        center_margin_a = center_consistency_a - cross_center_a
        center_margin_ae = center_consistency_ae - cross_center_ae
        center_margin_gap = center_margin_a - center_margin_ae

        reliability_feat = torch.cat(
            [
                agreement,
                energy_gap,
                norm_residual,
                torch.abs(energy_gap),
                energy_ratio_a,
                energy_ratio_ae,
                center_margin_a,
                center_margin_ae,
                center_margin_gap,
                center_consistency_a - center_consistency_ae,
            ],
            dim=1,
        )

        logits = self.attn_mlp(
            torch.cat(
                [
                    hidden_a,
                    hidden_ae,
                    signed_diff,
                    abs_diff,
                    hidden_a * hidden_ae,
                    shared_mean,
                    reliability_feat,
                ],
                dim=1,
            )
        )
        weights = F.softmax(logits / self.temperature, dim=1)
        if self.adaptive_bias_target is not None and self.adaptive_bias_cap > 0.0:
            return self._apply_branch_bias_fusion(weights, self.adaptive_bias_target, self.adaptive_bias_cap)
        if self.enable_branch_bias_fusion:
            return self._apply_branch_bias_fusion(weights, self.branch_bias_target, self.branch_bias_cap)
        ### <--- [MODIFIED] ---------------------------------------
        if self.min_weight > 0:
            weights = self.min_weight + (1.0 - 2.0 * self.min_weight) * weights
        ### ---------------------------------------
        return weights


def fuse_dual_views(hidden_a, hidden_ae, fusion_mode="mean", fusion_module=None, fixed_raw_weight=0.5):
    if fusion_mode == "attn":
        if fusion_module is None:
            raise ValueError("fusion_module must be provided when fusion_mode='attn'.")
        weights = fusion_module(hidden_a, hidden_ae)
    elif fusion_mode == "fixed":
        raw_weight = min(1.0, max(0.0, float(fixed_raw_weight)))
        weights = torch.empty(
            (hidden_a.shape[0], 2),
            dtype=hidden_a.dtype,
            device=hidden_a.device
        )
        weights[:, 0] = raw_weight
        weights[:, 1] = 1.0 - raw_weight
    else:
        weights = torch.full(
            (hidden_a.shape[0], 2),
            0.5,
            dtype=hidden_a.dtype,
            device=hidden_a.device
        )

    w_a = weights[:, 0:1]
    w_ae = weights[:, 1:2]
    fused = w_a * hidden_a + w_ae * hidden_ae
    return fused, weights
### ---------------------------------------
