import math

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
        enable_learnable_boundary_gate=False,
        boundary_gate_max_floor=0.20,
        boundary_gate_init_floor=0.03,
        boundary_gate_min_threshold=0.75,
        boundary_gate_max_threshold=0.98,
        boundary_gate_init_threshold=0.92,
        boundary_gate_sharpness=30.0,
        adaptive_bias_mode="cap",
        adaptive_boundary_sharpness=24.0,
        adaptive_boundary_tighten=0.0,
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
        self.last_adaptive_prior_state = None
        self.collapse_guard_floor = 0.0
        self.enable_learnable_boundary_gate = bool(enable_learnable_boundary_gate)
        self.boundary_gate_max_floor = min(0.49, max(0.0, float(boundary_gate_max_floor)))
        self.boundary_gate_min_threshold = min(0.999, max(0.5, float(boundary_gate_min_threshold)))
        self.boundary_gate_max_threshold = min(
            0.999,
            max(self.boundary_gate_min_threshold + 1e-6, float(boundary_gate_max_threshold)),
        )
        self.boundary_floor_logit = None
        self.boundary_threshold_logit = None
        self._boundary_gate_init_floor = float(boundary_gate_init_floor)
        self._boundary_gate_init_threshold = float(boundary_gate_init_threshold)
        self.boundary_gate_sharpness = max(1e-6, float(boundary_gate_sharpness))
        if adaptive_bias_mode not in ("boundary", "cap"):
            raise ValueError("adaptive_bias_mode must be either 'boundary' or 'cap'.")
        self.adaptive_bias_mode = adaptive_bias_mode
        self.adaptive_boundary_sharpness = max(1e-6, float(adaptive_boundary_sharpness))
        self.adaptive_boundary_tighten = min(0.95, max(0.0, float(adaptive_boundary_tighten)))
        self.adaptive_prior_gate = None
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
        if self.adaptive_bias_mode == "boundary":
            self.adaptive_prior_gate = nn.Sequential(
                nn.Linear(10, max(8, hidden_dim // 4)),
                nn.LeakyReLU(0.2),
                nn.Linear(max(8, hidden_dim // 4), 1),
            )
            nn.init.zeros_(self.adaptive_prior_gate[-1].weight)
            nn.init.constant_(self.adaptive_prior_gate[-1].bias, -1.10)
        if self.enable_learnable_boundary_gate:
            init_floor = min(self.boundary_gate_max_floor, max(0.0, self._boundary_gate_init_floor))
            init_floor_ratio = init_floor / max(self.boundary_gate_max_floor, 1e-6)
            init_floor_ratio = min(1.0 - 1e-6, max(1e-6, init_floor_ratio))
            self.boundary_floor_logit = nn.Parameter(torch.logit(torch.tensor(init_floor_ratio, dtype=torch.float32)))
            init_threshold = min(
                self.boundary_gate_max_threshold,
                max(self.boundary_gate_min_threshold, self._boundary_gate_init_threshold),
            )
            threshold_span = self.boundary_gate_max_threshold - self.boundary_gate_min_threshold
            init_threshold_ratio = (init_threshold - self.boundary_gate_min_threshold) / max(threshold_span, 1e-6)
            init_threshold_ratio = min(1.0 - 1e-6, max(1e-6, init_threshold_ratio))
            self.boundary_threshold_logit = nn.Parameter(torch.logit(torch.tensor(init_threshold_ratio, dtype=torch.float32)))

    def set_adaptive_branch_bias(self, target=None, cap=0.0):
        if target not in (None, "raw", "ae"):
            raise ValueError("adaptive branch bias target must be None, 'raw', or 'ae'.")
        self.adaptive_bias_target = target
        self.adaptive_bias_cap = min(0.49, max(0.0, float(cap)))

    def set_collapse_guard(self, floor=0.0):
        self.collapse_guard_floor = min(0.49, max(0.0, float(floor)))

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

    def _apply_learnable_boundary_gate(self, weights):
        if (
            not self.enable_learnable_boundary_gate
            or self.boundary_gate_max_floor <= 0.0
            or self.boundary_floor_logit is None
            or self.boundary_threshold_logit is None
        ):
            return weights

        floor = self.boundary_gate_max_floor * torch.sigmoid(self.boundary_floor_logit)
        threshold_span = self.boundary_gate_max_threshold - self.boundary_gate_min_threshold
        threshold = self.boundary_gate_min_threshold + threshold_span * torch.sigmoid(self.boundary_threshold_logit)
        dominant = torch.max(weights, dim=1, keepdim=True).values
        gate = torch.sigmoid((dominant - threshold) * self.boundary_gate_sharpness)
        corrected = floor + (1.0 - 2.0 * floor) * weights
        return (1.0 - gate) * weights + gate * corrected

    def boundary_gate_state(self):
        if (
            not self.enable_learnable_boundary_gate
            or self.boundary_floor_logit is None
            or self.boundary_threshold_logit is None
        ):
            return None
        with torch.no_grad():
            floor = self.boundary_gate_max_floor * torch.sigmoid(self.boundary_floor_logit)
            threshold_span = self.boundary_gate_max_threshold - self.boundary_gate_min_threshold
            threshold = self.boundary_gate_min_threshold + threshold_span * torch.sigmoid(self.boundary_threshold_logit)
        return float(threshold.detach().cpu().item()), float(floor.detach().cpu().item())

    def _adaptive_prior_max_margin(self):
        cap = min(0.49, max(1e-4, float(self.adaptive_bias_cap)))
        # Preserve the old cap scale semantically: cap=0.10 corresponds to a
        # maximum 90/10 prior margin, but the learned gate can reduce it to zero.
        return float(math.log((1.0 - cap) / cap))

    def _apply_adaptive_prior_logits(self, logits, reliability_feat):
        if self.adaptive_bias_target not in ("raw", "ae") or self.adaptive_bias_cap <= 0.0:
            self.last_adaptive_prior_state = None
            return logits
        if self.adaptive_bias_mode == "cap" or self.adaptive_prior_gate is None:
            return logits

        gate = torch.sigmoid(self.adaptive_prior_gate(reliability_feat))
        half_margin = 0.5 * self._adaptive_prior_max_margin() * gate
        if self.adaptive_bias_target == "raw":
            offset = torch.cat([half_margin, -half_margin], dim=1)
        else:
            offset = torch.cat([-half_margin, half_margin], dim=1)
        with torch.no_grad():
            self.last_adaptive_prior_state = {
                "target": self.adaptive_bias_target,
                "gate_mean": float(gate.mean().detach().cpu().item()),
                "gate_std": float(gate.std(unbiased=False).detach().cpu().item()),
                "max_margin": self._adaptive_prior_max_margin(),
            }
        return logits + offset

    def adaptive_prior_state(self):
        return self.last_adaptive_prior_state

    def _apply_adaptive_target_boundary(self, weights):
        if self.adaptive_bias_target not in ("raw", "ae") or self.adaptive_bias_cap <= 0.0:
            return weights
        if self.adaptive_bias_mode != "boundary":
            return weights

        # The structure prior does not directly set the final weights. It only
        # defines the reliable branch used when attention drifts across the
        # opposite-side majority boundary.
        wrong = weights[:, 1:2] if self.adaptive_bias_target == "raw" else weights[:, 0:1]
        gate = torch.sigmoid((wrong - 0.5) * self.adaptive_boundary_sharpness)
        cap = min(0.49, max(0.0, float(self.adaptive_bias_cap)))
        # Once the wrong branch dominates, the correction should be stronger
        # than the trigger boundary itself. The weak branch upper bound shrinks
        # as the boundary evidence grows, while the inactive side remains nearly
        # unchanged when gate is close to zero.
        effective_cap = cap * (1.0 - self.adaptive_boundary_tighten * gate)
        if self.adaptive_bias_target == "raw":
            weak = effective_cap * weights[:, 1:2]
            corrected = torch.cat([1.0 - weak, weak], dim=1)
        else:
            weak = effective_cap * weights[:, 0:1]
            corrected = torch.cat([weak, 1.0 - weak], dim=1)
        with torch.no_grad():
            self.last_adaptive_prior_state = {
                "target": self.adaptive_bias_target,
                "gate_mean": float(gate.mean().detach().cpu().item()),
                "gate_std": float(gate.std(unbiased=False).detach().cpu().item()),
                "max_margin": self._adaptive_prior_max_margin(),
            }
        return (1.0 - gate) * weights + gate * corrected

    def _apply_adaptive_cap_reparam(self, weights):
        if self.adaptive_bias_mode != "cap":
            return None
        if self.adaptive_bias_target not in ("raw", "ae") or self.adaptive_bias_cap <= 0.0:
            return None
        corrected = self._apply_branch_bias_fusion(
            weights,
            self.adaptive_bias_target,
            self.adaptive_bias_cap,
        )
        with torch.no_grad():
            self.last_adaptive_prior_state = {
                "target": self.adaptive_bias_target,
                "gate_mean": 1.0,
                "gate_std": 0.0,
                "max_margin": self._adaptive_prior_max_margin(),
            }
        return corrected

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
        logits = self._apply_adaptive_prior_logits(logits, reliability_feat)
        weights = F.softmax(logits / self.temperature, dim=1)
        adaptive_cap_weights = self._apply_adaptive_cap_reparam(weights)
        if adaptive_cap_weights is not None:
            return adaptive_cap_weights
        weights = self._apply_adaptive_target_boundary(weights)
        if self.enable_branch_bias_fusion:
            return self._apply_branch_bias_fusion(weights, self.branch_bias_target, self.branch_bias_cap)
        ### <--- [MODIFIED] ---------------------------------------
        if self.collapse_guard_floor > 0:
            floor = self.collapse_guard_floor
            weights = floor + (1.0 - 2.0 * floor) * weights
            return weights
        if self.min_weight > 0:
            weights = self.min_weight + (1.0 - 2.0 * self.min_weight) * weights
            return weights
        weights = self._apply_learnable_boundary_gate(weights)
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
