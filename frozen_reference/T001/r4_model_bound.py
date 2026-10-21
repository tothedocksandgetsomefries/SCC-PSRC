"""R4_CSI torch scoring primitives for the preregistered 2026-09-07 recipe.

This module consumes only the registered online tensors. It performs no file I/O,
retrieval, optimizer construction, training, or model selection. Synthetic
qualification of these formulas is not evidence of ranking gains.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


EPSILON = 1e-6
GAMMA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _require_finite(values: Tensor, label: str) -> None:
    if not bool(torch.isfinite(values).all()):
        raise ValueError("non-finite valid " + label)


def _validate_mask(values: Tensor, valid: Tensor) -> None:
    if values.ndim not in (2, 3) or valid.shape != values.shape[:2]:
        raise ValueError("expected values[B,L] or [B,L,M] and valid[B,L]")
    if valid.dtype != torch.bool or valid.device != values.device:
        raise ValueError("valid must be bool and on the values device")
    if not values.is_floating_point():
        raise ValueError("values must be floating point")


def masked_moments(values: Tensor, valid: Tensor, epsilon: float = EPSILON):
    """Population moments over L; padding NaN/Inf does not enter reductions.

    Empty operations have mean=0 and std=epsilon. The candidate dimension is
    retained so the outputs broadcast to [B,L] or [B,L,M].
    """
    _validate_mask(values, valid)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    _require_finite(values[valid], "values")
    mask = valid if values.ndim == 2 else valid.unsqueeze(-1)
    safe = torch.where(mask, values, torch.zeros_like(values))
    count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
    if values.ndim == 3:
        count = count.unsqueeze(-1)
    mean = safe.sum(dim=1, keepdim=True) / count
    _require_finite(mean, "pool mean (reduction overflow)")
    centered = torch.where(mask, safe - mean, torch.zeros_like(safe))
    variance = centered.square().sum(dim=1, keepdim=True) / count
    # Clamp variance before sqrt: sqrt(0) otherwise produces undefined gradients.
    std = variance.clamp_min(epsilon * epsilon).sqrt()
    _require_finite(std, "pool std (reduction overflow)")
    return mean, std


def masked_standardize(values: Tensor, valid: Tensor, epsilon: float = EPSILON):
    mean, std = masked_moments(values, valid, epsilon)
    mask = valid if values.ndim == 2 else valid.unsqueeze(-1)
    safe = torch.where(mask, values, torch.zeros_like(values))
    standardized = torch.where(mask, (safe - mean) / std, torch.zeros_like(safe))
    _require_finite(standardized, "standardized values")
    return standardized, mean, std


def center_valid(values: Tensor, valid: Tensor) -> Tensor:
    mean, _ = masked_moments(values, valid)
    safe = torch.where(valid, values, torch.zeros_like(values))
    return torch.where(valid, safe - mean, torch.zeros_like(safe))


class QueryRMSScaler(nn.Module):
    """Immutable-at-forward FIT-only population mean/std per subspace.

    The caller must bind the FIT rows, parent, roles and scaler hash in its run
    manifest. This type cannot establish data provenance from a tensor alone.
    Repeated qids, if supplied, contribute repeated rows; the caller controls the
    preregistered FIT query sampling and must record it.
    """

    def __init__(self, mean: Tensor, std: Tensor):
        super().__init__()
        if mean.ndim != 1 or std.shape != mean.shape or mean.numel() < 2:
            raise ValueError("scaler mean/std must have shape [M>=2]")
        if not mean.is_floating_point() or std.dtype != mean.dtype:
            raise ValueError("scaler mean/std must share a floating dtype")
        if std.device != mean.device:
            raise ValueError("scaler mean/std devices differ")
        _require_finite(mean, "scaler mean")
        _require_finite(std, "scaler std")
        if bool((std < EPSILON).any()):
            raise ValueError("scaler std must be >=1e-6")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())

    @classmethod
    def fit(cls, query_rms: Tensor, *, role: str) -> "QueryRMSScaler":
        if role != "FIT_TRAIN":
            raise ValueError("query RMS scaler may only fit FIT_TRAIN")
        if query_rms.ndim != 2 or query_rms.shape[0] < 1:
            raise ValueError("FIT query RMS must be nonempty [Q,M]")
        if not query_rms.is_floating_point():
            raise ValueError("FIT query RMS must be floating point")
        _require_finite(query_rms, "FIT query RMS")
        with torch.no_grad():
            mean = query_rms.mean(dim=0)
            std = (query_rms - mean).square().mean(dim=0).clamp_min(EPSILON**2).sqrt()
        return cls(mean, std)

    def forward(self, query_rms: Tensor) -> Tensor:
        if query_rms.ndim != 2 or query_rms.shape[1:] != self.mean.shape:
            raise ValueError("query RMS shape is incompatible with FIT scaler")
        if query_rms.dtype != self.mean.dtype or query_rms.device != self.mean.device:
            raise ValueError("query RMS and FIT scaler dtype/device differ")
        _require_finite(query_rms, "query RMS")
        scaled = (query_rms - self.mean) / self.std
        _require_finite(scaled, "scaled query RMS")
        return scaled


@dataclass(frozen=True)
class PoolStatistics:
    """Request/pool statistics, usable as fixed constants in algebra checks."""

    u_mean: Tensor
    u_std: Tensor
    parent_mean: Tensor
    parent_std: Tensor


@dataclass(frozen=True)
class PreparedInputs:
    codes: Tensor
    u_scaled: Tensor
    query_rms_scaled: Tensor
    valid: Tensor
    stats: PoolStatistics


def _validate_inputs(codes: Tensor, U: Tensor, query_rms: Tensor,
                     parent: Tensor, valid: Tensor, M: int) -> None:
    if codes.ndim != 3 or codes.shape[-1] != M or U.shape != codes.shape:
        raise ValueError("expected codes/U[B,L,M] with bound M")
    if parent.shape != codes.shape[:2] or valid.shape != parent.shape:
        raise ValueError("parent/valid shape mismatch")
    if query_rms.shape != (codes.shape[0], M):
        raise ValueError("query RMS must be [B,M]")
    if codes.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("codes must be integer 8-bit PQ identities")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be bool")
    tensors = (codes, U, query_rms, valid)
    if any(t.device != parent.device for t in tensors):
        raise ValueError("all inputs must share a device")
    if not parent.is_floating_point() or U.dtype != parent.dtype or query_rms.dtype != parent.dtype:
        raise ValueError("parent/U/query RMS must share a floating dtype")
    _require_finite(parent[valid], "parent")
    _require_finite(U[valid], "U")
    _require_finite(query_rms[valid.any(dim=1)], "query RMS")
    valid_codes = codes[valid]
    if bool(((valid_codes < 0) | (valid_codes > 255)).any()):
        raise ValueError("valid code identities must be in [0,255]")


def _validate_stats(stats: PoolStatistics, U: Tensor, parent: Tensor) -> None:
    shapes = ((U.shape[0], 1, U.shape[2]),) * 2 + ((parent.shape[0], 1),) * 2
    for label, value, shape in zip(
            ("u_mean", "u_std", "parent_mean", "parent_std"),
            (stats.u_mean, stats.u_std, stats.parent_mean, stats.parent_std), shapes):
        if value.shape != shape or value.dtype != U.dtype or value.device != U.device:
            raise ValueError("fixed pool statistic shape/dtype/device mismatch: " + label)
        _require_finite(value, label)
        if label.endswith("std") and bool((value < EPSILON).any()):
            raise ValueError("fixed pool std must be >=1e-6")


def pair_fast(h: Tensor, weights: Tensor) -> Tensor:
    """O(B*L*M*r) low-rank pair expression; no pair tables are materialized."""
    if h.ndim not in (3, 4) or h.shape[-2] < 2 or weights.shape != h.shape[-1:]:
        raise ValueError("expected h[...,M>=2,r] and weights[r]")
    M = h.shape[-2]
    terms = h.sum(dim=-2).square() - h.square().sum(dim=-2)
    return (0.5 * terms * weights).sum(dim=-1) / math.sqrt(M * (M - 1) / 2)


def pair_explicit(h: Tensor, weights: Tensor) -> Tensor:
    """Slow qualification oracle; never selected by the production forward."""
    if h.ndim not in (3, 4) or h.shape[-2] < 2 or weights.shape != h.shape[-1:]:
        raise ValueError("expected h[...,M>=2,r] and weights[r]")
    value = torch.zeros_like(h[..., 0, 0])
    M = h.shape[-2]
    for m in range(M):
        for n in range(m + 1, M):
            value = value + (h[..., m, :] * h[..., n, :] * weights).sum(dim=-1)
    return value / math.sqrt(M * (M - 1) / 2)


def raw_skip(parent: Tensor, residual: Tensor, parent_std: Tensor,
             valid: Tensor, gamma: float) -> Tensor:
    """New-model raw-parent contract; gamma0 returns the identical tensor.

    This is intentionally not a replacement for legacy CSRC's Z(A)+0 semantics.
    Padded parent values are preserved and must remain excluded by valid.
    """
    if gamma not in GAMMA_GRID:
        raise ValueError("gamma is outside the preregistered grid")
    if parent.shape != residual.shape or parent.shape != valid.shape:
        raise ValueError("raw skip tensor shape mismatch")
    if parent_std.shape != (parent.shape[0], 1):
        raise ValueError("parent std must be [B,1]")
    if gamma == 0:
        return parent
    return torch.where(valid, parent + gamma * parent_std * residual, parent)


class _R4Base(nn.Module):
    def __init__(self, M: int, rank: int, scaler: QueryRMSScaler):
        super().__init__()
        if M < 2 or rank not in (4, 8):
            raise ValueError("bound M>=2 and registered rank4/rank8 are required")
        if scaler.mean.numel() != M:
            raise ValueError("scaler subspaces differ from model")
        self.M, self.rank = int(M), int(rank)
        self.scaler = scaler

    def prepare(self, codes: Tensor, U: Tensor, query_rms: Tensor,
                parent: Tensor, valid: Tensor,
                stats: Optional[PoolStatistics] = None) -> PreparedInputs:
        _validate_inputs(codes, U, query_rms, parent, valid, self.M)
        if parent.dtype != self.scaler.mean.dtype or parent.device != self.scaler.mean.device:
            raise ValueError("inputs and model/scaler dtype/device differ")
        if stats is None:
            u_mean, u_std = masked_moments(U, valid)
            p_mean, p_std = masked_moments(parent, valid)
            stats = PoolStatistics(u_mean, u_std, p_mean, p_std)
        else:
            _validate_stats(stats, U, parent)
        safe_U = torch.where(valid.unsqueeze(-1), U, torch.zeros_like(U))
        u_scaled = torch.where(valid.unsqueeze(-1),
                               (safe_U - stats.u_mean) / stats.u_std,
                               torch.zeros_like(U))
        _require_finite(u_scaled, "scaled U")
        safe_codes = torch.where(valid.unsqueeze(-1), codes, torch.zeros_like(codes)).long()
        active = valid.any(dim=1, keepdim=True)
        safe_q = torch.where(active, query_rms, self.scaler.mean.unsqueeze(0))
        q_scaled = self.scaler(safe_q)
        return PreparedInputs(safe_codes, u_scaled, q_scaled, valid, stats)

    def raw_residual(self, prepared: PreparedInputs) -> Tensor:
        raise NotImplementedError

    def residual(self, codes: Tensor, U: Tensor, query_rms: Tensor,
                 parent: Tensor, valid: Tensor,
                 stats: Optional[PoolStatistics] = None) -> Tensor:
        prepared = self.prepare(codes, U, query_rms, parent, valid, stats)
        return center_valid(self.raw_residual(prepared), valid)

    def forward(self, codes: Tensor, U: Tensor, query_rms: Tensor,
                parent: Tensor, valid: Tensor, gamma: float = 1.0,
                stats: Optional[PoolStatistics] = None) -> Tensor:
        if gamma not in GAMMA_GRID:
            raise ValueError("gamma is outside the preregistered grid")
        if gamma == 0:
            # Validate the registered input boundary without evaluating features.
            _validate_inputs(codes, U, query_rms, parent, valid, self.M)
            if parent.dtype != self.scaler.mean.dtype or parent.device != self.scaler.mean.device:
                raise ValueError("inputs and model/scaler dtype/device differ")
            return parent
        prepared = self.prepare(codes, U, query_rms, parent, valid, stats)
        prediction = center_valid(self.raw_residual(prepared), valid)
        return raw_skip(parent, prediction, prepared.stats.parent_std, valid, gamma)

    def parameter_counts(self) -> Dict[str, int]:
        return {"trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "scaler_buffer_values": sum(b.numel() for b in self.scaler.buffers())}


class R4CSI(_R4Base):
    def __init__(self, M: int, rank: int, scaler: QueryRMSScaler, arm: str = "FULL"):
        super().__init__(M, rank, scaler)
        if arm not in ("FULL", "ADD_ONLY"):
            raise ValueError("only FULL and ADD_ONLY are registered CSI arms")
        self.arm = arm
        options = {"dtype": scaler.mean.dtype, "device": scaler.mean.device}
        self.E = nn.Parameter(torch.empty(M, 256, rank, **options))
        self.w_u = nn.Parameter(torch.empty(rank, **options))
        self.w_q = nn.Parameter(torch.empty(rank, **options))
        self.b = nn.Parameter(torch.empty(M, rank, **options))
        self.w_add = nn.Parameter(torch.zeros(M, rank, **options))
        if arm == "FULL":
            self.w_pair = nn.Parameter(torch.zeros(rank, **options))
        else:
            self.register_parameter("w_pair", None)
        # Small nonzero local initialization is fixed before any fit; all output
        # heads are zero. No parameter matching through unused tensors is used.
        nn.init.normal_(self.E, mean=0.0, std=0.02)
        nn.init.normal_(self.w_u, mean=0.0, std=0.02)
        nn.init.normal_(self.w_q, mean=0.0, std=0.02)
        nn.init.normal_(self.b, mean=0.0, std=0.02)

    def local_atoms(self, prepared: PreparedInputs) -> Tensor:
        subspaces = torch.arange(self.M, device=prepared.codes.device)
        embeddings = self.E[subspaces, prepared.codes]
        return torch.tanh(embeddings
                          + prepared.u_scaled.unsqueeze(-1) * self.w_u
                          + prepared.query_rms_scaled[:, None, :, None] * self.w_q
                          + self.b)

    def raw_residual(self, prepared: PreparedInputs) -> Tensor:
        h = self.local_atoms(prepared)
        additive = (h * self.w_add).sum(dim=(-2, -1)) / math.sqrt(self.M)
        return additive if self.w_pair is None else additive + pair_fast(h, self.w_pair)


class MatchedMLP(_R4Base):
    """Same online information; per-subspace [E(code), U_z, qRMS_z] -> 32/32.

    The model is a new control, not a reproduction of Tsmall or a published
    decoder. Raw/standardized parent totals are excluded from this residual head.
    """
    def __init__(self, M: int, rank: int, scaler: QueryRMSScaler):
        super().__init__(M, rank, scaler)
        options = {"dtype": scaler.mean.dtype, "device": scaler.mean.device}
        self.E = nn.Parameter(torch.empty(M, 256, rank, **options))
        nn.init.normal_(self.E, mean=0.0, std=0.02)
        self.network = nn.Sequential(
            nn.Linear(M * (rank + 2), 32, **options), nn.ReLU(),
            nn.Linear(32, 32, **options), nn.ReLU(),
            nn.Linear(32, 1, **options))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def local_features(self, prepared: PreparedInputs) -> Tensor:
        subspaces = torch.arange(self.M, device=prepared.codes.device)
        embedding = self.E[subspaces, prepared.codes]
        q = prepared.query_rms_scaled[:, None, :, None].expand(
            *prepared.codes.shape, 1)
        fields = torch.cat((embedding, prepared.u_scaled.unsqueeze(-1), q), dim=-1)
        return fields.flatten(start_dim=-2)

    def raw_residual(self, prepared: PreparedInputs) -> Tensor:
        return self.network(self.local_features(prepared)).squeeze(-1)


def residual_target(exact: Tensor, parent: Tensor, valid: Tensor) -> Tensor:
    """Offline supervised target. Never call this in an online scorer worker."""
    if exact.shape != parent.shape:
        raise ValueError("exact/parent shape mismatch")
    exact_z, _, _ = masked_standardize(exact, valid)
    parent_z, _, _ = masked_standardize(parent, valid)
    return exact_z - parent_z


def operation_smooth_l1(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """One SmoothL1(beta1) mean per operation; zero-candidate loss is zero.

    The outer training scheduler must apply the frozen qid/predicate equal
    weighting; candidate counts must not become operation weights.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction/target shape mismatch")
    _validate_mask(prediction, valid)
    _require_finite(prediction[valid], "prediction")
    _require_finite(target[valid], "target")
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    losses = F.smooth_l1_loss(safe_prediction, safe_target, reduction="none", beta=1.0)
    return losses.sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(losses.dtype)
