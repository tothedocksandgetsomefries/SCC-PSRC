"""N2: change only the reference population for input U normalization.

Import r4_model_bound and training_primitives from the frozen experiment sources.
No training, data access, role changes, or query-scaler fitting occurs on import.
"""
import numpy as np
import torch
from .r4_model_bound import EPSILON, MatchedMLP, PoolStatistics, masked_moments
from .training_primitives import operation_weights


@torch.no_grad()
def fit_reference_moments(U, valid, data, fit_rows, chunk_size=32):
    """Weighted population moments; equal candidate weights within each pool."""
    rows = np.asarray(fit_rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
        raise ValueError('Expected unique nonempty FIT operation indices')
    if U.ndim != 3 or valid.shape != U.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('Expected U[B,L,M] and boolean valid[B,L]')
    if chunk_size < 1 or rows.min() < 0 or rows.max() >= U.shape[0]:
        raise ValueError('Invalid chunk size or operation index')
    if not np.all(np.asarray(data['role'])[rows] == 0):
        raise ValueError('Reference moments may use FIT role 0 only')
    expected_rows = np.flatnonzero(np.asarray(data['role']) == 0)
    if not np.array_equal(np.sort(rows), expected_rows):
        raise ValueError('Must bind the complete frozen FIT operation set')
    weights = np.asarray(operation_weights(data, rows), dtype=np.float64)
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError('Nonfinite/nonpositive operation weights')
    first = torch.zeros(U.shape[-1], dtype=torch.float64, device=U.device)
    second = torch.zeros_like(first)
    denominator = torch.zeros((), dtype=torch.float64, device=U.device)
    empty, candidates = [], 0
    for off in range(0, len(rows), chunk_size):
        ii = rows[off:off + chunk_size]
        index = torch.as_tensor(ii, dtype=torch.long, device=U.device)
        x = U[index].to(torch.float64)
        mask = valid[index]
        if not bool(torch.isfinite(x[mask]).all()):
            raise ValueError('Nonfinite valid U value')
        counts = mask.sum(1)
        active = counts > 0
        empty.extend(ii[~active.cpu().numpy()].tolist())
        candidates += int(counts.sum())
        x = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))
        n = counts.clamp_min(1).to(torch.float64).unsqueeze(-1)
        w = torch.as_tensor(weights[off:off + len(ii)], dtype=torch.float64, device=U.device)
        w = w * active
        first += ((x.sum(1) / n) * w[:, None]).sum(0)
        second += ((x.square().sum(1) / n) * w[:, None]).sum(0)
        denominator += w.sum()
    if not bool(denominator > 0):
        raise ValueError('No nonempty FIT pools')
    mean = first / denominator
    raw_variance = second / denominator - mean.square()
    std = raw_variance.clamp_min(EPSILON ** 2).sqrt()
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError('Reference moment overflow')
    return {'mean': mean.detach(), 'std': std.detach(), 'metadata': {
        'role': 'FIT_TRAIN', 'operation_count': len(rows), 'valid_candidates': candidates,
        'empty_operation_indices': empty, 'nonempty_weight_sum': float(denominator),
        'all_operation_weight_sum': float(weights.sum()),
        'epsilon': EPSILON, 'accumulation_dtype': 'float64',
        'formula': 'mean=sum(w*pool_mean)/sum(w_nonempty); variance=sum(w*pool_second_moment)/sum(w_nonempty)-mean^2',
        'negative_roundoff_variance_dimensions': int((raw_variance < 0).sum()),
        'empty_pool_policy': 'exclude from moments denominator; preserve training operations unchanged'}}


class ReferenceMatchedMLP(MatchedMLP):
    def __init__(self, M, rank, scaler, reference='POOL_REF', fit_mean=None, fit_std=None):
        if reference not in ('POOL_REF', 'FIT_REF'):
            raise ValueError('Unknown reference arm')
        if rank != 8:
            raise ValueError('Frozen N2 embedding rank is 8')
        super().__init__(M, rank, scaler)
        self.reference = reference
        if reference == 'FIT_REF':
            mean = torch.as_tensor(fit_mean, dtype=scaler.mean.dtype, device=scaler.mean.device).detach().clone()
            std = torch.as_tensor(fit_std, dtype=scaler.std.dtype, device=scaler.std.device).detach().clone()
            if mean.shape != (M,) or std.shape != (M,):
                raise ValueError('FIT moments must be [M]')
            if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std >= EPSILON).all()):
                raise ValueError('Invalid FIT reference moments')
            self.register_buffer('fit_u_mean', mean)
            self.register_buffer('fit_u_std', std)

    def prepare(self, codes, U, query_rms, parent, valid, stats=None):
        if self.reference == 'POOL_REF':
            return super().prepare(codes, U, query_rms, parent, valid, stats)
        # Parent/output scales remain current-pool quantities in both arms.
        if stats is None:
            p_mean, p_std = masked_moments(parent, valid)
        else:
            p_mean, p_std = stats.parent_mean, stats.parent_std
        fixed = PoolStatistics(self.fit_u_mean[None, None, :].expand(U.shape[0], 1, -1),
                               self.fit_u_std[None, None, :].expand(U.shape[0], 1, -1),
                               p_mean, p_std)
        return super().prepare(codes, U, query_rms, parent, valid, fixed)
