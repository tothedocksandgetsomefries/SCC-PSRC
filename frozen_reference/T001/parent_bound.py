from __future__ import annotations
import numpy as np, torch
from torch import nn
from types import SimpleNamespace
KSUB=256
BANK_ROWS=4096
RANK=8
class CanonicalA(nn.Module):
    """Dataset-agnostic frozen hash-bank/rank-8/bound action family."""

    def __init__(self, centroids: np.ndarray, metric: str, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        centroids = np.asarray(centroids, dtype=np.float32)
        if centroids.ndim != 3 or centroids.shape[0] < 2 or centroids.shape[1] != KSUB:
            raise ValueError("centroid shape drift")
        if metric not in {"L2_ADC", "DOT"}:
            raise ValueError(metric)
        self.metric = metric
        self.m = int(centroids.shape[0])
        self.dsub = int(centroids.shape[2])
        self.register_buffer("centroids", torch.from_numpy(centroids.copy()))
        self.bank = nn.Parameter(torch.zeros(BANK_ROWS, self.m, RANK))
        self.W = nn.Parameter(torch.empty(self.m, RANK, KSUB, self.dsub))
        nn.init.xavier_uniform_(self.W)

    def _lut_from(self, query: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        q = query.reshape(-1, self.m, self.dsub)
        if self.metric == "L2_ADC":
            return torch.sum(2.0 * q[:, :, None, :] * codebook - codebook.square(), dim=-1)
        return torch.sum(q[:, :, None, :] * codebook, dim=-1)

    def lut(self, query: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = torch.tanh(self.bank[keys].sum(dim=1))
        raw = torch.einsum("bmr,mrkd->bmkd", alpha, self.W)
        rms = torch.sqrt(torch.mean(self.centroids.square(), dim=(1, 2))).clamp_min(1e-12)
        bound = (0.10 * rms)[None, :, None, None]
        delta = bound * torch.tanh(raw / bound)
        codebook = self.centroids[None] + delta
        return self._lut_from(query, codebook), delta, raw

    def base_lut(self, query: torch.Tensor) -> torch.Tensor:
        return self._lut_from(query, self.centroids[None].expand(len(query), -1, -1, -1))

    @staticmethod
    def gather(lut: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        if lut.shape[1] != codes.shape[2]:
            raise ValueError("LUT/code subquantizer drift")
        parts = torch.stack([lut[:, m].gather(1, codes[:, :, m].long()) for m in range(codes.shape[2])], dim=2)
        return parts.sum(dim=2)

    def score(self, query: torch.Tensor, keys: torch.Tensor, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lut, delta, raw = self.lut(query, keys)
        if codes.shape[2] != self.m:
            raise ValueError("score/code subquantizer drift")
        parts = torch.stack([lut[:, m].gather(1, codes[:, :, m].long()) for m in range(self.m)], dim=2)
        rms = torch.sqrt(torch.mean(self.centroids.square(), dim=(1, 2))).clamp_min(1e-12)
        bound = (0.10 * rms)[None, :, None, None]
        saturation = torch.mean((torch.abs(raw / bound) >= 2.0).float())
        return parts.sum(dim=2), parts, delta.square().mean(), saturation

    def base_score(self, query: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        return self.gather(self.base_lut(query), codes)

def explicit_reduce_torch(parts: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(parts.shape[:2], dtype=parts.dtype, device=parts.device)
    for subspace in range(parts.shape[2]):
        result = result + parts[:, :, subspace]
    return result
frozen=SimpleNamespace(ct=SimpleNamespace(CanonicalA=CanonicalA),explicit_reduce_torch=explicit_reduce_torch)

class Parent(frozen.ct.CanonicalA):
    """Same A state contract; global and soft modes are attribution only."""

    def __init__(self, kind, centroids, seed, e0=False):
        if kind not in ('A_HASH', 'A_GLOBAL', 'A_SOFT_ANCHOR_ONLY'):
            raise ValueError('Unregistered A kind: ' + str(kind))
        super().__init__(centroids, 'L2_ADC', seed)
        self.kind = kind
        self.e0 = bool(e0)
        if kind == 'A_GLOBAL':
            del self.bank
            self.g = nn.Parameter(torch.zeros(self.m, 8))
        # Derived buffers are excluded from checkpoints, preserving legacy keys.
        self.register_buffer('_r4_bound', self._bound(), persistent=False)

    def _bound(self):
        rms = torch.sqrt(torch.mean(self.centroids.square(), dim=(1, 2))).clamp_min(1e-12)
        return (0.10 * rms)[None, :, None, None]

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self._r4_bound = self._bound()
        return result

    def _alpha(self, query, keys):
        if self.kind == 'A_GLOBAL':
            return torch.tanh(self.g)[None].expand(len(query), -1, -1)
        summed = self.bank[keys].sum(dim=1)
        return summed if self.kind == 'A_SOFT_ANCHOR_ONLY' else torch.tanh(summed)

    def lut(self, query, keys):
        # Global-A has one query-independent codebook. Compute its contraction
        # at one fixed batch shape; query scoring below remains independent.
        global_shared = self.kind == 'A_GLOBAL'
        alpha = self._alpha(query[:1] if global_shared else query, keys)
        raw = torch.einsum('bmr,mrkd->bmkd', alpha, self.W)
        if self.kind == 'A_SOFT_ANCHOR_ONLY':
            delta = raw
        else:
            bound = self._r4_bound if self.e0 else self._bound()
            delta = bound * torch.tanh(raw / bound)
        lut = self._lut_from(query, self.centroids[None] + delta)
        if global_shared:
            # Retain the original per-query return shapes, without score reuse.
            delta = delta.expand(len(query), -1, -1, -1)
            raw = raw.expand(len(query), -1, -1, -1)
        return lut, delta, raw

    def score_fast(self, query, keys, codes):
        """Identical score reduction without unused action summaries."""
        if codes.shape[2] != self.m:
            raise ValueError('score/code subquantizer drift')
        lut, _, _ = self.lut(query, keys)
        parts = torch.stack([lut[:, m].gather(1, codes[:, :, m].long())
                             for m in range(self.m)], dim=2)
        # The imported reference replaces CanonicalA.score with an explicit
        # left-to-right FP32 reduction. A torch.sum here is not E0-equivalent.
        return frozen.explicit_reduce_torch(parts), parts

    def action(self, query, keys):
        """Separate action tensors; caller controls detaching and aggregation.

        For SOFT, ``saturation`` describes |raw / reference_bound| >= 2;
        the arm has no displacement tanh and this is not clipping frequency.
        """
        alpha = self._alpha(query, keys)
        _, delta, raw = self.lut(query, keys)
        bound = self._r4_bound if self.e0 else self._bound()
        return dict(alpha=alpha, delta=delta, raw=raw,
                    delta_mse=delta.square().mean(),
                    saturation=(torch.abs(raw / bound) >= 2.0).float().mean(),
                    reference_bound=bound)

def _score(self,query,keys,codes):
    s,u=self.score_fast(query,keys,codes)
    return s,u,None,None
Parent.score=_score
