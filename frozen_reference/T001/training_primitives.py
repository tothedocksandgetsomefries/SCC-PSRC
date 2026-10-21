from __future__ import annotations
import numpy as np, torch


def stable_order(ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.lexsort((np.asarray(ids, dtype=np.int64), -np.asarray(scores, dtype=np.float64)))

def _choice(values: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=np.int32)
    if not len(values):
        return np.empty(0, dtype=np.int32)
    return np.asarray(rng.choice(values, count, replace=len(values) < count), dtype=np.int32)

def w3_style_gt(full_gt: np.ndarray, rng: np.random.Generator, count: int = 64) -> np.ndarray:
    truth = np.asarray(full_gt, dtype=np.int32)
    truth = truth[truth >= 0]
    if not len(truth):
        return truth
    if count != 64:
        boundary = min(count // 2, len(truth))
        return np.concatenate((truth[-boundary:], _choice(truth, count - boundary, rng))).astype(np.int32)
    if len(truth) >= 100:
        return np.concatenate((truth[69:100], _choice(truth[:69], 33, rng))).astype(np.int32)
    boundary = min(31, len(truth))
    return np.concatenate((truth[-boundary:], _choice(truth, 64 - boundary, rng))).astype(np.int32)

def sample_positive_ids(
    contract: str,
    candidate_ids: np.ndarray,
    candidate_labels: np.ndarray,
    full_gt: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    candidate_ids = np.asarray(candidate_ids, dtype=np.int32)
    candidate_labels = np.asarray(candidate_labels, dtype=bool)
    reachable = np.unique(candidate_ids[candidate_labels & (candidate_ids >= 0)])
    full = np.asarray(full_gt, dtype=np.int32)
    full = np.unique(full[full >= 0])
    if contract == "CANDIDATE_ONLY":
        chosen = _choice(reachable, 64, rng)
    elif contract == "W3_STYLE_GT":
        chosen = w3_style_gt(full, rng)
        if not len(chosen):
            chosen = _choice(reachable, 64, rng)
    elif contract == "BALANCED_GT_TEACHER":
        if len(reachable) and len(full):
            chosen = np.concatenate((_choice(reachable, 32, rng), w3_style_gt(full, rng, 32)))
        elif len(full):
            chosen = w3_style_gt(full, rng)
        else:
            chosen = _choice(reachable, 64, rng)
    else:
        raise ValueError(contract)
    if not len(chosen):
        raise ValueError("operation has no legal positive teacher")
    if len(chosen) != 64:
        chosen = _choice(chosen, 64, rng)
    candidate_set = set(map(int, candidate_ids[candidate_ids >= 0]))
    outside = sum(int(value) not in candidate_set for value in chosen)
    return chosen.astype(np.int32), {
        "candidate_positive_unique": float(len(reachable)),
        "full_gt_unique": float(len(full)),
        "outside_candidate_fraction": outside / 64.0,
    }

def static_random_negative_ids(
    ids: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
    base: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.asarray(valid, dtype=bool) & ~np.asarray(labels, dtype=bool)
    values = np.asarray(ids[mask], dtype=np.int32)
    score = np.asarray(base[mask], dtype=np.float32)
    order = stable_order(values, score)[: min(256, len(values))]
    return _choice(values[order], 64, rng)

def zscore_tensor(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean(dim=1, keepdim=True)) / values.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)

def stress_weights(exact: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    separation = np.full(len(exact), np.nan, dtype=np.float32)
    for row in range(len(exact)):
        mask = np.asarray(valid[row], dtype=bool)
        positive = np.asarray(labels[row], dtype=bool) & mask
        negative = ~np.asarray(labels[row], dtype=bool) & mask
        if not positive.any() or not negative.any():
            continue
        candidates = exact[row, mask]
        span = float(np.quantile(candidates, 0.90) - np.quantile(candidates, 0.10))
        if span > 1e-12:
            separation[row] = max((float(np.min(exact[row, positive])) - float(np.max(exact[row, negative]))) / span, 0.0)
    positive_values = separation[np.isfinite(separation) & (separation > 0)]
    scale = float(np.median(positive_values)) if len(positive_values) else 1.0
    weights = np.clip(1.0 + 3.0 * np.exp(-separation / max(scale, 1e-6)), 1.0, 4.0).astype(np.float32)
    weights[~np.isfinite(weights)] = 4.0
    return weights, separation, scale

def operation_weights(data,rows):
    qids=data['qids'][rows];pred=data['predicate'][rows];uq=np.unique(qids);weights=np.empty(len(rows),'float64')
    for q in uq:
        ix=np.flatnonzero(qids==q);up=np.unique(pred[ix])
        for p in up:
            jj=ix[pred[ix]==p];weights[jj]=len(rows)/(len(uq)*len(up)*len(jj))
    return weights

def select_epoch(summary,initial,best):
    safe=(summary['R100']-initial['R100']>=-.001 and summary['strictR10']-initial['strictR10']>=-.0025)
    J=.5*(summary['R100']+summary['strictR10'])
    return safe and J>best['J'],safe,J