"""Deterministic artificial software fixture, never a scientific result."""
from types import SimpleNamespace
import numpy as np
import torch
from .core.parent_bound import Parent
from .core.r4_model_bound import QueryRMSScaler, MatchedMLP
from .core.replay_bound import Deployment, top

def artificial_fixture(seed=17, rotation=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m, ds, n = 2, 4, 64
    c = rng.normal(size=(m, 256, ds)).astype('float32')
    codes = rng.integers(0, 256, (n, m), dtype=np.uint8)
    transform = np.eye(m * ds, dtype=np.float32)
    if rotation:
        transform = np.linalg.qr(rng.normal(size=transform.shape))[0].astype('float32')
    b = SimpleNamespace(C=c, codes=codes, rotation=transform, residual=False)
    parent = Parent('A_GLOBAL', c, seed).eval()
    with torch.no_grad():
        parent.g.copy_(torch.from_numpy(rng.normal(0, .02, parent.g.shape).astype('float32')))
    head = MatchedMLP(m, 8, QueryRMSScaler(torch.zeros(m), torch.ones(m))).eval()
    with torch.no_grad():
        head.network[-1].weight.copy_(torch.from_numpy(rng.normal(0, .05, head.network[-1].weight.shape).astype('float32')))
    scorer = Deployment.__new__(Deployment)
    scorer.b, scorer.arm, scorer.parent, scorer.model, scorer.gamma = b, 'FINAL', parent, head, .5
    scorer.checkpoint = {'selected_epoch': None, 'role': 'ARTIFICIAL_SOFTWARE_FIXTURE'}
    query = rng.normal(size=m * ds).astype('float32')
    ids = np.arange(n, dtype=np.int64)
    qt = query @ transform.T
    with torch.inference_mode():
        native = parent.base_score(torch.from_numpy(qt[None]), torch.from_numpy(codes.astype('int64')[None]))[0].numpy()
    return b, scorer, query, ids, native

def run_demo(execution_mode='reference'):
    b, scorer, query, ids, native = artificial_fixture()
    if execution_mode == 'full-cm':
        from .materialized import materialize_scorer
        from .binding import geometry_identity
        import hashlib
        scorer = materialize_scorer(scorer, codec_identity=geometry_identity(b),
            transform_identity=hashlib.sha256(b.rotation.tobytes()).hexdigest())
    elif execution_mode != 'reference':
        raise ValueError('Unknown execution mode')
    legal = ids % 3 == 0
    candidate_ids = ids[legal]
    scores = scorer.score_transformed(query @ b.rotation.T, candidate_ids, native[legal])
    ranked = top(candidate_ids, scores, 10)
    assert len(ranked) == 10 and np.isin(ranked, candidate_ids).all()
    return {'status': 'SOFTWARE_DEMO_PASS', 'evidence_role': 'ARTIFICIAL_SOFTWARE_FIXTURE',
            'raw_database_required': False, 'execution_mode': execution_mode, 'gamma': scorer.gamma, 'candidate_count': len(candidate_ids),
            'top10_ids': ranked.tolist(), 'scientific_performance_claim': False}
