"""Identity checks at public loader boundaries; no scoring or selection changes.

Certificates are provenance records, not signatures. Treat manifests and PyTorch
checkpoints as trusted inputs. The frozen reference classes remain low-level APIs.
"""
import hashlib
import json
import numpy as np
import torch
from .core.replay_bound import state_hash, verify

def geometry_identity(b):
    h = hashlib.sha256()
    fields = ['C', 'codes', 'rotation'] + (['coarse', 'lists'] if b.residual else [])
    h.update(str(bool(b.residual)).encode())
    for name in fields:
        a = np.ascontiguousarray(getattr(b, name))
        h.update(json.dumps([name, str(a.dtype), list(a.shape)], separators=(',', ':')).encode())
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def scaler_identity(state):
    return state_hash({k: v for k, v in state.items() if k.startswith('scaler.') or k in ('fit_u_mean', 'fit_u_std')})

def certificate(b, ref, seed, method, checkpoint=None, parent_state=None):
    """Build only from independently verified source provenance, never infer a seed."""
    result = dict(schema=1, artifact_sha256=ref['sha256'], geometry_identity=geometry_identity(b),
                  seed=int(seed), method=method, parent_state_sha256=parent_state)
    if checkpoint is not None:
        result.update(state_sha256=state_hash(checkpoint['model']), scaler_sha256=scaler_identity(checkpoint['model']),
                      selected_gamma=checkpoint.get('selected_gamma'), selected_epoch=checkpoint.get('selected_epoch'))
    return result

def validate_bindings(b, models, seed, arm):
    gid = geometry_identity(b)
    needed = ['A'] if arm == 'SCC_ONLY' else (['A', arm] if arm not in ('BASE_MLP', 'AQ', 'NN_RECON', 'NN_QUERY_MATCHED') else [arm])
    loaded = {}
    for method in needed:
        ref = models[f'{seed}_{method}']; cert = ref.get('binding')
        if not cert or cert.get('schema') != 1:
            raise ValueError('Missing model binding certificate; use a provenance-qualified manifest')
        expected = dict(artifact_sha256=ref['sha256'], geometry_identity=gid, seed=int(seed), method=method)
        for key, value in expected.items():
            if cert.get(key) != value:
                raise ValueError('Model binding mismatch: ' + key)
        path = verify(ref)
        if method == 'AQ':
            continue
        cp = torch.load(path, map_location='cpu', weights_only=False)
        if cp.get('config', {}).get('seed', int(seed)) != int(seed):
            raise ValueError('Checkpoint seed mismatch')
        for key, value in dict(state_sha256=state_hash(cp['model']), scaler_sha256=scaler_identity(cp['model']),
                               selected_gamma=cp.get('selected_gamma'), selected_epoch=cp.get('selected_epoch')).items():
            if cert.get(key) != value:
                raise ValueError('Model binding mismatch: ' + key)
        if cp['selected_state_sha'] != cert['state_sha256']:
            raise ValueError('Selected tensor identity mismatch')
        loaded[method] = cp
    if 'A' in loaded and arm != 'SCC_ONLY':
        cert = models[f'{seed}_{arm}']['binding']
        if cert.get('parent_state_sha256') != loaded['A']['selected_state_sha']:
            raise ValueError('Model binding mismatch: parent_state_sha256')
    return loaded
