"""Validated artifact binding around the unchanged frozen scoring implementation."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .core.replay_bound import Deployment, verify, state_hash, parent_model, parent_forward, top
from .core.reference_control import ReferenceMatchedMLP
from .core.r4_model_bound import QueryRMSScaler

ALIASES = {'full': 'FINAL', 'native_parent_psrc': 'BASE_MLP', 'Base_MLP': 'BASE_MLP',
           'pool_ref': 'POOL_REF', 'fit_ref': 'FIT_REF', 'scc_only': 'SCC_ONLY'}

def load_geometry(reference):
    with np.load(verify(reference), allow_pickle=False) as z:
        b = SimpleNamespace(**{k: z[k] for k in z.files})
    if bool(b.residual):
        raise NotImplementedError('List-residual geometry is not qualified by this global-PQ adapter')
    if b.C.ndim != 3 or b.C.shape[1] != 256 or b.codes.dtype != np.uint8:
        raise ValueError('Expected 8-bit product quantizer and database codes')
    if b.rotation.shape != (b.C.shape[0] * b.C.shape[2],) * 2:
        raise ValueError('Codec/coordinate transform dimension mismatch')
    if b.codes.shape[1] != b.C.shape[0]:
        raise ValueError('Code width mismatch')
    b.residual = False
    return b

def load_scorer(geometry, models, seed, method='full', execution_mode='reference'):
    arm = ALIASES.get(method, method)
    if execution_mode not in ('reference', 'full-cm'):
        raise ValueError('Unknown execution mode: ' + execution_mode)
    if execution_mode == 'full-cm' and arm != 'FINAL':
        raise NotImplementedError('Full-CM requires the formal Full scorer')
    if arm not in ('FINAL', 'BASE_MLP', 'SCC_ONLY', 'POOL_REF', 'FIT_REF'):
        raise NotImplementedError(f'Unregistered frozen deployment: {method}')
    from .binding import validate_bindings
    validate_bindings(geometry, models, seed, arm)
    if arm in ('FINAL', 'BASE_MLP'):
        scorer = Deployment(geometry, models[f'{seed}_A'], models[f'{seed}_{arm}'], seed, arm)
        if execution_mode == 'full-cm':
            from .materialized import materialize_scorer
            from .binding import geometry_identity
            import hashlib
            scorer = materialize_scorer(scorer, codec_identity=geometry_identity(geometry),
                transform_identity=hashlib.sha256(geometry.rotation.tobytes()).hexdigest())
        return scorer
    if arm == 'SCC_ONLY':
        parent = parent_model(geometry, seed).eval()
        cp = torch.load(verify(models[f'{seed}_A']), map_location='cpu', weights_only=False)
        if cp['phase'] != 'A' or cp['selected_state_sha'] != state_hash(cp['model']):
            raise ValueError('Parent state identity mismatch')
        parent.load_state_dict(cp['model'], strict=True)
        class SCCOnly:
            gamma = None
            checkpoint = cp
            def score_transformed(self, q, ids, native):
                if not len(ids):
                    return np.empty(0, np.float32)
                with torch.inference_mode():
                    p, _, _ = parent_forward(parent, geometry, np.asarray(q, np.float32)[None], np.asarray(ids)[None], True)
                return p[0].numpy().copy()
        scorer = SCCOnly()
        scorer.parent = parent
        return scorer
    if arm not in ('POOL_REF', 'FIT_REF'):
        raise NotImplementedError(f'Unregistered frozen deployment: {method}')
    cp = torch.load(verify(models[f'{seed}_{arm}']), map_location='cpu', weights_only=False)
    if cp['phase'] != 'QC' or cp['arm'] != 'FINAL' or cp['selected_state_sha'] != state_hash(cp['model']):
        raise ValueError('Reference-control checkpoint mismatch')
    m = Deployment.__new__(Deployment)
    m.b, m.arm = geometry, 'FINAL'
    m.parent = parent_model(geometry, seed).eval()
    acp = torch.load(verify(models[f'{seed}_A']), map_location='cpu', weights_only=False)
    if acp['selected_state_sha'] != state_hash(acp['model']):
        raise ValueError('Parent state identity mismatch')
    m.parent.load_state_dict(acp['model'], strict=True)
    st = cp['model']
    m.model = ReferenceMatchedMLP(len(geometry.C), 8,
        QueryRMSScaler(st['scaler.mean'], st['scaler.std']), reference=arm,
        fit_mean=st.get('fit_u_mean'), fit_std=st.get('fit_u_std'))
    m.model.load_state_dict(st, strict=True)
    m.model.eval()
    m.gamma, m.checkpoint = cp['selected_gamma'], cp
    return m

def resolve_manifest(manifest_path, artifact_dir):
    """Manifest paths are relative to an explicitly supplied artifact directory."""
    m = json.loads(Path(manifest_path).read_text(encoding='utf8'))
    root = Path(artifact_dir).resolve()
    def resolve(x):
        if isinstance(x, dict):
            if 'path' in x and 'sha256' in x:
                p = (root / x['path']).resolve()
                if not p.is_relative_to(root):
                    raise ValueError('Artifact reference escapes artifact directory')
                x = dict(x, path=str(p), protected=False)
                verify(x)
            return {k: resolve(v) for k, v in x.items()}
        return [resolve(v) for v in x] if isinstance(x, list) else x
    return resolve(m)
