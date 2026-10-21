"""Ccal-only materialization for frozen global-PQ Parent; scoring order unchanged.

This is an execution mode, not a new learned state. No query LUT, projection,
candidate feature, or final score is cached. Original g/W remain resident.
"""
import copy,hashlib,time
import numpy as np
import torch
from .core.parent_bound import Parent
from .core.replay_bound import state_hash

class MaterializedParent(Parent):
    def __init__(self, original, *, codec_identity, transform_identity):
        if original.kind!='A_GLOBAL' or next(original.parameters()).device.type!='cpu':
            raise NotImplementedError('This mode is qualified only for frozen CPU global-PQ parents')
        # Do not perturb the caller's random stream when constructing a copy.
        with torch.random.fork_rng(devices=[]):
            super().__init__('A_GLOBAL',original.centroids.detach().numpy(),0,e0=original.e0)
        self.load_state_dict(original.state_dict(),strict=True);self.eval().requires_grad_(False)
        tick=time.perf_counter()
        with torch.inference_mode():
            q=torch.zeros((1,self.m*self.dsub),dtype=self.centroids.dtype)
            keys=torch.zeros((1,1),dtype=torch.long)
            alpha=self._alpha(q,keys)
            raw=torch.einsum('bmr,mrkd->bmkd',alpha,self.W)
            bound=self._r4_bound if self.e0 else self._bound()
            delta=bound*torch.tanh(raw/bound)
            ccal=self.centroids[None]+delta
        # Only Ccal persists. The reference score_fast discards delta/raw;
        # no additional train-time construction tensors are cached.
        self.register_buffer('_cm_codebook',ccal.clone(),persistent=False)
        self.materialization_seconds=time.perf_counter()-tick
        self.binding=dict(state_sha256=state_hash(original.state_dict()),codec_identity=codec_identity,transform_identity=transform_identity,dtype=str(ccal.dtype),device=str(ccal.device),Ccal_sha256=hashlib.sha256(ccal.numpy().tobytes()).hexdigest())
        self._cm_versions=self._versions()
    def _versions(self):
        # Public tensor updates (load_state_dict/copy_/in-place edits) invalidate.
        return tuple((name,id(t),t.data_ptr(),t._version,str(t.dtype),str(t.device)) for name,t in [('C',self.centroids),('g',self.g),('W',self.W)])
    def validate_cache(self, *, codec_identity=None, transform_identity=None, full_hash=False):
        if self._versions()!=self._cm_versions:raise RuntimeError('Materialized codebook invalidated by model change')
        for name,value in [('codec_identity',codec_identity),('transform_identity',transform_identity)]:
            if value is not None and value!=self.binding[name]:raise RuntimeError('Materialized codebook binding mismatch: '+name)
        if full_hash and state_hash(self.state_dict())!=self.binding['state_sha256']:raise RuntimeError('Materialized parameter fingerprint mismatch')
    def lut(self,query,keys):
        self.validate_cache()
        lut=self._lut_from(query,self._cm_codebook)
        return lut,None,None
    @property
    def cache_tensor_bytes(self):
        return self._cm_codebook.numel()*self._cm_codebook.element_size()

def materialize_scorer(scorer, *, codec_identity, transform_identity):
    if getattr(scorer,'arm',None)!='FINAL' or bool(scorer.b.residual):
        raise NotImplementedError('Full global-PQ execution only; other modes need their own qualification')
    rotation=np.asarray(scorer.b.rotation)
    if not np.array_equal(rotation,np.eye(len(rotation),dtype=rotation.dtype)):
        raise NotImplementedError('OPQ transforms are outside the qualified Full-CM scope')
    out=copy.copy(scorer)
    out.parent=MaterializedParent(scorer.parent,codec_identity=codec_identity,transform_identity=transform_identity)
    return out
