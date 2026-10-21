"""Explicit frozen controls and list-residual scoring; no fallback or training."""
from types import SimpleNamespace
import numpy as np
import torch
from .model import load_scorer,load_geometry
from .core.replay_bound import Deployment,verify,state_hash
from .core.residual_bound import ResidualParent
from .core.decoder_reference import NNDecoder
from .core.child_models_v2 import CHILD_ARMS,child_model
from .core.r4_model_bound import QueryRMSScaler

ALIASES={'full':'FINAL','native_parent_psrc':'BASE_MLP','Base_MLP':'BASE_MLP',
         'FixedCode_AQ':'AQ','FixedCode_NN_RECON':'NN_RECON',
         'FixedCode_NN_QUERY_MATCHED':'NN_QUERY_MATCHED','scc_only':'SCC_ONLY'}
def load_residual_geometry(ref):
    with np.load(verify(ref),allow_pickle=False) as z:b=SimpleNamespace(**{k:z[k] for k in z.files})
    if not bool(b.residual):raise ValueError('Expected list-residual codec')
    M,K,ds=b.C.shape;D=M*ds
    if K!=256 or b.codes.dtype!=np.uint8 or b.codes.shape[1]!=M:raise ValueError('Residual 8-bit code geometry')
    if b.rotation.shape!=(D,D) or b.coarse.ndim!=2 or b.coarse.shape[1]!=D:raise ValueError('Residual coordinate dimensions')
    if b.lists.shape!=(len(b.codes),) or not np.issubdtype(b.lists.dtype,np.integer) or np.any(b.lists<0) or np.any(b.lists>=len(b.coarse)):raise ValueError('Per-ID coarse-list binding')
    if not all(np.isfinite(a).all() for a in [b.C,b.coarse,b.rotation]):raise ValueError('Nonfinite geometry')
    b.residual=True;return b
def same_code_transfer(source,target):
    fields=['C','codes','rotation']+(['coarse','lists'] if source.residual else [])
    if source.residual!=target.residual:raise ValueError('Residual codec mismatch')
    for field in fields:
        if not np.array_equal(getattr(source,field),getattr(target,field)):raise ValueError('Transfer requires per-ID codec identity: '+field)
    return fields
class DecoderDeployment:
    def __init__(self,b,selected,method,source_codec):
        same_code_transfer(source_codec,b)
        if b.residual:raise NotImplementedError('Decoder list-residual inference was not qualified')
        self.b=b;self.method=method
        if method=='AQ':
            self.lut=np.load(verify(selected),allow_pickle=False)
            if self.lut.shape!=(len(b.C),256,len(b.C)*b.C.shape[2]) or not np.isfinite(self.lut).all():raise ValueError('AQ LUT shape/values')
        elif method in ('NN_RECON','NN_QUERY_MATCHED'):
            cp=torch.load(verify(selected),map_location='cpu',weights_only=False)
            if cp['phase']!=method or cp['selected_state_sha']!=state_hash(cp['model']):raise ValueError('Decoder selected checkpoint identity')
            self.model=NNDecoder(b.C);self.model.load_state_dict(cp['model'],strict=True);self.model.eval()
        else:raise NotImplementedError('Unqualified decoder: '+str(method))
    def score_transformed(self,q,ids,native=None):
        if not len(ids):return np.empty(0,np.float32)
        z=self.b.codes[ids].astype('int64')
        with torch.inference_mode():
            x=self.lut[np.arange(len(self.b.C)),z].sum(1) if self.method=='AQ' else self.model.decode(torch.as_tensor(z)).numpy()
        return -np.square(x-np.asarray(q,dtype=np.float32)).sum(-1)
class ResidualDeployment(Deployment):
    def __init__(self,b,selected_A,selected_QC,seed,arm):
        if not b.residual:raise ValueError('List-residual geometry required')
        self.b=b;self.arm=arm;self.parent=ResidualParent(b.C,b.coarse,seed).eval()
        if arm!='BASE_MLP':
            a=torch.load(verify(selected_A),map_location='cpu',weights_only=False)
            if a['phase']!='A' or a['selected_state_sha']!=state_hash(a['model']):raise ValueError('A selected binding')
            self.parent.load_state_dict(a['model'],strict=True)
        c=torch.load(verify(selected_QC),map_location='cpu',weights_only=False)
        if c['phase']!='QC' or c['arm']!=arm or c['selected_state_sha']!=state_hash(c['model']):raise ValueError('QC selected binding')
        if not np.array_equal(self.parent.centroids.numpy(),b.C) or not np.array_equal(self.parent.coarse.numpy(),b.coarse):raise ValueError('Residual codec/model mismatch')
        s=c['model'];m=child_model(len(b.C),8,QueryRMSScaler(s['scaler.mean'],s['scaler.std']),arm);m.load_state_dict(s,strict=True)
        self.model=m.eval();self.gamma=c['selected_gamma'];self.checkpoint=c
def load_control(b,models,seed,method='full',source_codec=None):
    arm=ALIASES.get(method,method)
    if arm not in CHILD_ARMS|{'AQ','NN_RECON','NN_QUERY_MATCHED','SCC_ONLY','POOL_REF','FIT_REF'}:
        raise NotImplementedError('Unqualified control: '+str(method))
    if arm not in ('FINAL','BASE_MLP','SCC_ONLY','POOL_REF','FIT_REF') or b.residual:
        from .binding import validate_bindings
        validate_bindings(b,models,seed,arm)
    if arm in ('AQ','NN_RECON','NN_QUERY_MATCHED'):
        if source_codec is None:raise ValueError('Decoder requires the bound training codec')
        return DecoderDeployment(b,models[f'{seed}_{arm}'],arm,source_codec)
    if b.residual:
        if arm not in CHILD_ARMS:raise NotImplementedError('Unqualified residual control: '+str(method))
        return ResidualDeployment(b,models[f'{seed}_A'],models[f'{seed}_{arm}'],seed,arm)
    if arm in CHILD_ARMS-{'FINAL','BASE_MLP'}:return Deployment(b,models[f'{seed}_A'],models[f'{seed}_{arm}'],seed,arm)
    return load_scorer(b,models,seed,arm)
