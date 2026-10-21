"""Matched controls for the fixed protocol; feature omissions apply at fit and serve."""
import torch
from torch import nn
from r4_model_bound import _R4Base,MatchedMLP,R4CSI
FEATURE_ARMS={'NO_CODE_EMBEDDING','NO_PARENT_PARTS','NO_QUERY_RMS'}
CHILD_ARMS={'FINAL','BASE_MLP','G_ADD_MATCHED'}|FEATURE_ARMS
class FeatureAblation(_R4Base):
    def __init__(self,M,rank,scaler,arm):
        super().__init__(M,rank,scaler)
        if arm not in FEATURE_ARMS:raise ValueError('Unregistered feature arm')
        self.arm=arm
        if arm!='NO_CODE_EMBEDDING':
            self.E=nn.Parameter(torch.empty(M,256,rank));nn.init.normal_(self.E,mean=0,std=.02)
        else:self.register_parameter('E',None)
        width=(0 if arm=='NO_CODE_EMBEDDING' else rank)+(arm!='NO_PARENT_PARTS')+(arm!='NO_QUERY_RMS')
        self.network=nn.Sequential(nn.Linear(M*width,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
    def raw_residual(self,p):
        fields=[]
        if self.E is not None:fields.append(self.E[torch.arange(self.M),p.codes])
        if self.arm!='NO_PARENT_PARTS':fields.append(p.u_scaled.unsqueeze(-1))
        if self.arm!='NO_QUERY_RMS':fields.append(p.query_rms_scaled[:,None,:,None].expand(*p.codes.shape,1))
        return self.network(torch.cat(fields,dim=-1).flatten(start_dim=-2)).squeeze(-1)
def child_model(M,rank,scaler,arm):
    if arm in ('FINAL','BASE_MLP'):return MatchedMLP(M,rank,scaler)
    # Archived r5_benchmark.METHODS binds G_ADD to ADD_ONLY / A_GLOBAL / r8.
    if arm=='G_ADD_MATCHED':return R4CSI(M,rank,scaler,arm='ADD_ONLY')
    if arm in FEATURE_ARMS:return FeatureAblation(M,rank,scaler,arm)
    raise NotImplementedError('Unregistered child arm '+str(arm))
