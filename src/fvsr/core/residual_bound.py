"""Exact AST extraction of archived R8 ResidualParent; no upstream side-effect imports."""
import numpy as np, torch
from .parent_bound import Parent

class ResidualParent(Parent):
    def __init__(self,C,coarse,seed):
        super().__init__('A_GLOBAL',C,seed,e0=False)
        self.register_buffer('coarse',torch.as_tensor(coarse,dtype=torch.float32).clone())
    def current_C(self):
        alpha=torch.tanh(self.g)
        raw=torch.einsum('mr,mrkd->mkd',alpha,self.W)
        bound=self._bound()[0]
        return self.centroids+bound*torch.tanh(raw/bound)
    def parts(self,q,lists,codes,calibrated=True):
        C=self.current_C() if calibrated else self.centroids
        m=torch.arange(self.m,device=q.device)
        v=self.coarse[lists].reshape(*lists.shape,self.m,-1)+C[m,codes.long()]
        u=(2*q.reshape(len(q),1,self.m,-1)*v-v.square()).sum(-1)
        parent=torch.zeros_like(u[:,:,0])
        for j in range(self.m):parent=parent+u[:,:,j]
        return parent,u
    def score(self,q,lists,codes):
        parent,u=self.parts(q,lists,codes);return parent,u,None,None
    def base_score(self,q,codes,lists=None):
        if lists is None:raise ValueError('Residual parent requires list identity')
        return self.parts(q,lists,codes,False)[0]
