import torch
from torch import nn
class NNDecoder(nn.Module):
    """REIMPLEMENTATION: additive M*256*D LUT -> BN(D) -> FC(D,2D) -> ReLU -> FC(2D,D).
    Paper 2112.09568v2 3.5, 5.1; hidden width 2D and scheduler patience10
    are explicit local choices, not verified author implementation; no residual skip.
    """
    def __init__(self,C):
        super().__init__();C=torch.as_tensor(C,dtype=torch.float32);M,K,ds=C.shape;D=M*ds
        lut=torch.zeros(M,K,D)
        for m in range(M):lut[m,:,m*ds:(m+1)*ds]=C[m]
        self.lut=nn.Parameter(lut);self.block=nn.Sequential(nn.BatchNorm1d(D),nn.Linear(D,2*D),nn.ReLU(),nn.Linear(2*D,D))
        self.M=M;self.D=D
    def decode(self,z):
        x=self.lut[torch.arange(self.M,device=z.device),z.long()].sum(-2)
        return self.block(x)
