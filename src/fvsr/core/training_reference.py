import copy,os,random,time
import numpy as np,torch
from torch import nn
from scipy import sparse
from scipy.sparse.linalg import lsqr
from .training_io import guard,record,identity,charge
RECIPES={
 'A':dict(optimizer='AdamW',lr=.001,weight_decay=0.,batch=16,max_epochs=30,min_epochs=2,patience=5,clip=1.),
 'B':dict(optimizer='AdamW',lr=.001,weight_decay=.0001,batch=16,max_epochs=30,min_epochs=10,patience=8,clip=1.),
 'QC':dict(optimizer='AdamW',lr=.0001,weight_decay=.0001,batch=16,max_epochs=12,min_epochs=4,patience=4,clip=1.),
 'NN_RECON':dict(optimizer='Adam',lr=.0005,weight_decay=0.,batch=256,max_epochs=300,plateau_factor=.5,plateau_patience=10,clip=None),
 'NN_QUERY_MATCHED':dict(optimizer='AdamW',lr=.0001,weight_decay=.0001,batch=16,max_epochs=12,min_epochs=4,patience=4,clip=1.)
}

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

def aq_fixed(codes,x,l2=1e-5,iter_limit=200,block_dim=16):
    codes=np.asarray(codes,dtype=np.int64);N,M=codes.shape
    if codes.min()<0 or codes.max()>255:raise ValueError('K256 codes')
    I=sparse.csr_matrix((np.ones(N*M),(np.repeat(np.arange(N),M),(codes+256*np.arange(M)[None]).ravel())),shape=(N,M*256))
    C=np.empty((M*256,x.shape[1]));status=[]
    for b in range(0,x.shape[1],block_dim):
        for d in range(b,min(b+block_dim,x.shape[1])):
            r=lsqr(I,x[:,d],damp=np.sqrt(l2),atol=1e-8,btol=1e-8,iter_lim=iter_limit)
            C[:,d]=r[0];status.append(dict(d=d,istop=r[1],iterations=r[2],residual=r[3]))
    return C.reshape(M,256,-1).astype('float32'),status

def snapshot(model,opt,scheduler,epoch,sampler,selected,config):
    return dict(schema='PE3_COMPLETE_EPOCH_1',model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
                scheduler=copy.deepcopy(scheduler.state_dict()) if scheduler else {'kind':'NOT_APPLICABLE'},
                epoch=epoch,next_epoch=epoch+1,sampler=copy.deepcopy(sampler),selected=copy.deepcopy(selected),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if any(p.is_cuda for p in model.parameters()) else [],config_sha=identity(config))

def restore(cp,model,opt,scheduler,config):
    required={'schema','model','optimizer','scheduler','epoch','next_epoch','sampler','selected','python_rng','numpy_rng','torch_rng','cuda_rng','config_sha'}
    if not required<=cp.keys() or cp['schema']!='PE3_COMPLETE_EPOCH_1' or cp['next_epoch']!=cp['epoch']+1:raise ValueError('Incomplete epoch checkpoint')
    if cp['config_sha']!=identity(config):raise ValueError('Checkpoint identity drift')
    model.load_state_dict(cp['model'],strict=True);opt.load_state_dict(cp['optimizer'])
    if scheduler:scheduler.load_state_dict(cp['scheduler'])
    elif cp['scheduler']!={'kind':'NOT_APPLICABLE'}:raise ValueError('Scheduler mismatch')
    random.setstate(cp['python_rng']);np.random.set_state(cp['numpy_rng']);torch.set_rng_state(cp['torch_rng'])
    if cp['cuda_rng']:torch.cuda.set_rng_state_all(cp['cuda_rng'])
    return cp['sampler'],cp['selected']

def checkpoint(path,payload):
    tick=time.perf_counter()
    p=guard(path,True)
    if p.exists():raise FileExistsError(p)
    tmp=p.with_suffix('.tmp')
    with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.rename(tmp,p)
    ref=record(p,'TECHNICAL_SMOKE_CHECKPOINT');ref['serialization_and_hash_seconds']=time.perf_counter()-tick
    return ref

def step(model,opt,loss,clip):
    opt.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):raise FloatingPointError('Loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Gradient')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True) if clip else torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))
    opt.step()
    return dict(loss=float(loss.detach()),grad_norm=float(norm),clipped=bool(clip and norm>clip),optimizer_steps=1)
