import hashlib,json,pathlib
import numpy as np,torch
from .parent_bound import Parent
from .child_models_v2 import child_model
from .r4_model_bound import QueryRMSScaler
def identity(obj):

    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def verify(ref):
 p=pathlib.Path(ref["path"]);assert ref.get("protected") is False and hashlib.sha256(p.read_bytes()).hexdigest()==ref["sha256"];return p
def state_hash(s):
    h=hashlib.sha256()
    for k,v in sorted(s.items()):
        a=v.detach().cpu().contiguous().detach().cpu().numpy();h.update(identity(dict(name=k,shape=list(a.shape),dtype=str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()

def parent_model(b,seed):
    return ResidualParent(b.C,b.coarse,seed) if b.residual else Parent('A_GLOBAL',b.C,seed)

def parent_forward(A,b,q,ids,calibrated=True):
    device=next(A.parameters()).device
    z=torch.as_tensor(b.codes[np.asarray(ids)].astype('int64'),device=device);qt=torch.as_tensor(q,dtype=torch.float32,device=device)
    if b.residual:
        lists=torch.as_tensor(b.lists[np.asarray(ids)].astype('int64'),device=device);p,u=A.parts(qt,lists,z,calibrated)
    elif calibrated:p,u,_,_=A.score(qt,torch.zeros((len(q),1),dtype=torch.long,device=device),z)
    else:
        lut=A.base_lut(qt);u=torch.stack([lut[:,m].gather(1,z[:,:,m]) for m in range(A.m)],-1)
        p=torch.zeros(u.shape[:2],device=device); 
        for m in range(A.m):p=p+u[:,:,m]
    return p,u,z
class Deployment:

    def __init__(self,b,selected_A,selected_QC,seed,arm):

        self.b=b;self.arm=arm;self.parent=parent_model(b,seed).eval()

        if arm!='BASE_MLP':

            a=torch.load(verify(selected_A),map_location='cpu',weights_only=False)

            if a['phase']!='A' or a['selected_state_sha']!=state_hash(a['model']):raise ValueError('A selected binding')

            self.parent.load_state_dict(a['model'],strict=True)

        c=torch.load(verify(selected_QC),map_location='cpu',weights_only=False)

        if c['phase']!='QC' or c['arm']!=arm or c['selected_state_sha']!=state_hash(c['model']):raise ValueError('QC selected binding')

        if not np.array_equal(self.parent.centroids.numpy(),b.C):raise ValueError('Codec/model mismatch')

        s=c['model'];m=child_model(len(b.C),8,QueryRMSScaler(s['scaler.mean'],s['scaler.std']),arm);m.load_state_dict(s,strict=True)

        self.model=m.eval();self.gamma=c['selected_gamma'];self.checkpoint=c

    def score_transformed(self,q,ids,native):

        if not len(ids):return np.empty(0,np.float32)

        with torch.inference_mode():

            p,u,z=parent_forward(self.parent,self.b,np.asarray(q,dtype=np.float32)[None],np.asarray(ids)[None],self.arm!='BASE_MLP')

            if self.arm=='BASE_MLP':p=torch.as_tensor(np.asarray(native,dtype=np.float32)[None])

            rms=torch.as_tensor(q.reshape(len(self.b.C),-1))[None].square().mean(-1).sqrt();v=torch.ones((1,len(ids)),dtype=torch.bool)

            return self.model(z,u,rms,p,v,gamma=self.gamma)[0].numpy().copy()
def top(ids,scores,k,valid=None):
    ids=np.asarray(ids);scores=np.asarray(scores)
    if ids.ndim!=1 or scores.shape!=ids.shape or k<0:raise ValueError('Rank shape/k')
    if valid is None:valid=np.ones(len(ids),bool)
    if np.asarray(valid).dtype!=np.bool_ or len(valid)!=len(ids):raise ValueError('Explicit boolean validity')
    ii=ids[valid];ss=scores[valid]
    if np.any(ii<0) or len(np.unique(ii))!=len(ii) or not np.isfinite(ss).all():raise ValueError('Invalid/duplicate ID or nonfinite score')
    return ii[np.lexsort((ii,-ss))[:k]]
def query_mask(attr,p,q):

    kind=p['kind'];N=len(attr['rank']) if 'rank' in attr else len(attr.get('EM',attr.get('aligned')))

    if kind=='LEGACY_FILTER':

        old=p['predicate']

        if 'regime' in old:return attr[old['regime']]//old['percent']==old['label']

        if old['condition']=='EM':return attr['EM']==old['label']

        found=np.flatnonzero(attr['EMIS_LABELS']==old['label'])

        if len(found)!=1:raise ValueError('Original EMIS label mapping')

        return attr['EMIS'][:,int(found[0])].astype(bool,copy=False)

    if kind=='ALL':return np.ones(N,bool)

    if kind=='RANGE_FRACTION':return attr[p['field']]<round(N*p['fraction'])

    if kind in ('EQ_AND_RANGE','EQ_OR_EQ'):

        l=query_mask(attr,p['left'],q);r=query_mask(attr,p['right'],q)

        return l&r if kind=='EQ_AND_RANGE' else l|r

    if kind!='FIXED_ATTRIBUTE_WINDOW':return mask_for(attr,p)

    count=round(N*p['selectivity']);corr=p['correlation']

    if not 0<=count<=N:raise ValueError('Illegal selectivity')

    if count==N:return np.ones(N,bool)

    order=attr['projection_order'];position=int(np.searchsorted(attr['projection'][order],float(q@attr['direction'])))

    if corr=='near':start=min(max(position-count//2,0),N-count)

    elif corr=='far':start=0 if position>=N//2 else N-count

    elif corr=='neutral':

        order=np.argsort(attr['neutral'],kind='stable')

        start=int.from_bytes(hashlib.sha256(np.asarray(q,dtype='float32').tobytes()).digest()[:8],'little')%(N-count+1)

    else:raise ValueError('Unknown correlation')

    mask=np.zeros(N,bool);mask[order[start:start+count]]=True;return mask

def cached_mask(attr,p,q):
 assert p['kind']=='FIXED_ATTRIBUTE_WINDOW' and p['correlation'] in ['near','neutral']
 N=len(attr['rank']);count=round(N*p['selectivity'])
 if p['correlation']=='near':
  order=attr['projection_order'];pos=int(np.searchsorted(attr['_projection_sorted'],float(q@attr['direction'])));start=min(max(pos-count//2,0),N-count)
 else:
  order=attr['_neutral_order'];start=int.from_bytes(hashlib.sha256(np.asarray(q,dtype='float32').tobytes()).digest()[:8],'little')%(N-count+1)
 mask=np.zeros(N,bool);mask[order[start:start+count]]=True;return mask
