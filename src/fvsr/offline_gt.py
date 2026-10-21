"""Offline exhaustive GT utility. Never imported by the serving adapters."""
import numpy as np
def exact_legal_topk(database,query,legal,k=100,chunk_size=2048):
    x=np.asarray(database);q=np.asarray(query,dtype=np.float64);mask=np.asarray(legal)
    if x.ndim!=2 or q.shape!=(x.shape[1],) or mask.shape!=(len(x),) or mask.dtype!=np.bool_ or k<0:raise ValueError('GT shape, boolean mask, or k')
    if not np.isfinite(q).all():raise ValueError('Nonfinite query')
    if chunk_size<1:raise ValueError('Positive chunk size required')
    ids=np.flatnonzero(mask);best=np.empty(0,np.int64);scores=np.empty(0,np.float64)
    for start in range(0,len(ids),chunk_size):
        ii=ids[start:start+chunk_size];delta=x[ii].astype(np.float64)-q
        if not np.isfinite(delta).all():raise ValueError('Nonfinite database vector')
        ss=-(delta*delta).sum(1);jj=np.r_[best,ii];vv=np.r_[scores,ss];order=np.lexsort((jj,-vv))[:k];best=jj[order];scores=vv[order]
    return best,scores
