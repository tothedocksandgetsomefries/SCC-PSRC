"""Frozen code-only IVF inference; full is default, GT remains offline."""
import argparse,hashlib,json,os
from pathlib import Path
if os.name=='nt':
    os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
    try:import faiss
    except ImportError:pass  # The backend still reports its required optional extra.
import numpy as np
import torch
from fvsr.model import resolve_manifest,load_geometry
from fvsr.controls import load_control
from fvsr.core.replay_bound import top
def main():
    p=argparse.ArgumentParser();p.add_argument('--demo',action='store_true');p.add_argument('--case');p.add_argument('--artifact-dir');p.add_argument('--queries');p.add_argument('--attributes');p.add_argument('--requests');p.add_argument('--method',default='full');p.add_argument('--execution-mode',choices=['reference','full-cm'],default='reference');p.add_argument('--seed',type=int,help='Caller-supplied model identity matching the manifest and checkpoint');p.add_argument('--nprobe',type=int,default=256);p.add_argument('--L',type=int,default=400);p.add_argument('--k',type=int,default=100);p.add_argument('--output');a=p.parse_args()
    if a.demo:
        from fvsr.demo import run_demo
        print(json.dumps(run_demo(a.execution_mode)));return
    if a.seed is None and a.method not in ('Native_L','Native_direct_k'):
        p.error('--seed is required for learned-model evaluation and must match the supplied manifest')
    if not all([a.case,a.artifact_dir,a.queries,a.attributes,a.requests,a.output]):p.error('case/artifact-dir/queries/attributes/requests/output are required')
    if a.L<a.k or a.k<1:raise ValueError('Require L >= k >= 1')
    from fvsr.backends import IVFPQAdapter
    from fvsr.attributes import real_mask
    torch.set_num_threads(1);case=resolve_manifest(a.case,a.artifact_dir);b=load_geometry(case['geometry']);backend=IVFPQAdapter(b,case['index'],a.nprobe)
    with np.load(a.queries,allow_pickle=False) as z:queries=z['queries'];qids=z['qids']
    with np.load(a.attributes,allow_pickle=False) as z:attrs={k:z[k] for k in z.files}
    if len(set(qids))!=len(qids) or queries.shape!=(len(qids),b.rotation.shape[0]):raise ValueError('Unique query IDs and coordinate dimensions required')
    byid={int(q):i for i,q in enumerate(qids)};requests=json.loads(Path(a.requests).read_text())
    if a.execution_mode=='full-cm':
        if a.method not in ('full','FINAL'):raise NotImplementedError('Full-CM requires --method full')
        from fvsr.model import load_scorer
        scorer=load_scorer(b,case['models'],a.seed,a.method,execution_mode='full-cm')
    else:scorer=None if a.method in ('Native_L','Native_direct_k') else load_control(b,case['models'],a.seed,a.method)
    dest=Path(a.output);dest.mkdir(parents=True,exist_ok=False);ranks=[];pools=[];records=[]
    for req in requests:
        qid=int(req['qid']);q=queries[byid[qid]];mask=real_mask(attrs,req['filter'],req['request']);legal=np.flatnonzero(mask)
        if 'legal_ids_sha256' in req and hashlib.sha256(legal.astype('<i8').tobytes()).hexdigest()!=req['legal_ids_sha256']:raise ValueError('Predicate/database row binding mismatch')
        ids,ss=backend.search(q,mask,a.k if a.method=='Native_direct_k' else a.L)
        if scorer is not None:ss=scorer.score_transformed(np.asarray(q,np.float32)@b.rotation.T,ids,ss)
        rank=top(ids,ss,a.k);padded=np.full(a.k,-1,np.int64);padded[:len(rank)]=rank;ranks.append(padded);pp=np.full(a.L,-1,np.int64);pp[:len(ids)]=ids;pools.append(pp);records.append(dict(qid=qid,filter=req['filter'],legal_count=len(legal),pool_size=len(ids)))
    np.savez_compressed(dest/'PREDICTIONS.npz',ranked_ids=np.asarray(ranks),pool_ids=np.asarray(pools));(dest/'REQUESTS.json').write_text(json.dumps(records,indent=2));print(json.dumps(dict(status='FROZEN_INFERENCE_COMPLETE',method=a.method,execution_mode=a.execution_mode,requests=len(records),GT_opened=False,raw_database_opened=False,quality_evaluated=False)))
if __name__=='__main__':main()
