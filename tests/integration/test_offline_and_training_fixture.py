import json
import numpy as np
import torch
import pytest
from fvsr.offline_gt import exact_legal_topk
from fvsr.demo import artificial_fixture

def test_exhaustive_gt_legal_short_ties_and_empty():
    db=np.array([[1,0],[-1,0],[0,0],[0,2]],np.float32);q=np.array([0,0],np.float32)
    ids,scores=exact_legal_topk(db,q,np.array([True,True,False,True]),k=100,chunk_size=1)
    assert ids.tolist()==[0,1,3] and scores.tolist()==[-1.,-1.,-4.]
    assert exact_legal_topk(db,q,np.zeros(4,bool))[0].size==0
    assert exact_legal_topk(db,q,np.ones(4,bool),k=0)[0].size==0

@pytest.mark.parametrize('reference',[None,'POOL_REF','FIT_REF'])
def test_archived_trainer_one_artificial_update(tmp_path,reference):
    """Software only: 64 artificial vectors, six queries, one technical B update."""
    from fvsr.core import training_io
    from fvsr.core.trainer_v2 import SharedTrainer
    b,m,q,ids,native=artificial_fixture();rng=np.random.default_rng(99);db=rng.normal(size=(64,8)).astype('float32');queries=rng.normal(size=(6,8)).astype('float32');pool=np.tile(ids,(6,1));valid=np.ones(pool.shape,bool);exact=-((db[None]-queries[:,None])**2).sum(-1)
    gt=np.full((6,100),-1,np.int64)
    for i,query in enumerate(queries):rank,_=exact_legal_topk(db,query,np.ones(64,bool));gt[i,:len(rank)]=rank
    with torch.no_grad():base=np.asarray([m.parent.base_score(torch.from_numpy(x[None]),torch.from_numpy(b.codes.astype('int64')[None]))[0].numpy() for x in queries])
    d=dict(role=np.array([0,0,1,1,2,2]),role_schema='FIT0_SELECT_CAL1_SELECT_CHECK2',qids=np.arange(6),predicate=np.array(['SOFTWARE_ONLY']*6),query=queries,ids=pool,valid=valid,base=base,exact=exact,gt=gt,legal_count=np.full(6,64),db_learn_ids=np.arange(32),db_cal_ids=np.arange(32,64),dataset='ARTIFICIAL_SOFTWARE_FIXTURE',identity='ARTIFICIAL_NOT_A_SCIENCE_DATASET',backend_identity='ARTIFICIAL_PQ_CODES')
    training_io.configure(tmp_path/'software_training');trainer=SharedTrainer(b,d,db,'TECHNICAL_QUALIFICATION',17,'SOFTWARE_FIXTURE_ONLY',max_updates=1)
    from contextlib import nullcontext
    from fvsr.reference_training import reference_factory
    from fvsr.core import trainer_v2 as tv
    original_factory=tv.child_model
    context=reference_factory(m.parent,b,d,dict(model=m.model.state_dict()),reference) if reference else nullcontext()
    with context:result=trainer.run('B',tmp_path/'software_training/stage',A=m.parent,arm='FINAL')
    assert tv.child_model is original_factory
    assert result['status']=='COMPLETE' and result['optimizer_updates']==1 and result['role']=='TECHNICAL_QUALIFICATION'
    assert (tmp_path/'software_training/stage/epoch000.pt').exists() and (tmp_path/'software_training/stage/epoch001.pt').exists()
    cp=torch.load(result['selected']['path'],weights_only=False,map_location='cpu');assert cp['phase']=='B' and cp['config']['recipe']['mode']=='TECHNICAL_QUALIFICATION'
