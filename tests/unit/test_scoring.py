import numpy as np
import pytest
import torch
from fvsr.demo import artificial_fixture,run_demo
from fvsr.core.replay_bound import top,parent_forward
from fvsr.core.r4_model_bound import masked_moments,operation_smooth_l1
from fvsr.model import load_scorer
from fvsr.backends import backend_adapter
from fvsr.attributes import real_mask

def test_mask_ties_short_and_empty_pool():
    assert top(np.array([7,2,4]),np.array([1.,1.,9.]),10,np.array([True,True,False])).tolist()==[2,7]
    b,m,q,ids,native=artificial_fixture()
    assert m.score_transformed(q,ids[:0],native[:0]).shape==(0,)
    for size in [1,2,7]:
        s=m.score_transformed(q,ids[:size],native[:size]);assert np.isfinite(s).all();assert len(top(ids[:size],s,100))==size
    with pytest.raises(ValueError):top(np.array([1,1]),np.ones(2),1)

def test_invalid_slots_do_not_change_population_statistics():
    v=torch.tensor([[True,True,False],[False,False,False]])
    x=torch.tensor([[1.,3.,float('nan')],[float('nan')]*3])
    mean,std=masked_moments(x,v)
    assert mean[0].item()==2 and std[0].item()==1
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()

def test_nonzero_gamma_changes_scores_and_backward_is_finite():
    b,m,q,ids,native=artificial_fixture();m.model.train()
    p,u,z=parent_forward(m.parent,b,q[None],ids[None],True)
    rms=torch.from_numpy(q.reshape(2,-1))[None].square().mean(-1).sqrt();valid=torch.ones_like(p,dtype=torch.bool)
    full=m.model(z,u,rms,p,valid,gamma=.5);zero=m.model(z,u,rms,p,valid,gamma=0)
    assert torch.equal(zero,p) and not torch.equal(full,p)
    target=torch.zeros_like(full);loss=operation_smooth_l1(full,target,valid).mean();loss.backward()
    assert any(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in m.model.parameters())

def test_real_closed_range_and_membership():
    a={'update_date':np.array([1,2,3]),'category_labels':np.array(['a','b']),'category_membership':np.array([[1,0],[1,1],[0,1]],bool)}
    assert real_mask(a,'R',{'range_start':1,'range_end':2}).tolist()==[True,True,False]
    assert real_mask(a,'EMIS',{'label':'b'}).tolist()==[False,True,True]
    assert not real_mask(a,'EMIS',{'label':'missing'}).any()

def test_unsupported_paths_fail_and_demo_runs():
    with pytest.raises(NotImplementedError):backend_adapter('racorn_pq')
    b,*_=artificial_fixture()
    with pytest.raises(NotImplementedError):load_scorer(b,{},1,'unknown_decoder')
    assert run_demo()['status']=='SOFTWARE_DEMO_PASS'
