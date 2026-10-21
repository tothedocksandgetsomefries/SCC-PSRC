import hashlib,importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from fvsr.demo import artificial_fixture
from fvsr.controls import load_control,DecoderDeployment,ResidualDeployment,same_code_transfer
from fvsr.core.replay_bound import state_hash,parent_forward,top
from fvsr.core.residual_bound import ResidualParent
from fvsr.core.decoder_reference import NNDecoder
from fvsr.core.child_models_v2 import child_model,FEATURE_ARMS
from fvsr.core.r4_model_bound import QueryRMSScaler
from fvsr.core.reference_control import ReferenceMatchedMLP
from fvsr.binding import certificate

def bind(b,models,seed,method):
    ref=models[f'{seed}_{method}'];cp=None if method=='AQ' else torch.load(ref['path'],weights_only=False)
    parent=None
    if method not in ('A','BASE_MLP','AQ','NN_RECON','NN_QUERY_MATCHED'):
        parent=torch.load(models[f'{seed}_A']['path'],weights_only=False)['selected_state_sha']
    ref['binding']=certificate(b,ref,seed,method,cp,parent)

def checkpoint(tmp,name,model,phase='QC',arm='FINAL'):
    p=tmp/(name+'.pt');s=model.state_dict();torch.save(dict(model=s,phase=phase,arm=arm,selected_state_sha=state_hash(s),selected_gamma=.5),p)
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),protected=False)
def test_control_checkpoints_load_without_fallback(tmp_path):
    b,m,q,ids,ns=artificial_fixture();seed=17;models={f'{seed}_A':checkpoint(tmp_path,'A',m.parent,'A')}
    bind(b,models,seed,'A')
    for arm in ['FINAL','BASE_MLP','G_ADD_MATCHED']+sorted(FEATURE_ARMS)+['POOL_REF','FIT_REF']:
        if arm in ('POOL_REF','FIT_REF'):head=ReferenceMatchedMLP(2,8,QueryRMSScaler(torch.zeros(2),torch.ones(2)),arm,torch.zeros(2),torch.ones(2))
        else:head=child_model(2,8,QueryRMSScaler(torch.zeros(2),torch.ones(2)),arm)
        models[f'{seed}_{arm}']=checkpoint(tmp_path,arm,head,arm='FINAL' if arm.endswith('_REF') else arm)
        bind(b,models,seed,arm)
        score=load_control(b,models,seed,arm).score_transformed(q,ids,ns)
        assert score.shape==(64,) and np.isfinite(score).all()
    assert np.array_equal(load_control(b,models,seed,'Base_MLP').score_transformed(q,ids,ns),load_control(b,models,seed,'native_parent_psrc').score_transformed(q,ids,ns))
    with pytest.raises(NotImplementedError):load_control(b,models,seed,'UNKNOWN_METHOD')
    bad=dict(models);bad[f'{seed}_BASE_MLP']=models[f'{seed}_FINAL']
    with pytest.raises(ValueError,match='binding mismatch'):load_control(b,bad,seed,'native_parent_psrc')
    for method in ['NN_RECON','NN_QUERY_MATCHED']:
        dec=NNDecoder(b.C).eval();ref=checkpoint(tmp_path,method,dec,method);models[f'{seed}_{method}']=ref
        bind(b,models,seed,method)
        actual=load_control(b,models,seed,method,source_codec=b).score_transformed(q,ids,ns)
        with torch.inference_mode():expected=-np.square(dec.decode(torch.from_numpy(b.codes.astype('int64'))).numpy()-q).sum(-1)
        assert np.array_equal(actual,expected)
    lut=np.zeros((2,256,8),np.float32);lut[0,:,:4]=b.C[0];lut[1,:,4:]=b.C[1];p=tmp_path/'AQ.npy';np.save(p,lut);models[f'{seed}_AQ']=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),protected=False)
    bind(b,models,seed,'AQ')
    actual=load_control(b,models,seed,'FixedCode_AQ',source_codec=b).score_transformed(q,ids,ns)
    expected=-np.square(lut[np.arange(2),b.codes].sum(1)-q).sum(-1);assert np.array_equal(actual,expected)

def test_list_residual_nonzero_gamma_original_intermediates(tmp_path):
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'frozen_reference/T001'))
    import replay_bound as original
    spec=importlib.util.spec_from_file_location('original_residual',root/'frozen_reference/controls/residual_bound.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    b,m,q,ids,_=artificial_fixture();b.residual=True;rng=np.random.default_rng(31);b.coarse=rng.normal(size=(4,8)).astype('float32');b.lists=rng.integers(0,4,size=64,dtype=np.int64)
    a=ResidualParent(b.C,b.coarse,17).eval()
    with torch.no_grad():a.g.fill_(.03)
    acp=checkpoint(tmp_path,'A',a,'A');cp=checkpoint(tmp_path,'QC',m.model)
    new=ResidualDeployment(b,acp,cp,17,'FINAL');old=original.Deployment.__new__(original.Deployment);old.b=b;old.arm='FINAL';old.parent=module.ResidualParent(b.C,b.coarse,17).eval();old.parent.load_state_dict(a.state_dict(),strict=True);old.model=m.model;old.gamma=.5
    for size in [1,7,64]:
        ix=ids[:size];ns=np.zeros(size,np.float32)
        with torch.inference_mode():
            p,u,z=parent_forward(new.parent,b,q[None],ix[None],True);rp,ru,rz=original.parent_forward(old.parent,b,q[None],ix[None],True)
            assert torch.equal(p,rp) and torch.equal(u,ru) and torch.equal(z,rz)
            rms=torch.from_numpy(q.reshape(2,-1))[None].square().mean(-1).sqrt();valid=torch.ones_like(p,dtype=torch.bool);x=new.model.prepare(z,u,rms,p,valid);y=old.model.prepare(rz,ru,rms,rp,valid)
            assert torch.equal(x.u_scaled,y.u_scaled) and torch.equal(x.query_rms_scaled,y.query_rms_scaled) and torch.equal(new.model.raw_residual(x),old.model.raw_residual(y))
        left=new.score_transformed(q,ix,ns);right=old.score_transformed(q,ix,ns)
        assert np.array_equal(left,right) and np.array_equal(top(ix,left,10),original.top(ix,right,10))
    bad=type(b)(**vars(b));bad.lists=b.lists.copy();bad.lists[0]=(bad.lists[0]+1)%4
    with pytest.raises(ValueError,match='lists'):same_code_transfer(b,bad)
    with pytest.raises(NotImplementedError,match='list-residual'):DecoderDeployment(b,acp,'NN_RECON',b)
