import copy
import hashlib
import numpy as np
import pytest
import torch
from fvsr.binding import certificate
from fvsr.controls import load_control
from fvsr.demo import artificial_fixture
from fvsr.core.replay_bound import state_hash

def fixture(tmp_path):
    b,m,q,ids,ns=artificial_fixture();models={}
    for method,model in [('A',m.parent),('FINAL',m.model),('BASE_MLP',m.model)]:
        cp=dict(model=model.state_dict(),phase='A' if method=='A' else 'QC',arm='FINAL' if method=='A' else method,
                selected_gamma=.5,selected_epoch=1,selected_state_sha=state_hash(model.state_dict()),config=dict(seed=17))
        p=tmp_path/(method+'.pt');torch.save(cp,p)
        ref=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),protected=False)
        ref['binding']=certificate(b,ref,17,method,cp,state_hash(m.parent.state_dict()) if method=='FINAL' else None)
        models[f'17_{method}']=ref
    return b,models,q,ids,ns

@pytest.mark.parametrize('field',['C','codes','rotation'])
def test_same_shape_wrong_codec_fails(tmp_path,field):
    b,models,*_=fixture(tmp_path);a=getattr(b,field).copy();a.flat[0]+=1;setattr(b,field,a)
    with pytest.raises(ValueError,match='geometry_identity'):load_control(b,models,17)

@pytest.mark.parametrize('field,value',[('seed',18),('method','POOL_REF'),('parent_state_sha256','wrong'),('scaler_sha256','wrong'),('selected_gamma',0)])
def test_wrong_deployment_identity_fails(tmp_path,field,value):
    b,models,*_=fixture(tmp_path);models['17_FINAL']['binding'][field]=value
    with pytest.raises(ValueError,match='binding mismatch'):load_control(b,models,17)

def test_missing_certificate_and_cross_seed_swap_fail(tmp_path):
    b,models,*_=fixture(tmp_path);bad=copy.deepcopy(models);bad['17_FINAL'].pop('binding')
    with pytest.raises(ValueError,match='Missing model binding'):load_control(b,bad,17)
    swapped={k.replace('17_','18_'):v for k,v in models.items()}
    with pytest.raises(ValueError,match='seed'):load_control(b,swapped,18)

def test_gamma_zero_preserves_corresponding_parent(tmp_path):
    from fvsr.core.replay_bound import parent_forward
    b,models,q,ids,ns=fixture(tmp_path)
    for method in ['FINAL','BASE_MLP']:
        ref=models['17_'+method];cp=torch.load(ref['path'],weights_only=False);cp['selected_gamma']=0
        torch.save(cp,ref['path']);ref['sha256']=hashlib.sha256(open(ref['path'],'rb').read()).hexdigest()
        ref['binding']=certificate(b,ref,17,method,cp,models['17_A']['binding']['state_sha256'] if method=='FINAL' else None)
        scorer=load_control(b,models,17,method)
        with torch.inference_mode():expected=parent_forward(scorer.parent,b,q[None],ids[None],True)[0][0].numpy() if method=='FINAL' else ns
        assert np.array_equal(scorer.score_transformed(q,ids,ns),expected)
