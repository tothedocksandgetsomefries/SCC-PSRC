from pathlib import Path
import sys
import numpy as np
import torch
import pytest
from fvsr.demo import artificial_fixture
from fvsr.core.replay_bound import parent_forward,top

@pytest.mark.parametrize('rotation',[False,True])
def test_global_pq_and_opq_coordinate_fixture(rotation):
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'frozen_reference/T001'))
    import replay_bound as original
    import parent_bound as parent
    import r4_model_bound as child
    b,m,q,ids,native=artificial_fixture(rotation=rotation)
    ref=original.Deployment.__new__(original.Deployment);ref.b=b;ref.arm='FINAL';ref.parent=parent.Parent('A_GLOBAL',b.C,17).eval();ref.parent.load_state_dict(m.parent.state_dict(),strict=True)
    ref.model=child.MatchedMLP(2,8,child.QueryRMSScaler(torch.zeros(2),torch.ones(2))).eval();ref.model.load_state_dict(m.model.state_dict(),strict=True);ref.gamma=.5
    qt=q@b.rotation.T
    for size in [1,7,64]:
        ix=ids[:size];ns=native[:size]
        with torch.inference_mode():
            p,u,z=parent_forward(m.parent,b,qt[None],ix[None],True);rp,ru,rz=original.parent_forward(ref.parent,b,qt[None],ix[None],True)
            assert torch.equal(u,ru) and torch.equal(p,rp) and torch.equal(z,rz)
            rms=torch.from_numpy(qt.reshape(2,-1))[None].square().mean(-1).sqrt();valid=torch.ones_like(p,dtype=torch.bool)
            x=m.model.prepare(z,u,rms,p,valid);y=ref.model.prepare(rz,ru,rms,rp,valid)
            assert torch.equal(x.u_scaled,y.u_scaled) and torch.equal(x.query_rms_scaled,y.query_rms_scaled)
            assert torch.equal(m.model.raw_residual(x),ref.model.raw_residual(y))
        a=m.score_transformed(qt,ix,ns);c=ref.score_transformed(qt,ix,ns)
        assert np.array_equal(a,c) and np.array_equal(top(ix,a,10),original.top(ix,c,10))
