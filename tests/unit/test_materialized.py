"""Public-mode regression checks; full scientific qualification is separate."""
import hashlib
import numpy as np
import pytest
import torch
from test_artifact_binding import fixture
from fvsr.model import load_scorer
from fvsr.demo import artificial_fixture
from fvsr.materialized import materialize_scorer
from fvsr.binding import geometry_identity
from fvsr.core.replay_bound import top

def test_bound_loader_scores_ranks_and_serialization(tmp_path):
    b,models,q,ids,ns=fixture(tmp_path)
    ref=load_scorer(b,models,17)
    cm=load_scorer(b,models,17,execution_mode='full-cm')
    a=ref.score_transformed(q,ids,ns);z=cm.score_transformed(q,ids,ns)
    assert np.array_equal(a,z) and np.array_equal(top(ids,a,10),top(ids,z,10))
    assert '_cm_codebook' not in cm.parent.state_dict()
    assert set(cm.parent.state_dict())==set(ref.parent.state_dict())
    assert cm.parent.cache_tensor_bytes==b.C.nbytes
    cm.parent.validate_cache(full_hash=True)
    with torch.no_grad():cm.parent.g.add_(.01)
    with pytest.raises(RuntimeError,match='invalidated'):cm.score_transformed(q,ids,ns)

def test_unqualified_mode_and_parent_rejected(tmp_path):
    b,models,*_=fixture(tmp_path)
    with pytest.raises(NotImplementedError,match='formal Full'):
        load_scorer(b,models,17,'native_parent_psrc',execution_mode='full-cm')
    with pytest.raises(ValueError,match='Unknown execution mode'):
        load_scorer(b,models,17,execution_mode='unregistered')
    b,m,*_=artificial_fixture(rotation=True)
    with pytest.raises(NotImplementedError,match='OPQ'):
        materialize_scorer(m,codec_identity=geometry_identity(b),transform_identity=hashlib.sha256(b.rotation.tobytes()).hexdigest())
