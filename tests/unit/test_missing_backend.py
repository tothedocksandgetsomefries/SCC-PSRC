import sys
import pytest
from fvsr.backends import backend_adapter,IVFPQAdapter
from fvsr.demo import artificial_fixture

def test_external_backend_never_falls_back():
    for name in ['RangePQ','Rii','HNSWPQ','ACORN','RACORN','unknown']:
        with pytest.raises(NotImplementedError,match='no Native fallback'):backend_adapter(name)

def test_missing_faiss_has_explicit_install_error(monkeypatch):
    monkeypatch.setitem(sys.modules,'faiss',None)
    b,*_=artificial_fixture()
    with pytest.raises(ImportError,match='ivf extra'):IVFPQAdapter(b,{},1)
