"""Native search remains the original Faiss IVF-PQ search plus an IDSelector."""
import numpy as np
from .core.replay_bound import verify, top

class IVFPQAdapter:
    def __init__(self, geometry, index_ref, nprobe=256):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError('IVF native search requires the ivf extra (faiss-cpu 1.9.x); no fallback is performed') from exc
        from .binding import geometry_identity
        bound=index_ref.get('binding',{})
        if bound.get('geometry_identity')!=geometry_identity(geometry) or bound.get('artifact_sha256')!=index_ref['sha256']:
            raise ValueError('Index/geometry identity certificate mismatch')
        self.faiss = faiss
        faiss.omp_set_num_threads(1)
        self.b = geometry
        self.index = faiss.read_index(str(verify(index_ref)))
        ix = self.index
        if type(ix).__name__ != 'IndexIVFPQ' or ix.by_residual or ix.metric_type != faiss.METRIC_L2:
            raise NotImplementedError('Only non-residual L2 IndexIVFPQ is qualified by this adapter')
        if ix.ntotal != len(geometry.codes) or ix.pq.M != len(geometry.C):
            raise ValueError('Index/database-code geometry mismatch')
        if not np.array_equal(faiss.vector_to_array(ix.pq.centroids).reshape(geometry.C.shape), geometry.C):
            raise ValueError('Index/model codebook mismatch')
        seen=[]
        for li in range(ix.nlist):
            count=ix.invlists.list_size(li)
            if not count:continue
            ids=faiss.rev_swig_ptr(ix.invlists.get_ids(li),count).copy()
            codes=faiss.rev_swig_ptr(ix.invlists.get_codes(li),count*ix.code_size).reshape(count,ix.code_size)
            if np.any(ids<0) or np.any(ids>=len(geometry.codes)) or not np.array_equal(codes,geometry.codes[ids]):
                raise ValueError('Index/database per-ID code mismatch')
            seen.extend(ids.tolist())
        if not np.array_equal(np.sort(seen),np.arange(len(geometry.codes))):
            raise ValueError('Index database ID coverage/uniqueness mismatch')
        if not 1 <= nprobe <= ix.nlist:
            raise ValueError('nprobe outside existing IVF list count')
        self.nprobe = nprobe

    def search(self, query, legal, limit=400):
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.b.rotation.shape[0],) or not np.isfinite(q).all():
            raise ValueError('Query geometry or finiteness mismatch')
        if legal.dtype != np.bool_ or legal.shape != (len(self.b.codes),):
            raise ValueError('Expected a boolean legal-set mask in original database row order')
        if limit < 1:
            raise ValueError('Search limit must be positive')
        qt = np.ascontiguousarray((q @ self.b.rotation.T)[None])
        allowed = np.flatnonzero(legal).astype('int64')
        selector = self.faiss.IDSelectorBatch(allowed)
        params = self.faiss.SearchParametersIVF(sel=selector)
        params.nprobe = self.nprobe
        distances, found = self.index.search(qt, limit, params=params)
        valid = found[0] >= 0
        ids, scores = found[0, valid], -distances[0, valid]
        order = np.lexsort((ids, -scores))
        ids, scores = ids[order], scores[order]
        if len(np.unique(ids)) != len(ids) or not legal[ids].all():
            raise RuntimeError('Native search produced duplicate or illegal row IDs')
        return ids, scores

    def rerank(self, query, legal, scorer, limit=400, k=100):
        ids, native = self.search(query, legal, limit)
        q = np.asarray(query, np.float32) @ self.b.rotation.T
        scores = scorer.score_transformed(q, ids, native)
        return top(ids, scores, k), ids, scores

def backend_adapter(name, *args, **kwargs):
    if name != 'ivf_pq':
        raise NotImplementedError(f'{name}: requires its original external build and adapter; no Native fallback')
    return IVFPQAdapter(*args, **kwargs)
