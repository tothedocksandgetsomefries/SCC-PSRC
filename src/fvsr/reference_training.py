"""Scoped port of the original paired Pool/FIT trainer factory.

The original implementation installs these factories in trainer_v2. This context
restores them after one explicit call; it adds no recipe or mathematical changes.
Use one training invocation per process (the underlying module factory is global).
"""
import copy
from contextlib import contextmanager
import numpy as np
from .core import trainer_v2 as tv
from .core.reference_control import ReferenceMatchedMLP, fit_reference_moments
from .core.r4_model_bound import QueryRMSScaler

@contextmanager
def reference_factory(parent, geometry, data, full_checkpoint, reference):
    if reference not in ('POOL_REF', 'FIT_REF'):
        raise ValueError('Unknown paired reference')
    if geometry.residual:
        raise NotImplementedError('Original paired Pool/FIT recipe was global PQ only')
    if data.get('role_schema') != 'FIT0_SELECT_CAL1_SELECT_CHECK2':
        raise PermissionError('FIT/SELECT role schema required')
    previous_model, previous_cache = tv.child_model, tv.feature_cache
    cache = previous_cache(parent, geometry, data, True)
    moments = fit_reference_moments(cache['inputs'][1], cache['inputs'][4], data, np.flatnonzero(data['role'] == 0))
    state = full_checkpoint['model']
    scaler = QueryRMSScaler(state['scaler.mean'], state['scaler.std'])
    def factory(M, rank, unused_scaler, arm):
        if arm != 'FINAL':
            raise ValueError('Paired reference uses the original FINAL arm route')
        return ReferenceMatchedMLP(M, rank, copy.deepcopy(scaler), reference=reference,
                                   fit_mean=moments['mean'] if reference == 'FIT_REF' else None,
                                   fit_std=moments['std'] if reference == 'FIT_REF' else None)
    try:
        tv.child_model, tv.feature_cache = factory, lambda *args, **kwargs: cache
        yield moments
    finally:
        tv.child_model, tv.feature_cache = previous_model, previous_cache
