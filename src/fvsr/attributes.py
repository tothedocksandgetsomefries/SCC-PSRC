"""Original row IDs; closed integer range and set-membership predicates."""
import numpy as np

def real_mask(attributes, family, request):
    if family == 'R':
        dates = attributes['update_date']
        return (dates >= request['range_start']) & (dates <= request['range_end'])
    if family == 'EMIS':
        labels = attributes['category_labels'].tolist()
        if request['label'] not in labels:
            return np.zeros(len(attributes['update_date']), dtype=bool)
        return attributes['category_membership'][:, labels.index(request['label'])].copy()
    raise NotImplementedError(f'Unregistered real filter family: {family}')
