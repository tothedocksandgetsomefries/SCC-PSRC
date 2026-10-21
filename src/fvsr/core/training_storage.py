"""Preserve all local recovery evidence; the original server retention policy is not used."""
def retain_recovery(out,keep_latest=2):
    from .training_io import guard
    guard(out,True)
    return {'status':'ALL_RECOVERY_CHECKPOINTS_RETAINED','requested_keep_latest':keep_latest}
