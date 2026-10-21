"""Verify a frozen codec; does not train, re-encode, or rebuild an index."""
import argparse,json
from fvsr.model import resolve_manifest,load_geometry
from fvsr.controls import load_residual_geometry
import numpy as np
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--artifact-dir',required=True);p.add_argument('--residual',action='store_true');a=p.parse_args();m=resolve_manifest(a.manifest,a.artifact_dir);b=(load_residual_geometry if a.residual else load_geometry)(m['geometry'])
    if not np.isfinite(b.C).all() or not np.isfinite(b.rotation).all():raise ValueError('Nonfinite codec')
    print(json.dumps(dict(status='FROZEN_CODEC_VERIFIED',N=len(b.codes),D=b.rotation.shape[0],M=len(b.C),K=b.C.shape[1],code_dtype=str(b.codes.dtype),residual=b.residual,new_codec_training=False,new_encoding=False)))
if __name__=='__main__':main()
