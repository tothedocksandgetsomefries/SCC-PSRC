import argparse,json
from fvsr.model import resolve_manifest
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--artifact-dir',required=True);a=p.parse_args();m=resolve_manifest(a.manifest,a.artifact_dir);print(json.dumps(dict(status='ARTIFACT_HASHES_PASS',manifest_keys=list(m))))
if __name__=='__main__':main()
