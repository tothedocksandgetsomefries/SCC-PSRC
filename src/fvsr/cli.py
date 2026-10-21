import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description='Frozen filtered-vector-search reference')
    sub=p.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='Run an artificial code-only software fixture')
    demo.add_argument('--execution-mode',choices=['reference','full-cm'],default='reference')
    check=sub.add_parser('verify',help='Verify a supplied artifact manifest');check.add_argument('--manifest',required=True);check.add_argument('--artifact-dir',required=True)
    cfg=sub.add_parser('check-config',help='Validate explicit frozen experiment settings');cfg.add_argument('config')
    a=p.parse_args(argv)
    if a.command=='demo':
        from .demo import run_demo
        value=run_demo(a.execution_mode)
    elif a.command=='verify':
        from .model import resolve_manifest
        value={'status':'ARTIFACT_HASHES_PASS','manifest_keys':list(resolve_manifest(a.manifest,a.artifact_dir))}
    else:
        value=json.loads(Path(a.config).read_text())
        required={'dataset','backend','quantizer','code_bytes','method','nprobe','L','k','protocol_id','evidence_role'}
        if not required<=value.keys():raise ValueError('Missing required config fields: '+str(sorted(required-value.keys())))
        if value['k']<1 or value['L']<value['k'] or value['code_bytes']<1:raise ValueError('Invalid sizes')
        if value['backend']!='ivf_pq':raise NotImplementedError('This executable adapter currently requires ivf_pq')
        value={'status':'CONFIG_SCHEMA_PASS','config':value}
    print(json.dumps(value,indent=2))

if __name__=='__main__':main()
