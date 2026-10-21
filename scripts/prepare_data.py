"""Verify pinned source files, or explicitly retrieve them from their source."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--data-dir',required=True,type=Path);p.add_argument('--download',action='store_true');a=p.parse_args();m=json.loads(a.manifest.read_text());root=a.data_dir.resolve();root.mkdir(parents=True,exist_ok=True);rows=[]
    for r in m['files']:
        target=(root/r['path']).resolve()
        if not target.is_relative_to(root):raise ValueError('Data path escapes declared directory')
        if not target.exists() and a.download:
            target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix(target.suffix+'.partial')
            if tmp.exists():raise FileExistsError(tmp)
            with urllib.request.urlopen(r['url']) as src,tmp.open('xb') as out:
                while True:
                    b=src.read(8<<20)
                    if not b:break
                    out.write(b)
            if sha(tmp)!=r['sha256']:raise ValueError('Downloaded hash mismatch; partial file retained')
            tmp.rename(target)
        if not target.exists():raise FileNotFoundError(str(target)+'; supply the source file or explicitly use --download')
        actual=sha(target)
        if actual!=r['sha256']:raise ValueError('Source hash mismatch: '+r['path'])
        rows.append(dict(file=r['path'],sha256=actual,bytes=target.stat().st_size))
    print(json.dumps(dict(status='PINNED_DATA_BYTES_VERIFIED',revision=m['revision'],files=rows,GT_computed=False,protected_query_feedback=False),indent=2))
if __name__=='__main__':main()
