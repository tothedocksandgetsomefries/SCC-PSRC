"""Explicit local output scope for the archived trainer; no controller or network access."""
import hashlib,json,os
from pathlib import Path
from .replay_bound import identity,verify
ROOT=None
def configure(root):
    global ROOT
    p=Path(root).resolve();p.mkdir(parents=True,exist_ok=False);ROOT=p
def guard(path,write=False):
    p=Path(path).resolve()
    if write and (ROOT is None or not p.is_relative_to(ROOT)):raise PermissionError('Training output must stay inside the explicit fresh output directory')
    return p
def record(path,role='TRAINING_ARTIFACT'):
    p=guard(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,protected=False,role=role)
def atomic(path,value):
    p=guard(path,True);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def charge(event,key,details=None):
    if ROOT is None:raise PermissionError('Explicit training output scope required')
    with (ROOT/'EXECUTION_LEDGER.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(event=event,key=key,details=details))+'\n')
def stop_requested():return ROOT is None or (ROOT/'STOP_REQUEST').exists()
