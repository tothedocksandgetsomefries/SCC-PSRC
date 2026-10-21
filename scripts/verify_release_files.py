"""Verify the local draft's exact file inventory; no scientific execution."""
from pathlib import Path
import hashlib, json
from public_files import public_files

def main():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'FILE_MANIFEST.json').read_text(encoding='utf-8'))
    problems=[]
    expected=[r['path'] for r in manifest['files']]
    if len(expected)!=len(set(expected)):
        problems.append(dict(error='duplicate_inventory_path'))
    actual={p.relative_to(root).as_posix() for p in public_files(root)}
    for path in sorted(actual-set(expected)):
        problems.append(dict(path=path,error='unlisted_public_file'))
    for path in sorted(set(expected)-actual):
        problems.append(dict(path=path,error='nonpublic_or_missing_inventory_entry'))
    for r in manifest['files']:
        p=(root/r['path']).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            problems.append(dict(path=r['path'],error='missing_or_outside_root'));continue
        if p.stat().st_size!=r['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:
            problems.append(dict(path=r['path'],error='checksum_mismatch'))
    print(json.dumps(dict(files_checked=len(manifest['files']),problems=problems,
        scope='File integrity only. See RELEASE_STATUS.json for publication readiness.'),indent=2))
    if problems:raise SystemExit(1)

if __name__=='__main__':main()
