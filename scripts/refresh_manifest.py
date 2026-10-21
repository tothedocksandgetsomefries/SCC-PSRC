"""Refresh the reviewed public-source inventory."""
from pathlib import Path
import hashlib
import json
from public_files import public_files

root = Path(__file__).resolve().parents[1]
files = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
              sha256=hashlib.sha256(p.read_bytes()).hexdigest())
         for p in public_files(root)]
(root / 'FILE_MANIFEST.json').write_text(json.dumps({
    'scope': 'Source, configuration, documentation and saved numerical views; see RELEASE_STATUS.json',
    'files': files,
}, indent=2) + '\n', encoding='utf-8')
print(f'Manifested {len(files)} files')
