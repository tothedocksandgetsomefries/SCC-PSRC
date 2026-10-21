"""Public-source inventory rules shared by refresh and verification."""
from pathlib import Path

ROOT_EXCLUSIONS = {
    '.git', '.idea', '.vscode', '.local_audit', 'submission', 'paper',
    'historical_implementation', 'artifacts', 'local_runs', 'data',
}
CACHE_COMPONENTS = {
    '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
    '.venv', 'venv', 'build', 'dist',
}
PRIVATE_SUFFIXES = ('.pyc', '.pyo', '.iml', '.pt', '.pth', '.pyd', '.so',
                    '.faiss', '.index', '.tex', '.bib', '.docx', '.doc',
                    '.aux', '.bbl', '.bcf', '.blg', '.log', '.out',
                    '.run.xml', '.synctex.gz', '.abs', '.pem', '.key')

def is_public_file(relative: Path) -> bool:
    parts = relative.parts
    name = relative.name
    if not parts or parts[0] in ROOT_EXCLUSIONS or relative.as_posix() == 'FILE_MANIFEST.json':
        return False
    if any(p in CACHE_COMPONENTS or p.endswith('.egg-info') for p in parts):
        return False
    if name == '.env' or (name.startswith('.env.') and name != '.env.example'):
        return False
    return not name.endswith(PRIVATE_SUFFIXES)

def public_files(root: Path):
    root = root.resolve()
    for path in sorted(root.rglob('*')):
        if path.is_file() and is_public_file(path.relative_to(root)):
            if not path.resolve().is_relative_to(root):
                raise ValueError('Public file resolves outside the project root')
            yield path
