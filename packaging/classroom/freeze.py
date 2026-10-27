"""Build one platform on that platform, recording exact source provenance."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
manifest = {
    'status': 'private release candidate; see redistribution notes',
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'platform': platform.platform(), 'python': sys.version,
    'sources_sha256': {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('python_src/wavedispx/classroom', 'packaging/classroom/native', 'cpp')
        for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
    },
}
(root/'packaging/classroom/build_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
                str(root/'packaging/classroom/classroom.spec')], cwd=root, check=True)
