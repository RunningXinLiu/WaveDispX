"""Fetch a pinned OFL font on the build machine, verifying Git blob identities."""
import hashlib
import json
from pathlib import Path
import urllib.request

revision = 'a85815a42757630ce188fdad368c2dfc444d4773'
base = f'https://raw.githubusercontent.com/google/fonts/{revision}/ofl/notosanssc/'
files = [
    ('NotoSansSC%5Bwght%5D.ttf', 'NotoSansSC.ttf', 'fb0637bafbcd804fe32152370a1225990745b4bc'),
    ('OFL.txt', 'OFL.txt', '1c9f43281b8f216c5461fe9ac729afbade7724e4'),
]
out = Path(__file__).resolve().parent/'fonts'
out.mkdir(exist_ok=True)
manifest = {'repository': 'https://github.com/google/fonts', 'revision': revision, 'files': {}}
for remote, local, blob in files:
    with urllib.request.urlopen(base+remote, timeout=90) as response:
        data = response.read()
    actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if actual != blob:
        raise RuntimeError(f'Font source identity mismatch: {local}')
    (out/local).write_bytes(data)
    manifest['files'][local] = {'git_blob': blob, 'sha256': hashlib.sha256(data).hexdigest()}
(out/'SOURCE.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
