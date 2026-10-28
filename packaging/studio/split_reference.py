"""Lossless per-case storage of the server baseline for reliable source transfer."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).parent/'reference'
source=root/'baseline.npz'
provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
assert hashlib.sha256(source.read_bytes()).hexdigest()==provenance['baseline_sha256']
manifest={'source_baseline_sha256':provenance['baseline_sha256'],'cases':{}}
with np.load(source,allow_pickle=False) as original:
    for name in provenance['cases']:
        path=root/(name+'.npz')
        assert not path.exists(),path
        fields={k.split('__',1)[1]:original[k] for k in original.files if k.startswith(name+'__')}
        np.savez_compressed(path,**fields)
        with np.load(path,allow_pickle=False) as saved:
            for k,v in fields.items(): np.testing.assert_array_equal(saved[k],v)
        manifest['cases'][name]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        print(name,path.stat().st_size)
(root/'FILES.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
