"""Copy a bounded source-only snapshot for private CI; never copy data/binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--qed', type=Path, required=True)
parser.add_argument('--cps', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise SystemExit('Refusing to overwrite an existing source snapshot')
records = {}

def copy(source, relative):
    destination = args.out / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    records[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()

for folder in ['include', 'src', 'fortran']:
    for p in (args.qed/folder).rglob('*'):
        if p.is_file() and p.suffix in ['.hpp','.h','.cc','.cpp','.f90','.f','.txt']:
            copy(p, Path('qed')/p.relative_to(args.qed))
for dependency in ['fmt','toml11','highfive']:
    base = args.qed/'external'/dependency
    for p in base.rglob('*'):
        if p.is_file() and ('include' in p.relative_to(base).parts or p.name.startswith(('LICENSE','COPYING','COPYRIGHT'))):
            copy(p, Path('qed/external')/dependency/p.relative_to(base))
for name in ['LICENSE','QEDISP_NUMERICS_VERSION.json','README.md']:
    copy(args.qed/name,Path('qed')/name)
for name in ['sprep96','sdisp96','sdpsrf96','sio']:
    copy(args.cps/'VOLIII/src'/f'{name}.f',Path('cps/VOLIII/src')/f'{name}.f')
for name in ['igetmod','lgstr','mchdep','mnmarg','mgtarg','grphsubf']:
    copy(args.cps/'SUBS'/f'{name}.f',Path('cps/SUBS')/f'{name}.f')
for p in (args.cps/'CALPLOT/src/flib').glob('*.f'):
    copy(p,Path('cps')/p.relative_to(args.cps))
for relative in ['flib/dfsubs.c','cmd/dpsubs.c','clib/clcplt.c','clib/grd.c','clib/ponintr.c']:
    p=args.cps/'CALPLOT/src'/relative
    copy(p,Path('cps')/p.relative_to(args.cps))
for p in (args.cps/'CALPLOT/src/include').glob('*.h'):
    copy(p,Path('cps')/p.relative_to(args.cps))
(args.out/'SOURCE_MANIFEST.json').write_text(json.dumps({
    'purpose':'Private build candidate; not a public student release',
    'source_sha256':records,
    'qed_scope':'Existing classroom phase forward source snapshot; no upgrade',
    'distribution_review':'Preserve original notices. QED LICENSE says LGPLv3 while source headers say GPLv3; review before public release. CPS exact snapshot license provenance needs confirmation.'
},indent=2))
print(f'{len(records)} source files, {sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())} bytes')
