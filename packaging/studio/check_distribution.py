"""Test relocated Studio with no developer runtime, then archive only a pass."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

parser=argparse.ArgumentParser()
parser.add_argument('--label',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
out=root/'studio-smoke-output'
out.mkdir(exist_ok=True)
release=root/'studio-release'
release.mkdir(exist_ok=True)
name='WaveDispX-Studio'
source=root/'dist'/(name+'.app' if sys.platform=='darwin' else name)
with tempfile.TemporaryDirectory(prefix='studio-relocated-') as temp:
    target=Path(temp)/'用户 软件'/source.name
    shutil.copytree(source,target,symlinks=True)
    executable=target/('Contents/MacOS/'+name if sys.platform=='darwin' else name+('.exe' if os.name=='nt' else ''))
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','QEDISPINV_FORWARD','QEDISPINV_GROUP_LIBRARY','CPS_BIN','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH','NUMBA_CACHE_DIR'):
        env.pop(key,None)
    env['PATH']=str(Path(os.environ['SystemRoot'])/'System32') if os.name=='nt' else '/usr/bin:/bin'
    env['QT_QPA_PLATFORM']='windows' if os.name=='nt' else 'offscreen'
    done=subprocess.run([str(executable),'--self-test',str(out)],cwd=temp,env=env,capture_output=True,timeout=1800)
    (out/'process_stdout.txt').write_bytes(done.stdout)
    (out/'process_stderr.txt').write_bytes(done.stderr)
    if done.returncode:
        print(done.stderr.decode('utf-8',errors='replace'))
        if (out/'self_test.json').exists(): print((out/'self_test.json').read_text(encoding='utf-8'))
        raise SystemExit(done.returncode)
    report=json.loads((out/'self_test.json').read_text(encoding='utf-8'))
    assert report['passed'],report
    if os.name=='nt':
        package=Path(shutil.make_archive(str(release/(name+'-'+args.label)),'zip',root_dir=target.parent,base_dir=target.name))
    else:
        package=release/(name+'-'+args.label+'.tar.gz')
        with tarfile.open(package,'w:gz') as archive: archive.add(target,arcname=target.name)
    (release/'SHA256SUMS.txt').write_text(hashlib.sha256(package.read_bytes()).hexdigest()+'  '+package.name+'\n')
print('PASS',args.label)
