"""Relocation smoke test; the packaged executable, not Python, runs the checks."""
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

parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
out = root/'smoke-output'
out.mkdir(exist_ok=True)
release = root/'release'
release.mkdir(exist_ok=True)
name = 'WaveDispX-Classroom'
source = root/'dist'/(name+'.app' if sys.platform == 'darwin' else name)
with tempfile.TemporaryDirectory(prefix='classroom-relocated-') as temp:
    target = Path(temp)/'学生 软件'/source.name
    shutil.copytree(source, target, symlinks=True)
    executable = target/('Contents/MacOS/'+name if sys.platform == 'darwin' else name+('.exe' if os.name == 'nt' else ''))
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'QEDISPINV_FORWARD', 'CPS_BIN', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
        env.pop(key, None)
    # Exclude compiler DLL directories and developer-installed solvers.
    env['PATH'] = str(Path(os.environ['SystemRoot'])/'System32') if os.name == 'nt' else '/usr/bin:/bin'
    # Qt's Windows offscreen plugin has no usable font database; exercise the
    # real Windows platform plugin used by students, including font rendering.
    env['QT_QPA_PLATFORM'] = 'windows' if os.name == 'nt' else 'offscreen'
    done = subprocess.run([str(executable), '--self-test', str(out)], cwd=temp, env=env,
                          capture_output=True, timeout=300)
    (out/'process_stdout.txt').write_bytes(done.stdout)
    (out/'process_stderr.txt').write_bytes(done.stderr)
    if done.returncode:
        print(done.stderr.decode('utf-8', errors='replace'))
        if (out/'self_test.json').exists(): print((out/'self_test.json').read_text(encoding='utf-8'))
        raise SystemExit(done.returncode)
    report = json.loads((out/'self_test.json').read_text(encoding='utf-8'))
    if not report['passed']:
        raise RuntimeError(report)
    # Create a student candidate only after the relocated executable passes.
    if os.name == 'nt':
        package = Path(shutil.make_archive(str(release/(name+'-'+args.label)), 'zip', root_dir=target.parent, base_dir=target.name))
    else:
        package = release/(name+'-'+args.label+'.tar.gz')
        with tarfile.open(package, 'w:gz') as archive:
            archive.add(target, arcname=target.name)
    (release/'SHA256SUMS.txt').write_text(hashlib.sha256(package.read_bytes()).hexdigest()+'  '+package.name+'\n')
print('PASS:', args.label)
