"""Explicit elastic model contract, native CPS comparison and ideal f-c response."""
from __future__ import annotations

import csv
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from wavedispx.config import Layer, ModelConfig
from wavedispx.core import global_stiffness
from wavedispx.forward import _model_arrays
from wavedispx.processing.qedisp import find_qedisp_forward, run_qedisp_forward
from .runtime import bundled_solver

COLUMNS = ('thickness_m', 'vp_mps', 'vs_mps', 'density_kgm3', 'qp', 'qs')
PRESETS = {
    '两层：软覆盖层': [(10, 750, 320, 1800), (np.inf, 1700, 750, 2150)],
    '四层：低速夹层': [(10, 750, 320, 1800), (10, 600, 220, 1750),
                     (20, 1100, 450, 1950), (np.inf, 1700, 750, 2150)],
    '均匀半空间：极限检查': [(np.inf, 1700, 750, 2150)],
}


def validate_model(model):
    if not 1 <= len(model.layers) <= 40:
        raise ValueError('教学版支持 1–40 层（含最后的半空间）。')
    for i, layer in enumerate(model.layers):
        h, vp, vs, rho, qp, qs = (getattr(layer, k) for k in COLUMNS)
        if not np.all(np.isfinite([vp, vs, rho])) or min(vp, vs, rho) <= 0:
            raise ValueError(f'第 {i+1} 层：Vp、Vs、密度必须是有限正数；不支持流体。')
        if vp * vp <= (4 / 3) * vs * vs:
            raise ValueError(f'第 {i+1} 层：正体积模量要求 Vp > sqrt(4/3) × Vs。')
        if i == len(model.layers)-1:
            if h != np.inf:
                raise ValueError('最后一层厚度必须是 inf（半空间）。')
        elif not np.isfinite(h) or h <= 0:
            raise ValueError(f'第 {i+1} 层：有限层厚度必须为正数。')
        if qp != np.inf or qs != np.inf:
            raise ValueError('此版本为无耗散弹性介质，Qp、Qs 必须为 inf；不会忽略有限 Q。')
    return model


def model_from_rows(rows):
    return validate_model(ModelConfig(tuple(Layer(*map(float, row)) for row in rows)))


def model_csv(model):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(COLUMNS)
    writer.writerows([[getattr(layer, key) for key in COLUMNS] for layer in model.layers])
    return stream.getvalue()


def read_model(text):
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    if reader.fieldnames != list(COLUMNS):
        raise ValueError('CSV 表头必须按顺序为：' + ','.join(COLUMNS))
    rows = []
    for row in reader:
        if None in row or any(row[k] is None for k in COLUMNS):
            raise ValueError('CSV 每行必须恰好有六列。')
        rows.append([row[k] for k in COLUMNS])
    return model_from_rows(rows)


def axis(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0) or np.any(np.diff(values) <= 0):
        raise ValueError(f'{name} 必须是严格递增的有限正数。')
    return values


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cps_programs(directory=None):
    directory = directory or os.environ.get('CPS_BIN')
    paths = {}
    for name in ('sprep96', 'sdisp96', 'sdpsrf96'):
        filename = name + ('.exe' if os.name == 'nt' else '')
        path = Path(directory) / filename if directory else (bundled_solver(name) or Path(shutil.which(name) or filename))
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f'找不到原生 CPS {name}；请设置 CPS_BIN 或在界面指定目录。')
        paths[name] = path.resolve()
    return paths


def cps_curves(model, frequency, modes, directory=None):
    validate_model(model)
    frequency = axis(frequency, 'frequency_hz')
    if not isinstance(modes, int) or not 1 <= modes <= 12:
        raise ValueError('模态数必须为 1–12，含基阶。')
    programs = cps_programs(directory)
    logs = []
    with tempfile.TemporaryDirectory(prefix='wavedispx-classroom-cps-') as tmp:
        work = Path(tmp)
        header = ['MODEL.01', 'WaveDispX classroom elastic SI input converted to km',
                  'ISOTROPIC', 'KGS', 'FLAT EARTH', '1-D', 'CONSTANT VELOCITY',
                  'LINE08', 'LINE09', 'LINE10', 'LINE11',
                  'H(KM) VP(KM/S) VS(KM/S) RHO(GM/CC) QP QS ETAP ETAS FREFP FREFS']
        for layer in model.layers:
            h = 0 if np.isinf(layer.thickness_m) else layer.thickness_m / 1000
            header.append(f'{h:.12g} {layer.vp_mps/1000:.12g} {layer.vs_mps/1000:.12g} {layer.density_kgm3/1000:.12g} 0 0 0 0 1 1')
        (work / 'model.mod').write_text('\n'.join(header)+'\n')
        np.savetxt(work / 'frequencies.txt', frequency, fmt='%.12g')
        commands = [('sprep96', '-M', 'model.mod', '-FARR', 'frequencies.txt', '-NMOD', str(modes), '-R', '-L'),
                    ('sdisp96',), ('sdpsrf96', '-R', '-ASC'), ('sdpsrf96', '-L', '-ASC')]
        for name, *args in commands:
            done = subprocess.run([str(programs[name]), *args], cwd=work, text=True,
                                  capture_output=True, timeout=120, check=True,
                                  stdin=subprocess.DEVNULL,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            logs.append({'command': [name, *args], 'stdout': done.stdout, 'stderr': done.stderr})
        curves = {}
        for wave, filename in [('rayleigh', 'SDISPR.ASC'), ('love', 'SDISPL.ASC')]:
            result = np.full((len(frequency), modes), np.nan)
            path = work / filename
            if not path.is_file():
                raise RuntimeError(f'CPS 未生成 {filename}；不能将程序失败标成无模态。')
            for line in path.read_text().splitlines()[1:]:
                fields = line.split()
                if not fields:
                    continue
                mode, _, _, f, c = map(float, fields)
                index = int(np.argmin(abs(frequency - f)))
                if not np.isclose(frequency[index], f, rtol=2e-6, atol=1e-7):
                    raise RuntimeError('CPS 输出频率与输入不匹配。')
                if not 0 <= int(mode) < modes or c <= 0 or not np.isfinite(c):
                    raise RuntimeError('CPS 输出模态/速度无效。')
                if np.isfinite(result[index, int(mode)]):
                    raise RuntimeError('CPS 输出重复频率/模态。')
                result[index, int(mode)] = c * 1000
            curves[wave] = result
    return curves, logs, {name: {'path': str(path), 'sha256': sha(path)} for name, path in programs.items()}


def ideal_response(model, frequency, velocity, gamma, progress=None):
    """K(k, omega-i*gamma) u = e; k=2*pi*f/c. Unit surface traction.

    Returns unnormalised complex VV and TT surface compliance; no spatial
    aperture, source spectrum, modal weighting, taper or curve broadening.
    """
    validate_model(model)
    frequency, velocity = axis(frequency, 'frequency_hz'), axis(velocity, 'velocity_mps')
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('gamma 必须 > 0；严格零阻尼极点不能用有限像素图表示。')
    if len(frequency)*len(velocity) > 200_000:
        raise ValueError('教学版单次最多 200000 个 f-c 点。大网格请在服务器运行。')
    arrays = _model_arrays(model)
    rayleigh = np.empty((len(frequency), len(velocity)), complex)
    love = np.empty_like(rayleigh)
    for i, f in enumerate(frequency):
        for j, c in enumerate(velocity):
            sh, psv = global_stiffness(*arrays, 2*np.pi*f/c, complex(2*np.pi*f, -gamma))
            load_r = np.zeros(len(psv)); load_r[1] = 1
            load_l = np.zeros(len(sh)); load_l[0] = 1
            rayleigh[i, j] = np.linalg.solve(psv, load_r)[1]
            love[i, j] = np.linalg.solve(sh, load_l)[0]
        if progress:
            progress(f'理想响应：{i+1}/{len(frequency)} 个频率')
    if not np.all(np.isfinite(rayleigh)) or not np.all(np.isfinite(love)):
        raise RuntimeError('理想响应含非有限值，请检查数值设置；未绘制替代图。')
    return {'rayleigh': rayleigh, 'love': love}


def calculate(model, frequency, modes, velocity, gamma, energy=False,
              cps_bin=None, qedisp=None, progress=None, backend='qedisp'):
    """Run only the selected curve backend. An unselected backend is never probed."""
    validate_model(model)
    frequency, velocity = axis(frequency, 'frequency_hz'), axis(velocity, 'velocity_mps')
    if not isinstance(modes, int) or not 1 <= modes <= 12:
        raise ValueError('模态数必须为 1–12，含基阶。')
    if len(frequency) > 501:
        raise ValueError('教学版最多 501 个频率点。')
    if backend not in ('qedisp', 'cps'):
        raise ValueError('backend 必须是 qedisp 或 cps。')
    tick = time.perf_counter()
    logs = []
    if backend == 'qedisp':
        if progress: progress('QEDispInv：计算 Rayleigh / Love 频散曲线')
        executable = find_qedisp_forward(qedisp or (None if os.environ.get('QEDISPINV_FORWARD') else bundled_solver('qedisp')))
        native = run_qedisp_forward(model, frequency, rayleigh_modes=modes,
                                    love_modes=modes, executable=executable)
        curves = {'rayleigh': native.rayleigh_mps, 'love': native.love_mps}
        binaries = {'qedisp': {'path': str(executable), 'sha256': sha(executable)}}
    else:
        if progress: progress('CPS：计算 Rayleigh / Love 频散曲线')
        curves, logs, binaries = cps_curves(model, frequency, modes, cps_bin)
    result = dict(model=model, frequency=frequency, velocity=velocity, modes=modes,
                  gamma=gamma, backend=backend, curves=curves, response=None,
                  binaries=binaries, logs=logs,
                  timings={'curves_s': time.perf_counter()-tick, 'ideal_s': 0.0})
    return add_ideal(result, velocity, gamma, progress) if energy else result


def add_ideal(result, velocity, gamma, progress=None):
    """Compute an optional overlay without re-evaluating or mutating any curves."""
    tick = time.perf_counter()
    velocity = axis(velocity, 'velocity_mps')
    response = ideal_response(result['model'], result['frequency'], velocity, gamma, progress)
    from wavedispx.core import _dwm_core
    return {**result, 'velocity': velocity, 'gamma': gamma, 'response': response,
            'timings': {**result['timings'], 'ideal_s': time.perf_counter()-tick},
            'binaries': {**result['binaries'], 'wavedispx_core': {
                'path': _dwm_core.__file__, 'sha256': sha(_dwm_core.__file__)}}}
