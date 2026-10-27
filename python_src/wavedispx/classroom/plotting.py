"""Display-only figures and inspectable export; raw arrays are never normalised."""
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from .engine import model_csv


def draw_model(figure, model):
    """One depth plot: Vp/Vs on the bottom axis, density on a separate top axis."""
    figure.clear()
    figure.set_layout_engine('constrained')
    ax = figure.subplots()
    density_ax = ax.twiny()
    bottom = max(20.0, sum(x.thickness_m for x in model.layers[:-1]) * 1.35)
    lines = []
    for name, color, label, target, style in [
        ('vp_mps', '#d97706', 'Vp', ax, '-'),
        ('vs_mps', '#167b9c', 'Vs', ax, '-'),
        ('density_kgm3', '#9255a6', 'ρ', density_ax, '--'),
    ]:
        x, z, depth = [], [], 0.0
        for layer in model.layers:
            end = bottom if np.isinf(layer.thickness_m) else depth + layer.thickness_m
            x.extend([getattr(layer, name)]*2); z.extend([depth, end]); depth = end
        lines.extend(target.plot(x, z, label=label, color=color, lw=2, ls=style))
    ax.set(xlabel='Vp / Vs (m/s)', ylabel='Depth (m)', ylim=(bottom, 0))
    density_ax.set_xlabel('Density ρ (kg/m³)', color='#9255a6', labelpad=9)
    density_ax.tick_params(axis='x', colors='#9255a6', labelsize=8)
    density_ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    halfspace_top = sum(x.thickness_m for x in model.layers[:-1])
    ax.axhspan(halfspace_top, bottom, color='#64748b', alpha=.06)
    ax.text(.03, .02, 'Bottom: half-space', transform=ax.transAxes, fontsize=8, color='#64748b')
    ax.grid(alpha=.18)
    ax.legend(handles=lines, loc='best', fontsize=9)


def draw_wave(figure, result, wave, overlay=False, visible_modes=None):
    figure.clear()
    figure.set_layout_engine('constrained')
    # Reserve the same colourbar space in both states; overlay cannot move axes.
    grid = figure.add_gridspec(1, 2, width_ratios=[1, .035])
    ax = figure.add_subplot(grid[0, 0])
    color_ax = figure.add_subplot(grid[0, 1]); color_ax.set_visible(False)
    f, c = result['frequency'], result['velocity']
    show_energy = overlay and result['response'] is not None
    if show_energy:
        power = abs(result['response'][wave])**2
        reference = np.max(power, axis=1, keepdims=True)
        db = 10*np.log10(np.maximum(power/np.maximum(reference,np.finfo(float).tiny),1e-8))
        image = ax.pcolormesh(f,c,db.T,cmap='magma',vmin=-40,vmax=0,shading='auto',rasterized=True)
        color_ax.set_visible(True)
        figure.colorbar(image,cax=color_ax,label='Power (dB)')
    colors = ['#0072b2','#009e73','#cc79a7','#e69f00','#56b4e9','#d55e00']
    n = min(visible_modes or result['modes'], result['modes'])
    stats=[]
    from matplotlib import patheffects
    for m in range(n):
        curve = result['curves'][wave][:,m]
        line, = ax.plot(f,curve,color=colors[m%len(colors)],lw=1.9,label=f'M{m}',zorder=3)
        if show_energy:
            line.set_path_effects([patheffects.Stroke(linewidth=3.1,foreground='white',alpha=.8),patheffects.Normal()])
        stats.append(f'M{m}：{np.isfinite(curve).sum()}/{len(f)} 个有效根')
    backend = 'QEDispInv' if result['backend']=='qedisp' else 'CPS'
    ax.set(xlabel='Frequency (Hz)',ylabel='Phase velocity (m/s)',
           title=f'{wave.title()} · {backend}',xlim=(f[0],f[-1]),ylim=(c[0],c[-1]))
    ax.legend(loc='best',fontsize=9,ncol=min(n,4))
    ax.grid(alpha=.15)
    return '  ·  '.join(stats)


def export_result(path, result):
    payload={'model.csv':model_csv(result['model']).encode()}
    arrays={'frequency_hz':result['frequency'],'velocity_mps':result['velocity']}
    for wave in ('rayleigh','love'):
        stream=io.StringIO(); writer=csv.writer(stream)
        writer.writerow(['frequency_hz','mode','phase_velocity_mps','backend'])
        curves=result['curves'][wave]
        for i,f in enumerate(result['frequency']):
            for m in range(result['modes']):
                writer.writerow([f,m,curves[i,m],result['backend']])
        payload[wave+'_curves.csv']=stream.getvalue().encode()
        arrays[wave+'_phase_velocity_mps']=curves
        if result['response'] is not None:
            arrays[wave+'_complex_compliance']=result['response'][wave]
            arrays[wave+'_raw_power']=abs(result['response'][wave])**2
        for overlay in ([False,True] if result['response'] is not None else [False]):
            fig=Figure(figsize=(10,7)); draw_wave(fig,result,wave,overlay)
            buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=140)
            payload[wave+('_ideal' if overlay else '_curves')+'.png']=buf.getvalue()
    fig=Figure(figsize=(4.5,6)); draw_model(fig,result['model'])
    buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=140); payload['model.png']=buf.getvalue()
    buf=io.BytesIO(); np.savez_compressed(buf,**arrays); payload['arrays.npz']=buf.getvalue()
    payload['solver_logs.json']=json.dumps(result['logs'],ensure_ascii=False,indent=2).encode()
    config={'schema':'wavedispx-classroom/2','backend':result['backend'],'ideal_operator':'wavedispx-classroom/1','physics':'isotropic solid, flat layers, elastic, free surface, half-space',
            'frequency_hz':result['frequency'].tolist(),'velocity_mps':result['velocity'].tolist(),
            'mode_count_including_fundamental':result['modes'],'gamma_s_inv':result['gamma'],
            'ideal_enabled':result['response'] is not None,
            'operator':'k=2*pi*f/c; K(k,2*pi*f-i*gamma)u=e; surface unit traction; VV and TT; P=abs(u)^2',
            'display':'10log10(P/max_c P) per frequency, clipped [-40,0] dB; no root masks',
            'boundary':'no source radius, taper, waveforms or receiver array; gamma is not material Q',
            'mode_numbering':'backend root ordinal, zero-based; no branch tracking',
            'status':'teaching diagnostic; no universal numerical validation'}
    payload['config.json']=json.dumps(config,ensure_ascii=False,indent=2).encode()
    manifest={'binaries':result['binaries'],'timings':result['timings'],'platform':platform.platform(),
              'python':platform.python_version(),
              'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
              'files_sha256':{key:hashlib.sha256(value).hexdigest() for key,value in payload.items()}}
    from .runtime import resource_root
    build_manifest = resource_root()/'provenance/build_manifest.json'
    if build_manifest.is_file():
        manifest['build'] = json.loads(build_manifest.read_text(encoding='utf-8'))
    payload['manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as out:
        for name,content in payload.items(): out.writestr(name,content)
