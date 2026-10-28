"""Explicit Studio products v1, separate from the legacy all-stage workflow."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from .processing import theoretical_dispersion, surface_wave_dispersion, combine_dispersion_images


@dataclass(frozen=True)
class ExperimentRequest:
    product: str = "legacy"
    waves: str = "both"
    theory_step_hz: float = .1
    hvsr_smoothing_b: float = 40.
    hvsr_floor_ratio: float = 1e-6

    @classmethod
    def from_mapping(cls, mapping):
        raw = mapping.get("experiment", {})
        unknown = set(raw)-set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("未知 experiment 参数："+", ".join(sorted(unknown)))
        r = cls(**raw)
        if r.product not in {"legacy", "theory", "energy", "energy_theory", "hvsr", "energy_hvsr"} or r.waves not in {"rayleigh", "love", "both"}:
            raise ValueError("experiment.product / waves 无效")
        if not np.isfinite(r.theory_step_hz) or r.theory_step_hz <= 0:
            raise ValueError("experiment.theory_step_hz 必须 >0")
        if not np.isfinite(r.hvsr_smoothing_b) or r.hvsr_smoothing_b < 0 or not 0 < r.hvsr_floor_ratio < 1:
            raise ValueError("HVSR: smoothing_b>=0，floor_ratio在(0,1)内")
        return r


@dataclass
class ExperimentResult:
    request: ExperimentRequest
    statuses: dict = field(default_factory=dict)
    images: dict = field(default_factory=dict)
    theory: object = None
    hvsr: object = None
    elapsed_s: float = 0.


def theory_context(mapping, base_directory):
    """Validate only model/theory inputs; acquisition requirements do not apply."""
    from types import SimpleNamespace
    from .config import ModelConfig
    p = mapping["processing"]
    low, high = float(p["frequency_min_hz"]), float(p["frequency_max_hz"])
    if not np.isfinite([low, high]).all() or not 0 < low < high:
        raise ValueError("理论频率必须满足0 < frequency_min_hz < frequency_max_hz")
    modes = {key: int(p.get(key, 3)) for key in ("rayleigh_modes", "love_modes")}
    if any(value < 1 or value != p.get(key, 3) for key,value in modes.items()):
        raise ValueError("理论模态数必须是正整数")
    output = Path(base_directory) / str(mapping.get("run", {}).get("output_directory", "output_theory"))
    config = SimpleNamespace(model=ModelConfig.from_root(mapping), processing=SimpleNamespace(
        frequency_min_hz=low, frequency_max_hz=high, **modes), output_directory=output.resolve(), mode="theory")
    return SimpleNamespace(config=config, geometry=None, shot=None, correlation=None)


def wave_unavailable(config, wave):
    if wave == "rayleigh":
        return None  # Actual amplitude/measurement checks follow the calculation.
    if config.das.enabled:
        return "当前 DAS 路径仅支持垂直源；尚未实现/验收 Love-DAS。可另算 Love 理论曲线，不能替代 DAS 能量。"
    if config.mode == "active" and config.active_source.component == "vertical":
        return "当前主动垂直点力未激发 TT 响应。请选择矢量水平力或旧水平载荷基响应；不自动修改源。"
    if config.mode == "active" and config.active_source.component == "vector" and abs(config.active_source.force_inclination_deg) == 90:
        return "当前矢量源为纯垂直力，没有Love横向激发；可单独计算Love理论曲线。"
    if config.mode == "passive" and config.ambient_sources.component != "three_component":
        return "当前被动流程只保存 ZZ，缺少 TT 相关记录。需要 three_component 的 XYZ 观测和合适的水平源激发。"
    if config.mode == "passive" and config.ambient_sources.polarization != "legacy_xyz" and config.ambient_sources.horizontal_power_fraction == 0:
        return "当前源水平功率占比为0；此模型的垂直点力不激发 Love。TT投影不应被解释为纯Love能量。"
    return None


def run_experiment(workflow, request):
    start = time.perf_counter()
    result = ExperimentResult(request)
    c, p = workflow.config, workflow.config.processing
    if "hvsr" in request.product and (c.mode != "passive" or c.das.enabled or c.ambient_sources.component != "three_component"):
        raise ValueError("节点HVSR需要被动three_component的XYZ观测且关闭DAS；源水平功率可以为0。不从NCF或DAS应变计算HVSR。")
    selected = ("rayleigh", "love") if request.waves == "both" else (request.waves,)
    for wave in ("rayleigh", "love"):
        result.statuses[wave] = "未选择此波型。" if wave not in selected else "尚未计算。"
    wants_energy = request.product in {"legacy", "energy", "energy_theory", "energy_hvsr"}
    wants_theory = request.product in {"legacy", "theory", "energy_theory", "energy_hvsr"}
    if wants_energy and getattr(p, "compare_methods", False):
        raise ValueError("新计算目标暂不支持processing.compare_methods；请设为false，仅使用dispersion_method，或返回旧流程比较方法。")
    if wants_energy:
        available = []
        for wave in selected:
            reason = wave_unavailable(c, wave)
            if reason:
                result.statuses[wave] = reason
            else:
                available.append(wave)
        if available:
            observations = workflow.simulate_observations()
            if c.mode == "active":
                s = workflow.shot
                if s.vector_xyz is not None:
                    from .source_vector import project_green_to_rtz
                    az = np.arctan2(s.receiver_y_m-c.active_source.y_m, s.receiver_x_m-c.active_source.x_m)
                    r,t,z = np.moveaxis(project_green_to_rtz(s.components,az,s.force_xyz),-1,0)
                    raw = {"rayleigh": [(z, "vertical"), (r, "radial")], "love": [(t, "transverse")]}
                else:
                    r = [(s.vertical_vertical, "vertical"), (s.vertical_radial, "radial")]
                    h = [(s.horizontal_vertical, "vertical"), (s.horizontal_radial, "radial")]
                    raw = {"rayleigh": r if c.active_source.component == "vertical" else h if c.active_source.component == "horizontal" else r+h,
                           "love": [(s.horizontal_tangential, "transverse")]}
                if c.das.enabled:
                    raw["rayleigh"] = [(s.das_record.data, "vertical")]
                distances, dt = s.distance_m, c.active_source.dt_s
            else:
                ncfs = observations.component_ncfs
                primary = "das" if c.das.enabled else "zz"
                raw = {"rayleigh": [(ncfs[primary].data, "vertical")], "love": []}
                if "rr" in ncfs:
                    raw["rayleigh"].append((ncfs["rr"].data, "radial"))
                if "tt" in ncfs:
                    raw["love"] = [(ncfs["tt"].data, "transverse")]
                distances, dt = observations.ncf.distance_m, c.noise.dt_s
            velocity = p.velocity_min_mps+p.velocity_step_mps*np.arange(int(round((p.velocity_max_mps-p.velocity_min_mps)/p.velocity_step_mps))+1)
            for wave in available:
                data = raw[wave]
                if not data or not any(np.any(np.abs(a) > 0) for a, _ in data):
                    result.statuses[wave] = "当前观测响应为零或缺少对应分量；请检查源方向与接收方向。未生成有效能量图。"
                    continue
                images = [surface_wave_dispersion(a, distances, dt, velocity, component=component,
                    method=p.dispersion_method, frequency_min_hz=p.frequency_min_hz, frequency_max_hz=p.frequency_max_hz,
                    frequency_step_hz=p.frequency_step_hz, zero_padding=p.zero_padding,
                    distance_taper_fraction=p.distance_taper_fraction, frequency_block_size=c.frequency_block_size,
                    radial_kernel_lookup_step=p.radial_kernel_lookup_step, workers=c.workers) for a, component in data]
                result.images[wave] = combine_dispersion_images(images) if len(images)>1 else images[0]
                result.statuses[wave] = "已计算观测能量；并非纯模态分离，也不代表数值收敛。"
    if wants_theory:
        if result.images:
            frequency = next(iter(result.images.values())).frequency_hz
        else:
            count = int(np.floor((p.frequency_max_hz-p.frequency_min_hz)/request.theory_step_hz))+1
            if count > 10000:
                raise ValueError("理论频率网格超过10000点；请增大experiment.theory_step_hz")
            frequency = p.frequency_min_hz+np.arange(count)*request.theory_step_hz
        result.theory = theoretical_dispersion(c.model, frequency, rayleigh_modes=p.rayleigh_modes if "rayleigh" in selected else 0,
                            love_modes=p.love_modes if "love" in selected else 0, backend="qedisp")
        if request.product == "theory":
            for wave in selected:
                curves = getattr(result.theory, wave+"_mps")
                result.statuses[wave] = ("QED理论曲线已计算；不使用源、阵列或记录时长，不是频散能量。" if np.isfinite(curves).any()
                                        else "本次QED未返回有效理论根；不能仅据此断言模型不存在该波型。")
    if request.product in {"hvsr", "energy_hvsr"}:
        from .hvsr import compute_passive_hvsr
        result.hvsr = compute_passive_hvsr(workflow, smoothing_b=request.hvsr_smoothing_b, floor_ratio=request.hvsr_floor_ratio)
        result.statuses["hvsr"] = ("已计算逐台站原始XYZ谱比；窗间散度不是置信区间。"
            if np.isfinite(result.hvsr.geometric_mean).any() else
            "HVSR无有效频点：请检查垂直谱、源频带和窗长；每频点至少10周期、3个有效窗。不用小常数补出谱比。")
        if request.product == "hvsr":
            for wave in selected:
                result.statuses[wave] = "本次只计算节点HVSR，没有请求频散图；HVSR不是Rayleigh椭圆率。"
    result.elapsed_s = time.perf_counter()-start
    return result


def save_experiment(result, workflow, mapping):
    """New immutable output subdirectory, separate from legacy results.h5."""
    from .app_service import toml_dumps
    directory = Path(workflow.config.output_directory) / ("experiment_"+str(time.time_ns()))
    directory.mkdir(parents=True, exist_ok=False)
    text = toml_dumps(mapping)
    (directory/"input.toml").write_text(text, encoding="utf-8")
    arrays = {}
    for name, image in result.images.items():
        arrays.update({name+"_f": image.frequency_hz, name+"_c": image.velocity_mps, name+"_energy": image.energy})
    if result.theory is not None:
        arrays.update(theory_f=result.theory.frequency_hz, theory_rayleigh=result.theory.rayleigh_mps, theory_love=result.theory.love_mps)
    if result.hvsr is not None:
        arrays.update({"hvsr_"+k: v for k,v in asdict(result.hvsr).items()})
    if workflow.shot is not None:
        shot = workflow.shot
        arrays.update(time_s=shot.time_s, distance_m=shot.distance_m, unit_load_basis=shot.components)
        if shot.vector_xyz is not None:
            arrays.update(vector_xyz=shot.vector_xyz, force_xyz=shot.force_xyz)
    if workflow.geometry is not None:
        arrays.update({"geometry_"+k:v for k,v in asdict(workflow.geometry).items()})
    if workflow.correlation is not None:
        arrays.update(preview_record=workflow.correlation.preview_record, preview_time_s=workflow.correlation.preview_time_s)
        for name,ncf in workflow.correlation.component_ncfs.items():
            arrays.update({name+"_ncf":ncf.data, name+"_lag":ncf.lag_time_s, name+"_distance":ncf.distance_m})
    if result.hvsr is not None and hasattr(workflow, "hvsr_preview"):
        arrays["hvsr_first_window_preview"] = workflow.hvsr_preview
    np.savez_compressed(directory/"arrays.npz", **arrays)
    metadata = {"schema": "wavedispx-experiment/v1", "request": asdict(result.request), "statuses": result.statuses,
                "elapsed_s": result.elapsed_s, "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "arrays_sha256": hashlib.sha256((directory/"arrays.npz").read_bytes()).hexdigest(),
                "coordinate_convention": "+X toward +Y azimuth; Z uses native Green sign; no ENU/NED inference",
                "hvsr_note": "full-duration raw XYZ window log statistics; not NCF; log_std is scatter not confidence; NaN marks invalid bins"}
    code = Path(__file__).parent
    evidence = [code/name for name in ("experiment.py", "source_vector.py", "hvsr.py", "active.py", "noise.py", "passive.py", "config.py")]
    if result.images or result.hvsr is not None:
        from .core import _dwm_core
        evidence.append(Path(_dwm_core.__file__))
    if result.theory is not None:
        from .processing import find_qedisp_forward
        evidence.append(Path(find_qedisp_forward()))
    metadata["implementation_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    (directory/"manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return directory
