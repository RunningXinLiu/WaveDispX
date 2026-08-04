"""Compare active high-mode visibility for a small set of layered models.

This is an intentionally reproducible engineering benchmark, not an optimizer.
It uses the exact production forward/imaging functions and reports normalized
energy sampled on each independent theoretical Rayleigh branch.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from wavedispx.active import analyze_active_shot, simulate_active_shot
from wavedispx.config import Layer, ModelConfig, load_config


ROOT = Path(__file__).resolve().parents[1]


MODELS = {
    "two_layer_20m_220_440": (
        Layer(20.0, 380.0, 220.0, 2000.0),
        Layer(np.inf, 760.0, 440.0, 2000.0),
    ),
    "two_layer_30m_220_440": (
        Layer(30.0, 380.0, 220.0, 2000.0),
        Layer(np.inf, 760.0, 440.0, 2000.0),
    ),
    "strong_contrast_20m_160_600": (
        Layer(20.0, 320.0, 160.0, 1800.0),
        Layer(np.inf, 1050.0, 600.0, 2200.0),
    ),
    "three_layer_5m_180_15m_300_600": (
        Layer(5.0, 340.0, 180.0, 1850.0),
        Layer(15.0, 560.0, 300.0, 2000.0),
        Layer(np.inf, 1050.0, 600.0, 2200.0),
    ),
    "three_layer_10m_180_20m_300_600": (
        Layer(10.0, 340.0, 180.0, 1850.0),
        Layer(20.0, 560.0, 300.0, 2000.0),
        Layer(np.inf, 1050.0, 600.0, 2200.0),
    ),
}


def branch_support(energy, velocity, theory):
    values = []
    for mode in range(theory.shape[1]):
        valid = np.isfinite(theory[:, mode])
        closest = np.abs(velocity[:, None] - theory[None, :, mode]).argmin(axis=0)
        sampled = energy[closest, np.arange(theory.shape[0])][valid]
        values.append(
            (
                float(np.median(sampled)) if sampled.size else np.nan,
                float(np.mean(sampled >= 0.35)) if sampled.size else np.nan,
                int(sampled.size),
            )
        )
    return values


def main() -> None:
    base = load_config(ROOT / "examples/active_high_modes_3c.toml")
    base = replace(
        base,
        processing=replace(base.processing, compare_methods=False),
    )
    print("model,mode,median_energy,fraction_ge_0.35,valid_frequency_count")
    for name, layers in MODELS.items():
        config = replace(base, model=ModelConfig(layers))
        shot = simulate_active_shot(
            config.model,
            config.active_source,
            config.receivers,
            workers=config.workers,
            distance_block_size=config.frequency_block_size * 8,
        )
        analysis = analyze_active_shot(config, shot)
        support = branch_support(
            analysis.rayleigh_combined.energy,
            analysis.rayleigh_combined.velocity_mps,
            analysis.theory.rayleigh_mps,
        )
        for mode, (median, fraction, count) in enumerate(support):
            print(f"{name},{mode},{median:.6f},{fraction:.6f},{count}")


if __name__ == "__main__":
    main()
