#!/usr/bin/env python3
"""Run/resume a fixed-geometry NNWS benchmark over multiple noise seeds."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from wavedispx.config import load_config
from wavedispx.nnws_qc import extract_seed_summary_rows, write_seed_summary_csv
from wavedispx.runner import run


def _parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start_text, stop_text = value.split(":", 1)
        seeds = list(range(int(start_text), int(stop_text)))
    else:
        seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/passive_nnws_anisotropic_benchmark.toml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output_python/nnws_seed_benchmark"),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=_parse_seeds("314159:314169"),
        help="comma list or Python-style stop-exclusive range, e.g. 314159:314169",
    )
    parser.add_argument(
        "--std-mode",
        choices=("population_eps", "upstream_sample"),
        default="population_eps",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if (
        config.mode != "passive"
        or config.noise is None
        or not (config.nnws.enabled or config.nnws_v2.enabled)
    ):
        parser.error("the benchmark config must enable passive NNWS v1 or v2")
    if config.nnws.enabled and config.nnws_v2.enabled:
        parser.error("seed benchmarking expects exactly one NNWS version enabled")
    method_version = "v1" if config.nnws.enabled else "v2"
    summary_strategies = None
    if method_version == "v2":
        summary_strategies = (
            "baseline_linear",
            "nnws_v2_random_neff_control",
            "nnws_v2_shared_oof",
            "nnws_v2_tau_p_directional_guard_frequency",
            "nnws_v2_tau_p_directional_guard_frequency_permuted_time_control",
            "nnws_v2_tau_p_branch_guard_frequency",
            "nnws_v2_tau_p_branch_guard_frequency_permuted_time_control",
            "nnws_v2_tau_p_diffuse_prearrival_frequency",
            "nnws_v2_tau_p_diffuse_prearrival_frequency_permuted_time_control",
            "pair_crossfit_tau_p_diffuse_prearrival_hard_frequency",
            "pair_crossfit_tau_p_diffuse_prearrival_hard_frequency_"
            "permuted_time_control",
            "pair_crossfit_coherence",
            "pair_crossfit_coherence_permuted_time_control",
            "pair_prearrival_reweighting",
            "pair_prearrival_reweighting_permuted_time_control",
            "pair_crossfit_tau_p_spz_frequency",
            "pair_crossfit_tau_p_spz_frequency_permuted_time_control",
            "pair_crossfit_tau_p_diffuse_soft_frequency",
            "pair_crossfit_tau_p_diffuse_soft_frequency_permuted_time_control",
            "pair_crossfit_tau_p_diffuse_hard_frequency",
            "pair_crossfit_tau_p_diffuse_hard_frequency_permuted_time_control",
            *(
                (
                    "oracle_pair_spz_random_neff_control",
                    "oracle_pair_spz",
                    "nnws_v2_spz_blend_010",
                    "nnws_v2_spz_blend_020",
                    "nnws_v2_spz_blend_030",
                    "nnws_v2_spz_blend_040",
                    "nnws_v2_spz_frequency_blend_010",
                    "nnws_v2_spz_frequency_blend_020",
                    "nnws_v2_spz_frequency_blend_030",
                    "nnws_v2_spz_frequency_blend_040",
                    "nnws_v2_spz_frequency_blend_050",
                    "nnws_v2_spz_phase_hybrid_010",
                    "nnws_v2_spz_phase_hybrid_020",
                    "nnws_v2_spz_phase_hybrid_030",
                    "nnws_v2_spz_phase_hybrid_040",
                    "nnws_v2_spz_phase_hybrid_050",
                    "nnws_v2_spz_phase_hybrid_permuted_time_control_020",
                    "nnws_v2_spz_v2_amplitude_phase_hybrid_020",
                    "nnws_v2_spz_v2_amplitude_phase_hybrid_"
                    "permuted_time_control_020",
                    "nnws_v2_spz_branch_frequency_blend_020",
                    "nnws_v2_spz_branch_frequency_blend_010",
                    "nnws_v2_spz_branch_phase_hybrid_010",
                    "nnws_v2_spz_branch_phase_hybrid_"
                    "permuted_time_control_010",
                    "nnws_v2_spz_branch_phase_hybrid_020",
                    "nnws_v2_spz_branch_v2_amplitude_phase_hybrid_020",
                    "nnws_v2_spz_branch_frequency_permuted_time_control_020",
                    "nnws_v2_spz_branch_phase_hybrid_"
                    "permuted_time_control_020",
                    "nnws_v2_spz_branch_v2_amplitude_phase_hybrid_"
                    "permuted_time_control_020",
                    "supervised_egf_oracle_upper_bound",
                    "supervised_egf_oracle_permuted_time_control",
                    "supervised_pair_egf_oracle_upper_bound",
                    "supervised_pair_egf_oracle_permuted_time_control",
                    "oracle_pair_branch_spz",
                    "oracle_pair_branch_spz_random_neff_control",
                    "nnws_v2_pair_branch_spz_blend_020",
                    "nnws_v2_pair_branch_spz_blend_permuted_time_control_020",
                    "oracle_pair_branch_spz_frequency",
                    "oracle_pair_branch_spz_frequency_random_neff_control",
                    "nnws_v2_pair_branch_spz_frequency_blend_020",
                    "nnws_v2_pair_branch_spz_frequency_blend_"
                    "permuted_time_control_020",
                    "linear_spz_frequency_blend_010",
                    "linear_spz_frequency_blend_020",
                    "linear_spz_frequency_blend_030",
                    "linear_spz_frequency_blend_040",
                    "linear_spz_frequency_blend_050",
                    "nnws_v2_spz_random_neff_control",
                    "nnws_v2_spz_constrained",
                )
                if config.pair_spz.enabled
                else ()
            ),
        )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    completed: list[int] = []
    started = time.perf_counter()
    for run_index, seed in enumerate(args.seeds, start=1):
        run_directory = output_root / f"seed_{seed}"
        manifest_path = run_directory / "manifest.json"
        if args.force or not manifest_path.is_file():
            method_updates = (
                {
                    "nnws": replace(
                        config.nnws,
                        std_mode=args.std_mode,
                        epochs=args.epochs or config.nnws.epochs,
                    )
                }
                if method_version == "v1"
                else {
                    "nnws_v2": replace(
                        config.nnws_v2,
                        epochs=args.epochs or config.nnws_v2.epochs,
                    )
                }
            )
            run_config = replace(
                config,
                name=f"{config.name}_seed_{seed}",
                output_directory=run_directory,
                noise=replace(config.noise, seed=seed),
                config_path=None,
                **method_updates,
            )
            method_config = (
                run_config.nnws if method_version == "v1" else run_config.nnws_v2
            )
            print(
                f"[{run_index}/{len(args.seeds)}] running noise seed {seed} "
                f"({method_version}, {method_config.epochs} epochs)",
                flush=True,
            )
            run(run_config)
        else:
            print(
                f"[{run_index}/{len(args.seeds)}] reusing completed noise seed {seed}",
                flush=True,
            )
        if summary_strategies is None:
            all_rows.extend(extract_seed_summary_rows(run_directory, seed))
        else:
            all_rows.extend(
                extract_seed_summary_rows(
                    run_directory, seed, strategies=summary_strategies
                )
            )
        completed.append(seed)
        write_seed_summary_csv(all_rows, output_root / "seed_metrics.csv")

    metadata = {
        "base_config": str(args.config.resolve()),
        "output_root": str(output_root),
        "noise_seeds": completed,
        "varied_parameter": "noise.seed only",
        "method_version": method_version,
        "pair_spz_enabled": config.pair_spz.enabled,
        "fixed_nnws_optimizer_seed": (
            config.nnws.seed if method_version == "v1" else config.nnws_v2.seed
        ),
        "std_mode": args.std_mode if method_version == "v1" else None,
        "epochs": args.epochs
        or (
            config.nnws.epochs
            if method_version == "v1"
            else config.nnws_v2.epochs
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "summary_csv": str((output_root / "seed_metrics.csv").resolve()),
    }
    (output_root / "benchmark_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"completed {len(completed)} seeds in {metadata['elapsed_seconds']:.1f} s; "
        f"summary: {output_root / 'seed_metrics.csv'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
