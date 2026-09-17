#!/usr/bin/env python3
"""Run the controlled active/passive array-method dispersion benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_src"))

from wavedispx.dispersion_benchmark import (  # noqa: E402
    DEFAULT_CASES,
    DEFAULT_METHODS,
    run_controlled_dispersion_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PS/FJ/FH/FK/tau-p/LS-Radon/HR-Radon/Capon/MUSIC/SPAC/"
            "ESAC from 1 to 24 Hz on controlled active, stationary-phase, "
            "off-stationary-phase, and diffuse cases."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output_python" / "dispersion_method_benchmark",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES, default=DEFAULT_CASES)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_controlled_dispersion_benchmark(
        args.output,
        case_names=args.cases,
        method_names=args.methods,
        seed=args.seed,
        quick=args.quick,
        make_plot=not args.no_plot,
    )
    print(f"metrics: {result.metrics_csv}")
    print(f"arrays:  {result.results_h5}")
    print(f"manifest:{result.manifest_json}")
    if result.summary_png is not None:
        print(f"overview:{result.summary_png}")
        print(f"vector:  {result.summary_pdf}")
        print(f"matrix:  {result.method_matrix_png}")
        print(f"wavefield methods:{result.passive_wavefield_png}")
        print(f"SPAC methods:     {result.passive_spac_png}")
        print(f"all-method atlas: {result.broadband_atlas_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
