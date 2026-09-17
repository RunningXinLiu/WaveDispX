#!/usr/bin/env python3
"""Run the MATLAB PS/FH/pFH/FJ kernel-variant benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_src"))

from wavedispx.dispersion_benchmark import (  # noqa: E402
    DEFAULT_CASES,
    MATLAB_KERNEL_METHODS,
    run_controlled_dispersion_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the 16 PhaseShift2_modes3 PS/FH/pFH modes plus J0/Y0 "
            "FJ real/imag/abs variants on the controlled active/passive cases."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output_python" / "dispersion_kernel_benchmark",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES, default=DEFAULT_CASES)
    parser.add_argument("--methods", nargs="+", default=MATLAB_KERNEL_METHODS)
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
    if result.matlab_kernel_atlas_png is not None:
        print(f"kernel atlas: {result.matlab_kernel_atlas_png}")
        print(f"kernel PDF:   {result.matlab_kernel_atlas_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
