#!/usr/bin/env python3
"""Generate source, weight, waveform, dispersion, and multi-seed NNWS QC."""

from __future__ import annotations

import argparse
from pathlib import Path

from wavedispx.nnws_qc import (
    plot_dispersion_qc,
    plot_multiseed_summary,
    plot_ncf_gathers,
    plot_source_and_weight_qc,
    plot_waveform_overlays,
    write_effectiveness_gate,
    write_qc_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path("output_python/passive_nnws_anisotropic_benchmark"),
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=None)
    args = parser.parse_args()

    output_directory = args.output_directory or args.run_directory / "qc"
    outputs: list[Path] = []
    outputs.extend(plot_source_and_weight_qc(args.run_directory, output_directory))
    outputs.extend(plot_waveform_overlays(args.run_directory, output_directory))
    outputs.extend(plot_ncf_gathers(args.run_directory, output_directory))
    outputs.extend(plot_dispersion_qc(args.run_directory, output_directory))
    if args.summary_csv is not None:
        outputs.extend(plot_multiseed_summary(args.summary_csv, output_directory))
    gate = write_effectiveness_gate(
        args.run_directory, output_directory / "nnws_effectiveness_gate.json"
    )
    manifest = write_qc_manifest(
        output_directory,
        outputs,
        run_directory=args.run_directory,
        summary_csv=args.summary_csv,
    )
    print(f"wrote {len(outputs)} figure files, {gate}, and {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
