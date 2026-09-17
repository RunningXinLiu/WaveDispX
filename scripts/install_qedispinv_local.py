#!/usr/bin/env python3
"""Copy a complete QEDispInv checkout into WaveDispX's ignored tool directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import shutil
import subprocess


def _build_group_library(destination: Path) -> Path:
    compiler = shutil.which("gfortran")
    if compiler is None:
        raise SystemExit(
            "gfortran is required to build QEDispInv's native group-velocity "
            "helper. Install a Fortran compiler or use --skip-group-library."
        )
    sources = (
        destination / "fortran" / "sregn96.f90",
        destination / "fortran" / "slegn96.f90",
    )
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise SystemExit("QEDispInv group sources are missing: " + ", ".join(missing))
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    output = destination / "lib" / f"libqedisp_group{suffix}"
    subprocess.run(
        [compiler, "-shared", "-fPIC", "-O2", "-o", str(output), *(str(path) for path in sources)],
        check=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="QEDispInv repository containing bin/forward, lib/, and LICENSE",
    )
    parser.add_argument(
        "--skip-group-library",
        action="store_true",
        help="copy QEDispInv without compiling its native group-velocity helper",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    required = (source / "bin" / "forward", source / "lib", source / "LICENSE")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Not a complete QEDispInv checkout; missing: " + ", ".join(missing))

    destination = (
        args.project_root.expanduser().resolve()
        / ".wavedispx-tools"
        / "QEDispInv"
    )
    if destination.exists():
        raise SystemExit(
            f"Destination already exists: {destination}. "
            "Remove or rename it explicitly before reinstalling."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=True)
    if not (destination / "bin" / "forward").is_file():
        raise SystemExit("Copy completed but bin/forward is missing")
    group_library = None
    if not args.skip_group_library:
        group_library = _build_group_library(destination)
    print(destination)
    if group_library is not None:
        print(group_library)


if __name__ == "__main__":
    main()
