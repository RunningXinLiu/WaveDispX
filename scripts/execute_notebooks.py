"""Execute WaveDispX notebooks without reading machine-wide nbconvert plugins."""

from __future__ import annotations

import argparse
from pathlib import Path

from nbclient import NotebookClient
import nbformat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOKS = (
    ROOT / "notebooks/WaveDispX_workflow.ipynb",
    ROOT / "notebooks/01_active_quickstart.ipynb",
    ROOT / "notebooks/02_passive_quickstart.ipynb",
    ROOT / "notebooks/03_methods_and_validation.ipynb",
    ROOT / "notebooks/04_das_active_passive.ipynb",
)


def execute(path: Path, *, timeout: int) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute(cwd=str(ROOT))
    nbformat.write(notebook, path)
    print(f"executed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    targets = tuple(path.resolve() for path in args.notebooks) or DEFAULT_NOTEBOOKS
    for path in targets:
        execute(path, timeout=args.timeout)


if __name__ == "__main__":
    main()
