"""Command-line launcher for the WaveDispX PySide6 workbench."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the WaveDispX native desktop workbench")
    parser.add_argument("--template", type=Path, help="TOML loaded when the window starts")
    parser.add_argument(
        "--page",
        choices=("parameters", "review", "run", "results", "guide"),
        default="parameters",
        help="initial visible page",
    )
    parser.add_argument("--screenshot", type=Path, help="save a real window screenshot and exit")
    parser.add_argument("--width", type=int, default=1580)
    parser.add_argument("--height", type=int, default=980)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError as exc:  # pragma: no cover - environment diagnostic
        print(
            "WaveDispX Desktop 需要 PySide6。请在项目 Python 环境安装 PySide6 后重试。",
            file=sys.stderr,
        )
        return 2

    from .app import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("WaveDispX")
    app.setOrganizationName("WaveDispX")
    window = MainWindow(args.template)
    window.resize(args.width, args.height)
    if args.page == "results":
        window.refresh_results()
    window.select_page(args.page)
    window.show()

    if args.screenshot:
        destination = args.screenshot.expanduser().resolve()

        def capture() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(destination)):
                print(f"无法保存截图：{destination}", file=sys.stderr)
                app.exit(1)
                return
            print(destination)
            app.exit(0)

        QTimer.singleShot(1200, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
