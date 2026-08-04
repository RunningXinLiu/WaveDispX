"""Launch the local Streamlit application with the current Python interpreter."""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is not installed. Run: "
            "conda run -n seisloc python -m pip install -e '.[ui]' --no-build-isolation"
        ) from exc
    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
